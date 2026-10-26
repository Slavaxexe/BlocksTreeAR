"""Post-search stability checks: fixed choices, common splits, no LLM feedback."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time
from contextlib import nullcontext

import numpy as np
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from .contracts import Preparation
from .data import file_hash, within
from .finalize import FinalRefit
from .storage import load, save
from .validation import split_plan, split_rows


def extra_splits(frame, available, policy, targets, count=2):
    """Preserve the chosen strategy and original row positions, excluding outer validation."""
    plan = split_plan(policy, targets)
    source = frame.iloc[available].reset_index(drop=True)
    splits, signatures = [], set()
    for number in range(count):
        positions = np.arange(len(source))
        candidate = plan.model_copy(update={'seed': plan.seed + 104729 * (number + 1),
                                            'cutoff': None, 'validation_rows': None})
        if plan.strategy in {'time', 'group_time'}:
            from .supervised_features import ordered_time
            times = ordered_time(source[plan.time_column])
            periods = np.unique(times)
            if len(periods) - number < 2:
                break
            positions = positions[times <= periods[-1 - number]]
            candidate = candidate.model_copy(update={'validation_periods': plan.validation_periods or 1})
        elif plan.strategy == 'ordered':
            positions = positions[:int(len(source) * (1 - plan.fraction) ** number)]
        try:
            train, valid = split_rows(source.iloc[positions].reset_index(drop=True), candidate, targets)
        except ValueError:
            continue
        train, valid = available[positions[train]], available[positions[valid]]
        signature = tuple(valid)
        if signature not in signatures:
            signatures.add(signature)
            splits.append((train, valid))
    return splits


def supported(root, node_id, seen=None):
    """Avoid replay paths which would load stored training labels or run new searches."""
    from .model_evidence import frozen_available
    seen = set() if seen is None else seen
    if node_id in seen:
        return True
    seen.add(node_id)
    source = root / 'nodes' / node_id
    manifest = json.loads((source / 'assembly.json').read_text(encoding='utf-8'))
    for block in manifest['blocks']:
        call = block.get('operation')
        if call is None:
            return False
        if call['kind'] == 'automl':
            facts = json.loads((source / f"{call['parameters']['engine']}_full_diagnostics.json").read_text(encoding='utf-8'))
            if not frozen_available(facts):
                return False
        if call['kind'] == 'saved_ensemble' and not all(
                supported(root, node, seen) for node in call['parameters']['source_nodes']):
            return False
    return True


class FoldRefit(FinalRefit):
    def __init__(self, root, train_rows, valid_rows, destination, seconds):
        super().__init__(root, seconds)
        self.train_rows = train_rows
        self.reference = self.train.iloc[valid_rows].reset_index(drop=True).copy()
        self.train = self.train.iloc[train_rows].copy()
        self.test = self.reference.drop(columns=self.preparation.data.target_columns).copy()
        self.destination = destination
        runtime = self.metadata['runtime']
        runtime.update(run_directory=str(destination), work_directory=str(destination),
            is_final_refit=False, is_validation_recheck=True, fit_row_positions=train_rows)

    def ensemble(self, source, plan, state):
        from .ensemble_data import make_split
        from .ensembles import blend, learned
        from .operations import _classification
        if plan.method not in {'stacking', 'gating'} and not plan.optimize_weights:
            return super().ensemble(source, plan, state)
        # Refit combiners inside this fold. The old combiner may have seen its labels.
        classification = _classification(state, self.defaults)
        train, meta = make_split(self.train, self.preparation.data.target_columns,
                                 self.preparation.validation, classification)
        inner = FoldRefit(self.root, self.train_rows[train], self.train_rows[meta],
                          self.destination / 'inner' / source.name, max(1., self.deadline - time.monotonic()))
        pairs = []
        for node_id in plan.source_nodes:
            held_out = inner.run_node(node_id)['test']
            held_out.attrs['meta_held_out'] = True
            pairs.append({**self.run_node(node_id), 'meta': held_out})
        targets = self.preparation.data.target_columns
        state = {**state, 'auxiliary': {'ensemble_meta': self.train.iloc[meta].reset_index(drop=True)}}
        if plan.optimize_weights:
            from .weight_search import optimize
            metadata = {**state['preparation'], 'runtime': {**state['preparation']['runtime'],
                        'run_directory': str(self.root)}}
            weights = optimize(pairs, {**state, 'preparation': metadata}, self.defaults,
                               plan.model_copy(update={'weights': []}))
            return blend(pairs, state, self.defaults, weights, plan.method)
        classes = LabelEncoder().fit(self.train[targets[0]]).classes_ if classification else None
        return learned(pairs, state, self.defaults, plan, classes)


def evaluate(root, node_id, split, output, seconds, work_directory=None):
    # Only a score/row provenance survives; replay models and native AutoML files are disposable.
    from .task_types import score_predictions
    if not supported(root, node_id):
        raise ValueError('This saved pipeline has no safe fixed-choice replay')
    workspace = nullcontext(str(work_directory)) if work_directory else tempfile.TemporaryDirectory(prefix='recheck-')
    with workspace as temporary:
        replay = FoldRefit(root, split['train'], split['valid'], Path(temporary), seconds)
        prediction = replay.run_node(node_id)['test']
        value = score_predictions(replay.reader, replay.reference, prediction, replay.config,
                                  replay.defaults.get('task_type'))
        if len(prediction) != len(replay.reference) or not np.isfinite(value):
            raise ValueError('Recheck produced invalid predictions or metric')
        save(output, {'node_id': node_id, 'status': 'completed', 'value': float(value),
                      'fit_rows': len(split['train']), 'validation_rows': len(split['valid'])})


def choose(incumbent, candidates, scores, direction, folds):
    """Promote only a challenger consistent on every extra split and better overall."""
    complete = {row['node_id']: scores[row['node_id']] for row in candidates
                if len(scores.get(row['node_id'], [])) == folds
                and all(value is not None and np.isfinite(value) for value in scores[row['node_id']])}
    if folds < 2 or incumbent not in complete:
        return incumbent
    sign = 1 if direction == 'minimize' else -1
    original = {row['node_id']: row['value'] for row in candidates}
    overall = lambda node: sign * np.mean([original[node], *complete[node]])
    eligible = [node for node in complete if all(sign * a <= sign * b
                for a, b in zip(complete[node], complete[incumbent]))]
    best = min(eligible, key=overall)
    return best if overall(best) < overall(incumbent) else incumbent


async def recheck(root, delta, options, process, deadline, progress):
    incumbent = delta.best_node_id
    destination = root / 'final_validation'
    destination.mkdir(exist_ok=True)
    report = {'status': 'skipped', 'search_best_node': incumbent, 'selected_node': incumbent,
        'scope': 'Repeated validation inside original fit; not an untouched test or unbiased performance estimate.',
        'selection_rule': 'No worse on every extra split and better mean including the original validation score.',
        'candidates': [], 'splits': [], 'errors': []}
    save(destination / 'report.json', report)
    preparation = load(root / 'preparation.json', Preparation)
    from .data import module
    reader = module(root / 'prepare.py', preparation.code_sha256)
    tables = reader.load_data(Path(preparation.task_directory), preparation.model_dump(mode='json'))
    with np.load(within(root, preparation.validation['split_file']), allow_pickle=False) as split:
        available = split['fit']
    splits = extra_splits(tables['train'], available, preparation.validation,
                          preparation.data.target_columns, options.validation_folds)
    ranked = sorted((row for row in delta.nodes if row.result.status == 'completed' and row.result.value is not None),
                    key=lambda row: row.result.value, reverse=delta.metric.direction == 'maximize')
    ranked.sort(key=lambda row: row.node_id != incumbent)
    for row in ranked:
        if len(report['candidates']) >= options.validation_finalists:
            break
        try:
            if supported(root, row.node_id):
                report['candidates'].append({'node_id': row.node_id, 'value': row.result.value})
            else:
                report['errors'].append({'node_id': row.node_id, 'error': 'No safe fixed-choice replay'})
        except (ValueError, OSError, KeyError) as error:
            report['errors'].append({'node_id': row.node_id, 'error': str(error)})
    if (len(splits) < 2 or len(report['candidates']) < 2
            or incumbent not in {row['node_id'] for row in report['candidates']}):
        report['reason'] = 'Need an eligible incumbent, two candidates and two valid extra splits'
        save(destination / 'report.json', report)
        return report
    report['status'] = 'running'
    scores = {row['node_id']: [] for row in report['candidates']}
    for number, (train, valid) in enumerate(splits):
        split_file = destination / f'split_{number:02d}.npz'
        np.savez(split_file, train=train, valid=valid)
        report['splits'].append({'file': split_file.name, 'sha256': file_hash(split_file),
                                'fit_rows': len(train), 'validation_rows': len(valid)})
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            break
        per_job = min(options.experiment_timeout, remaining / (len(splits) - number))
        parallel = min(options.parallel, len(report['candidates']))
        per_job /= int(np.ceil(len(report['candidates']) / parallel))
        slots = asyncio.Semaphore(parallel)
        async def job(row):
            output = destination / f"{row['node_id']}_{number:02d}.json"
            try:
                async with slots:
                    # The parent owns cleanup even if the worker is killed on timeout.
                    with tempfile.TemporaryDirectory(prefix='recheck-') as workspace:
                        await process(['-m', 'agent.final_validation', '--run-directory', str(root),
                            '--node-id', row['node_id'], '--split', str(split_file), '--output', str(output),
                            '--work-directory', workspace,
                            '--time-budget', str(max(1., per_job - 1))], root, output.with_suffix('.log'), deadline, per_job)
                return json.loads(output.read_text(encoding='utf-8'))['value']
            except (ValueError, OSError, TimeoutError) as error:
                report['errors'].append({'node_id': row['node_id'], 'split': number, 'error': str(error)})
                return None
        values = await asyncio.gather(*(job(row) for row in report['candidates']))
        for row, value in zip(report['candidates'], values):
            scores[row['node_id']].append(value)
        report.update(scores=scores)
        save(destination / 'report.json', report)
        progress(report)
    report['selected_node'] = choose(incumbent, report['candidates'], scores, delta.metric.direction, len(splits))
    report['summary'] = []
    for row in report['candidates']:
        values = [value for value in scores[row['node_id']] if value is not None]
        report['summary'].append({**row, 'completed_splits': len(values),
            'extra_mean': float(np.mean(values)) if values else None,
            'extra_std': float(np.std(values)) if values else None,
            'combined_mean': float(np.mean([row['value'], *values])) if values else None})
    report['status'] = 'completed' if all(len(v) == len(splits) and None not in v for v in scores.values()) else 'partial'
    save(destination / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--node-id', required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-budget', type=float, required=True)
    parser.add_argument('--work-directory', type=Path)
    args = parser.parse_args()
    with np.load(args.split, allow_pickle=False) as split, threadpool_limits(limits=1):
        evaluate(args.run_directory.resolve(), args.node_id, split, args.output, args.time_budget, args.work_directory)


if __name__ == '__main__':
    main()
