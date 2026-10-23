"""Our orchestration/bridge; margo source remains an external read-only checkout."""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from margo_handoff import build_training_handoff
from storage import atomic_bytes, save
from training_component import run_job, run_training


async def run_margo_pipeline(task_dir, selection, source_root, output, *, target, metric,
                             time_budget=4800., preparation_timeout=600., trial_timeout=180.,
                             max_trials=12, preparation_llm=True, prepared_dir=None,
                             model_selection_root=None, selection_timeout=450., probe_seconds=90.,
                             probe_max_rows=12000, probe_folds=2, selection_llm=True,
                             id_columns=None, submission=None):
    if any(not math.isfinite(value) or value <= 0 for value in (time_budget, preparation_timeout, trial_timeout)):
        raise ValueError('Time limits must be positive and finite')
    started = time.monotonic()
    deadline = started+time_budget
    output, source_root = Path(output).resolve(), Path(source_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'stage': 'preparation', 'time_budget_seconds': time_budget}

    def checkpoint(**updates):
        summary.update(updates, elapsed_seconds=time.monotonic()-started)
        save(output/'pipeline_report.json', summary)

    try:
        checkpoint()
        if prepared_dir is None:
            prepared_dir = output/'preparation'
            prepared_dir.mkdir()
            source_hashes = {str(path.relative_to(source_root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted((source_root/'data_preparation').glob('*.py'))}
            if not source_hashes:
                raise ValueError('margo source_root must contain data_preparation/*.py')
            selection_data = json.loads(Path(selection).read_text(encoding='utf-8-sig')) if selection else {}
            job = {'output_dir': str(prepared_dir), 'source': {'source_root': str(source_root), 'source_hashes': source_hashes},
                   'use_llm': preparation_llm, 'config': {'task_dir': str(Path(task_dir).resolve()),
                   'target_columns': [target], 'metric_name': metric, 'id_columns': selection_data.get('id_columns', id_columns if id_columns is not None else ['id']),
                   'output_dir': str(prepared_dir)}}
            save(prepared_dir/'job.json', job)
            preparation = await run_job(prepared_dir/'job.json', prepared_dir, deadline,
                                        min(preparation_timeout, time_budget*.25), module='margo_preparation_worker')
            checkpoint(preparation=preparation)
        if model_selection_root:
            checkpoint(stage='model_selection')
            selection_output = output/'model_selection'
            selection_output.mkdir()
            save(selection_output/'job.json', {
                'output_dir': str(selection_output),
                'source_root': str(Path(model_selection_root).resolve()),
                'preparation_dir': str(Path(prepared_dir).resolve()),
                'margo_root': str(source_root), 'probe_seconds': probe_seconds,
                'probe_max_rows': probe_max_rows, 'probe_folds': probe_folds,
                'use_llm': selection_llm,
                'submission': submission,
            })
            selected = await run_job(selection_output/'job.json', selection_output, deadline,
                                     min(selection_timeout, time_budget*.2), module='model_selection_worker')
            selection = selection_output/'training_selection.json'
            checkpoint(model_selection=selected)
        if not selection:
            raise ValueError('Provide selection JSON or model_selection_root')
        checkpoint(stage='handoff')
        training_input = build_training_handoff(prepared_dir, selection, source_root, output/'handoff')
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('No budget remains after preparation')
        checkpoint(stage='training', training_input=str(training_input))
        report = await run_training(training_input, output/'training', time_budget=remaining,
                                    trial_timeout=trial_timeout, max_trials=max_trials)
        submission = report.get('submission')
        if submission and Path(submission).is_file():
            atomic_bytes(output/Path(submission).name, Path(submission).read_bytes())
            summary['submission'] = str(output/Path(submission).name)
        checkpoint(status=report['status'], stage='complete', primary_metric=report['primary_metric'],
                   winner=report['winner'], search_stop_reason=report.get('search_stop_reason'),
                   full_training_rows=report.get('final_refit', {}).get('full_training_rows'),
                   training_report=str(output/'training/report.json'), error=report.get('error'))
    except Exception as error:
        checkpoint(status='failed', stage='complete', error=str(error)[-2000:])
    return summary


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description='Run margo preparation and our training component without changing margo')
    parser.add_argument('task_dir', type=Path)
    parser.add_argument('--margo-root', type=Path, required=True)
    selection_source = parser.add_mutually_exclusive_group(required=True)
    selection_source.add_argument('--selection', type=Path)
    selection_source.add_argument('--model-selection-root', type=Path)
    parser.add_argument('--target')
    parser.add_argument('--metric')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--time-budget', type=float, default=4800.)
    parser.add_argument('--preparation-timeout', type=float, default=600.)
    parser.add_argument('--trial-timeout', type=float, default=180.)
    parser.add_argument('--max-trials', type=int, default=12)
    parser.add_argument('--no-preparation-llm', action='store_true')
    parser.add_argument('--selection-timeout', type=float, default=450.)
    parser.add_argument('--probe-seconds', type=float, default=90.)
    parser.add_argument('--probe-max-rows', type=int, default=12000)
    parser.add_argument('--probe-folds', type=int, default=2)
    parser.add_argument('--no-selection-llm', action='store_true')
    parser.add_argument('--id-columns', nargs='*', default=['id'])
    args = parser.parse_args(argv)
    if args.prepared_dir is None and (not args.target or not args.metric):
        parser.error('--target and --metric are required when running preparation; prepared artifacts already declare them')
    report = asyncio.run(run_margo_pipeline(args.task_dir, args.selection, args.margo_root, args.output_dir,
        target=args.target, metric=args.metric, time_budget=args.time_budget, preparation_timeout=args.preparation_timeout,
        trial_timeout=args.trial_timeout, max_trials=args.max_trials, preparation_llm=not args.no_preparation_llm,
        prepared_dir=args.prepared_dir, model_selection_root=args.model_selection_root,
        selection_timeout=args.selection_timeout, probe_seconds=args.probe_seconds,
        probe_max_rows=args.probe_max_rows, probe_folds=args.probe_folds,
        selection_llm=not args.no_selection_llm, id_columns=args.id_columns))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'completed' else 2
