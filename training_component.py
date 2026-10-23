"""Standalone training/validation stage; upstream owns task analysis, EDA and model selection."""
import argparse
import asyncio
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from agents import JSONAgents
from config import Settings
from data import file_hash, installed_libraries
from storage import atomic_bytes, save
from training_contracts import TrainingInput
from training_templates import template_context
from training_roles import TrainingAnalyst, TrainingPlanner
from training_coordinator import TrainingCoordinator, candidate_fingerprint
from margo_recipe import training_source_paths, register_margo


RULES = ('You implement ONLY training and validation for upstream_report.task_type. An upstream Qwen has already chosen '
         'model families and supplied EDA. Do not repeat task analysis, model-family selection, feature '
         'engineering or a business-model stage. Use only the selected_models and installed packages '
         'allowed by requirements.txt. upstream_report.primary_metric is the task scoring metric supplied '
         'by the upstream pipeline, not a metric for you to choose. It is mandatory and immutable. '
         'The primary metric/direction and validation constraints are fixed. Training loss and auxiliary '
         'diagnostics may differ, but cannot replace the supplied primary metric for ranking candidates. '
         'Hidden test labels are unavailable; never use sample_submission values as ground truth. '
         'Choose training strategy and justified bounded hyperparameters, optimizer where applicable, '
         'and regularization based on upstream EDA plus recorded experimental results. '
         'SGD/Adam/momentum/Nesterov apply to compatible differentiable models, not boosting trees. '
         'The ordinary-model evaluator imputes, scales and one-hot encodes categorical columns inside '
         'training folds; do not pass original categorical_feature/cat_features names to those estimators. '
         'Identifiers and group/time split keys are excluded from model features. input_facts.feature_columns '
         'is the authoritative actual feature list, even if upstream EDA lists additional columns. '
         'Do not invent an encoded feature count or include split keys in it. '
         'Put executable solver/loss/learning-rate/regularization choices in parameters; optimizer and '
         'regularization strings only explain them. For LightGBM, subsample has no effect unless '
         'subsample_freq (bagging frequency) is positive. Do not infer causality from one validation '
         'split or attribute a score difference to one parameter when several parameters changed. '
         'The evaluator owns preprocessing, split, score and full-data final refit. '
         'model_templates supplies the actual Python templates used by the worker for selected models. '
         'Read their API notes and adapt parameters to EDA, measured results and time budget. '
         'Template examples are starting points; you may use other supported hyperparameters. '
         'Do not edit Python template source or invent template APIs. '
         'When upstream_report.preparation is present, its outer split and feature recipe are locked. '
         'Do not change the outer validation fraction; choose only the internal validation strategy. '
         'Ordinary models refit the recipe within each CV fold. AutoML with this external recipe '
         'supports inner holdout only, so use inner_validation=holdout when scheduling FLAML/AutoGluon. '
         'Return structured decisions, not Python code. Do not invent measurements or claim unexecuted work. ')

CLASSIFICATION_RULES = (
    'This is the CLASSIFICATION branch. Use classifier templates, never regressors or regression losses. '
    'The target can contain string class labels; backend handles encoding and preserves the original labels '
    'in submissions. f1_macro gives equal weight to classes, so inspect imbalance and per-class diagnostics. '
    'Optimize the supplied metric only. All classification metrics maximize except log_loss, which minimizes. '
    'Never pass label names as numeric class_weight indices: backend label encoding is internal. '
    'For automatic CatBoost class balancing use auto_class_weights=Balanced, not class_weights=Balanced. '
    'Use stratified inner folds and ensure rare classes are present. Use inner holdout for FLAML/AutoGluon. '
    'An experiment estimate includes every inner fold, the outer-fit model and prediction on train, validation '
    'and hidden test. Multiclass boosting can be much more expensive than regression at the same tree count. '
    'Candidate parameters override selected_models.parameters; omitting a key retains the upstream value. '
    'XGBoost class_weight is a backend policy translated to fold-local sample_weight, not a native XGBoost '
    'constructor option. Use class_weight=null to turn an upstream balancing policy off explicitly. '
    'HistGradientBoosting uses backend threadpool limits; it has no native n_jobs constructor argument. '
    'actual_parameters.backend_threads and class_weight_adapter describe backend behavior; do not copy '
    'those reporting fields into candidate.parameters. Inspect actual_inner_folds and warnings before '
    'claiming that all requested inner CV folds completed. The common outer score remains comparable. '
    'Preserve the fixed outer_split and do not tune thresholds/calibration on its labels. '
    'The backend aligns probabilities to classes; submit labels or probabilities according to submission.prediction_type. ')


def _table(path, ids):
    return pd.read_parquet(path) if path.suffix.lower() == '.parquet' else pd.read_csv(path, dtype={name: str for name in ids})


def make_split(train, contract, decision):
    indexes = np.arange(len(train))
    constraints = contract.validation_constraints
    if contract.preparation or contract.outer_split:
        frozen_path = contract.preparation.split if contract.preparation else contract.outer_split
        frozen = json.loads(Path(frozen_path).read_text(encoding='utf-8'))
        for key in ('fit', 'valid'):
            if any(isinstance(value, bool) or not isinstance(value, int) for value in frozen[key]):
                raise ValueError('Prepared split must contain integer row positions')
        fit, valid = np.asarray(frozen['fit'], dtype=int), np.asarray(frozen['valid'], dtype=int)
        joined = np.concatenate([fit, valid])
        if len(joined) != len(train) or not np.array_equal(np.sort(joined), indexes):
            raise ValueError('Prepared split must partition every training row exactly once')
        if constraints.strategy == 'group':
            groups = pd.util.hash_pandas_object(train[constraints.group_columns], index=False).to_numpy()
            if set(groups[fit]) & set(groups[valid]):
                raise ValueError('Prepared split violates the group constraint')
        elif constraints.strategy == 'time':
            times = pd.to_datetime(train[constraints.time_column], errors='raise', utc=True)
            if times.isna().any() or times.iloc[fit].max() >= times.iloc[valid].min():
                raise ValueError('Prepared split violates the chronological constraint')
    elif constraints.strategy == 'group':
        groups = pd.util.hash_pandas_object(train[constraints.group_columns], index=False).to_numpy()
        fit, valid = next(GroupShuffleSplit(n_splits=1, test_size=decision.validation_fraction,
                                            random_state=decision.seed).split(indexes, groups=groups))
    elif constraints.strategy == 'time':
        times = pd.to_datetime(train[constraints.time_column], errors='raise')
        if times.isna().any():
            raise ValueError('Time column contains missing timestamps')
        ordered = np.argsort(times.to_numpy(), kind='stable')
        cut = max(1, min(len(train)-1, int(len(train)*(1-decision.validation_fraction))))
        boundary = times.iloc[ordered[cut]]
        fit, valid = indexes[(times < boundary).to_numpy()], indexes[(times >= boundary).to_numpy()]
    else:
        fit, valid = train_test_split(indexes, test_size=decision.validation_fraction, random_state=decision.seed,
                                      stratify=train[contract.target] if contract.task_type == 'classification' else None)
    if not len(fit) or not len(valid) or np.intersect1d(fit, valid).size:
        raise ValueError('Validation split is empty or overlapping')
    return np.asarray(fit), np.asarray(valid)


async def run_job(job_path, output, deadline, budget, *, module='training_worker'):
    remaining = min(budget, deadline-time.monotonic())
    if remaining <= 0:
        raise TimeoutError('Stage deadline exhausted')
    env = {'PATH': os.defpath, 'PYTHONPATH': str(Path(__file__).resolve().parent),
           'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    for key in ('SystemRoot', 'USERPROFILE', 'TEMP', 'TMP', 'APPDATA', 'LOCALAPPDATA', 'HOME'):
        if key in os.environ:
            env[key] = os.environ[key]
    # These two workers call Qwen themselves. Platform credentials arrive in env,
    # whereas ordinary model-training workers do not need access to them.
    if module in {'margo_preparation_worker', 'model_selection_worker'}:
        settings = Settings.from_env()
        env.update(OPENAI_API_KEY=settings.api_key, OPENAI_BASE_URL=settings.base_url,
                   OPENAI_MODEL=settings.model)
        if settings.timeout_seconds is not None:
            env['QWEN_TIMEOUT_SECONDS'] = str(settings.timeout_seconds)
        if settings.max_tokens is not None:
            env['QWEN_MAX_TOKENS'] = str(settings.max_tokens)
    with (output/'execution.log').open('wb') as stream:
        child = await asyncio.create_subprocess_exec(sys.executable, '-m', module,
            '--job', str(job_path), '--time-budget', str(remaining*.92), cwd=output, env=env,
            stdout=stream, stderr=asyncio.subprocess.STDOUT)
        try:
            await asyncio.wait_for(child.wait(), remaining)
        except (TimeoutError, asyncio.CancelledError):
            import psutil
            try:
                descendants = psutil.Process(child.pid).children(recursive=True)
                for descendant in reversed(descendants):
                    try:
                        descendant.kill()
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            if child.returncode is None:
                child.kill()
            await child.wait()
            raise
    result_path = output/'result.json'
    if not result_path.exists():
        tail = (output/'execution.log').read_text(encoding='utf-8', errors='replace')[-2500:]
        raise ValueError('Worker produced no result: '+tail)
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if child.returncode or result.get('status') != 'completed':
        raise WorkerExecutionError(result)
    return result


class WorkerExecutionError(ValueError):
    """Keep the failing worker's diagnostic available for bounded Qwen repair."""
    def __init__(self, result):
        super().__init__(result.get('error', 'Worker failed'))
        self.diagnostic = {key: str(result[key])[-4000:]
                           for key in ('error_type', 'error', 'traceback') if key in result}


async def run_training(input_path, output, *, time_budget=4800., trial_timeout=300., max_trials=12,
                       patience_rounds=2, relative_min_improvement=1e-4, agents=None):
    started = time.monotonic()
    if not math.isfinite(time_budget) or not math.isfinite(trial_timeout) or time_budget <= 0 or trial_timeout <= 0 or max_trials < 1:
        raise ValueError('Positive finite time_budget, trial_timeout and max_trials required')
    coordinator = TrainingCoordinator(started=started, time_budget=time_budget, trial_timeout=trial_timeout,
        max_trials=max_trials, patience_rounds=patience_rounds, relative_min_improvement=relative_min_improvement)
    deadline, search_deadline = coordinator.deadline, coordinator.search_deadline
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_path = Path(input_path).resolve()
    contract = TrainingInput.model_validate_json(input_path.read_text(encoding='utf-8'))
    raw = contract.model_dump(mode='json')
    for role, value in raw['files'].items():
        if value:
            raw['files'][role] = str((input_path.parent/value).resolve())
    if raw.get('outer_split'):
        raw['outer_split'] = str((input_path.parent/raw['outer_split']).resolve())
    if raw.get('preparation'):
        for key in ('source_root', 'recipe', 'split'):
            raw['preparation'][key] = str((input_path.parent/raw['preparation'][key]).resolve())
        register_margo(raw['preparation'])
    contract = TrainingInput.model_validate(raw)
    normalized = output/'input.json'
    save(normalized, raw)
    paths = training_source_paths(raw, input_path.parent)
    hashes = {key: file_hash(path) for key, path in paths.items()}
    train, test = (_table(Path(raw['files'][role]), contract.id_columns) for role in ('train', 'test'))
    if contract.target not in train or contract.target in test:
        raise ValueError('Train must contain target; hidden test must not contain target')
    if len(train) < 30 or not len(test) or not train.columns.is_unique or not test.columns.is_unique:
        raise ValueError('Need at least 30 train rows, nonempty test and unique column names')
    if contract.task_type == 'regression':
        if not np.isfinite(pd.to_numeric(train[contract.target], errors='raise')).all():
            raise ValueError('Regression labels must be finite')
    elif train[contract.target].isna().any() or train[contract.target].nunique() < 2:
        raise ValueError('Classification needs nonmissing labels and at least two classes')
    for column in contract.id_columns:
        if column not in test or column not in train:
            raise ValueError('Missing declared identifier: '+column)
    metric = contract.primary_metric
    direction = coordinator.direction = ('minimize' if metric == 'log_loss' else 'maximize') if contract.task_type == 'classification' else ('maximize' if metric == 'r2' else 'minimize')
    agents = agents or JSONAgents(Settings.from_env(), search_deadline, max_calls=0)
    agents.deadline = search_deadline
    agents.root.log_directory = output
    role_rules = RULES + (CLASSIFICATION_RULES if contract.task_type == 'classification' else '')
    analyst, planner = TrainingAnalyst(agents, role_rules), TrainingPlanner(agents, role_rules)
    root = Path(__file__).resolve().parent
    excluded = {contract.target, *contract.id_columns, *contract.validation_constraints.group_columns}
    if contract.validation_constraints.time_column:
        excluded.add(contract.validation_constraints.time_column)
    context = {'upstream_report': raw, 'allowed_requirements': (root/'requirements.txt').read_text(encoding='utf-8'),
               'installed_libraries': installed_libraries(), 'input_facts': {'train_rows': len(train),
               'test_rows': len(test), 'feature_columns': [name for name in train if name not in excluded],
               'excluded_columns': sorted(excluded & set(train.columns))},
               'primary_metric': metric, 'direction': direction, 'trial_timeout_seconds': trial_timeout}
    if contract.task_type == 'classification':
        from classification_templates import template_context as classification_template_context
        selected_template_context = classification_template_context
        context['input_facts']['class_counts'] = {str(key): int(value) for key, value in train[contract.target].value_counts().items()}
    else:
        selected_template_context = template_context
    context['model_templates'] = selected_template_context(raw['selected_models'], context['allowed_requirements'],
                                                context['installed_libraries'])
    if contract.preparation or contract.outer_split:
        split_path = contract.preparation.split if contract.preparation else contract.outer_split
        split = json.loads(Path(split_path).read_text(encoding='utf-8'))
        fraction = len(split['valid'])/len(train)
        if not .05 <= fraction <= .4:
            raise ValueError('Prepared validation fraction must be between 0.05 and 0.4')
        context['locked_outer_split'] = {'source': 'margo' if contract.preparation else 'classification_preparation', 'validation_fraction': fraction,
                                         'fit_rows': len(split['fit']), 'validation_rows': len(split['valid'])}
        if contract.preparation:
            context['input_facts']['raw_feature_columns'] = context['input_facts']['feature_columns']
            context['input_facts']['feature_columns'] = contract.preparation.feature_names
    save(output/'model_templates.json', context['model_templates'])
    report = {'status': 'running', 'task_type': contract.task_type, 'stage': 'validation_decision', 'primary_metric': metric,
              'direction': direction, 'trials': [], 'repairs': [], 'rounds': [], 'role_history': [],
              'winner': None, 'final_refit': {}, 'time_budget_seconds': time_budget,
              'final_reserved_seconds': coordinator.reserve}

    def checkpoint(**values):
        report.update(values, elapsed_seconds=time.monotonic()-started, coordinator=coordinator.state())
        save(output/'report.json', report)

    def live_context():
        return {**context, 'coordinator': coordinator.state()}

    async def role_call(stage, role, coroutine, *, until=None):
        role_deadline = min(search_deadline, until if until is not None else search_deadline)
        remaining = role_deadline-time.monotonic()
        if remaining <= 0:
            coroutine.close()
            raise TimeoutError('No search time left for Qwen role')
        entry = {'role': role, 'stage': stage, 'status': 'running'}
        report['role_history'].append(entry)
        checkpoint(stage=stage, active_role=role, active_trial=None)
        previous_deadline, role_started = agents.deadline, time.monotonic()
        agents.deadline = role_deadline
        try:
            result = await asyncio.wait_for(coroutine, remaining)
            entry['status'] = 'completed'
            return result
        except Exception as error:
            if isinstance(error, TimeoutError):
                message = f'Qwen {role} exceeded its remaining role budget ({remaining:.1f}s)'
                entry.update(status='failed', error=message, error_type='TimeoutError')
                raise TimeoutError(message) from error
            entry.update(status='failed', error=str(error)[-1000:], error_type=type(error).__name__)
            raise
        finally:
            elapsed = time.monotonic()-role_started
            entry['duration_seconds'] = elapsed
            coordinator.role_durations.append(elapsed)
            agents.deadline = previous_deadline
            checkpoint(active_role=None)

    async def analysis_then_plan(*, round_id, failed_trial=None, pair_budget):
        repair = failed_trial is not None
        until = time.monotonic()+pair_budget
        analysis = await role_call('repair_analysis' if repair else 'result_analysis', 'analyst',
            analyst.analyze(live_context(), report['trials'], report['winner'], failed_trial=failed_trial), until=until)
        analysis_file = f'repair_analysis_{round_id:02d}.json' if repair else f'analysis_{round_id:02d}.json'
        save(output/analysis_file, analysis)
        plan = await role_call('repair_planning' if repair else 'next_plan', 'planner',
            planner.next(live_context(), analysis, report['trials'], report['winner'], failed_trial=failed_trial), until=until)
        plan_file = f'repair_plan_{round_id:02d}.json' if repair else f'plan_{round_id:02d}.json'
        save(output/plan_file, plan)
        return analysis, plan

    queue = deque()
    try:
        checkpoint()
        validation = await role_call('validation_decision', 'planner', planner.validation(live_context()))
        fit, valid = make_split(train, contract, validation)
        if contract.preparation or contract.outer_split:
            validation = validation.model_copy(update={'validation_fraction': len(valid)/len(train)})
        np.savez(output/'split.npz', fit=fit, valid=valid)
        split_hash = file_hash(output/'split.npz')
        save(output/'validation_decision.json', {**validation.model_dump(), 'strategy': contract.validation_constraints.strategy,
             'fit_rows': len(fit), 'validation_rows': len(valid), 'split_sha256': split_hash})
        context['validation_decision'] = validation.model_dump()
        context['input_facts'].update(outer_fit_rows=len(fit), outer_validation_rows=len(valid))
        plan = await role_call('training_plan', 'planner', planner.initial(live_context()))
        save(output/'training_plan.json', plan)
        covered = {candidate.model_index for candidate in plan.candidates}
        if set(range(len(contract.selected_models))) - covered - {int(key) for key in plan.skipped_models}:
            raise ValueError('Each upstream model needs trials or an explicit skip reason')
        if len(covered) > max_trials:
            raise ValueError('max_trials cannot cover the proposed upstream models; increase budget or explicitly skip models')
        first, extra, scheduled = [], [], set()
        for candidate in plan.candidates:
            if candidate.model_index in scheduled:
                extra.append(candidate)
            else:
                first.append(candidate)
                scheduled.add(candidate.model_index)
        # Every new search round is informed by the analyst; preplanned extra
        # variants are suggestions for the next planner, not an unchecked batch.
        queue.extend(first)
        context['deferred_initial_candidates'] = [candidate.model_dump(mode='json') for candidate in extra]
        checkpoint(initial_deferred_candidates=context['deferred_initial_candidates'])
        feedback_round = 0
        while True:
            reason = coordinator.stopping_reason()
            if reason:
                coordinator.stop_reason = reason
                break
            if not queue:
                coordinator.finish_round()
                reason = coordinator.stopping_reason()
                if reason or feedback_round >= max_trials or not report['trials']:
                    coordinator.stop_reason = reason or ('planning_round_limit' if report['trials'] else 'no_affordable_candidates')
                    break
                pair_budget = coordinator.planning_budget()
                if not pair_budget:
                    coordinator.stop_reason = 'insufficient_time_for_planning_and_trial'
                    break
                feedback_round += 1
                try:
                    analysis, next_plan = await analysis_then_plan(round_id=feedback_round, pair_budget=pair_budget)
                    report['rounds'].append({'round_id': feedback_round, 'analysis': analysis.model_dump(mode='json'),
                                             'plan': next_plan.model_dump(mode='json')})
                    checkpoint()
                except Exception as error:
                    coordinator.stop_reason = 'planning_unavailable'
                    checkpoint(analysis_warning='Follow-up planning stopped: '+str(error)[-1000:])
                    break
                if next_plan.stop or not next_plan.next_candidates:
                    coordinator.stop_reason = 'planner_stop' if next_plan.stop else 'no_new_candidates'
                    break
                queue.extend(next_plan.next_candidates)
            candidate = queue.popleft()
            frozen = candidate.model_dump(mode='json')
            trial_id = f'trial_{len(report["trials"]):03d}'
            if candidate.model_index >= len(contract.selected_models):
                result = {'trial_id': trial_id, 'status': 'failed', 'candidate': frozen,
                          'error': 'Candidate references a model outside upstream selection'}
                coordinator.observe(result, frozen, 'unknown', 0.)
                report['trials'].append(result)
                checkpoint()
                continue
            family = contract.selected_models[candidate.model_index].family
            if not coordinator.admit(candidate, family):
                checkpoint()
                continue
            trial_dir = output/'trials'/trial_id
            trial_dir.mkdir(parents=True)
            job = {'input_path': str(normalized), 'split_path': str(output/'split.npz'),
                   'candidate': frozen, 'validation': validation.model_dump(), 'phase': 'evaluate',
                   'output_dir': str(trial_dir), 'expected_source_hashes': hashes}
            save(trial_dir/'job.json', job)
            checkpoint(stage='training', active_role='coordinator', active_trial={'trial_id': trial_id, 'candidate': frozen})
            trial_started = time.monotonic()
            try:
                if {key: file_hash(path) for key, path in paths.items()} != hashes:
                    raise ValueError('Input data changed during training')
                result = await run_job(trial_dir/'job.json', trial_dir, search_deadline, trial_timeout)
                if (result.get('primary_metric') != metric or result.get('direction') != direction
                        or result.get('split_sha256') != split_hash
                        or result.get('input_sha256') != file_hash(normalized)):
                    raise ValueError('Worker result does not match the frozen input, split and primary metric')
                value = float(result['score'])
                if not math.isfinite(value):
                    raise ValueError('Primary score is not finite')
                result.update(trial_id=trial_id, candidate=frozen, score=value)
                actual = result.get('actual_parameters', {})
                if (result.get('family') == 'lightgbm' and actual.get('subsample', 1.) < 1.
                        and actual.get('subsample_freq', actual.get('bagging_freq', 0)) == 0):
                    result['parameter_warnings'] = ['subsample is set below 1 but bagging frequency is 0; row subsampling is inactive']
            except (ValueError, TimeoutError, OSError) as error:
                result = {'trial_id': trial_id, 'candidate': frozen, 'family': family, 'status': 'failed', 'error': str(error)[-3000:]}
                if isinstance(error, WorkerExecutionError):
                    result['diagnostic'] = error.diagnostic
            improved = coordinator.observe(result, frozen, family, time.monotonic()-trial_started)
            report['trials'].append(result)
            if improved:
                candidate_file = trial_dir/contract.submission.filename
                if candidate_file.exists():
                    atomic_bytes(output/contract.submission.filename, candidate_file.read_bytes())
                checkpoint(winner={'trial_id': trial_id, 'score': result['score'], 'candidate': frozen,
                           'score_scope': 'fixed held-out validation before full-data refit'})
            checkpoint(active_trial=None, active_role=None)
            pair_budget = coordinator.planning_budget()
            if (result['status'] == 'failed' and len(report['repairs']) < 2
                    and not coordinator.stopping_reason() and pair_budget):
                repair = {'failed_trial_id': trial_id, 'status': 'reviewing'}
                report['repairs'].append(repair)
                try:
                    analysis, repair_plan = await analysis_then_plan(round_id=len(report['repairs']),
                        failed_trial=result, pair_budget=pair_budget)
                    repair.update(status='reviewed', analysis=analysis.model_dump(mode='json'),
                                  plan=repair_plan.model_dump(mode='json'))
                    if not repair_plan.stop and repair_plan.next_candidates:
                        corrected = repair_plan.next_candidates[0]
                        if len(repair_plan.next_candidates) != 1 or corrected.model_index != candidate.model_index:
                            raise ValueError('Repair must contain one configuration of the same upstream model')
                        if candidate_fingerprint(corrected.model_dump(mode='json')) in coordinator.seen:
                            raise ValueError('Repair repeated an already evaluated configuration')
                        queue.appendleft(corrected)
                        repair.update(status='retry_queued', candidate=corrected.model_dump(mode='json'))
                except Exception as error:
                    repair.update(status='review_failed', error=str(error)[-1000:])
                checkpoint(stage='training', active_role=None)
        best, best_candidate = coordinator.best, coordinator.best_candidate
        checkpoint(stage='coordinator_select', active_trial=None, active_role='coordinator',
                   search_stop_reason=coordinator.stop_reason)
        if best is None:
            raise ValueError('No candidate completed with a valid primary score')
        if coordinator.remaining_search() > 5:
            try:
                final_analysis = await role_call('final_analysis', 'analyst',
                    analyst.analyze(live_context(), report['trials'], report['winner'], final=True),
                    until=time.monotonic()+min(120., max(30., max(coordinator.role_durations or [0.])*1.5), coordinator.remaining_search()))
                save(output/'final_analysis.json', final_analysis)
                checkpoint(final_analysis=final_analysis.model_dump(mode='json'))
            except Exception as error:
                checkpoint(final_analysis_warning=str(error)[-1000:])
        checkpoint(stage='final_refit', active_role='coordinator')
        final = output/'final'
        final.mkdir()
        job = {'input_path': str(normalized), 'split_path': str(output/'split.npz'), 'candidate': best_candidate,
               'validation': validation.model_dump(), 'phase': 'final', 'output_dir': str(final),
               'source_result_dir': str(output/'trials'/best['trial_id']), 'expected_source_hashes': hashes}
        save(final/'job.json', job)
        if {key: file_hash(path) for key, path in paths.items()} != hashes:
            raise ValueError('Input data changed before final refit')
        final_result = await run_job(final/'job.json', final, deadline, deadline-time.monotonic())
        if int(final_result.get('full_training_rows', final_result.get('training_rows', -1))) != len(train):
            raise ValueError('Full-data refit did not report all labelled rows')
        submission = final/contract.submission.filename
        if not submission.exists():
            raise ValueError('Final refit produced no submission')
        atomic_bytes(output/contract.submission.filename, submission.read_bytes())
        checkpoint(status='completed', stage='complete', active_trial=None, active_role=None, final_refit=final_result,
                   submission=str(output/contract.submission.filename),
                   score_scope='Winner score is held-out evaluation BEFORE final full-data training; no hidden-test score')
    except Exception as error:
        checkpoint(status='partial' if coordinator.best else 'failed', stage='complete', active_trial=None, active_role=None,
                   error=str(error)[-3000:],
                   final_refit={'status': 'failed', 'error': str(error)[-3000:]} if coordinator.best else {},
                   submission=str(output/contract.submission.filename) if (output/contract.submission.filename).exists() else None)
    return report


def main(argv=None):
    # Qwen explanations can contain symbols outside the Windows legacy codepage.
    # Printing a completed report must not turn a successful run into exit code 1.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description='Train and validate regression models chosen by an upstream agent')
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--time-budget', type=float, default=4800.)
    parser.add_argument('--trial-timeout', type=float, default=300.)
    parser.add_argument('--max-trials', type=int, default=12)
    parser.add_argument('--patience-rounds', type=int, default=2)
    parser.add_argument('--relative-min-improvement', type=float, default=1e-4)
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(run_training(args.input, args.output_dir, time_budget=args.time_budget,
                                         trial_timeout=args.trial_timeout, max_trials=args.max_trials,
                                         patience_rounds=args.patience_rounds,
                                         relative_min_improvement=args.relative_min_improvement))
    except Exception as error:
        report = {'status': 'failed', 'stage': 'input_validation', 'error': str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'completed' else 2
