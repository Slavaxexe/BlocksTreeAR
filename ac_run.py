"""Classification counterpart of ar_run; no imports of the regression research loop."""
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ar_config import Settings, load_env
from ar_llm import LLM
from ar_memory import Memory
from ar_validation import save_folds
from ac_task import read_task
from ac_data import Features, detect_time, design_folds, discover_entity
from ac_models import positive_index
from ac_research import Lab, research_loop
from ac_final import build_ensemble, refit_members, write_submission


@dataclass
class Budget:
    total_s: float
    started: float = field(default_factory=time.monotonic)

    @property
    def deadline(self):
        return self.started+self.total_s-min(60., self.total_s*.05)

    def elapsed(self):
        return time.monotonic()-self.started


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def distribution_shift(train, test, columns, n_jobs):
    """Train-vs-test classifier uses only public features, never task targets."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    count = min(5000, len(train), len(test))
    if count < 3:return {'auc': None, 'reason': 'Too few rows'}
    data = pd.concat([train[columns].sample(count, random_state=42), test[columns].sample(count, random_state=42)], ignore_index=True)
    for column in data:
        if not pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].astype('string').fillna('__NA__').astype('category')
    labels = np.r_[np.zeros(count), np.ones(count)]
    model = LGBMClassifier(n_estimators=50, num_leaves=15, n_jobs=n_jobs, verbosity=-1, random_state=42)
    probabilities = cross_val_predict(model, data, labels, cv=StratifiedKFold(3, shuffle=True, random_state=42), method='predict_proba')[:, 1]
    return {'auc': float(roc_auc_score(labels, probabilities)), 'scope': 'public features only; sampled train-vs-test discrimination'}


def run(task_path, out_dir=None, no_llm=False, total_budget=10800., max_experiments=60,
        llm_fn=None, write_next_to_task=True, llm_calls=None, n_jobs=None):
    if not np.isfinite(total_budget) or total_budget < 20 or max_experiments < 1:
        raise ValueError('Classification requires a finite budget >=20s and at least one experiment')
    if n_jobs is not None and (not isinstance(n_jobs, int) or n_jobs < 1):
        raise ValueError('CPU count must be a positive integer')
    root = Path(__file__).resolve().parent
    load_env(root)
    settings = Settings(llm_enabled=not no_llm, max_experiments=max_experiments)
    settings.n_jobs = min(8, n_jobs or settings.n_jobs)
    if llm_calls:settings.llm_max_calls = llm_calls
    budget = Budget(total_budget)
    run_dir = Path(out_dir).resolve() if out_dir else root/'artifacts'/('classification_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except PermissionError:
        run_dir = Path(tempfile.mkdtemp(prefix='classification_'))
    mem = Memory(run_dir)
    llm = LLM(settings, mem, fn=llm_fn)
    mem.fact('run', {'task_type': 'classification', 'budget_s': total_budget, 'n_jobs': settings.n_jobs,
                    'llm': llm.enabled, 'model': settings.model if llm.enabled else None})
    log(f'Classification: {run_dir} | Qwen={llm.enabled} | CPU={settings.n_jobs} | budget={total_budget:.0f}s')
    task = read_task(task_path)
    train = pd.read_csv(task.files['train'], dtype={name: str for name in task.id_cols})
    test = pd.read_csv(task.files['test'], dtype={name: str for name in task.id_cols})
    original_rows = len(train)
    train = train.loc[train[task.target].notna()].reset_index(drop=True)
    if len(train) < 30 or not len(test):raise ValueError('Need at least 30 labelled rows and a nonempty hidden test')
    encoder = LabelEncoder().fit(train[task.target])
    classes = encoder.classes_.tolist()
    if len(classes) < 2:raise ValueError('Classification requires at least two classes')
    if task.metric == 'f1' and len(classes) != 2:raise ValueError('Binary F1 requires exactly two classes')
    y = encoder.transform(train[task.target])
    for name, value in {'task.target': task.target, 'task.id_cols': task.id_cols, 'task.metric': task.metric,
                        'task.metric_quote': task.metric_quote, 'task.type': 'classification',
                        'direction': 'minimize' if task.metric == 'log_loss' else 'maximize',
                        'class_labels': classes}.items():mem.fact(name, value)
    mem.stage('0_task', 'ok', {'target': task.target, 'metric': task.metric, 'classes': classes,
                              'missing_target_rows_removed': original_rows-len(train)})
    time_info = detect_time(train, test, task.target, task.id_cols)
    time_column = time_info['col'] if time_info else None
    feature_columns = [name for name in test if name not in task.id_cols]
    mem.fact('dataset_interface', {'feature_columns': feature_columns,
        'dtypes': {name: str(train[name].dtype) for name in feature_columns},
        'dictionary': task.dictionary[:80], 'hidden_test_labels_available': False})
    entity = discover_entity(train, test, feature_columns, time_column)
    group_columns = [name for name in feature_columns if name != time_column
        and not pd.api.types.is_numeric_dtype(train[name]) and 2 <= train[name].nunique() <= 300][:4]
    audit = {'time': time_info, 'entity': entity, 'class_counts': {str(label): int(value) for label, value in train[task.target].value_counts().items()},
        'missing': {name: float(train[name].isna().mean()) for name in feature_columns if train[name].isna().any()},
        'group_columns': group_columns, 'entity_search_scope': 'structure-only, no labels used for key selection'}
    try:
        audit['adversarial'] = distribution_shift(train, test, [name for name in feature_columns if name != time_column], settings.n_jobs)
    except Exception as error:
        audit['adversarial'] = {'error': str(error)[:400]}
    for key, value in audit.items():mem.fact('audit.'+key, value)
    mem.stage('1_audit', 'ok', audit)
    log(f'Audit: time={time_column}, inferred entity={entity and entity["key"]}, classes={len(classes)}')
    design = design_folds(train, y, time_column, entity['key'] if entity else None, settings.seed)
    folds_path = run_dir/'folds.npz'
    fold_hash = save_folds(design, folds_path)
    mem.fact('validation', {'kind': design['kind'], 'folds': design['labels'], 'sizes': design['sizes'],
        'weights': design['weights'], 'sha256': fold_hash, 'history_scope': 'fit-local; validation labels excluded'})
    mem.stage('2_validation', 'ok', {'kind': design['kind'], 'folds': design['labels']})
    all_features = pd.concat([train.drop(columns=[task.target]), test], ignore_index=True)
    feature_path, label_path = run_dir/'features.parquet', run_dir/'labels.npy'
    all_features.to_parquet(feature_path, index=False);np.save(label_path, y, allow_pickle=False)
    feature_store = Features(all_features, task.target, task.id_cols, len(classes), time_column,
                             entity['key'] if entity else None, group_columns, settings.seed)
    mem.fact('feature_blocks', feature_store.available())
    mem.stage('3_features', 'ok', {'available': feature_store.available(), 'fitting': 'inside each training partition only'})
    context = {'features_path': str(feature_path), 'labels_path': str(label_path), 'folds_path': str(folds_path),
        'fold_weights': design['weights'], 'n_train': len(train), 'classes': classes, 'target': task.target,
        'id_cols': task.id_cols, 'time_column': time_column, 'entity_key': entity['key'] if entity else None,
        'group_columns': group_columns, 'seed': settings.seed, 'metric': task.metric, 'n_jobs': settings.n_jobs,
        'positive_index': positive_index(classes, task.positive_label)}
    (run_dir/'context.json').write_text(json.dumps(context, ensure_ascii=False), encoding='utf-8')
    lab = Lab(mem, settings, budget, feature_store, context, run_dir)
    paths = [run_dir/'submission.csv']+([Path(task.task_dir)/'submission.csv'] if write_next_to_task else [])
    safety = {'checks': None, 'record': None}
    def save_safety(record):
        if record.get('status') != 'ok':return
        prediction, information = refit_members({record['id']: 1.}, mem, context, run_dir/'safety',
            min(budget.deadline, time.monotonic()+max(20., lab.reserve())), log, max_seeds=1)
        if prediction is not None:
            safety['checks'] = write_submission(prediction, test, task, classes, paths, record['spec']['training']['threshold'])
            safety['record'] = record['id']
            mem.log('safety_submission', checks=safety['checks'], members=information)
            log('Safety submission saved after full-data baseline fit')
    stop = research_loop(lab, llm, log, save_safety)
    best = mem.best()
    if best is None:raise ValueError('All classifier experiments failed; inspect execution logs')
    mem.stage('5_research', 'ok', {'stop_reason': stop, 'experiments': len(mem.state['experiments']),
                                  'best': {'id': best['id'], 'cv': best['cv']}})
    ensemble = build_ensemble(mem, run_dir, y, task.metric, context['positive_index'],
                             deadline=min(budget.deadline, time.monotonic()+20))
    (run_dir/'ensemble.json').write_text(json.dumps(ensemble, ensure_ascii=False, indent=2), encoding='utf-8')
    mem.stage('6_ensemble', 'ok', ensemble)
    prediction, members = refit_members(ensemble['weights'], mem, context, run_dir, budget.deadline, log)
    checks, final_status = safety['checks'], 'fallback_safety'
    if prediction is not None:
        checks = write_submission(prediction, test, task, classes, paths, ensemble['threshold'])
        final_status = 'ok'
    if not checks:raise ValueError('Classification produced no valid full-data submission')
    mem.stage('7_final', final_status, {'checks': checks, 'members': members, 'full_training_rows': len(train),
                                      'score': None})
    report = {'task_type': 'classification', 'task': {'target': task.target, 'metric': task.metric,
        'metric_quote': task.metric_quote, 'classes': classes, 'id': task.id_cols},
        'stages': mem.state['stages'], 'facts': {key: value['value'] for key, value in mem.state['facts'].items()},
        'validation': {'kind': design['kind'], 'labels': design['labels'], 'sizes': design['sizes']},
        'experiments': mem.state['experiments'], 'best_single': {'id': best['id'], 'cv': best['cv'], 'summary': best['summary']},
        'ensemble': ensemble, 'final_cv': ensemble['ensemble_score'] if not members.get('weights_renormalized') and final_status == 'ok' else None,
        'final_score_scope': 'No hidden-test labels available; full-data training score is null',
        'submission_checks': checks, 'final_status': final_status, 'final_refit': members,
        'stop_reason': stop, 'llm': mem.state['llm'], 'elapsed_s': budget.elapsed()}
    (run_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame([{key: value.get(key) for key in ('id', 'summary', 'status', 'cv', 'train_cv', 'seconds', 'verdict')}
        for value in mem.state['experiments']]).to_csv(run_dir/'leaderboard.csv', index=False)
    from html import escape
    (run_dir/'dashboard.html').write_text('<meta charset="utf-8"><title>Classification AutoResearch</title>'
        '<h1>Classification AutoResearch</h1><p>Primary metric: '+escape(task.metric)
        +'; scores below describe OOF selection, not hidden-test quality.</p><pre>'
        +escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre>', encoding='utf-8')
    mem.save()
    log(f'Classification complete: {stop}, best={best["cv"]:.6f}, final={report["final_cv"]}, rows={checks["rows"]}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--out')
    parser.add_argument('--budget', type=float, default=10800.)
    parser.add_argument('--max-experiments', type=int, default=60)
    parser.add_argument('--llm-calls', type=int)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--no-llm', action='store_true')
    args = parser.parse_args()
    run(args.task, args.out, args.no_llm, args.budget, args.max_experiments,
        llm_calls=args.llm_calls, n_jobs=args.cpu)


if __name__ == '__main__':
    main()
