"""OOF classifier blend, deadline-aware full-data refits, labels/probability CSV."""
import json
import re
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ac_experiment import ExperimentSpec, run_in_subprocess
from ac_models import labels_from_probabilities, normalize, positive_index, score


def build_ensemble(mem, root, y, metric, positive, deadline=None):
    minimize = metric == 'log_loss'
    ordered = sorted(mem.ok_experiments(), key=lambda value: value['cv'], reverse=not minimize)
    if not ordered:raise ValueError('No valid classification experiment completed')
    best = ordered[0]
    threshold = best['spec']['training']['threshold']
    candidates, predictions = [], []
    for record in ordered[:8]:
        path = root/'experiments'/record['id']/'oof.npy'
        if path.is_file():
            candidates.append(record);predictions.append(np.load(path, allow_pickle=False))
    mask = np.logical_and.reduce([np.isfinite(prediction).all(axis=1) for prediction in predictions])
    if not mask.any():raise ValueError('No common OOF rows for classification ensemble')
    target = y[mask]
    probabilities = [prediction[mask] for prediction in predictions]
    counts = np.zeros(len(candidates), dtype=int)
    scores = [score(metric, target, value, threshold, positive) for value in probabilities]
    first = int(np.argmin(scores) if minimize else np.argmax(scores))
    counts[first] = 1
    current = probabilities[first].copy()
    best_counts, best_score = counts.copy(), scores[first]
    for step in range(1, 30):
        if deadline and time.monotonic() >= deadline:break
        values = [score(metric, target, (current+value)/(step+1), threshold, positive) for value in probabilities]
        chosen = int(np.argmin(values) if minimize else np.argmax(values))
        current += probabilities[chosen];counts[chosen] += 1
        if values[chosen] < best_score if minimize else values[chosen] > best_score:
            best_score, best_counts = values[chosen], counts.copy()
    gain = (best['cv']-best_score if minimize else best_score-best['cv'])/max(abs(best['cv']), 1e-12)
    used = gain >= .003 and np.count_nonzero(best_counts) >= 2
    weights = ({candidates[i]['id']: float(count/best_counts.sum()) for i, count in enumerate(best_counts) if count}
               if used else {best['id']: 1.})
    return {'used': bool(used), 'weights': weights, 'threshold': threshold,
        'best_single': best['id'], 'best_single_score': best['cv'],
        'ensemble_score': float(best_score if used else best['cv']), 'rel_gain_pct': float(100*gain),
        'validation_rows': int(mask.sum()), 'score_scope': 'OOF selection on repeatedly inspected fixed outer folds, not hidden-test score',
        'reason': 'OOF blend improved at least 0.3%' if used else 'Retain best individual classifier'}


def refit_members(weights, mem, context, root, deadline, log=print, max_seeds=3):
    records = {value['id']: value for value in mem.ok_experiments()}
    folds = max(1, len(context['fold_weights']))
    estimates = {name: records[name]['seconds']/folds*1.8+5 for name in weights}
    seed_count = max_seeds if deadline-time.monotonic() > sum(estimates.values())*max_seeds*1.5+30 else 1
    predictions, accepted, details = [], [], {}
    for name, weight in sorted(weights.items(), key=lambda value: -value[1]):
        record = records[name]
        spec = ExperimentSpec.model_validate(record['spec'])
        iterations = record.get('best_iters') or [250]
        rounds = max(20, int(np.mean(iterations)*1.15))
        runs, seeds = [], []
        for seed in range(seed_count if spec.model.model_id not in {'logistic_regression', 'sgd'} else 1):
            remaining = deadline-time.monotonic()
            if remaining < estimates[name]+2:
                runs.append({'status': 'skipped', 'reason': 'Insufficient measured final-refit budget'});break
            output = root/'final'/f'{name}_seed{seed}'
            result = run_in_subprocess(spec, {**context, 'mode': 'final', 'n_rounds': rounds,
                'seed': context['seed']+seed}, output, min(remaining-1., max(30., estimates[name]*3)))
            runs.append(result)
            path = output/'test_pred.npy'
            if result.get('status') == 'ok' and result.get('full_training_rows') == context['n_train'] and path.is_file():
                seeds.append(normalize(np.load(path, allow_pickle=False)))
        details[name] = {'runs': runs, 'successful_seeds': len(seeds)}
        if seeds:
            predictions.append(np.mean(seeds, axis=0));accepted.append(weight)
        log(f'Full-data classifier {name}: {len(seeds)} seed(s), {len(runs)} run(s)')
    if not predictions:return None, details
    actual = np.asarray(accepted)/sum(accepted)
    details['weights_renormalized'] = len(predictions) != len(weights)
    details['full_training_rows'] = context['n_train']
    details['score'] = None
    return normalize(sum(weight*value for weight, value in zip(actual, predictions))), details


def write_submission(probabilities, test, task, classes, paths, threshold=.5):
    probabilities = normalize(probabilities)
    if len(probabilities) != len(test):raise ValueError('Prediction row count differs from hidden test')
    positive = positive_index(classes, task.positive_label)
    result = test[task.id_cols].copy().reset_index(drop=True)
    columns = task.prediction_columns
    if task.prediction_type == 'labels':
        if len(columns) != 1:raise ValueError('Class-label output needs one prediction column')
        result[columns[0]] = np.asarray(classes)[labels_from_probabilities(probabilities, threshold, positive)]
    elif len(columns) == 1:
        if len(classes) != 2:raise ValueError('Single probability column requires binary classification; provide all class columns for multiclass')
        result[columns[0]] = probabilities[:, positive]
    else:
        if len(columns) != len(classes):raise ValueError('Multiclass probability output requires one column per class')
        for column in columns:
            normalized_name = re.sub(r'^(?:probability|proba|prob|p)[_-]', '', column, flags=re.I)
            matches = [index for index, label in enumerate(classes) if str(label) == column or str(label) == normalized_name]
            if len(matches) != 1:raise ValueError(f'Cannot unambiguously map probability output column {column} to a class')
            result[column] = probabilities[:, matches[0]]
    if task.files.get('sample_submission'):
        sample = pd.read_csv(task.files['sample_submission'], dtype={name: str for name in task.id_cols})
        if len(sample) != len(result) or set(sample) != set(result):raise ValueError('Submission shape differs from sample header/rows')
        if task.id_cols and not sample[task.id_cols].equals(result[task.id_cols]):
            if sample.duplicated(task.id_cols).any() or result.duplicated(task.id_cols).any():
                raise ValueError('Ambiguous identifier order: duplicate keys in sample')
            result = sample[task.id_cols].merge(result, on=task.id_cols, how='left', sort=False, validate='one_to_one')
            if result[columns].isna().any().any():raise ValueError('Sample identifiers do not match test identifiers')
        result = result[list(sample)]
    checks = {'ok': True, 'rows': len(result), 'columns': list(result), 'probabilities_finite': bool(np.isfinite(probabilities).all()),
              'class_labels': classes, 'prediction_type': task.prediction_type, 'score': None}
    for path in paths:
        path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix('.csv.tmp');result.to_csv(pending, index=False);pending.replace(path)
    return checks
