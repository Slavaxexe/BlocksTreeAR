"""Bounded, aggregate-only validation diagnostics computed after prediction."""

from collections import Counter
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quantiles(values):
    return {str(q): float(v) for q, v in zip((0, .05, .25, .5, .75, .95, 1),
                                            np.quantile(values, (0, .05, .25, .5, .75, .95, 1)))}


def _errors(actual, predicted):
    residual = predicted - actual
    absolute = np.abs(residual)
    denominator = float(np.square(actual - actual.mean()).sum())
    return {'rows': int(len(actual)), 'rmse': float(np.sqrt(np.square(residual).mean())),
            'mae': float(absolute.mean()), 'median_absolute_error': float(np.median(absolute)),
            'mean_residual': float(residual.mean()),
            'r2': float(1 - np.square(residual).sum() / denominator) if len(actual) > 1 and denominator > 0 else None}


def _regression(reference, predictions, preparation, targets):
    ids = preparation['data'].get('id_columns', [])
    columns = [c for c in predictions if c not in ids]
    if all(target in columns for target in targets):
        mapping = {target: target for target in targets}
    elif len(targets) == len(columns) == 1:
        mapping = {targets[0]: columns[0]}
    else:
        raise ValueError('Regression prediction columns do not unambiguously map to targets')
    result = {}
    for target, column in mapping.items():
        actual = pd.to_numeric(reference[target], errors='raise').to_numpy(dtype=float)
        predicted = pd.to_numeric(predictions[column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError('Regression targets or predictions contain missing/non-finite values')
        residual = predicted - actual
        report = {'metrics': _errors(actual, predicted), 'prediction_column': str(column),
                  'residual_definition': 'prediction - actual', 'residual_quantiles': _quantiles(residual),
                  'absolute_error_quantiles': _quantiles(np.abs(residual)), 'target_bins': [], 'groups': []}
        # Bins describe validation errors, never transformations passed back to fitting.
        if np.unique(actual).size > 1:
            bins = pd.qcut(pd.Series(actual), q=min(5, np.unique(actual).size), duplicates='drop')
            for interval in bins.cat.categories:
                mask = (bins == interval).to_numpy()
                if mask.any():
                    report['target_bins'].append({'target_min': float(actual[mask].min()),
                        'target_max': float(actual[mask].max()), **_errors(actual[mask], predicted[mask])})
        excluded = {*targets, *ids}
        group_columns = [c for c in reference if c not in excluded
                         and not pd.api.types.is_numeric_dtype(reference[c])
                         and 2 <= reference[c].nunique(dropna=False) <= 20][:3]
        for group_column in group_columns:
            # Actual group names and aggregate measurements only; no row identifiers.
            groups = reference[group_column].astype('string').fillna('<missing>').reset_index(drop=True)
            for group, count in groups.value_counts().head(8).items():
                if count < 20:
                    continue
                mask = (groups == group).to_numpy()
                report['groups'].append({'column': str(group_column), 'value': str(group)[:120],
                                         **_errors(actual[mask], predicted[mask])})
        result[str(target)] = report
    return result


def _classification(reference, predictions, preparation, requirements, target):
    ids = preparation['data'].get('id_columns', [])
    columns = [c for c in predictions if c not in ids]
    actual = reference[target].reset_index(drop=True)
    if actual.isna().any():
        raise ValueError('Classification validation target contains missing values')
    observed = [_plain(v) for v in pd.unique(actual)]
    declared = requirements.get('problem', {}).get('class_labels', [])
    ordered = []
    for label in declared:
        matches = [value for value in observed if str(value) == str(label)]
        if len(matches) == 1 and matches[0] not in ordered:
            ordered.append(matches[0])
    ordered.extend(value for value in observed if value not in ordered)
    metric = re.sub(r'[^a-z0-9]', '', str(preparation.get('metric', {}).get('name', '')).lower())
    prediction_type = str(requirements.get('output', {}).get('prediction_type') or '').lower()
    probability = metric == 'auc' or metric.startswith('rocauc') or 'logloss' in metric or metric == 'crossentropy' or any(
        word in prediction_type for word in ('probab', 'вероят'))
    conversion = 'supplied labels'
    if probability:
        probability_classes = ordered + [label for label in declared if not any(str(label) == str(value) for value in ordered)]
        if len(columns) == len(probability_classes) and all(sum(str(label) == str(column) for label in probability_classes) == 1
                                               for column in columns):
            classes = [next(label for label in probability_classes if str(label) == str(column)) for column in columns]
            values = predictions[columns].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('Classification probabilities are outside [0, 1] or non-finite')
            if not np.allclose(values.sum(axis=1), 1, atol=1e-3):
                raise ValueError('Multiclass probability rows do not sum to one')
            predicted = pd.Series([classes[i] for i in values.argmax(axis=1)])
            conversion = 'argmax over probability columns explicitly named by classes'
        elif len(columns) == 1 and len(ordered) == 2:
            output = requirements.get('output', {})
            positive = output.get('positive_class')
            if positive is None:
                positive = requirements.get('problem', {}).get('positive_class')
            if positive is None:
                raise ValueError('Binary probability diagnostics require explicit positive_class; class order is not inferred')
            matches = [value for value in ordered if str(value) == str(positive)]
            if len(matches) != 1:
                raise ValueError('Explicit positive_class does not match validation classes')
            positive = matches[0]
            negative = next(label for label in ordered if label != positive)
            values = predictions[columns[0]].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('Binary probabilities are outside [0, 1] or non-finite')
            predicted = pd.Series(np.where(values >= .5, positive, negative))
            conversion = 'explicit positive-class probability, fixed diagnostic threshold 0.5 (not tuned)'
        else:
            raise ValueError('Probability columns do not explicitly map to class labels')
    elif target in columns:
        predicted = predictions[target].reset_index(drop=True)
    elif len(columns) == 1:
        predicted = predictions[columns[0]].reset_index(drop=True)
    else:
        raise ValueError('Classification prediction columns do not unambiguously identify labels')
    if predicted.isna().any():
        raise ValueError('Classification predictions contain missing labels')
    for series in (actual, predicted):
        if pd.api.types.is_numeric_dtype(series) and not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError('Classification labels contain non-finite values')
    ordered.extend(_plain(value) for value in pd.unique(predicted) if _plain(value) not in ordered)
    indexes = {label: i for i, label in enumerate(ordered)}
    truth = np.array([indexes[_plain(value)] for value in actual], dtype=int)
    guess = np.array([indexes[_plain(value)] for value in predicted], dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, guess, labels=np.arange(len(ordered)), zero_division=0)
    predicted_counts = np.bincount(guess, minlength=len(ordered))
    per_class = [{'label': label, 'support': int(support[i]), 'predicted_count': int(predicted_counts[i]),
                  'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
                 for i, label in enumerate(ordered)]
    worst = sorted((row for row in per_class if row['support']), key=lambda row: (row['f1'], -row['support']))[:10]
    pairs = Counter((int(a), int(b)) for a, b in zip(truth, guess) if a != b)
    confusions = [{'actual': ordered[a], 'predicted': ordered[b], 'count': int(count),
                   'fraction_of_actual_class': float(count / support[a])}
                  for (a, b), count in pairs.most_common(10)]
    matrix = None
    if len(ordered) <= 30:
        matrix = np.zeros((len(ordered), len(ordered)), dtype=int)
        np.add.at(matrix, (truth, guess), 1)
        matrix = {'labels': ordered, 'rows': 'actual', 'columns': 'predicted', 'counts': matrix.tolist()}
    return {'metrics': {'accuracy': float(accuracy_score(truth, guess)),
                        'macro_f1': float(f1.mean()), 'weighted_f1': float(np.average(f1, weights=support))},
            'label_policy': 'union of observed validation targets and predictions; absent declared classes not scored',
            'prediction_conversion': conversion, 'class_count': len(ordered),
            'per_class': per_class if len(ordered) <= 30 else worst,
            'per_class_omitted': max(0, len(ordered) - (len(per_class) if len(ordered) <= 30 else len(worst))),
            'confusion_matrix': matrix, 'top_confusions': confusions, 'worst_classes': worst}


def build_diagnostics(reference, predictions, preparation, requirements):
    """Describe an already executed validation prediction; never fit or tune anything."""
    targets = list(preparation.get('data', {}).get('target_columns', []))
    task = str(requirements.get('problem', {}).get('task_type') or '').lower()
    report = {'schema_version': 1, 'status': 'unavailable', 'task_type': task, 'rows': len(reference),
              'targets': targets, 'sections': {}, 'warnings': [],
              'scope': 'aggregate outer-validation diagnostics after prediction; not training/tuning data'}
    try:
        if not len(reference) or len(reference) != len(predictions):
            raise ValueError('Validation and prediction rows are empty or differ in count')
        if not targets or any(target not in reference for target in targets):
            raise ValueError('Validation targets are not explicitly available')
        if any(kind in task for kind in ('multilabel', 'rank', 'forecast', 'survival', 'cluster')):
            raise ValueError('This task type does not have a standard diagnostic mapping')
        if any(kind in task for kind in ('class', 'binary')):
            if len(targets) != 1:
                raise ValueError('Multi-target classification diagnostics are not mapped')
            report['sections']['classification'] = _classification(reference, predictions, preparation, requirements, targets[0])
        elif 'regress' in task:
            report['sections']['regression'] = _regression(reference, predictions, preparation, targets)
        else:
            raise ValueError('Unknown task type; diagnostic metric semantics are not guessed')
        report['status'] = 'completed'
        if 'classification' in report['sections'] and report['sections']['classification']['per_class_omitted']:
            report['warnings'].append('Per-class output truncated; confusion matrix omitted above 30 classes')
        # Strict serialization also rejects arithmetic overflow in otherwise finite inputs.
        json.dumps(report, allow_nan=False)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        report.update(status='unavailable', sections={}, reason=str(error)[:500])
    return report


def quality_summary(report):
    """Keep model-search context compact; full details stay in the node artifact."""
    summary = {key: report[key] for key in ('status', 'rows') if key in report}
    summary['task_type'] = str(report.get('task_type', ''))[:120]
    summary['targets'] = [str(name)[:120] for name in report.get('targets', [])[:3]]
    summary['interpretation'] = 'Measured validation aggregates, not causal explanations; do not tune on outer labels'
    if report.get('status') != 'completed':
        summary['reason'] = report.get('reason', 'Diagnostics unavailable')
        return summary
    sections = report.get('sections', {})
    if 'classification' in sections:
        report = sections['classification']
        summary.update(metrics=report['metrics'], class_count=report['class_count'],
                       worst_classes=report['worst_classes'][:5], top_confusions=report['top_confusions'][:5])
    elif 'regression' in sections:
        summary['regression'] = {name: {'metrics': entry['metrics'],
            'worst_target_bins': sorted(entry['target_bins'], key=lambda row: -row['rmse'])[:2],
            'worst_groups': sorted(entry['groups'], key=lambda row: -row['rmse'])[:2]}
            for name, entry in list(sections['regression'].items())[:3]}
    if len(json.dumps(summary, ensure_ascii=False)) > 4000:
        summary.pop('worst_classes', None)
        summary.pop('top_confusions', None)
        if 'regression' in summary:
            summary['regression'] = {name: {'metrics': entry['metrics']}
                                     for name, entry in summary['regression'].items()}
        summary['details_omitted'] = 'Full aggregates are in quality_diagnostics.json'
    if len(json.dumps(summary, ensure_ascii=False)) > 4000:
        summary = {key: summary[key] for key in ('status', 'task_type', 'rows', 'targets', 'metrics', 'class_count')
                   if key in summary}
        summary['details_omitted'] = 'Full aggregates are in quality_diagnostics.json'
    return summary
