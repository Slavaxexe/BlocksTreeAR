"""Measured classification EDA, class-balance monitoring, and fit-local weights."""
import numpy as np
import pandas as pd


BALANCE_METRICS = {'f1', 'f1_macro', 'balanced_accuracy'}


def class_profile(y, classes):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=len(classes))
    observed = counts[counts > 0]
    majority = int(counts.max()) if len(counts) else 0
    return {'rows': len(y), 'classes': len(classes),
        'majority_to_minority_ratio': float(majority/observed.min()) if len(observed) else None,
        'missing_classes': [str(classes[i]) for i in range(len(classes)) if counts[i] == 0],
        'per_class': [{'label': str(label), 'count': int(counts[i]),
            'share': float(counts[i]/max(1, len(y))),
            'minority': bool(counts[i] > 0 and counts[i] < majority*.2),
            'few_examples': bool(counts[i] < 100)} for i, label in enumerate(classes)]}


def fit_class_weights(y, power, clip):
    """Only source labels enter weights; no validation/test class frequencies."""
    y = np.asarray(y, dtype=int)
    if not len(y) or power == 0:
        return None
    counts = np.bincount(y)
    weights = np.minimum((counts.max()/np.maximum(counts, 1))**power, clip)[y]
    return weights/weights.mean()


def dataset_profile(train, test, target, ids, max_rows=100000):
    columns = [name for name in test if name not in ids]
    sample = train.sample(min(max_rows, len(train)), random_state=42)
    other = test.sample(min(max_rows, len(test)), random_state=42)
    placeholders, low_cardinality = [], []
    for name in columns:
        values = sample[name]
        if pd.api.types.is_numeric_dtype(values):
            unique = int(values.nunique())
            if 2 < unique <= 20:
                low_cardinality.append({'feature': name, 'unique': unique})
        else:
            for value in ('-', '?', 'unknown', 'Unknown', ''):
                count = int(values.eq(value).sum())
                if count:
                    placeholders.append({'feature': name, 'value': value, 'rows': count,
                        'share': count/len(sample), 'interpretation': 'candidate absence/category, not automatically NaN'})
    result = {'sample_rows': len(sample), 'total_train_rows': len(train),
        'sampled': len(sample) < len(train), 'placeholder_candidates': placeholders[:20],
        'potential_categorical_numeric': low_cardinality[:20]}
    if len(columns) <= 200:
        keys = pd.util.hash_pandas_object(sample[columns], index=False)
        test_keys = pd.util.hash_pandas_object(other[columns], index=False)
        table = pd.DataFrame({'key': keys.to_numpy(), 'label': sample[target].to_numpy()})
        conflicts = table.groupby('key', sort=False).label.nunique()
        ambiguous = table.key.isin(conflicts.index[conflicts > 1])
        result['duplicates'] = {'extra_copies': int(keys.duplicated().sum()),
            'conflicting_label_groups': int((conflicts > 1).sum()),
            'rows_in_conflicting_groups': int(ambiguous.sum()),
            'conflicting_row_fraction': float(ambiguous.mean()),
            'test_feature_match_fraction': float(test_keys.isin(set(keys)).mean()),
            'test_sample_rows': len(other), 'scope': 'all non-ID input features; sampled when reported'}
    return result


def prediction_monitor(y, probabilities, classes, metric, training, threshold=.5, positive=1):
    from ac_models import labels_from_probabilities
    y = np.asarray(y, dtype=int)
    predicted = labels_from_probabilities(probabilities, threshold, positive)
    n_classes = len(classes)
    if len(y) != len(predicted):
        raise ValueError('Balance monitor labels/predictions differ in length')
    cm = np.bincount(y*n_classes+predicted, minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    support, predicted_count = cm.sum(axis=1), cm.sum(axis=0)
    tp = np.diag(cm)
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
    precision = np.divide(tp, predicted_count, out=np.zeros(n_classes), where=predicted_count > 0)
    f1 = np.divide(2*tp, support+predicted_count, out=np.zeros(n_classes), where=(support+predicted_count) > 0)
    train_by_class = {row['label']: row for row in training['per_class']}
    rows, alerts = [], []
    for i, label in enumerate(classes):
        info = train_by_class[str(label)]
        under = bool(support[i] > 0 and predicted_count[i] < .75*support[i] and recall[i] < .5)
        row = {'label': str(label), 'train_count': info['count'], 'train_share': info['share'],
            'validation_count': int(support[i]), 'predicted_count': int(predicted_count[i]),
            'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]),
            'minority': info['minority'], 'few_examples': bool(support[i] < 30),
            'underpredicted': under, 'zero_recall': bool(support[i] > 0 and tp[i] == 0)}
        rows.append(row)
        if under or row['zero_recall']:
            alerts.append({'label': str(label), 'underpredicted': under,
                'zero_recall': row['zero_recall'], 'support': int(support[i])})
    # At most 12 confusion pairs; avoid allocating a giant dense diagnostic list.
    errors = cm.copy();np.fill_diagonal(errors, 0)
    flat = errors.ravel();top = np.argsort(flat)[-12:][::-1]
    confusions = [{'true': str(classes[index//n_classes]), 'predicted': str(classes[index % n_classes]),
        'rows': int(flat[index]), 'share_of_true_class': float(flat[index]/support[index//n_classes])}
        for index in top if flat[index] > 0]
    ratio = training['majority_to_minority_ratio'] or 1.
    try_weights = bool(metric in BALANCE_METRICS and ratio >= 5 and any(
        row['minority'] and row['underpredicted'] for row in rows))
    return {'metric': metric, 'validation_rows': len(y), 'per_class': rows,
        'alerts': alerts, 'top_confusions': confusions,
        'soft_weights_worth_testing': try_weights,
        'recommended_actions': (['Test clipped soft class weights against the current winner on the SAME folds; keep only primary-metric improvements.'] if try_weights else [])
            + (['Inspect feature ambiguity and probability calibration; class frequency alone does not justify full balanced weights.'] if alerts else []),
        'scope': 'Observed validation labels/predictions; no hidden-test labels or inferred test class counts'}


def compact_monitor(monitor, max_classes=24):
    if not monitor:
        return None
    result = {key: value for key, value in monitor.items() if key != 'per_class'}
    rows = monitor['per_class']
    result['per_class'] = sorted(rows, key=lambda row: (not row['underpredicted'], row['f1']))[:max_classes]
    result['omitted_classes'] = max(0, len(rows)-max_classes)
    return result
