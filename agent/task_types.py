"""Small adapters shared by experiment fitting, prediction and scoring."""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score,
    calinski_harabasz_score, davies_bouldin_score, f1_score,
    log_loss, mean_absolute_error, mean_squared_error, ndcg_score, r2_score,
    roc_auc_score, silhouette_score)


def family(task_type, targets=(), frame=None):
    name = str(task_type or '').lower().replace('-', '_').replace(' ', '_')
    for key, words in [('clustering', ('cluster',)), ('ranking', ('rank',)),
                       ('forecasting', ('forecast', 'time_series', 'timeseries')),
                       ('classification', ('class', 'binary', 'multilabel', 'multi_label'))]:
        if any(word in name for word in words):
            return key
    if not name and targets and frame is not None:
        if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in targets):
            return 'classification'
    return 'regression'


def group_codes(frame, columns):
    if not columns:
        raise ValueError('Ranking requires query_columns, separate from document IDs')
    # Composite keys and missing values keep their own identities without string concatenation.
    return pd.factorize(pd.MultiIndex.from_frame(frame[columns]), sort=False)[0]


def fit_ranker(model, features, targets, source, columns, weights=None):
    codes = group_codes(source, columns)
    order = np.argsort(codes, kind='stable')
    x = features.iloc[order] if hasattr(features, 'iloc') else features[order]
    options = {'sample_weight': np.asarray(weights)[order]} if weights is not None else {}
    backend = type(model).__module__.split('.')[0]
    if backend == 'lightgbm':
        options['group'] = np.bincount(codes).tolist()
    elif backend == 'xgboost':
        options['qid'] = codes[order]
        if weights is not None:
            raise ValueError('XGBRanker expects query weights; row_weights cannot be used here')
    elif backend == 'catboost':
        options['group_id'] = codes[order]
    else:
        raise ValueError('Use LGBMRanker, XGBRanker or CatBoostRanker for grouped fitting')
    return model.fit(x, np.asarray(targets)[order], **options)


class TargetModels:
    """Independent target models; probabilities retain separate class spaces."""
    def __init__(self, models, classification):
        self.models, self.classification = models, classification

    def predict(self, features):
        return np.column_stack([model.predict(features).reshape(-1) for model in self.models])

    def predict_proba(self, features):
        return [model.predict_proba(features) for model in self.models]


def fit_targets(model, features, targets, weights, seeds, classification, runtime):
    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier
    from .model_fitting import fit_seeds
    if np.asarray(targets).ndim == 1:
        return fit_seeds(model, features, targets, weights, seeds, classification, runtime)
    fitted = []
    for column in np.asarray(targets).T:
        candidate = clone(model)
        if classification and len(np.unique(column)) == 1:
            fitted.append(DummyClassifier(strategy='most_frequent').fit(features, column, sample_weight=weights))
        else:
            fitted.append(fit_seeds(candidate, features, column, weights, seeds, classification, runtime))
    return TargetModels(fitted, classification)


def multi_probabilities(model, features, labels, state, defaults):
    from .operations import _prediction
    targets = state['preparation']['data']['target_columns']
    mapping = state['preparation']['output'].get('class_columns', {})
    result, positive = {}, []
    for target, encoder, estimator, probabilities in zip(targets, labels, model.models, model.predict_proba(features)):
        classes = encoder.inverse_transform(np.asarray(estimator.classes_, dtype=int))
        if target in mapping:
            for label, column in mapping[target].items():
                indices = np.flatnonzero(classes.astype(str) == str(label))
                result[column] = probabilities[:, indices[0]] if len(indices) else np.zeros(len(features))
        else:
            if not set(encoder.classes_).issubset({0, 1, False, True}):
                raise ValueError('Multiclass output probabilities need output.class_columns for each target')
            indices = np.flatnonzero(classes == 1)
            positive.append(probabilities[:, indices[0]] if len(indices) else np.zeros(len(features)))
    if result and positive:
        raise ValueError('Specify class_columns for every target, or use binary target columns for all outputs')
    return pd.DataFrame(result) if result else _prediction(np.column_stack(positive), state, defaults)


def ranking_score(reference, values, target, query_columns, name, parameters=None):
    parameters = parameters or {}
    match = re.search(r'@(\d+)|(?:at|k)(\d+)$', name)
    k = int(next(value for value in match.groups() if value)) if match else None
    k = parameters.get('top_k', k)
    if k is not None and (type(k) is not int or k < 1):
        raise ValueError('Ranking metric top_k must be a positive integer')
    codes = group_codes(reference, query_columns)
    scores = []
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        truth, prediction = reference[target].to_numpy()[rows], np.asarray(values).reshape(-1)[rows]
        if name.startswith('ndcg'):
            if (truth < 0).any():
                raise ValueError('NDCG requires nonnegative relevance')
            gain = parameters.get('gain', 'linear')
            if gain not in {'linear', 'exponential'}:
                raise ValueError('NDCG gain must be linear or exponential')
            if gain == 'exponential':
                truth = np.exp2(truth.astype(float)) - 1
            score = float(truth[0] > 0) if len(rows) == 1 else ndcg_score([truth], [prediction], k=k)
        else:
            # AP@k divides by min(number of relevant documents, k), including missed positives.
            ordered = (truth[np.argsort(-prediction, kind='stable')] > 0)[:k]
            denominator = min(int((truth > 0).sum()), k or len(truth))
            score = float((np.cumsum(ordered) / np.arange(1, len(ordered) + 1) * ordered).sum() / denominator) if denominator else 0.
        scores.append(score)
    return float(np.mean(scores))


def task_score(reference, predictions, config, task_type):
    """Verified standard metrics for new families; None keeps the task's custom scorer."""
    binding = config['data']
    targets = binding['target_columns']
    kind = family(task_type, targets, reference)
    name = re.sub(r'[^a-z0-9@]', '', str(config['metric']['name']).lower())
    parameters = config['metric'].get('parameters', {})
    values = predictions.to_numpy()
    if kind == 'ranking' and name.startswith(('ndcg', 'map', 'meanaverageprecision')):
        return ranking_score(reference, values, targets[0], binding.get('query_columns', []), name, parameters)
    if kind == 'clustering':
        labels = values.reshape(-1)
        if name in {'ari', 'adjustedrandindex', 'adjustedrandscore'} and targets:
            return float(adjusted_rand_score(reference[targets[0]], labels))
        if name in {'silhouette', 'silhouettescore', 'daviesbouldin', 'daviesbouldinscore', 'calinskiharabasz', 'calinskiharabaszscore'}:
            features = predictions.attrs.get('cluster_features')
            if features is None:
                raise ValueError('Cluster metric needs the fitted model feature representation')
            count = len(np.unique(labels))
            if not 1 < count < len(labels):
                raise ValueError('Cluster metric requires between 2 and n_samples-1 predicted clusters')
            if name.startswith('silhouette'):
                return float(silhouette_score(features, labels, sample_size=min(2000, len(labels)), random_state=42))
            scorer = davies_bouldin_score if name.startswith('davies') else calinski_harabasz_score
            return float(scorer(features, labels))
        return None
    if len(targets) <= 1:
        return None
    truth = reference[targets].to_numpy()
    if kind != 'classification':
        if name in {'rmse', 'rootmeansquarederror'}:
            if parameters.get('rmse_aggregation') == 'mean_target_rmse':
                return float(np.sqrt(mean_squared_error(truth, values, multioutput='raw_values')).mean())
            return float(np.sqrt(mean_squared_error(truth, values)))
        scorers = {'mse': mean_squared_error, 'meansquarederror': mean_squared_error,
                   'mae': mean_absolute_error, 'meanabsoluteerror': mean_absolute_error,
                   'r2': r2_score, 'r2score': r2_score}
        return float(scorers[name](truth, values)) if name in scorers else None
    # Multi-label indicators and multiple categorical targets have different averaging semantics.
    binary = bool(pd.DataFrame(truth).isin([0, 1, False, True]).all().all())
    if name in {'accuracy', 'subsetaccuracy', 'exactmatch'}:
        return float(np.mean(np.all(truth == values, axis=1)))
    if name in {'hammingloss', 'hamming'}:
        return float(np.mean(truth != values))
    if 'f1' in name:
        average = parameters.get('average') or next((value for value in ('micro', 'weighted', 'samples', 'macro') if value in name), 'macro')
        multilabel = any(word in str(task_type).lower().replace('-', '_') for word in ('multilabel', 'multi_label'))
        if binary and multilabel:
            return float(f1_score(truth.astype(int), values.astype(int), average=average, zero_division=0))
        return float(np.mean([f1_score(truth[:, index].tolist(), values[:, index].tolist(), average=average, zero_division=0)
                              for index in range(len(targets))]))
    if name in {'auc', 'rocauc', 'macrorocauc'} and binary:
        return float(roc_auc_score(truth.astype(int), values.astype(float), average=parameters.get('average', 'macro')))
    if name in {'logloss', 'crossentropy'}:
        mapping = config['output'].get('class_columns', {})
        losses = []
        for index, target in enumerate(targets):
            if target in mapping:
                classes = sorted(mapping[target])
                probabilities = predictions[[mapping[target][label] for label in classes]].to_numpy(dtype=float)
                losses.append(log_loss(reference[target].astype(str), probabilities, labels=classes))
            elif binary:
                losses.append(log_loss(truth[:, index].astype(int), values[:, index].astype(float), labels=[0, 1]))
            else:
                return None
        return float(np.mean(losses))
    return None


def score_predictions(reader, reference, predictions, config, task_type):
    value = task_score(reference, predictions, config, task_type)
    return float(reader.score(reference, predictions, config) if value is None else value)
