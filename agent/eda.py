"""Executable feature-analysis catalog. Outputs are measurements, never conclusions."""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from .features import create_features
from .plans import AnalysisRequest


CATALOG = {
    'quality': 'Missing/nonfinite rates, cardinality, constants and numeric quantiles.',
    'association': 'Univariate mutual information with fit target, in nats; not causality or validation quality.',
    'correlation': 'Pairwise numeric Spearman correlation; inspect redundancy, not causality.',
    'shift': 'Fit versus validation/test: numeric KS and categorical total variation/unseen rates; no hidden labels.',
    'groups': 'Feature summaries and available target distributions by group, with group sizes.',
    'redundancy': 'Exact duplicate columns and duplicated rows across specified features.',
    'missingness': 'Association between missing-value indicators and fit target.',
    'interactions': 'Joint predictive score versus each individual feature on the same internal holdout.',
    'predictive': 'A reproducible proxy model on selected features, using an internal fit-only holdout.',
    'permutation': 'Permutation importance on internal holdout, not training importance.',
    'ablation': 'Retrain with/without selected features or groups on the same internal holdout.',
    'feature': 'Create proposed feature recipes; compare baseline and augmented proxy models on internal holdout.',
    'stability': 'Association stability across specified groups, or two disjoint seeded fit subsets.',
}

SCORERS = {
    'rmse': 'neg_root_mean_squared_error', 'rootmeansquarederror': 'neg_root_mean_squared_error',
    'mse': 'neg_mean_squared_error', 'meansquarederror': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error', 'meanabsoluteerror': 'neg_mean_absolute_error',
    'r2': 'r2', 'r2score': 'r2', 'accuracy': 'accuracy', 'accuracyscore': 'accuracy',
    'f1': 'f1', 'macrof1': 'f1_macro', 'f1macro': 'f1_macro', 'weightedf1': 'f1_weighted',
    'logloss': 'neg_log_loss', 'rocauc': 'roc_auc', 'auc': 'roc_auc',
}


def analysis_key(request):
    from .storage import fingerprint
    request = request if isinstance(request, AnalysisRequest) else AnalysisRequest.model_validate(request)
    value = request.model_dump(exclude={'name'})
    value['columns'] = sorted(value['columns'])
    value['baseline_columns'] = sorted(value['baseline_columns'])
    return fingerprint(value)


def _number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _top(values, count):
    return values if count is None else values[:count]


def _population(frame, filters):
    for condition in filters:
        values = frame[condition.column]
        if condition.operation in {'isna', 'notna'}:
            mask = getattr(values, condition.operation)()
        else:
            mask = getattr(values, condition.operation)(condition.value)
        frame = frame.loc[mask.fillna(False)]
    return frame


def _matrix(frame, columns):
    frame = frame[columns].reset_index(drop=True).copy()
    for name in columns:
        if pd.api.types.is_numeric_dtype(frame[name]):
            frame[name] = frame[name].replace([np.inf, -np.inf], np.nan)
        else:
            frame[name] = frame[name].astype('string').fillna('__missing__').astype(str)
    return frame


def _association(frame, columns, target, classification, top_k, seed):
    if target not in frame:
        return {'status': 'unavailable', 'reason': 'This phase has no row-level target'}
    clean = frame.dropna(subset=[target])
    if len(clean) < 4 or clean[target].nunique() < 2:
        return {'status': 'unavailable', 'reason': 'Insufficient target variation/rows'}
    data, discrete, names = {}, [], []
    for name in columns:
        values = clean[name]
        if values.nunique(dropna=False) <= 1:
            continue
        categorical = not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
        if categorical:
            data[name] = pd.factorize(values.astype('string').fillna('__missing__'))[0]
        else:
            values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
            data[name] = values.fillna(values.median()).fillna(0)
        discrete.append(categorical)
        names.append(name)
    if not names:
        return {'status': 'completed', 'ranking': [], 'constant_features': columns}
    y = LabelEncoder().fit_transform(clean[target]) if classification else clean[target].to_numpy(dtype=float)
    finite = np.isfinite(y)
    function = mutual_info_classif if classification else mutual_info_regression
    scores = function(pd.DataFrame(data).to_numpy()[finite], y[finite], discrete_features=discrete,
                      random_state=seed, n_neighbors=min(3, int(finite.sum()) - 1), n_jobs=1)
    ranking = sorted([{'feature': name, 'mutual_information_nats': float(value)}
                      for name, value in zip(names, scores)], key=lambda row: -row['mutual_information_nats'])
    return {'status': 'completed', 'rows': int(finite.sum()), 'ranking': _top(ranking, top_k),
            'qualification': 'Univariate dependency estimate; zero is not proof of no joint signal.'}


def _model(frame, classification, estimator, seed):
    numeric = list(frame.select_dtypes(include='number').columns)
    categorical = [column for column in frame if column not in numeric]
    transform = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), numeric),
        ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical),
    ])
    if estimator == 'extra_trees':
        factory = ExtraTreesClassifier if classification else ExtraTreesRegressor
        model = factory(n_estimators=64, min_samples_leaf=2, n_jobs=1, random_state=seed)
    elif estimator == 'hist_gradient_boosting':
        factory = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        model = factory(max_iter=75, random_state=seed)
    else:
        model = LogisticRegression(max_iter=500) if classification else Ridge()
    return make_pipeline(transform, model)


def _inner_split(data, y, request, policy, classification):
    strategy = request.validation
    if strategy == 'auto':
        description = str(policy.get('strategy', '')).lower()
        strategy = 'group' if policy.get('group_columns') or 'group' in description else (
            'time' if any(word in description for word in ('time', 'temporal', 'chronological')) else 'random')
    if strategy == 'group':
        column = request.group_by or next(iter(policy.get('group_columns', [])), None)
        if not column:
            raise ValueError('Group proxy validation needs group_by')
        groups = data[column].astype('string').fillna('__missing__')
        return next(GroupShuffleSplit(n_splits=1, test_size=request.validation_fraction,
                                      random_state=request.seed).split(data, y, groups))
    if strategy == 'time':
        column = request.split_column or policy.get('split_column') or policy.get('time_column')
        if not column or data[column].isna().any():
            raise ValueError('Temporal proxy validation needs a non-missing split_column')
        periods = sorted(data[column].unique())
        if len(periods) < 2:
            raise ValueError('Temporal proxy validation needs at least two distinct periods')
        cut = max(1, int(len(periods) * (1 - request.validation_fraction)))
        validation = data[column].isin(periods[cut:]).to_numpy()
        return np.flatnonzero(~validation), np.flatnonzero(validation)
    return train_test_split(np.arange(len(data)), test_size=request.validation_fraction,
                            random_state=request.seed, stratify=y if classification else None)


def _predictive(frame, columns, target, classification, request, metric, cache, recipes=(), policy=None):
    scorer_name = SCORERS.get(''.join(c for c in metric.lower() if c.isalnum()))
    if scorer_name is None:
        raise ValueError(f'No verified proxy scorer for {metric!r}; choose a descriptive analysis')
    if not columns:
        raise ValueError('A proxy model needs at least one input feature')
    key = (tuple(frame.index), tuple(columns), target, request.estimator, request.seed, request.validation,
           request.validation_fraction, request.group_by, request.split_column,
           tuple(recipe.model_dump_json() for recipe in recipes))
    if key not in cache:
        data = frame.dropna(subset=[target]).reset_index(drop=True)
        y = LabelEncoder().fit_transform(data[target]) if classification else data[target].to_numpy(dtype=float)
        finite = np.isfinite(y)
        data, y = data.loc[finite].reset_index(drop=True), y[finite]
        train, valid = _inner_split(data, y, request, policy or {}, classification)
        # Recipe statistics are learned on the inner training rows, not on its holdout.
        # Recipe inputs may be needed for calculation without being model features.
        raw_columns = list(dict.fromkeys([*columns, *(column for recipe in recipes for column in recipe.columns if column in data)]))
        x_train, x_valid = create_features([data.iloc[train][raw_columns], data.iloc[valid][raw_columns]],
                                          recipes, forbidden=[target])
        all_columns = list(dict.fromkeys([*columns, *(recipe.name for recipe in recipes)]))
        x_train, x_valid = _matrix(x_train, all_columns), _matrix(x_valid, all_columns)
        model = _model(x_train, classification, request.estimator, request.seed)
        model.fit(x_train, y[train])
        scorer = get_scorer(scorer_name)
        score = float(scorer(model, x_valid, y[valid]))
        cache[key] = (model, x_valid, y[valid], scorer, score, len(train))
    model, valid, y_valid, scorer, score, train_rows = cache[key]
    negative = scorer_name.startswith('neg_')
    result = {'metric': metric, 'value': -score if negative else score,
              'direction': 'minimize' if negative else 'maximize', 'estimator': request.estimator,
              'inner_train_rows': train_rows, 'inner_validation_rows': len(valid),
              'feature_count': valid.shape[1], 'comparison_score': score,
              'scope': 'Proxy evaluation inside fit only; not the experiment outer-validation score'}
    return result, (model, valid, y_valid, scorer)


def analyze_features(state, value, *, task_type=None, cache=None):
    """Tool-like API: dataset state plus a validated request -> scoped measurements."""
    request = value if isinstance(value, AnalysisRequest) else AnalysisRequest.model_validate(value)
    cache = {} if cache is None else cache
    if request.dataset != 'fit' and request.kind not in {'quality', 'correlation', 'redundancy', 'groups'}:
        raise ValueError('Label-dependent and shift analyses must start from fit')
    frame = _population(state[request.dataset], request.filters)
    preparation = state['preparation']
    targets = preparation['data']['target_columns']
    ids = preparation['data'].get('id_columns', [])
    columns = sorted(request.columns or [c for c in frame if c not in {*targets, *ids}])
    if set(columns) & set(targets):
        raise ValueError('Use target= for the response; columns must contain input features only')
    if not set(columns) <= set(frame):
        raise ValueError(f'Unknown analysis features: {sorted(set(columns) - set(frame))}')
    target = request.target or (targets[0] if len(targets) == 1 else None)
    if target is not None and target not in targets:
        raise ValueError('Analysis target must be a declared task target')
    if request.sample_rows and len(frame) > request.sample_rows:
        frame = frame.sample(request.sample_rows, random_state=request.seed)
    classification = ('class' in str(task_type).lower() or 'binary' in str(task_type).lower()) if task_type else (
        target in frame and not pd.api.types.is_numeric_dtype(frame[target]))
    result = {'kind': request.kind, 'scope': request.dataset, 'rows_used': len(frame), 'rows_available': len(state[request.dataset]),
              'target_available': target in frame, 'features': columns,
              'filters': [item.model_dump() for item in request.filters]}
    if request.kind == 'quality':
        result['columns'] = {name: {'missing_rate': float(frame[name].isna().mean()),
            'unique': int(frame[name].nunique()), 'constant': bool(frame[name].nunique(dropna=False) <= 1),
            'quantiles': {str(q): _number(v) for q, v in frame[name].astype(float).quantile([0, .01, .5, .99, 1]).items()}
                         if pd.api.types.is_numeric_dtype(frame[name]) else {}}
                             for name in columns}
    elif request.kind in {'association', 'missingness'}:
        if request.kind == 'missingness':
            data = frame[columns].isna()
            if target in frame:
                data[target] = frame[target]
        else:
            data = frame
        result.update(_association(data, columns, target, classification, request.top_k, request.seed))
    elif request.kind == 'correlation':
        corr = frame[columns].select_dtypes(include='number').corr(method='spearman')
        pairs = [{'a': a, 'b': b, 'spearman': _number(corr.loc[a, b])}
                 for a, b in itertools.combinations(corr.columns, 2) if pd.notna(corr.loc[a, b])]
        result['pairs'] = _top(sorted(pairs, key=lambda row: -abs(row['spearman'])), request.top_k)
    elif request.kind == 'redundancy':
        duplicates = []
        for a, b in itertools.combinations(columns, 2):
            if frame[a].equals(frame[b]):
                duplicates.append([a, b])
        result.update(duplicate_columns=duplicates, duplicated_feature_rows=int(frame[columns].duplicated().sum()))
    elif request.kind == 'shift':
        other = _population(state[request.compare], request.comparison_filters)
        if request.sample_rows and len(other) > request.sample_rows:
            other = other.sample(request.sample_rows, random_state=request.seed)
        measurements = []
        for name in columns:
            a, b = frame[name], other[name]
            row = {'feature': name, 'missing_rate_fit': float(a.isna().mean()),
                   'missing_rate_other': float(b.isna().mean())}
            if pd.api.types.is_numeric_dtype(a):
                a, b = a.replace([np.inf, -np.inf], np.nan).dropna(), b.replace([np.inf, -np.inf], np.nan).dropna()
                row.update(statistic='KS', distance=float(ks_2samp(a, b).statistic) if len(a) and len(b) else None)
            else:
                a, b = a.astype('string').fillna('__missing__'), b.astype('string').fillna('__missing__')
                frequencies = pd.concat([a.value_counts(normalize=True), b.value_counts(normalize=True)], axis=1).fillna(0)
                row.update(statistic='total_variation', distance=float(abs(frequencies.iloc[:, 0] - frequencies.iloc[:, 1]).sum() / 2),
                           unseen_rate_other=float((~b.isin(set(a))).mean()) if len(b) else None)
            measurements.append(row)
        result.update(compare=request.compare, comparison_rows=len(other),
                      comparison_filters=[item.model_dump() for item in request.comparison_filters],
                      shifts=_top(sorted(measurements, key=lambda row: -(row['distance'] or 0)), request.top_k))
    elif request.kind == 'groups':
        if request.group_by is None or request.group_by in targets:
            raise ValueError('groups needs a non-target group_by column')
        groups = frame.groupby(request.group_by, dropna=False, observed=True)
        sizes = groups.size().sort_values(ascending=False)
        result['groups'] = []
        for group in _top(list(sizes.index), request.top_k):
            subset = frame[frame[request.group_by].isna()] if pd.isna(group) else groups.get_group(group)
            row = {'group': str(group), 'rows': len(subset),
                   'feature_means': {c: _number(subset[c].mean()) for c in columns if pd.api.types.is_numeric_dtype(subset[c])}}
            if target in subset:
                row['target_distribution'] = {str(k): float(v) for k, v in subset[target].value_counts(normalize=True).items()} if classification else {'mean': _number(subset[target].mean())}
            result['groups'].append(row)
    elif request.kind == 'stability':
        if request.group_by:
            groups = list(frame.groupby(request.group_by, dropna=False, observed=True))
        else:
            parts = np.array_split(frame.sample(frac=1, random_state=request.seed), 2)
            groups = list(enumerate(parts))
        result['subsets'] = [{'group': str(name), 'rows': len(part),
                              'association': _association(part, columns, target, classification, request.top_k, request.seed)}
                             for name, part in _top(groups, request.top_k)]
    else:
        if target not in frame:
            return {**result, 'status': 'unavailable', 'reason': 'Supervised evaluation requires fit-phase targets'}
        metric = preparation['metric']['name']
        baseline_columns = request.baseline_columns or (columns if request.kind != 'ablation' else
            [c for c in frame if c not in {*targets, *ids}])
        if set(baseline_columns) & set(targets):
            raise ValueError('Baseline inputs cannot contain targets')
        policy = preparation.get('validation', {})
        baseline, fitted = _predictive(frame, baseline_columns, target, classification, request, metric, cache, policy=policy)
        result['baseline'] = baseline
        if request.kind == 'permutation':
            model, valid, labels, scorer = fitted
            scores = permutation_importance(model, valid, labels, scoring=scorer, n_repeats=request.repeats,
                                            random_state=request.seed, n_jobs=1)
            result['importance'] = _top(sorted([{'feature': c, 'mean_score_drop': float(mean), 'std': float(std)}
                for c, mean, std in zip(valid.columns, scores.importances_mean, scores.importances_std)],
                key=lambda row: -row['mean_score_drop']), request.top_k)
        elif request.kind in {'ablation', 'feature'}:
            changed = [c for c in baseline_columns if c not in columns] if request.kind == 'ablation' else baseline_columns
            candidate, _ = _predictive(frame, changed, target, classification, request, metric, cache,
                                      recipes=request.features if request.kind == 'feature' else (), policy=policy)
            result.update(candidate=candidate,
                          improvement=candidate['comparison_score'] - baseline['comparison_score'])
        elif request.kind == 'interactions':
            singles = [_predictive(frame, [c], target, classification, request, metric, cache, policy=policy)[0] for c in columns]
            result.update(single_features=dict(zip(columns, singles)),
                          joint_gain_over_best_single=baseline['comparison_score'] - max(row['comparison_score'] for row in singles))
    return result


def run_analyses(state, requests, *, task_type=None):
    results, cache = {}, {}
    for value in requests:
        request = value if isinstance(value, AnalysisRequest) else AnalysisRequest.model_validate(value)
        results[request.name] = analyze_features(state, request, task_type=task_type, cache=cache)
    return results
