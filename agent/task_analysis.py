"""Task-specific EDA measurements, scoped to supplied tables and fit labels."""
import numpy as np
import pandas as pd

from .task_types import group_codes


def predictive_task(state, frame, request, columns, task_type):
    import time
    from .eda import _inner_split, SCORERS
    from .features import create_features
    from .operations import run_operation
    from .task_types import family, task_score
    from .validation import metadata
    binding = dict(state['preparation']['data'])
    targets = binding['target_columns']
    kind = family(task_type, targets, frame)
    if kind != 'clustering' and (not targets or not set(targets) <= set(frame)):
        return {'status': 'unavailable', 'reason': 'Supervised proxy requires fit-only targets'}
    policy = metadata(state['preparation'].get('validation', {}))
    if kind == 'ranking':
        binding['query_columns'] = request.query_columns or binding.get('query_columns', [])
        group_codes(frame, binding['query_columns'])
        policy.update(strategy='group', group_columns=binding['query_columns'])
    elif kind == 'forecasting':
        binding['time_column'] = request.split_column or binding.get('time_column')
        binding['series_columns'] = request.series_columns or binding.get('series_columns', [])
        policy.update(strategy='time', time_column=binding['time_column'])
    y = frame[targets].to_numpy() if targets else np.empty((len(frame), 0))
    train, valid = _inner_split(frame, y, request, policy, False)
    if request.sample_rows and len(frame) > request.sample_rows and kind not in {'ranking', 'forecasting'}:
        rng = np.random.default_rng(request.seed)
        train = np.sort(rng.choice(train, min(len(train), max(2, int(request.sample_rows * .75))), replace=False))
        valid = np.sort(rng.choice(valid, min(len(valid), max(2, int(request.sample_rows * .25))), replace=False))
    fit, reference = frame.iloc[train].copy(), frame.iloc[valid].copy()
    hidden = reference.drop(columns=targets)
    recipes = list({recipe.name: recipe.model_dump() for recipe in [*request.baseline_features, *request.features]}.values())
    fit, hidden = create_features([fit, hidden], recipes, forbidden=targets, policy=policy)
    inputs = list(dict.fromkeys([*columns, *(recipe['name'] for recipe in recipes)]))
    preparation = {**state['preparation'], 'data': binding, 'validation': policy,
        'runtime': {'is_smoke_check': True, 'num_cpus': 1, 'deadline': time.monotonic() + 60}}
    work = {'fit': fit, 'validation': hidden, 'test': hidden, 'preparation': preparation}
    defaults = {'task_type': task_type}
    classifier = kind == 'classification'
    factories = {
        'extra_trees': ('sklearn.ensemble.ExtraTreesClassifier' if classifier else 'sklearn.ensemble.ExtraTreesRegressor',
                        {'n_estimators': 32, 'min_samples_leaf': 2}),
        'hist_gradient_boosting': ('sklearn.ensemble.HistGradientBoostingClassifier' if classifier else 'sklearn.ensemble.HistGradientBoostingRegressor',
                                   {'max_iter': 75}),
        'linear': ('sklearn.linear_model.LogisticRegression' if classifier else 'sklearn.linear_model.Ridge',
                   {'max_iter': 500} if classifier else {}),
    }
    estimator, parameters = factories[request.estimator]
    if kind == 'forecasting':
        result = run_operation(work, {'kind': 'forecast', 'parameters': {
            'time_column': binding['time_column'], 'series_columns': binding['series_columns'],
            'lags': request.lags, 'feature_columns': inputs,
            'estimator': estimator, 'parameters': parameters}}, defaults)
    else:
        if kind == 'clustering':
            estimator, parameters = 'sklearn.cluster.KMeans', {'n_clusters': request.n_clusters, 'n_init': 3}
        work.update(run_operation(work, {'kind': 'preprocess', 'parameters': {'feature_columns': inputs}}, defaults))
        work.update(run_operation(work, {'kind': 'fit', 'parameters': {'estimator': estimator, 'parameters': parameters}}, defaults))
        result = run_operation(work, {'kind': 'predict'}, defaults)
    prediction = result['validation_predictions']
    value = task_score(reference, prediction, preparation, task_type)
    if value is None:
        from sklearn.metrics import get_scorer
        name = ''.join(char for char in preparation['metric']['name'].lower() if char.isalnum())
        scorer_name = SCORERS.get(name)
        if scorer_name is None or kind == 'forecasting':
            # Numeric forecast metrics use the same value definition, without a second fit.
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            functions = {'rmse': lambda a, b: np.sqrt(mean_squared_error(a, b)),
                         'mse': mean_squared_error, 'mae': mean_absolute_error, 'r2': r2_score}
            if name not in functions:
                return {'status': 'unavailable', 'reason': 'No verified proxy metric for this task'}
            value = functions[name](reference[targets].to_numpy(), prediction.to_numpy())
        else:
            model = work['model']['estimator']
            encoder = work['model']['labels']
            truth = encoder.transform(reference[targets[0]]) if encoder is not None else reference[targets[0]]
            score = get_scorer(scorer_name)(model, work['validation_features'], truth)
            value = -score if scorer_name.startswith('neg_') else score
    return {'status': 'completed', 'metric': preparation['metric']['name'], 'value': float(value),
        'inner_train_rows': len(fit), 'inner_validation_rows': len(reference),
        'feature_columns': inputs, 'recipe_names': [recipe['name'] for recipe in recipes], 'estimator': estimator,
        'task_family': kind, 'scope_note': 'Task-aware proxy inside fit only; not the outer score'}


def inspect_task(state, frame, request, columns, task_type):
    binding = state['preparation']['data']
    targets = binding['target_columns']
    if request.kind == 'targets':
        available = [column for column in targets if column in frame]
        if not available:
            return {'status': 'unavailable', 'reason': 'Target structure requires fit labels'}
        data = frame[available]
        detailed = ([request.target, *[column for column in available if column != request.target]]
                    if request.target else available)[:request.top_k]
        distributions = {column: {'missing': int(data[column].isna().sum()),
            'unique': int(data[column].nunique()),
            'frequent': {str(label): int(count) for label, count in data[column].value_counts().head(request.top_k).items()}}
            for column in detailed}
        numeric = data[detailed].select_dtypes(include='number')
        pairs = []
        for index, a in enumerate(numeric):
            for b in list(numeric)[index + 1:]:
                score = numeric[a].corr(numeric[b], method='spearman')
                if pd.notna(score):
                    pairs.append({'a': a, 'b': b, 'spearman': float(score)})
        result = {'status': 'completed', 'target_count': len(available), 'detailed_targets': detailed,
                  'omitted_target_details': len(available) - len(detailed),
                  'distributions': distributions,
                  'target_correlations': sorted(pairs, key=lambda pair: -abs(pair['spearman']))[:request.top_k]}
        if data.notna().all().all() and data.isin([0, 1, False, True]).all().all():
            binary = data.astype(int)
            result.update(positive_rates=binary[detailed].mean().to_dict(), label_cardinality=float(binary.sum(axis=1).mean()),
                empty_label_rows=int(binary.sum(axis=1).eq(0).sum()),
                cooccurrence=(binary[detailed].T @ binary[detailed]).to_dict())
        return result
    if request.kind == 'queries':
        keys = request.query_columns or binding.get('query_columns', [])
        if not keys:
            return {'status': 'unavailable', 'reason': 'Choose query_columns to measure ranking groups'}
        codes = group_codes(frame, keys)
        groups = frame.groupby(codes, sort=False)
        sizes = groups.size()
        result = {'status': 'completed', 'query_columns': keys, 'queries': len(sizes),
            'size_quantiles': {str(q): float(value) for q, value in sizes.quantile([0, .5, 1]).items()},
            'singleton_queries': int((sizes == 1).sum())}
        target = request.target or (targets[0] if len(targets) == 1 else None)
        if target in frame:
            result.update(constant_relevance_queries=int((groups[target].nunique() <= 1).sum()),
                no_positive_queries=int((groups[target].max() <= 0).sum()))
            centered_y = frame[target] - groups[target].transform('mean')
            signals = []
            for column in columns:
                if column in keys or not pd.api.types.is_numeric_dtype(frame[column]):
                    continue
                centered_x = frame[column] - groups[column].transform('mean')
                score = centered_x.corr(centered_y, method='spearman')
                if pd.notna(score):
                    signals.append({'feature': column, 'within_query_spearman': float(score)})
            result['signals'] = sorted(signals, key=lambda value: -abs(value['within_query_spearman']))[:request.top_k]
        from .eda import _population
        other = _population(state[request.compare], request.comparison_filters)
        result['query_overlap_rate_other'] = float(pd.MultiIndex.from_frame(other[keys]).isin(
            pd.MultiIndex.from_frame(frame[keys])).mean()) if len(other) else None
        return result
    if request.kind == 'series':
        from .supervised_features import ordered_time
        time_column = request.split_column or binding.get('time_column')
        keys = request.series_columns or binding.get('series_columns', [])
        if not time_column:
            return {'status': 'unavailable', 'reason': 'Choose split_column or data.time_column'}
        if any(lag < 1 for lag in request.lags):
            raise ValueError('Series lags must be positive observation counts')
        ordered = frame.assign(__time=ordered_time(frame[time_column])).sort_values('__time')
        grouper = keys if keys else lambda index: 0
        groups = ordered.groupby(grouper, dropna=False, sort=False)
        sizes, gaps = groups.size(), groups['__time'].diff().dropna()
        result = {'status': 'completed', 'time_column': time_column, 'series_columns': keys,
            'series_count': len(sizes), 'duplicate_series_times': int(frame.duplicated([*keys, time_column]).sum()),
            'length_quantiles': {str(q): float(value) for q, value in sizes.quantile([0, .5, 1]).items()},
            'time_step_quantiles': {str(q): float(value) for q, value in gaps.quantile([0, .5, 1]).items()} if len(gaps) else {},
            'lag_unit': 'observations within a series, not calendar intervals', 'target_autocorrelations': []}
        for target in targets:
            if target not in ordered or not pd.api.types.is_numeric_dtype(ordered[target]):
                continue
            for lag in request.lags:
                shifted = groups[target].shift(lag)
                score = ordered[target].corr(shifted)
                result['target_autocorrelations'].append({'target': target, 'lag': lag,
                    'pairs': int(shifted.notna().sum()), 'correlation': float(score) if pd.notna(score) else None})
        from .eda import _population
        other = _population(state[request.compare], request.comparison_filters)
        result['comparison_time_range'] = [str(other[time_column].min()), str(other[time_column].max())] if len(other) else []
        return result
    from sklearn.cluster import KMeans
    from .eda import _matrix, _model
    from .task_types import task_score
    # Reuse the fitted preprocessing recipe, with no target access or hidden-label evaluation.
    transform = _model(_matrix(frame, columns), False, 'linear', request.seed)[0]
    features = transform.fit_transform(_matrix(frame, columns))
    if len(frame) <= request.n_clusters:
        return {'status': 'unavailable', 'reason': 'Cluster proxy needs more rows than clusters'}
    labels = KMeans(n_clusters=request.n_clusters, n_init=3, random_state=request.seed).fit_predict(features)
    predictions = pd.DataFrame({'cluster': labels})
    predictions.attrs['cluster_features'] = features
    config = {**state['preparation'], 'metric': {'name': 'silhouette'}}
    value = task_score(frame, predictions, config, 'clustering')
    return {'status': 'completed', 'metric': 'silhouette', 'value': value,
        'cluster_sizes': {str(label): int(count) for label, count in pd.Series(labels).value_counts().items()},
        'scope_note': 'Fit-only exploratory geometry; not outer validation quality or proof of true classes'}
