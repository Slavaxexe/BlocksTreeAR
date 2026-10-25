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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from .features import SUPERVISED_GUIDE, create_features
from .plans import AnalysisRequest


CATALOG = {
    'targets': 'Fit-only multi-target distributions, target correlation and binary label prevalence/co-occurrence.',
    'queries': 'Ranking query sizes, constant/empty relevance groups, within-query feature association and query overlap. query_columns or data.query_columns is the composite query key.',
    'series': 'Series lengths, duplicate series/time rows, time spacing and fit-only target autocorrelation at observation lags. split_column=time; series_columns optional; lags=[1,...].',
    'clusters': 'Unsupervised fit-only KMeans proxy on columns; n_clusters controls clusters, with silhouette and cluster sizes. Scale/geometry evidence, not labelled accuracy.',
    'task_predictive': 'Ready task-aware proxy on an internal fit-only holdout: multi-target/multilabel, pointwise ranking scored per complete query, clustering or recursive forecasting. query_columns/series_columns/split_column/lags identify structure. columns selects inputs; features are fitted inside the holdout. Reports the primary metric and actual holdout sizes.',
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
    'ablation': 'Compare the same pipeline before/after removing columns. baseline_columns is the original raw input set; columns lists ONLY removed model inputs. features/baseline_features are applied in both models. At least one input must remain.',
    'feature': 'Compare baseline_columns + baseline_features against the same raw columns + baseline_features + features. Candidate recipes replace matching baseline recipe names. Keep existing encodings in baseline_features to isolate the new change.',
    'stability': 'Association stability across specified groups, or two disjoint seeded fit subsets.',
    'entities': 'Test proposed key_candidates (lists of columns), or columns as one composite key. Reports repeats, duplicate key/period ambiguity, comparison-table overlap and strictly earlier-history coverage. split_column supplies ordered time. Full filtered tables, no target access needed; overlap is not proof of entity identity.',
}
CATALOG['feature'] += ' ' + SUPERVISED_GUIDE
CATALOG['feature'] += ' For supervised recipes sample_rows caps proxy model fitting/scoring, not the available inner-training history or encoding source.'
for kind in ('predictive', 'permutation', 'interactions'):
    CATALOG[kind] += ' Applies all supplied features/baseline_features, fitted only on the inner training rows; reports actual model inputs.'

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
    return fingerprint({'implementation': 3, 'request': value})


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
    if finite.sum() < 4:
        return {'status': 'unavailable', 'reason': 'Insufficient finite target rows'}
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
    from .validation import metadata, structured_holdout
    policy = metadata(policy)
    if request.group_by:
        policy['group_columns'] = [request.group_by]
    if request.split_column:
        policy['time_column'] = request.split_column
    indices = structured_holdout(data, policy, request.validation_fraction, request.seed, request.validation)
    if indices is not None:
        return indices
    return train_test_split(np.arange(len(data)), test_size=request.validation_fraction,
                            random_state=request.seed, stratify=y if classification else None)


def _predictive(frame, columns, target, classification, request, metric, cache, recipes=(), policy=None, removed=()):
    from .validation import metadata
    policy = metadata(policy)
    scorer_name = SCORERS.get(''.join(c for c in metric.lower() if c.isalnum()))
    if scorer_name is None:
        raise ValueError(f'No verified proxy scorer for {metric!r}; choose a descriptive analysis')
    model_columns = list(dict.fromkeys([*columns, *(recipe.name for recipe in recipes)]))
    model_columns = [column for column in model_columns if column not in removed]
    if not model_columns:
        raise ValueError('A proxy model needs at least one input feature')
    key = (tuple(frame.index), tuple(columns), target, request.estimator, request.seed, request.validation,
           request.validation_fraction, request.group_by, request.split_column, request.sample_rows,
           tuple(recipe.model_dump_json() for recipe in recipes), tuple(removed),
           tuple(sorted((str(key), str(value)) for key, value in policy.items())))
    if key not in cache:
        data = frame.dropna(subset=[target]).reset_index(drop=True)
        y = LabelEncoder().fit_transform(data[target]) if classification else data[target].to_numpy(dtype=float)
        finite = np.isfinite(y)
        data, y = data.loc[finite].reset_index(drop=True), y[finite]
        train, valid = _inner_split(data, y, request, policy or {}, classification)
        reference_rows = len(train)
        model_rows = np.arange(len(train))
        if request.sample_rows and len(train) + len(valid) > request.sample_rows:
            # The complete inner train remains the statistics source. Only model fitting/scoring is sampled.
            train_budget = min(len(train), max(2, int(request.sample_rows * len(train) / (len(train) + len(valid)))))
            valid_budget = min(len(valid), max(2, request.sample_rows - train_budget))
            rng = np.random.default_rng(request.seed)
            model_rows = np.sort(rng.choice(len(train), train_budget, replace=False))
            valid = np.sort(rng.choice(valid, valid_budget, replace=False))
        # Recipe statistics are learned on the inner training rows, not on its holdout.
        # Recipe inputs may be needed for calculation without being model features.
        dependencies = [column for recipe in recipes for column in [*recipe.columns,
            recipe.parameters.get('time_column'), recipe.parameters.get('group_column'),
            recipe.parameters.get('value_column')] if column in data and column != target]
        policy_columns = [*(policy or {}).get('group_columns', []), (policy or {}).get('time_column'),
                          (policy or {}).get('split_column'), request.group_by, request.split_column]
        dependencies += [column for column in policy_columns if column in data and column != target]
        raw_columns = list(dict.fromkeys([*columns, *dependencies]))
        training = data.iloc[train][list(dict.fromkeys([*raw_columns, target]))]
        recipe_policy = dict(policy or {})
        if request.validation != 'auto':
            recipe_policy['strategy'] = request.validation
            recipe_policy['group_columns'] = [request.group_by] if request.validation == 'group' and request.group_by else []
            if request.split_column:
                recipe_policy['time_column'] = request.split_column
        x_train, x_valid = create_features([training, data.iloc[valid][raw_columns]],
                                          recipes, forbidden=[target], policy=recipe_policy)
        from .supervised_features import history
        history_masks = {recipe.name: np.isfinite(history([x_train, x_valid], recipe.columns,
            {**recipe.parameters, 'fallback': 'missing'})[1]) for recipe in recipes
            if recipe.operation == 'history' and recipe.name in model_columns}
        x_train, x_valid = _matrix(x_train.iloc[model_rows], model_columns), _matrix(x_valid, model_columns)
        model = _model(x_train, classification, request.estimator, request.seed)
        model.fit(x_train, y[train][model_rows])
        scorer = get_scorer(scorer_name)
        score = float(scorer(model, x_valid, y[valid]))
        slices = {}
        for name, covered in history_masks.items():
            slices[name] = {'coverage': float(covered.mean()),
                'coverage_definition': 'Key-specific strictly earlier source values, before any global fallback'}
            for scope, mask in [('with_history', covered), ('without_history', ~covered)]:
                subset_score = None
                if mask.any():
                    try:
                        measured = float(scorer(model, x_valid.loc[mask], y[valid][mask]))
                        subset_score = _number(-measured if scorer_name.startswith('neg_') else measured)
                    except ValueError:
                        pass  # Some classification metrics need classes absent from a small slice.
                slices[name][scope] = {'rows': int(mask.sum()), 'value': subset_score}
        from .storage import fingerprint
        split_hash = fingerprint({'reference': train.tolist(), 'train': train[model_rows].tolist(), 'validation': valid.tolist()})
        cache[key] = (model, x_valid, y[valid], scorer, score, len(model_rows), slices, split_hash, reference_rows)
    model, valid, y_valid, scorer, score, train_rows, slices, split_hash, reference_rows = cache[key]
    negative = scorer_name.startswith('neg_')
    result = {'metric': metric, 'value': -score if negative else score,
              'direction': 'minimize' if negative else 'maximize', 'estimator': request.estimator,
              'inner_train_rows': train_rows, 'inner_validation_rows': len(valid),
              'inner_reference_rows': reference_rows,
              'feature_count': valid.shape[1], 'feature_columns': list(valid.columns),
              'recipe_names': [recipe.name for recipe in recipes if recipe.name in valid.columns],
              'inner_split_sha256': split_hash,
              'comparison_score': score,
              'scope': 'Proxy evaluation inside fit only; not the experiment outer-validation score'}
    if slices:
        result['history_slices'] = slices
    estimator = model.steps[-1][1]
    result['recipe_configuration'] = {
        'target_columns': [target],
        'features': [recipe.model_dump() for recipe in recipes],
        'preprocess': {'feature_columns': list(valid.columns), 'encoding': 'ordinal',
                       'numeric_imputer': 'median', 'scaling': 'standard'},
        'fit': {'estimator': '.'.join(type(estimator).__module__.split('.')[:2]) + '.' + type(estimator).__name__,
                'parameters': estimator.get_params(deep=False)},
        'proxy_split_sha256': split_hash, 'proxy_sample_rows': request.sample_rows,
        'scope': 'Same feature/model recipe; main experiment refits on full fit and scores outer validation'}
    return result, (model, valid, y_valid, scorer)


def analyze_features(state, value, *, task_type=None, cache=None):
    """Tool-like API: dataset state plus a validated request -> scoped measurements."""
    request = value if isinstance(value, AnalysisRequest) else AnalysisRequest.model_validate(value)
    cache = {} if cache is None else cache
    if request.dataset != 'fit' and request.kind not in {'quality', 'correlation', 'redundancy', 'groups', 'queries', 'series'}:
        raise ValueError('Label-dependent and shift analyses must start from fit')
    frame = _population(state[request.dataset], request.filters)
    preparation = state['preparation']
    targets = preparation['data']['target_columns']
    ids = preparation['data'].get('id_columns', [])
    recipes = list({recipe.name: recipe for recipe in [*request.baseline_features, *request.features]}.values())
    columns = sorted(request.columns or [c for c in frame if c not in {*targets, *ids}])
    if request.kind in {'groups', 'stability'}:
        # The target has its own summary below; listing it twice is redundant, not an input feature.
        columns = [column for column in columns if column not in targets]
    if request.kind == 'shift' and set(columns) & set(targets):
        return {'kind': request.kind, 'scope': request.dataset, 'status': 'unavailable',
                'reason': 'Held-out target distributions are unavailable: validation/test labels are hidden'}
    if set(columns) & set(targets):
        raise ValueError('Use target= for the response; columns must contain input features only')
    available = set(frame) | ({recipe.name for recipe in recipes} if request.kind == 'ablation' else set())
    if not set(columns) <= available:
        raise ValueError(f'Unknown analysis features: {sorted(set(columns) - available)}')
    target = request.target or (targets[0] if len(targets) == 1 else None)
    if target is not None and target not in targets:
        raise ValueError('Analysis target must be a declared task target')
    preserve_reference = any(recipe.operation in {'history', 'target_encoding'} for recipe in recipes)
    if request.kind not in {'entities', 'series', 'queries', 'task_predictive'} and not preserve_reference and request.sample_rows and len(frame) > request.sample_rows:
        frame = frame.sample(request.sample_rows, random_state=request.seed)
    from .task_types import family
    classification = family(task_type, [target] if target in frame else [], frame) == 'classification'
    result = {'kind': request.kind, 'scope': request.dataset, 'rows_used': len(frame), 'rows_available': len(state[request.dataset]),
              'target_available': target in frame, 'features': columns,
              'filters': [item.model_dump() for item in request.filters]}
    if request.kind in {'targets', 'queries', 'series', 'task_predictive'}:
        result['target_available'] = bool(targets) and all(column in frame for column in targets)
    if frame.empty:
        return {**result, 'status': 'unavailable', 'reason': 'No rows match the selected population'}
    if request.group_by and request.group_by not in frame:
        raise ValueError(f'Unknown grouping column: {request.group_by}')
    if request.kind in {'targets', 'queries', 'series', 'clusters', 'task_predictive'}:
        from .task_analysis import inspect_task, predictive_task
        if request.kind == 'task_predictive':
            result.update(predictive_task(state, frame, request, columns, task_type))
        else:
            result.update(inspect_task(state, frame, request, columns, task_type))
    elif request.kind == 'entities':
        from .entity_analysis import inspect_entities
        other = _population(state[request.compare], request.comparison_filters)
        result.update(inspect_entities(frame, other, request, targets))
    elif request.kind == 'quality':
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
        if other.empty:
            return {**result, 'status': 'unavailable', 'comparison_rows': 0,
                    'reason': 'No rows match the comparison population'}
        if not set(columns) <= set(other):
            raise ValueError(f'Comparison table lacks columns: {sorted(set(columns) - set(other))}')
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
                row['target_distribution'] = {str(k): float(v) for k, v in subset[target].value_counts(normalize=True).items()} if classification else {
                    name: _number(getattr(subset[target], name)()) for name in ('mean', 'std', 'min', 'max', 'median')}
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
        if not set(baseline_columns) <= set(frame):
            raise ValueError(f'Unknown baseline inputs: {sorted(set(baseline_columns) - set(frame))}')
        if request.kind == 'ablation':
            inputs = set(baseline_columns) | {recipe.name for recipe in recipes}
            if not request.columns or not set(columns) <= inputs:
                raise ValueError('Ablation columns must name only removed baseline inputs; set baseline_columns explicitly')
            if not inputs - set(columns):
                raise ValueError('Ablation removes every model input; retain at least one feature')
        policy = preparation.get('validation', {})
        baseline_recipes = request.baseline_features if request.kind == 'feature' else recipes
        baseline, fitted = _predictive(frame, baseline_columns, target, classification, request, metric, cache,
                                       recipes=baseline_recipes, policy=policy)
        result['baseline'] = baseline
        if request.kind == 'permutation':
            model, valid, labels, scorer = fitted
            scores = permutation_importance(model, valid, labels, scoring=scorer, n_repeats=request.repeats,
                                            random_state=request.seed, n_jobs=1)
            result['importance'] = _top(sorted([{'feature': c, 'mean_score_drop': float(mean), 'std': float(std)}
                for c, mean, std in zip(valid.columns, scores.importances_mean, scores.importances_std)],
                key=lambda row: -row['mean_score_drop']), request.top_k)
        elif request.kind in {'ablation', 'feature'}:
            candidate, _ = _predictive(frame, baseline_columns, target, classification, request, metric, cache,
                                      recipes=recipes, policy=policy, removed=columns if request.kind == 'ablation' else ())
            result.update(candidate=candidate,
                          improvement=candidate['comparison_score'] - baseline['comparison_score'])
            result['absolute_improvement'] = result['improvement']
            result['relative_improvement'] = (result['improvement'] / abs(baseline['value'])
                                               if baseline['value'] else None)
            result['improvement_percent'] = (100 * result['relative_improvement']
                                              if result['relative_improvement'] is not None else None)
            previous = {recipe.name: recipe for recipe in baseline_recipes}
            result['comparison'] = {
                'added_recipes': [recipe.name for recipe in recipes if recipe.name not in previous],
                'modified_recipes': [recipe.name for recipe in recipes
                    if recipe.name in previous and recipe != previous[recipe.name]],
                'removed_inputs': columns if request.kind == 'ablation' else [],
                'same_inner_split': baseline['inner_split_sha256'] == candidate['inner_split_sha256']}
        elif request.kind == 'interactions':
            singles = [_predictive(frame, [c], target, classification, request, metric, cache,
                                   recipes=recipes, policy=policy)[0] for c in columns]
            result.update(single_features=dict(zip(columns, singles)),
                          joint_gain_over_best_single=baseline['comparison_score'] - max(row['comparison_score'] for row in singles),
                          qualification='Each single-input comparison keeps the same engineered recipes' if recipes else 'Single raw inputs versus joint model')
    return result


def run_analyses(state, requests, *, task_type=None):
    results, cache = {}, {}
    for value in requests:
        request = value if isinstance(value, AnalysisRequest) else AnalysisRequest.model_validate(value)
        results[request.name] = analyze_features(state, request, task_type=task_type, cache=cache)
    return results
