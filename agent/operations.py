"""Trusted experiment operations. Qwen selects configurations, never Python source."""
from importlib import import_module
import inspect
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.preprocessing import LabelEncoder

from .eda import SCORERS, _inner_split, _matrix
from .features import SUPERVISED_GUIDE, create_features
from .plans import AnalysisRequest, AutoMLPlan, SearchSpace, ToolCall


CATALOG = {
    'constant': {'requires': ['fit', 'validation', 'test', 'preparation'],
                 'provides': ['validation_predictions', 'test_predictions'], 'parameters': 'No settings: cheap mean/mode/prior baseline.'},
    'features': {'requires': ['fit', 'validation', 'test', 'preparation'],
                 'provides': ['fit_table', 'validation_table', 'test_table'], 'parameters': 'recipes: [{name,operation,columns,parameters}]. Operations: sum/mean/product/missing (one or more columns); difference/ratio (two); signed_log/frequency (one); clip (one, lower=.01/upper=.99 quantiles); group_mean ([group,numeric]); datetime (one, part=year/month/day/dayofweek/hour); expression: parameters.expression with x0/x1 or col(column_name), arithmetic, conditions, abs/log1p/log/exp/sqrt/sin/cos/tanh/minimum/maximum/where/clip/isna/fillna/length/lower/contains/concat/frequency/group_mean/group_std/group_count. Statistics are fit-only.'},
    'preprocess': {'requires': ['fit', 'validation', 'test', 'preparation'],
                   'provides': ['fit_features', 'fit_targets', 'validation_features', 'test_features', 'transformer'],
                   'parameters': 'feature_columns: COMPLETE feature subset, []=all features; NOT a list of categorical columns. Categoricals are detected by dtype automatically. drop_columns removes from that subset. numeric_imputer=median/mean/constant; encoding=ordinal/onehot/native; native_backend=catboost/lightgbm/xgboost; scaling=none/standard/robust/minmax; pca_components optional.'},
    'fit': {'requires': ['fit_features', 'fit_targets', 'transformer', 'fit', 'preparation'], 'provides': ['model'],
            'parameters': 'estimator: installed sklearn-compatible class path; parameters: constructor kwargs; search_space: domains {kind:int/float,low,high,log:false} or {kind:categorical,choices:[...]}; num_trials=10; validation=auto/random/group/time, group_by/split_column for one inner holdout (no CV/folds/OOF in this operation).'},
    'predict': {'requires': ['model', 'validation_features', 'test_features', 'preparation'],
                'provides': ['validation_predictions', 'test_predictions'], 'parameters': 'output=required/probabilities. Use outputs aliases for ensemble inputs.'},
    'automl': {'requires': ['fit', 'validation', 'test', 'preparation'],
               'provides': ['validation_predictions', 'test_predictions'], 'parameters': 'engine=flaml/autogluon; models:[str], model_parameters:{model:{kw:value}}, search_space:{model:{kw:typed_domain}}, num_trials; validation=holdout/cv, folds=3, holdout_fraction=.2; split_column/group_by optionally name the inner time/group column; otherwise use shared validation metadata. bag_folds=0/2+, stack_levels=0 (AutoGluon only); ensemble:bool; budget_fraction in (0,1]; columns/drop_columns, features:FeatureSpec list; row_weights:{time_column,half_life}. With recipes, weights or group/time policy: explicit fit-only holdout, no bagging/stacking (FLAML ensemble=false), then frozen refit on all fit. Without these constraints native CV is available.'},
    'saved_ensemble': {'requires': ['fit', 'validation', 'test', 'preparation', 'auxiliary'],
        'provides': ['validation_predictions', 'test_predictions'],
        'parameters': 'EnsemblePlan: source_nodes, method=mean/vote/probabilities/conditional/stacking/gating. Cached blending needs no training. Conditional rules use input features. Learned methods train only the combiner on saved shared inner-holdout predictions (meta_ready sources); estimator/parameters, feature_columns and passthrough control them. optimize_weights=true tunes blend weights on saved meta_ready predictions; weight_trials=64. No source retraining.'},
    'ensemble': {'requires': ['preparation'], 'provides': ['validation_predictions', 'test_predictions'],
                 'parameters': 'sources: [[validation_state_key,test_state_key],...]; weights optional; method=mean/vote/probabilities. List every source state key in block.requires.'},
}

CATALOG['features']['parameters'] += ' ' + SUPERVISED_GUIDE
CATALOG['fit']['parameters'] += (
    ' row_weights:{time_column,half_life} optionally discounts older fit rows exponentially; numeric time units '
    'or days for timestamps. Estimator must support sample_weight. seeds:[int,...] fits fixed parameters '
    'with each seed and averages predictions (probabilities/votes for classifiers), within the same node budget. '
    'Loss/objective and tree count are ordinary constructor parameters; no forced target transform.')


def _classification(state, defaults):
    task = str(defaults.get('task_type') or '').lower()
    if task:
        return 'class' in task or 'binary' in task
    targets = state['preparation']['data']['target_columns']
    return not pd.api.types.is_numeric_dtype(state['fit'][targets[0]])


def _prediction(values, state, defaults, labels=None, probabilities=False):
    binding, output = state['preparation']['data'], state['preparation']['output']
    columns = [c for c in output['columns'] if c not in binding.get('id_columns', [])]
    values = np.asarray(values)
    if probabilities:
        if len(columns) == 1 and values.shape[1] == 2:
            positive = defaults['class_labels'][-1] if defaults.get('class_labels') else labels[-1]
            index = [str(label) for label in labels].index(str(positive))
            return pd.DataFrame({columns[0]: values[:, index]})
        if labels is None:
            raise ValueError('Probability output needs class labels')
        lookup = {str(label): index for index, label in enumerate(labels)}
        if all(column in lookup for column in columns):
            values = values[:, [lookup[column] for column in columns]]
        elif len(columns) == len(labels) and defaults.get('class_labels'):
            values = values[:, [lookup[str(label)] for label in defaults['class_labels']]]
        else:
            raise ValueError('Probability output columns must map to explicit class labels')
    return pd.DataFrame(values.reshape(len(values), len(columns)), columns=columns)


def _requires_probabilities(state, defaults):
    metric = ''.join(c for c in state['preparation']['metric']['name'].lower() if c.isalnum())
    return metric in {'auc', 'rocauc', 'logloss', 'crossentropy'} or 'probab' in str(defaults.get('prediction_type')).lower()


def _constant(state, defaults):
    targets = state['preparation']['data']['target_columns']
    fit = state['fit']
    classification = _classification(state, defaults)
    result = {}
    for scope in ('validation', 'test'):
        if classification and _requires_probabilities(state, defaults):
            counts = fit[targets[0]].value_counts(normalize=True).sort_index()
            probabilities = np.tile(counts.to_numpy(), (len(state[scope]), 1))
            predictions = _prediction(probabilities, state, defaults, labels=list(counts.index), probabilities=True)
        else:
            metric = ''.join(c for c in state['preparation']['metric']['name'].lower() if c.isalnum())
            statistic = 'median' if metric in {'mae', 'meanabsoluteerror'} else 'mean'
            values = [fit[c].mode().iloc[0] if classification else float(getattr(fit[c], statistic)()) for c in targets]
            predictions = _prediction(np.tile(values, (len(state[scope]), 1)), state, defaults)
        if classification and len(targets) == 1:
            from .predictions import attach_probabilities
            counts = fit[targets[0]].value_counts(normalize=True).sort_index()
            attach_probabilities(predictions, np.tile(counts.to_numpy(), (len(state[scope]), 1)), counts.index)
        result[scope + '_predictions'] = predictions
    return result


def _preprocess(state, parameters):
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

    targets = state['preparation']['data']['target_columns']
    ids = state['preparation']['data'].get('id_columns', [])
    columns = parameters.get('feature_columns') or [c for c in state['fit'] if c not in {*targets, *ids}]
    columns = [c for c in columns if c not in parameters.get('drop_columns', [])]
    if set(columns) & set(targets):
        raise ValueError('Preprocessing features cannot include targets')
    frames = [_matrix(state[scope], columns) for scope in ('fit', 'validation', 'test')]
    numeric = list(frames[0].select_dtypes(include='number'))
    categorical = [c for c in columns if c not in numeric]
    steps = [SimpleImputer(strategy=parameters.get('numeric_imputer', 'median'), keep_empty_features=True)]
    scaling = parameters.get('scaling', 'none')
    scalers = {'standard': StandardScaler, 'robust': RobustScaler, 'minmax': MinMaxScaler}
    if scaling != 'none':
        steps.append(scalers[scaling]())
    if parameters.get('encoding') == 'native':
        from .native_categories import NativeCategories
        if scaling != 'none' or parameters.get('pca_components') is not None:
            raise ValueError('Native categories cannot be combined with scaling or PCA')
        transformer = NativeCategories(parameters.get('native_backend', 'catboost'),
                                       parameters.get('numeric_imputer', 'median'))
    else:
        encoding = (OneHotEncoder(handle_unknown='ignore', sparse_output=False) if parameters.get('encoding', 'ordinal') == 'onehot'
                    else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        transformer = ColumnTransformer([('numeric', make_pipeline(*steps), numeric), ('categorical', encoding, categorical)])
        if parameters.get('pca_components') is not None:
            transformer = make_pipeline(transformer, PCA(n_components=parameters['pca_components'], random_state=42))
    arrays = [transformer.fit_transform(frames[0]), *(transformer.transform(frame) for frame in frames[1:])]
    # Keep the exact input table, including new features and group/time columns.
    transformer.fit_source_ = state['fit']
    created = [recipe['name'] for recipe in state['fit'].attrs.get('feature_recipes', [])]
    transformer.diagnostics_ = {'selected_columns': columns, 'numeric_columns': numeric,
        'categorical_columns': categorical, 'input_feature_count': len(columns),
        'output_feature_count': int(arrays[0].shape[1]),
        'unused_created_features': [column for column in created if column not in columns]}
    return {'fit_features': arrays[0], 'fit_targets': state['fit'][targets].copy(),
            'validation_features': arrays[1], 'test_features': arrays[2], 'transformer': transformer}


def _estimator(path, parameters, runtime):
    from .data import ML_LIBRARIES
    module, name = path.rsplit('.', 1)
    if module.split('.')[0] not in set(ML_LIBRARIES.values()):
        raise ValueError('Choose an estimator from the installed ML libraries')
    factory = getattr(import_module(module), name)
    if not inspect.isclass(factory):
        raise ValueError('estimator must name an ML class, not an arbitrary function')
    def construct(value):
        if isinstance(value, dict) and 'class_path' in value:
            return _estimator(value['class_path'], value.get('parameters', {}), runtime)
        if isinstance(value, dict):
            return {key: construct(item) for key, item in value.items()}
        if isinstance(value, list):
            return [construct(item) for item in value]
        return value
    options = construct(dict(parameters))
    available = inspect.signature(factory).parameters
    if 'random_state' in available and 'random_seed' not in options:
        options.setdefault('random_state', 42)
    for key in ('n_jobs', 'thread_count'):
        if key in available:
            options[key] = int(runtime.get('num_cpus', 1))
    model = factory(**options)
    if callable(getattr(model, 'get_params', None)):
        if 'n_jobs' in model.get_params(deep=False):
            model.set_params(n_jobs=int(runtime.get('num_cpus', 1)))
    if not callable(getattr(model, 'fit', None)) or not callable(getattr(model, 'predict', None)):
        raise ValueError('This operation needs an estimator with fit/predict; choose a supported recipe')
    return model


def _suggest(trial, name, value):
    domain = SearchSpace.model_validate(value)
    if domain.kind == 'categorical':
        return trial.suggest_categorical(name, domain.choices)
    if domain.kind == 'int':
        return trial.suggest_int(name, int(domain.low), int(domain.high), log=domain.log)
    return trial.suggest_float(name, domain.low, domain.high, log=domain.log)


def _fit(state, parameters, defaults):
    from .model_fitting import fit_model, fit_seeds
    from .supervised_features import row_weights
    features, targets = state['fit_features'], state['fit_targets']
    classification = _classification(state, defaults)
    targets = np.asarray(targets)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets[:, 0]
    labels = None
    if classification:
        if targets.ndim != 1:
            raise ValueError('Use a multi-output recipe for multiple classification targets')
        labels = LabelEncoder().fit(targets)
        targets = labels.transform(targets)
    runtime = state['preparation'].get('runtime', {})
    options = dict(parameters.get('parameters', {}))
    from .native_categories import NativeCategories
    if isinstance(state['transformer'], NativeCategories):
        options = state['transformer'].estimator_parameters(parameters['estimator'], options)
    model = _estimator(parameters['estimator'], options, runtime)
    source = getattr(state['transformer'], 'fit_source_', state['fit']).reset_index(drop=True)
    weights = row_weights(source, parameters['row_weights']) if parameters.get('row_weights') else None
    search = parameters.get('search_space', {})
    diagnostics = {'estimator': parameters['estimator'], 'fit_rows': len(targets),
                   'parameters': options, 'inner_validation': None,
                   'row_weights': parameters.get('row_weights'), 'seeds': parameters.get('seeds', []),
                   'weight_range': [float(weights.min()), float(weights.max())] if weights is not None else None}
    if search:
        # Check parameter names during smoke, before the expensive search.
        probe = {name: (domain.get('choices', [None])[0] if domain['kind'] == 'categorical' else domain['low'])
                 for name, domain in search.items()}
        clone(model).set_params(**probe)
    if search and not runtime.get('is_smoke_check'):
        import optuna
        metric = ''.join(c for c in state['preparation']['metric']['name'].lower() if c.isalnum())
        scorer = get_scorer(SCORERS[metric])
        request = AnalysisRequest(name='tuning', kind='predictive', validation=parameters.get('validation', 'auto'),
                                  group_by=parameters.get('group_by'), split_column=parameters.get('split_column'))
        source = getattr(state['transformer'], 'fit_source_', state['fit']).reset_index(drop=True)
        train, valid = _inner_split(source, targets, request,
                                    state['preparation'].get('validation', {}), classification)
        recipes = source.attrs.get('feature_recipes', [])
        base = source.drop(columns=[recipe['name'] for recipe in recipes])
        inner_source = base.iloc[train]
        inner_weights = row_weights(inner_source, parameters['row_weights']) if weights is not None else None
        recipe_policy = dict(state['preparation'].get('validation', {}))
        if request.validation != 'auto':
            recipe_policy.update(strategy=request.validation,
                group_columns=[request.group_by] if request.validation == 'group' and request.group_by else [])
            if request.split_column:
                recipe_policy['time_column'] = request.split_column
        inner_fit, inner_valid = create_features([inner_source, base.iloc[valid]], recipes,
            forbidden=state['preparation']['data']['target_columns'], policy=recipe_policy)
        columns = list(state['transformer'].feature_names_in_)
        transformer = clone(state['transformer'])
        inner_fit = transformer.fit_transform(_matrix(inner_fit, columns))
        inner_valid = transformer.transform(_matrix(inner_valid, columns))
        def objective(trial):
            selected = {name: _suggest(trial, name, domain) for name, domain in search.items()}
            candidate = clone(model).set_params(**selected)
            fit_model(candidate, inner_fit, targets[train], inner_weights)
            return float(scorer(candidate, inner_valid, targets[valid]))
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        remaining = runtime.get('deadline', time.monotonic() + 60) - time.monotonic()
        study.optimize(objective, n_trials=parameters.get('num_trials', 10), timeout=max(1, remaining * 0.65))
        model.set_params(**study.best_params)
        diagnostics.update(best_parameters=study.best_params, inner_score=study.best_value,
            requested_trials=parameters.get('num_trials', 10),
            completed_trials=sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials),
            inner_validation={'method': 'single_holdout', 'splits': 1, 'strategy': request.validation,
                              'train_rows': len(train), 'validation_rows': len(valid)})
    model = fit_seeds(model, features, targets, weights, parameters.get('seeds', []), classification, runtime)
    return {'model': {'estimator': model, 'labels': labels, 'classification': classification,
                       'diagnostics': diagnostics}}


def _predict(state, parameters, defaults):
    fitted = state['model']
    model, labels = fitted['estimator'], fitted['labels']
    probability = parameters.get('output') == 'probabilities' or (
        fitted['classification'] and _requires_probabilities(state, defaults))
    result = {}
    for scope in ('validation', 'test'):
        features = state[scope + '_features']
        values = model.predict_proba(features) if probability else model.predict(features)
        if probability and parameters.get('output') == 'probabilities':
            prediction = pd.DataFrame(values, columns=labels.classes_)
        else:
            if labels is not None and not probability:
                codes = np.asarray(values)
                if not np.isfinite(codes).all() or not np.equal(codes, codes.astype(int)).all():
                    raise ValueError('A classifier must predict the fitted class codes; choose a classification estimator')
                values = labels.inverse_transform(codes.astype(int))
            prediction = _prediction(values, state, defaults, labels=labels.classes_ if labels is not None else None,
                                     probabilities=probability)
        if fitted['classification'] and callable(getattr(model, 'predict_proba', None)):
            from .predictions import attach_probabilities
            probabilities = values if probability else model.predict_proba(features)
            attach_probabilities(prediction, probabilities, labels.classes_)
        result[scope + '_predictions'] = prediction
    return result


def _ensemble(state, parameters, defaults):
    sources = parameters['sources']
    weights = np.asarray(parameters.get('weights', [1.] * len(sources)), dtype=float)
    if len(weights) != len(sources) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Ensemble weights must be finite, nonnegative, match sources and have positive sum')
    weights /= weights.sum()
    vote_classes = sorted(set(label for pair in sources for key in pair for label in state[key].iloc[:, 0])) if (
        parameters.get('method') == 'vote' and len(state[sources[0][0]].columns) == 1) else None
    result = {}
    for index, scope in enumerate(('validation', 'test')):
        frames = [state[pair[index]].reset_index(drop=True) for pair in sources]
        if any(list(frame.columns) != list(frames[0].columns) for frame in frames):
            raise ValueError('Ensemble prediction columns/class order differ')
        if parameters.get('method') == 'vote':
            values = np.stack([frame.to_numpy() for frame in frames])
            voted = np.empty(values.shape[1:], dtype=object)
            for row in range(values.shape[1]):
                for column in range(values.shape[2]):
                    scores = {}
                    for label, weight in zip(values[:, row, column], weights):
                        scores[label] = scores.get(label, 0.) + weight
                    voted[row, column] = max(scores, key=scores.get)
            output = pd.DataFrame(voted.tolist(), columns=frames[0].columns)
        else:
            values = sum(weight * frame.to_numpy(dtype=float) for weight, frame in zip(weights, frames))
            if parameters.get('method') == 'probabilities' and not _requires_probabilities(state, defaults):
                values = np.asarray(frames[0].columns)[values.argmax(axis=1)]
                output = _prediction(values, state, defaults)
            else:
                output = _prediction(values, state, defaults, labels=list(frames[0].columns),
                                     probabilities=parameters.get('method') == 'probabilities')
        from .predictions import attach_probabilities
        if parameters.get('method') == 'probabilities':
            attach_probabilities(output, sum(weight * frame.to_numpy(dtype=float) for weight, frame in zip(weights, frames)), frames[0].columns)
        elif vote_classes:
            votes = sum(weight * np.column_stack([frame.iloc[:, 0].to_numpy() == label for label in vote_classes])
                        for weight, frame in zip(weights, frames))
            attach_probabilities(output, votes, vote_classes)
        result[scope + '_predictions'] = output
    return result


def run_operation(state, value, defaults=None):
    call = value if isinstance(value, ToolCall) else ToolCall.model_validate(value)
    defaults = defaults or {}
    state = {**state, **{key: state[source] for key, source in call.inputs.items()}}
    if call.kind == 'constant':
        result = _constant(state, defaults)
    elif call.kind == 'features':
        targets = state['preparation']['data']['target_columns']
        frames = create_features([state[key] for key in ('fit', 'validation', 'test')],
                                 call.parameters.get('recipes', []), forbidden=targets,
                                 policy=state['preparation'].get('validation', {}))
        result = dict(zip(('fit_table', 'validation_table', 'test_table'), frames))
    elif call.kind == 'preprocess':
        result = _preprocess(state, call.parameters)
    elif call.kind == 'fit':
        from .training_cache import cached
        result = cached(state, call, defaults, lambda inputs: _fit(inputs, call.parameters, defaults))
    elif call.kind == 'predict':
        result = _predict(state, call.parameters, defaults)
    elif call.kind == 'automl':
        from .automl import run_automl
        plan = AutoMLPlan.model_validate(call.parameters)
        from .training_cache import cached
        from tempfile import TemporaryDirectory
        def train(inputs):
            return run_automl(inputs, engine=plan.engine, config=plan.model_dump(), **defaults)
        runtime = state['preparation'].get('runtime', {})
        if runtime.get('is_smoke_check'):
            # Tiny smoke predictors are disposable; do not keep another AutoGluon model tree.
            with TemporaryDirectory(prefix='automl-smoke-') as temporary:
                metadata = {**state['preparation'], 'runtime': {**runtime, 'work_directory': temporary}}
                result = train({**state, 'preparation': metadata})
        else:
            result = cached(state, call, defaults, train)
        if runtime.get('work_directory'):
            from .storage import save
            from pathlib import Path
            phase = 'smoke' if runtime.get('is_smoke_check') else 'full'
            save(Path(runtime['work_directory']) / f'{plan.engine}_{phase}_diagnostics.json',
                 result['automl_diagnostics'])
    elif call.kind == 'saved_ensemble':
        from .ensembles import run_saved_ensemble
        result = run_saved_ensemble(state, call.parameters, defaults)
    else:
        result = _ensemble(state, call.parameters, defaults)
    if call.kind in {'fit', 'preprocess'}:
        from pathlib import Path
        from .storage import fingerprint, save
        runtime = state['preparation'].get('runtime', {})
        key = 'model' if call.kind == 'fit' else 'transformer'
        diagnostics = result[key]['diagnostics'] if call.kind == 'fit' else result[key].diagnostics_
        scope = 'smoke' if runtime.get('is_smoke_check') else 'full'
        name = fingerprint(call.outputs.get(key, key))[:12]
        if runtime.get('work_directory'):
            save(Path(runtime['work_directory']) / f'{call.kind}_{name}_{scope}_diagnostics.json',
                 {'configuration': call.model_dump(), **diagnostics})
    return {call.outputs.get(key, key): value for key, value in result.items()}
