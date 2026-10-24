"""Fixed-candidate regression training for a model-selection handoff.

This module neither selects models nor reads external validation labels during
fitting. The coordinator owns ranking, the wall-clock deadline and publication.
"""

from pathlib import Path
import json
import math
import hashlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from max_training_templates import plain_definition
from max_training_template_automl import flaml as flaml_settings, autogluon as autogluon_settings


SUPPORTED_FAMILIES = (
    'ridge', 'elasticnet', 'sgd', 'random_forest', 'extra_trees',
    'hist_gradient_boosting', 'lightgbm', 'catboost', 'xgboost', 'mlp',
    'flaml', 'autogluon',
)


def read_table(path, ids=()):
    path = Path(path)
    if path.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={column: 'string' for column in ids})


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


class FrozenFlamlRecipe:
    """A single FLAML-selected estimator, with a freshly fitted data transformer."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        import copy
        from sklearn.base import clone
        from flaml.automl.data import DataTransformer
        self.transformer = DataTransformer()
        transformed, labels = self.transformer.fit_transform(x, y, 'regression')
        self.categories = {column: transformed[column].cat.categories
                           for column in transformed if isinstance(transformed[column].dtype, pd.CategoricalDtype)}
        self.model = clone(self.estimator.estimator)
        self.preprocess = copy.deepcopy(self.estimator)
        self.preprocess._model = None
        options = {}
        if type(self.model).__module__.startswith('catboost'):
            params = self.model.get_params()
            rounds_key = 'n_estimators' if 'n_estimators' in params else 'iterations'
            self.model.set_params(**{rounds_key: max(1, int(self.estimator.estimator.tree_count_)),
                                     'use_best_model': False, 'allow_writing_files': False, 'thread_count': 2})
            options['cat_features'] = list(self.categories)
        self.model.fit(self.preprocess._preprocess(transformed), labels, **options)
        return self

    def predict(self, x):
        transformed = self.transformer.transform(x)
        for column, categories in self.categories.items():
            transformed[column] = pd.Categorical(transformed[column], categories=categories)
            if '__NAN__' not in transformed[column].cat.categories:
                transformed[column] = transformed[column].cat.add_categories('__NAN__')
            transformed[column] = transformed[column].fillna('__NAN__')
        return self.model.predict(self.preprocess._preprocess(transformed))


class FrozenAutogluonRecipe:
    """A selected base model with frozen AG features and learned iteration count.

    AutoGluon 1.3.1 model templates support fitting without a validation set.
    This preserves the selected recipe; final training does not invoke AutoML.
    """

    def __init__(self, feature_generator, template):
        self.feature_generator = feature_generator
        self.template = template

    def fit(self, x, y, deadline, output):
        import copy
        self.model = copy.deepcopy(self.template)
        name = type(self.model).__name__
        resource_key = {'RFModel': 'n_jobs', 'XTModel': 'n_jobs', 'LGBModel': 'num_threads',
                        'CatBoostModel': 'thread_count', 'XGBoostModel': 'n_jobs'}.get(name)
        if resource_key:
            self.model._user_params[resource_key] = 2
        self.model.set_contexts(str(Path(output) / 'ag_frozen_model'))
        features = self.feature_generator.transform(x)
        self.model.fit(X=features, y=y, time_limit=_left(deadline), num_cpus=2, num_gpus=0)
        return self

    def predict(self, x):
        return self.model.predict(self.feature_generator.transform(x))


def _check_source_files(config, root, expected):
    for role, digest in expected.items():
        path = root / config['files'][role]
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input file changed since coordinator froze it: {role}')


def metric_score(y, prediction, metric):
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if len(y) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError('Predictions must have the expected row count and only finite values')
    functions = {'mae': mean_absolute_error, 'mse': mean_squared_error,
                 'r2': r2_score, 'mape': mean_absolute_percentage_error}
    score = float(np.sqrt(mean_squared_error(y, prediction))) if metric == 'rmse' else float(functions[metric](y, prediction))
    if not math.isfinite(score):
        raise ValueError(f'Primary metric {metric} is not finite')
    return score


def _left(deadline):
    value = deadline - time.monotonic()
    if value <= .1:
        raise TimeoutError('Training job time budget exhausted')
    return value


def _features(train, test, config):
    constraints = config.get('validation_constraints', {})
    excluded = {config['target'], *config.get('id_columns', []), *constraints.get('group_columns', [])}
    if constraints.get('time_column'):
        excluded.add(constraints['time_column'])
    columns = [name for name in train if name not in excluded]
    if not columns:
        raise ValueError('No usable features after excluding target, identifiers and split keys')
    missing = sorted(set(columns) - set(test.columns))
    if missing:
        raise ValueError(f'Test is missing feature columns: {missing}')
    frames = [frame.loc[:, columns].copy() for frame in (train, test)]
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(train[column])]
    categorical = [column for column in columns if column not in numeric]
    for frame in frames:
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        for column in categorical:
            frame[column] = frame[column].map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)
    return frames, numeric, categorical, sorted(excluded & set(train.columns))


def _preprocessor(numeric, categorical, dense=False):
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                              ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__', keep_empty_features=True)),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=not dense))]), categorical),
    ], sparse_threshold=0.0 if dense else 1.0)


def _model(family, parameters, seed):
    cls, defaults = plain_definition(family, seed)
    params = dict(parameters)
    if family == 'lightgbm':
        aliases = {'bagging_fraction': 'subsample', 'bagging_freq': 'subsample_freq',
                   'feature_fraction': 'colsample_bytree', 'lambda_l1': 'reg_alpha',
                   'lambda_l2': 'reg_lambda', 'min_data_in_leaf': 'min_child_samples',
                   'num_iterations': 'n_estimators', 'eta': 'learning_rate',
                   'num_threads': 'n_jobs', 'num_thread': 'n_jobs', 'nthread': 'n_jobs'}
        for old, canonical in aliases.items():
            if old in params:
                value = params.pop(old)
                if canonical in params and params[canonical] != value:
                    raise ValueError(f'Conflicting LightGBM aliases: {old} and {canonical}')
                params[canonical] = value
        if 'categorical_feature' in params:
            if params['categorical_feature'] != 'auto':
                raise ValueError('This evaluator owns categorical encoding; omit categorical_feature column names from model parameters')
            params.pop('categorical_feature')
    # Template defaults are already trusted executable settings. LightGBM exposes
    # native kwargs such as verbosity only after they are supplied to its wrapper.
    supported = set(cls().get_params()) | set(defaults)
    # XGBoost accepts **kwargs, but its wrapper does not validate arbitrary keys.
    # Restrict it to the advertised sklearn API and documented booster options.
    if family == 'xgboost':
        supported |= {'max_bin', 'grow_policy', 'max_leaves', 'sampling_method', 'max_cat_to_onehot', 'max_cat_threshold'}
    if family == 'catboost':
        import inspect
        supported |= set(inspect.signature(cls.__init__).parameters) - {'self'}
    unknown = sorted(set(params) - supported)
    if unknown:
        raise ValueError(f'Unsupported parameters for {family}: {unknown}. Use the installed model API names.')
    # Resource settings cannot be overridden by Qwen or a handoff report.
    params.update({key: value for key, value in defaults.items() if key not in params})
    if family in {'random_forest', 'extra_trees', 'lightgbm', 'xgboost'}:
        params['n_jobs'] = 2
    if family == 'lightgbm':
        for alias in ('num_threads', 'num_thread', 'nthread'):
            params.pop(alias, None)
    if family == 'catboost':
        params.update(thread_count=2, task_type='CPU', allow_writing_files=False)
    if family == 'xgboost':
        params['device'] = 'cpu'
        if params.get('tree_method') == 'gpu_hist':
            raise ValueError('GPU tree_method is unavailable; use tree_method=hist')
    if family == 'mlp' and isinstance(params.get('hidden_layer_sizes'), list):
        params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
    return cls(**params), params


def _freeze_full_fit(pipeline):
    """Reuse selected iteration count, while including every row in final fit."""
    from sklearn.base import clone
    result = clone(pipeline)
    fitted_model = pipeline.named_steps['model']
    params = fitted_model.get_params()
    overrides = {}
    if 'early_stopping' in params and params['early_stopping']:
        overrides['model__early_stopping'] = False
        if 'max_iter' in params and hasattr(fitted_model, 'n_iter_'):
            overrides['model__max_iter'] = max(1, int(fitted_model.n_iter_))
    result.set_params(**overrides)
    return result, overrides


def _groups(frame, constraints):
    columns = constraints.get('group_columns', [])
    if not columns:
        raise ValueError('Group validation requires group_columns')
    return pd.MultiIndex.from_frame(frame[columns].fillna('__missing__')).factorize()[0]


def _inner_splits(frame, config, validation):
    constraints = config.get('validation_constraints', {})
    strategy = constraints.get('strategy', 'random')
    seed = int(validation.get('seed', 42))
    cv = validation.get('inner_validation', 'holdout') == 'cv'
    folds = int(validation.get('folds', 3))
    if len(frame) < 4:
        raise ValueError('At least four fit rows are required for inner validation')
    if cv and folds < 2:
        raise ValueError('CV requires at least two folds')
    if strategy == 'time':
        column = constraints.get('time_column')
        if not column or column not in frame:
            raise ValueError('Time validation requires an existing time_column')
        dates = pd.to_datetime(frame[column], errors='raise', utc=True)
        if dates.isna().any():
            raise ValueError('Time split column contains missing timestamps')
        order = np.argsort(dates.to_numpy(), kind='stable')
        # Keep identical timestamps in the same fold to prevent simultaneous-row leakage.
        unique = np.sort(dates.unique())
        if cv:
            splitter = TimeSeriesSplit(n_splits=min(folds, len(unique) - 1))
            for left, right in splitter.split(unique):
                yield np.flatnonzero(dates.isin(unique[left])), np.flatnonzero(dates.isin(unique[right]))
        else:
            cut = max(1, min(len(unique) - 1, int(len(unique) * .8)))
            yield np.flatnonzero(dates.isin(unique[:cut])), np.flatnonzero(dates.isin(unique[cut:]))
    elif strategy == 'group':
        groups = _groups(frame, constraints)
        count = len(np.unique(groups))
        if count < 2:
            raise ValueError('At least two groups are required')
        splitter = GroupKFold(n_splits=min(folds, count)) if cv else GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
        yield from splitter.split(frame, groups=groups)
    elif strategy == 'random':
        if cv:
            yield from KFold(n_splits=min(folds, len(frame)), shuffle=True, random_state=seed).split(frame)
        else:
            yield train_test_split(np.arange(len(frame)), test_size=.2, random_state=seed)
    else:
        raise ValueError(f'Unsupported validation strategy: {strategy}')


def _fit_plain(family, parameters, x, y, numeric, categorical, seed, deadline):
    _left(deadline)
    if family == 'hist_gradient_boosting':
        size = len(x) * (len(numeric) + sum(x[c].nunique(dropna=False) for c in categorical))
        if size > 100_000_000:
            raise ValueError('Dense one-hot matrix would exceed 100 million values; choose a sparse-compatible model')
    model, actual = _model(family, parameters, seed)
    pipeline = Pipeline([('features', _preprocessor(numeric, categorical, dense=family == 'hist_gradient_boosting')),
                         ('model', model)])
    fit_options = {}
    if family == 'lightgbm':
        def callback(environment):
            _left(deadline)
        fit_options['model__callbacks'] = [callback]
    if family == 'catboost':
        class DeadlineCallback:
            def after_iteration(self, info):
                return time.monotonic() < deadline
        fit_options['model__callbacks'] = [DeadlineCallback()]
    pipeline.fit(x, y, **fit_options)
    _left(deadline)
    return pipeline, actual


def _fit_automl(family, parameters, x, y, raw_rows, config, validation, phase, deadline, output):
    # FLAML/AutoGluon implement preprocessing, fitted on their internal train part.
    # No external validation labels are passed to either library.
    work = x.copy()
    for column in work:
        if not pd.api.types.is_numeric_dtype(work[column]):
            work[column] = work[column].astype('category')
    budget = _left(deadline)
    train_idx, valid_idx = next(_inner_splits(raw_rows, config, {**validation, 'inner_validation': 'holdout'}))
    if family == 'flaml':
        from flaml import AutoML
        settings = flaml_settings(parameters, metric=config['primary_metric'],
                                  time_budget=budget, log_file=output/'flaml.log')
        if validation.get('inner_validation') == 'cv' and phase != 'final':
            # Iterator materialized before fit; indices refer only to outer fit.
            from sklearn.model_selection import BaseCrossValidator
            splits = list(_inner_splits(raw_rows, config, validation))
            class FixedCV(BaseCrossValidator):
                def get_n_splits(self, X=None, y=None, groups=None):
                    return len(splits)
                def split(self, X=None, y=None, groups=None):
                    yield from splits
            settings.update(eval_method='cv', split_type=FixedCV(), retrain_full=False)
            model = AutoML()
            model.fit(X_train=work, y_train=y, **settings)
        else:
            settings.update(eval_method='holdout', retrain_full=False)
            model = AutoML()
            model.fit(X_train=work.iloc[train_idx], y_train=y.iloc[train_idx],
                      X_val=work.iloc[valid_idx], y_val=y.iloc[valid_idx], **settings)
            # Explicit refit ensures *all* supplied rows are used, including final phase.
        best = model.model
        best_config = model.best_config
        best_estimator = model.best_estimator
        frozen = FrozenFlamlRecipe(best).fit(x, y)
        return frozen, {'family': family, 'settings': parameters, 'selected_estimator': best_estimator,
                        'selected_parameters': best_config}, work
    from autogluon.tabular import TabularPredictor
    settings = autogluon_settings(parameters, metric=config['primary_metric'])
    label = '__handoff_target__'
    while label in work:
        label += '_'
    data = work.copy()
    data[label] = np.asarray(y)
    predictor = TabularPredictor(label=label, problem_type='regression', eval_metric=settings['eval_metric'],
                                path=str(output / 'autogluon_model'), verbosity=0)
    if validation.get('inner_validation') == 'cv':
        raise ValueError('AutoGluon adapter currently supports explicit holdout only; choose inner_validation=holdout')
    predictor.fit(train_data=data.iloc[train_idx], tuning_data=data.iloc[valid_idx],
                  time_limit=max(1., budget * .65), num_cpus=2, num_gpus=0,
                  hyperparameters=settings['hyperparameters'],
                  presets=settings['presets'], num_bag_folds=0,
                  fit_weighted_ensemble=False)
    _left(deadline)
    best = predictor.model_best
    selected = predictor._trainer.load_model(best)
    template = selected.convert_to_refit_full_template()
    frozen = FrozenAutogluonRecipe(predictor._learner.feature_generator, template)
    frozen.fit(x, y, deadline, output)
    return frozen, {'family': family, 'settings': parameters, 'selected_model': best,
                    'selected_parameters': selected.get_trained_params(),
                    'ensemble_enabled': False, 'feature_preprocessing_frozen': True}, work


def _predict(model, x):
    values = np.asarray(model.predict(x), dtype=float).reshape(-1)
    if len(values) != len(x) or not np.isfinite(values).all():
        raise ValueError('Model produced invalid/non-finite predictions or unexpected row count')
    return values


def _submission(config, input_dir, train, test, prediction):
    specification = config.get('submission', {})
    ids = config.get('id_columns', [])
    sample_path = config.get('files', {}).get('sample_submission')
    column = specification.get('prediction_column')
    if sample_path:
        sample = read_table(input_dir / sample_path, ids)
        if len(sample) != len(test) or sample.columns.duplicated().any():
            raise ValueError('Sample submission schema or row count is invalid')
        output_columns = [name for name in sample if name not in ids]
        if column is None:
            if len(output_columns) != 1:
                raise ValueError('Specify prediction_column when sample has multiple non-ID columns')
            column = output_columns[0]
        if column not in sample or column in ids:
            raise ValueError('prediction_column must be a non-ID sample submission column')
        if set(output_columns) != {column}:
            raise ValueError('Regression submission requires exactly one prediction column')
        if ids:
            if not set(ids).issubset(test.columns) or not set(ids).issubset(sample.columns):
                raise ValueError('Identifiers must exist in both test and sample submission')
            if test.duplicated(ids).any() or sample.duplicated(ids).any():
                raise ValueError('Duplicate submission identifiers cannot be aligned safely')
            predicted = test[ids].copy()
            predicted[column] = prediction
            joined = sample[ids].merge(predicted, on=ids, how='left', sort=False, validate='one_to_one')
            if joined[column].isna().any():
                raise ValueError('Sample submission identifiers do not match test identifiers')
            sample[column] = joined[column].to_numpy()
        else:
            sample[column] = prediction
        return sample
    result = test.loc[:, ids].copy() if ids else pd.DataFrame(index=range(len(test)))
    result[column or config['target']] = prediction
    return result


def run_job(job, time_budget):
    started = time.monotonic()
    deadline = started + float(time_budget)
    if time_budget <= 0:
        raise ValueError('Worker time_budget must be positive')
    input_path = Path(job['input_path']).resolve()
    config = json.loads(input_path.read_text(encoding='utf-8-sig'))
    if config.get('task_type') != 'regression':
        raise ValueError('Training backend currently supports regression only')
    output = Path(job['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = input_path.parent
    expected_hashes = job.get('expected_source_hashes', {})
    _check_source_files(config, root, expected_hashes)
    ids = config.get('id_columns', [])
    train = read_table(root / config['files']['train'], ids)
    test = read_table(root / config['files']['test'], ids)
    target = config['target']
    if target not in train:
        raise ValueError(f'Training table is missing target {target}')
    if target in test:
        raise ValueError('Test table must not contain target; hidden/test labels must not enter training or selection')
    y = pd.to_numeric(train[target], errors='raise')
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError('All training rows must have a finite numeric regression target')
    (x, xtest), numeric, categorical, excluded = _features(train, test, config)
    candidate = job['candidate']
    selection = config['selected_models'][int(candidate['model_index'])]
    family = selection['family']
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f'Unsupported model family {family}; supported: {SUPPORTED_FAMILIES}')
    parameters = dict(selection.get('parameters', {}))
    parameters.update(candidate.get('parameters', {}))
    phase = job['phase']
    validation = job.get('validation', {})
    seed = int(validation.get('seed', 42))
    source = None
    candidate_hash = _hash(candidate)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    split_sha256 = hashlib.sha256(Path(job['split_path']).read_bytes()).hexdigest()
    if phase == 'final':
        if not job.get('source_result_dir'):
            raise ValueError('Final refit requires source_result_dir of the selected evaluated candidate')
        source = Path(job['source_result_dir'])
        source_result = json.loads((source / 'result.json').read_text(encoding='utf-8'))
        if source_result.get('status') != 'completed' or source_result.get('phase') != 'evaluate':
            raise ValueError('Final refit source must be a successful evaluated candidate')
        for key, value in {'candidate_hash': candidate_hash, 'input_sha256': input_sha256,
                           'split_sha256': split_sha256, 'primary_metric': config['primary_metric'], 'family': family}.items():
            if source_result.get(key) != value:
                raise ValueError(f'Final refit frozen source mismatch: {key}')
        if hashlib.sha256((source / 'model.joblib').read_bytes()).hexdigest() != source_result.get('model_sha256'):
            raise ValueError('Frozen selected model file hash does not match its evaluated result')
    if phase == 'evaluate':
        with np.load(job['split_path'], allow_pickle=False) as split:
            fit_idx, valid_idx = np.asarray(split['fit'], dtype=int), np.asarray(split['valid'], dtype=int)
        if min(len(fit_idx), len(valid_idx)) == 0:
            raise ValueError('Outer fit/validation partitions must be non-empty')
        if min(fit_idx.min(), valid_idx.min()) < 0 or max(fit_idx.max(), valid_idx.max()) >= len(train):
            raise ValueError('Outer split indices are out of bounds')
        if len(np.unique(fit_idx)) != len(fit_idx) or len(np.unique(valid_idx)) != len(valid_idx) or np.intersect1d(fit_idx, valid_idx).size:
            raise ValueError('Outer fit/validation partitions must contain unique disjoint indices')
    elif phase == 'final':
        fit_idx, valid_idx = np.arange(len(train)), None
    else:
        raise ValueError(f'Unknown phase: {phase}')
    xfit, yfit, raw_fit = x.iloc[fit_idx], y.iloc[fit_idx], train.iloc[fit_idx]
    inner_scores = []
    metric = config['primary_metric']
    if metric not in {'rmse', 'mae', 'mse', 'r2', 'mape'}:
        raise ValueError(f'Unsupported regression metric: {metric}')
    with threadpool_limits(limits=2):
        if validation.get('inner_validation') == 'cv' and phase == 'evaluate' and family not in {'flaml', 'autogluon'}:
            for inner_fit, inner_valid in _inner_splits(raw_fit, config, validation):
                model, _ = _fit_plain(family, parameters, xfit.iloc[inner_fit], yfit.iloc[inner_fit], numeric, categorical, seed, deadline)
                inner_scores.append(metric_score(yfit.iloc[inner_valid], _predict(model, xfit.iloc[inner_valid]), metric))
        if phase == 'final':
            from sklearn.base import clone
            frozen = joblib.load(source / 'model.joblib')
            if family == 'autogluon':
                model = FrozenAutogluonRecipe(frozen.feature_generator, frozen.template).fit(xfit, yfit, deadline, output)
                actual = source_result['actual_parameters']
            elif family == 'flaml':
                model = FrozenFlamlRecipe(frozen.estimator).fit(xfit, yfit)
                actual = source_result['actual_parameters']
            else:
                model, final_overrides = _freeze_full_fit(frozen)
                model.fit(xfit, yfit)
                actual = {**source_result['actual_parameters'],
                          'full_refit_overrides': final_overrides}
            _left(deadline)
        elif family in {'flaml', 'autogluon'}:
            model, actual, _ = _fit_automl(family, parameters, xfit, yfit, raw_fit, config, validation, phase, deadline, output)
        else:
            model, actual = _fit_plain(family, parameters, xfit, yfit, numeric, categorical, seed, deadline)
        score = None
        train_score = None
        _check_source_files(config, root, expected_hashes)
        if phase == 'evaluate':
            train_score = metric_score(yfit, _predict(model, xfit), metric)
            prediction = _predict(model, x.iloc[valid_idx])
            score = metric_score(y.iloc[valid_idx], prediction, metric)
            pd.DataFrame({'row_index': valid_idx, 'prediction': prediction}).to_csv(output / 'validation_predictions.csv', index=False)
            test_prediction = _predict(model, xtest)
            fallback = _submission(config, root, train, test, test_prediction)
            filename = config.get('submission', {}).get('filename', 'submission.csv')
            if Path(filename).name != filename:
                raise ValueError('submission.filename must be a plain filename')
            fallback.to_csv(output / filename, index=False)
        else:
            prediction = _predict(model, xtest)
            submission = _submission(config, root, train, test, prediction)
            filename = config.get('submission', {}).get('filename', 'submission.csv')
            if Path(filename).name != filename:
                raise ValueError('submission.filename must be a plain filename')
            submission.to_csv(output / filename, index=False)
        _left(deadline)
    model_path = output / 'model.joblib'
    _check_source_files(config, root, expected_hashes)
    joblib.dump(model, model_path)
    model_file = str(model_path)
    result = {'status': 'completed', 'phase': phase, 'candidate': candidate['name'], 'family': family,
              'primary_metric': metric, 'metric': metric, 'direction': 'maximize' if metric == 'r2' else 'minimize',
              'candidate_hash': candidate_hash, 'input_sha256': input_sha256, 'split_sha256': split_sha256,
              'score': score, 'score_scope': 'outer_validation' if phase == 'evaluate' else 'none_full_data_refit',
              'training_score': train_score,
              'inner_cv_score': float(np.mean(inner_scores)) if inner_scores else None,
              'inner_fold_scores': inner_scores, 'duration_seconds': time.monotonic() - started,
              'actual_training_rows': len(fit_idx), 'total_labelled_rows': len(train),
              'outer_fit_rows': len(fit_idx) if phase == 'evaluate' else source_result.get('outer_fit_rows'),
              'outer_validation_rows': len(valid_idx) if valid_idx is not None else source_result.get('outer_validation_rows'),
              'full_training_rows': len(train) if phase == 'final' else None,
              'validation_rows': len(valid_idx) if valid_idx is not None else 0,
              'prediction_rows': len(prediction), 'parameters': parameters, 'actual_parameters': actual,
              'excluded_columns': excluded, 'feature_columns': list(x.columns), 'model_path': model_file,
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'submission_path': str(output / filename), 'submission_scope': 'full_data_refit' if phase == 'final' else 'outer_fit_fallback'}
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    return result
