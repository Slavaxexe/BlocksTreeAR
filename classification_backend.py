"""Classification training with frozen validation, class alignment and full refit.

Only the outer holdout ranks candidates. AutoML sees a separate inner holdout;
final training reuses its selected estimator without another AutoML search.
"""
import copy
import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from classification_templates import plain_definition
from margo_recipe import MargoRecipeTransformer, register_margo
from training_backend import (_check_source_files, _features, _freeze_full_fit, _hash,
                               _inner_splits as regression_inner_splits, _left, _preprocessor,
                               _submission, read_table)
from training_template_classification import flaml as flaml_settings, autogluon as autogluon_settings


SUPPORTED_FAMILIES = ('logistic_regression', 'sgd', 'random_forest', 'extra_trees',
                      'hist_gradient_boosting', 'lightgbm', 'catboost', 'xgboost',
                      'mlp', 'knn', 'flaml', 'autogluon')
SUPPORTED_METRICS = {'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_weighted', 'roc_auc', 'log_loss'}


def positive_index(classes, specification):
    value = specification.get('positive_label')
    if value is None:
        return 1 if len(classes) == 2 else None
    matches = [i for i, label in enumerate(classes) if str(label) == str(value)]
    if len(matches) != 1:
        raise ValueError('submission.positive_label must uniquely match an observed training label')
    return matches[0]


def metric_score(y, predicted, probabilities, metric, class_count, positive=1):
    y, predicted = np.asarray(y), np.asarray(predicted)
    if len(y) != len(predicted):
        raise ValueError('Classifier prediction row count differs from expected labels')
    if metric == 'accuracy':
        score = accuracy_score(y, predicted)
    elif metric == 'balanced_accuracy':
        score = balanced_accuracy_score(y, predicted)
    elif metric in {'f1', 'f1_macro', 'f1_weighted'}:
        if metric == 'f1' and class_count != 2:
            raise ValueError('Binary f1 requires exactly two classes; use f1_macro or f1_weighted for multiclass')
        options = dict(average={'f1': 'binary', 'f1_macro': 'macro', 'f1_weighted': 'weighted'}[metric], zero_division=0)
        if metric == 'f1':
            options['pos_label'] = positive
        else:
            options['labels'] = np.arange(class_count)
        score = f1_score(y, predicted, **options)
    elif metric == 'log_loss':
        if probabilities is None:
            raise ValueError('log_loss requires an estimator with predict_proba')
        score = log_loss(y, probabilities, labels=np.arange(class_count))
    elif metric == 'roc_auc':
        if probabilities is None:
            raise ValueError('roc_auc requires an estimator with predict_proba')
        if class_count == 2:
            score = roc_auc_score((y == positive).astype(int), probabilities[:, positive])
        else:
            score = roc_auc_score(y, probabilities, labels=np.arange(class_count), multi_class='ovr', average='macro')
    else:
        raise ValueError(f'Unsupported classification metric: {metric}')
    if not math.isfinite(float(score)):
        raise ValueError('Classification primary metric is not finite')
    return float(score)


def _predict(model, x, class_count, require_probability=False):
    predicted = np.asarray(model.predict(x)).reshape(-1)
    if len(predicted) != len(x) or pd.isna(predicted).any():
        raise ValueError('Classifier returned missing predictions or wrong number of rows')
    if not np.isin(predicted, np.arange(class_count)).all():
        raise ValueError('Classifier returned unknown encoded class labels')
    predicted = predicted.astype(int)
    probabilities = None
    if hasattr(model, 'predict_proba'):
        raw = np.asarray(model.predict_proba(x), dtype=float)
        if raw.ndim == 1 and class_count == 2:
            raw = np.column_stack([1. - raw, raw])
        classes = np.asarray(getattr(model, 'classes_', np.arange(class_count)))
        if raw.shape != (len(x), len(classes)) or len(classes) != class_count or set(classes) != set(range(class_count)):
            raise ValueError('Classifier probability columns do not cover every training class')
        probabilities = raw[:, [int(np.flatnonzero(classes == label)[0]) for label in range(class_count)]]
        if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-8) or np.any(probabilities > 1. + 1e-8):
            raise ValueError('Classifier probabilities must be finite and between zero and one')
        if not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5):
            raise ValueError('Classifier probability rows must sum to one')
    if require_probability and probabilities is None:
        raise ValueError('This metric/submission requires predict_proba; SGD supports log_loss or modified_huber')
    return predicted, probabilities


def _inner_splits(frame, y, config, validation):
    if config.get('validation_constraints', {}).get('strategy', 'random') != 'random':
        splits = regression_inner_splits(frame, config, validation)
    else:
        seed = int(validation.get('seed', 42))
        counts = pd.Series(np.asarray(y)).value_counts()
        if counts.min() < 2:
            raise ValueError('Stratified inner validation requires at least two examples of every class in outer fit')
        if validation.get('inner_validation') == 'cv':
            folds = min(int(validation.get('folds', 3)), int(counts.min()))
            if folds < 2:
                raise ValueError('Classification CV requires at least two folds')
            splits = StratifiedKFold(folds, shuffle=True, random_state=seed).split(frame, y)
        else:
            nvalid = max(len(counts), int(math.ceil(.2 * len(frame))))
            if nvalid > len(frame) - len(counts):
                raise ValueError('Not enough class examples for an inner holdout')
            splits = [train_test_split(np.arange(len(frame)), test_size=nvalid, random_state=seed, stratify=y)]
    classes = set(np.asarray(y))
    for fit, valid in splits:
        if set(np.asarray(y)[fit]) != classes:
            raise ValueError('An inner training fold omits a class; change validation strategy or fold count')
        yield fit, valid


class ClassWeightedClassifier(ClassifierMixin, BaseEstimator):
    """Fit-local class weights for XGBoost, which accepts sample_weight only.

    Keeping this policy in a cloneable estimator also applies it to full refit;
    neither class frequencies nor weights are copied from a validation fold.
    Dictionary keys are the backend's explicitly recorded encoded class IDs.
    """
    def __init__(self, estimator, class_weight='balanced'):
        self.estimator, self.class_weight = estimator, class_weight

    def fit(self, x, y, **fit_options):
        policy = self.class_weight
        if isinstance(policy, dict):
            policy = {int(label): float(value) for label, value in policy.items()}
            if any(not math.isfinite(value) or value <= 0 for value in policy.values()):
                raise ValueError('Encoded class weights must be finite positive numbers')
            if set(np.unique(y)) - set(policy):
                raise ValueError('class_weight dictionary must cover every encoded training class')
        elif policy not in {None, 'balanced'}:
            raise ValueError('XGBoost class_weight supports balanced, null, or encoded-class dictionary')
        weights = compute_sample_weight(policy, y)
        if 'sample_weight' in fit_options:
            weights = weights * np.asarray(fit_options.pop('sample_weight'))
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, y, sample_weight=weights, **fit_options)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, x):
        return self.estimator_.predict(x)

    def predict_proba(self, x):
        return self.estimator_.predict_proba(x)


def _model(family, parameters, seed):
    cls, defaults = plain_definition(family, seed)
    params = dict(parameters)
    # Resource hints are owned by this backend's threadpool, not nonexistent
    # constructor parameters of HistGradientBoostingClassifier/MLPClassifier.
    if family in {'hist_gradient_boosting', 'mlp', 'sgd'}:
        params.pop('n_jobs', None)
    class_weight = params.pop('class_weight', None) if family == 'xgboost' else None
    if family == 'lightgbm':
        aliases = {'bagging_fraction': 'subsample', 'bagging_freq': 'subsample_freq',
                   'feature_fraction': 'colsample_bytree', 'lambda_l1': 'reg_alpha', 'lambda_l2': 'reg_lambda',
                   'min_data_in_leaf': 'min_child_samples', 'num_iterations': 'n_estimators', 'eta': 'learning_rate',
                   'num_threads': 'n_jobs', 'num_thread': 'n_jobs', 'nthread': 'n_jobs'}
        for alias, canonical in aliases.items():
            if alias in params:
                value = params.pop(alias)
                if canonical in params and params[canonical] != value:
                    raise ValueError(f'Conflicting LightGBM parameters: {alias}, {canonical}')
                params[canonical] = value
        if params.get('categorical_feature') == 'auto':
            params.pop('categorical_feature')
    supported = set(cls().get_params()) | set(defaults)
    if family == 'catboost':
        supported |= set(inspect.signature(cls.__init__).parameters) - {'self'}
    if family == 'xgboost':
        supported |= {'max_bin', 'grow_policy', 'max_leaves', 'sampling_method'}
    unknown = sorted(set(params) - supported)
    if unknown:
        raise ValueError(f'Unsupported parameters for classifier {family}: {unknown}')
    params = {**defaults, **params}
    if family in {'logistic_regression', 'random_forest', 'extra_trees', 'lightgbm', 'xgboost', 'knn'}:
        params['n_jobs'] = 2
    if family == 'catboost':
        params.update(thread_count=2, task_type='CPU', allow_writing_files=False)
    if family == 'xgboost':
        params['device'] = 'cpu'
        if params.get('tree_method') == 'gpu_hist':
            raise ValueError('GPU tree_method unavailable; select hist')
    if family == 'mlp' and isinstance(params.get('hidden_layer_sizes'), list):
        params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
    model = cls(**params)
    actual = {**params, 'backend_threads': 2}
    if family == 'xgboost' and class_weight is not None:
        model = ClassWeightedClassifier(model, class_weight)
        actual.update(class_weight=class_weight, class_weight_adapter='fit_local_sample_weight')
    return model, actual


def _fit_plain(family, parameters, x, y, numeric, categorical, seed, deadline, config):
    _left(deadline)
    if family == 'hist_gradient_boosting':
        size = len(x) * (len(numeric) + sum(x[c].nunique(dropna=False) for c in categorical))
        if size > 100_000_000:
            raise ValueError('Dense one-hot matrix exceeds safe size; choose sparse-compatible classifier')
    model, actual = _model(family, parameters, seed)
    steps = []
    if config.get('preparation'):
        steps.append(('upstream', MargoRecipeTransformer(config['preparation'], config['target'], config.get('id_columns', []))))
        numeric, categorical = make_column_selector(dtype_include=np.number), make_column_selector(dtype_exclude=np.number)
    pipeline = Pipeline([*steps, ('features', _preprocessor(numeric, categorical, dense=family == 'hist_gradient_boosting')),
                         ('model', model)])
    options = {}
    if family == 'lightgbm':
        def callback(environment):
            _left(deadline)
        options['model__callbacks'] = [callback]
    if family == 'catboost':
        class DeadlineCallback:
            def after_iteration(self, info):
                return time.monotonic() < deadline
        options['model__callbacks'] = [DeadlineCallback()]
    pipeline.fit(x, y, **options)
    _left(deadline)
    return pipeline, actual


class FrozenFlamlClassifier:
    """One native FLAML-selected classifier with newly fitted data transformer."""
    def __init__(self, estimator, class_count):
        self.estimator, self.class_count = estimator, class_count

    def fit(self, x, y, deadline=None, output=None):
        from sklearn.base import clone
        from flaml.automl.data import DataTransformer
        self.transformer = DataTransformer()
        transformed, encoded = self.transformer.fit_transform(x, y, 'classification')
        if not np.array_equal(np.asarray(encoded), np.asarray(y)):
            raise ValueError('FLAML changed the globally frozen class mapping')
        self.categories = {c: transformed[c].cat.categories for c in transformed if isinstance(transformed[c].dtype, pd.CategoricalDtype)}
        self.model = clone(self.estimator.estimator)
        self.preprocess = copy.deepcopy(self.estimator)
        self.preprocess._model = None
        options = {}
        if type(self.model).__module__.startswith('catboost'):
            params = self.model.get_params()
            key = 'n_estimators' if 'n_estimators' in params else 'iterations'
            self.model.set_params(**{key: max(1, int(self.estimator.estimator.tree_count_)),
                                     'use_best_model': False, 'allow_writing_files': False, 'thread_count': 2})
            options['cat_features'] = list(self.categories)
        self.model.fit(self.preprocess._preprocess(transformed), encoded, **options)
        self.classes_ = np.arange(self.class_count)
        return self

    def _transform(self, x):
        transformed = self.transformer.transform(x)
        for column, categories in self.categories.items():
            transformed[column] = pd.Categorical(transformed[column], categories=categories)
            if '__NAN__' not in transformed[column].cat.categories:
                transformed[column] = transformed[column].cat.add_categories('__NAN__')
            transformed[column] = transformed[column].fillna('__NAN__')
        return self.preprocess._preprocess(transformed)

    def predict(self, x):
        return self.model.predict(self._transform(x))

    def predict_proba(self, x):
        return self.model.predict_proba(self._transform(x))


class FrozenAutogluonClassifier:
    def __init__(self, feature_generator, template, class_count):
        self.feature_generator, self.template, self.class_count = feature_generator, template, class_count

    def fit(self, x, y, deadline, output):
        self.model = copy.deepcopy(self.template)
        key = {'RFModel': 'n_jobs', 'XTModel': 'n_jobs', 'LGBModel': 'num_threads',
               'CatBoostModel': 'thread_count', 'XGBoostModel': 'n_jobs'}.get(type(self.model).__name__)
        if key:
            self.model._user_params[key] = 2
        self.model.set_contexts(str(Path(output) / 'ag_frozen_classifier'))
        self.model.fit(X=self.feature_generator.transform(x), y=y, time_limit=_left(deadline), num_cpus=2, num_gpus=0)
        self.classes_ = np.arange(self.class_count)
        return self

    def predict_proba(self, x):
        values = np.asarray(self.model.predict_proba(self.feature_generator.transform(x)), dtype=float)
        return np.column_stack([1. - values, values]) if values.ndim == 1 else values

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)


class PreparedClassifier:
    """Feature recipe is always refitted on the rows used by the native model."""
    def __init__(self, preparation, target, ids, estimator):
        self.preparation, self.target, self.ids, self.estimator = preparation, target, ids, estimator

    def fit(self, x, y, deadline, output):
        self.features_ = MargoRecipeTransformer(self.preparation, self.target, self.ids).fit(x)
        self.estimator.fit(self.features_.transform(x), y, deadline, output)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, x):
        return self.estimator.predict(self.features_.transform(x))

    def predict_proba(self, x):
        return self.estimator.predict_proba(self.features_.transform(x))


class ClassificationMetric:
    """Pickleable scorer: AG persists its scorer along with every base model."""
    def __init__(self, metric, class_count, positive):
        self.metric, self.class_count, self.positive = metric, class_count, positive

    def __call__(self, y_true, y_pred):
        probabilities = None
        if self.metric in {'roc_auc', 'log_loss'}:
            probabilities = np.asarray(y_pred)
            if probabilities.ndim == 1:
                probabilities = np.column_stack([1. - probabilities, probabilities])
            predicted = probabilities.argmax(axis=1)
        else:
            predicted = y_pred
        return metric_score(y_true, predicted, probabilities, self.metric, self.class_count, self.positive)


def _fit_automl(family, parameters, x, y, raw_rows, config, validation, deadline, output, class_count, positive):
    if validation.get('inner_validation') == 'cv':
        raise ValueError('Classification AutoML adapters require inner_validation=holdout; native classifiers support CV')
    train_idx, valid_idx = next(_inner_splits(raw_rows, y, config, validation))
    if set(np.asarray(y)[valid_idx]) != set(range(class_count)):
        raise ValueError('AutoML inner holdout must include every training class')
    work = x.copy()
    if config.get('preparation'):
        prefix = MargoRecipeTransformer(config['preparation'], config['target'], config.get('id_columns', [])).fit(x.iloc[train_idx])
        work = prefix.transform(x)
    for column in work:
        if not pd.api.types.is_numeric_dtype(work[column]):
            work[column] = work[column].astype('category')
    metric = config['primary_metric']
    if family == 'flaml':
        from flaml import AutoML
        settings = flaml_settings(parameters, metric=metric, time_budget=_left(deadline), log_file=output/'flaml.log')
        # One callback ensures FLAML uses exactly the upstream averaging and positive class.
        def objective(X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, *args, **kwargs):
            predicted, probabilities = _predict(estimator, X_val, class_count, metric in {'log_loss', 'roc_auc'})
            score = metric_score(y_val, predicted, probabilities, metric, class_count, positive)
            return (score if metric == 'log_loss' else 1. - score), {'primary_metric': score}
        settings['metric'] = objective
        search = AutoML()
        search.fit(X_train=work.iloc[train_idx], y_train=y.iloc[train_idx],
                   X_val=work.iloc[valid_idx], y_val=y.iloc[valid_idx], **settings)
        frozen = FrozenFlamlClassifier(search.model, class_count)
        actual = dict(family=family, settings=parameters, selected_estimator=search.best_estimator,
                      selected_parameters=search.best_config)
    else:
        from autogluon.tabular import TabularPredictor
        from autogluon.core.metrics import make_scorer
        settings = autogluon_settings(parameters, metric=metric)
        scorer = make_scorer(name='handoff_primary', score_func=ClassificationMetric(metric, class_count, positive),
                             optimum=0. if metric == 'log_loss' else 1., greater_is_better=metric != 'log_loss',
                             needs_proba=metric in {'roc_auc', 'log_loss'})
        label = '__classification_target__'
        while label in work:
            label += '_'
        data = work.copy()
        data[label] = np.asarray(y)
        predictor = TabularPredictor(label=label, problem_type='binary' if class_count == 2 else 'multiclass',
                                     eval_metric=scorer, path=str(output/'autogluon_model'), verbosity=0)
        predictor.fit(train_data=data.iloc[train_idx], tuning_data=data.iloc[valid_idx],
                      time_limit=max(1., _left(deadline) * .6), num_cpus=2, num_gpus=0,
                      hyperparameters=settings['hyperparameters'], presets=settings['presets'],
                      num_bag_folds=0, fit_weighted_ensemble=False)
        selected = predictor._trainer.load_model(predictor.model_best)
        template = selected.convert_to_refit_full_template()
        frozen = FrozenAutogluonClassifier(predictor._learner.feature_generator, template, class_count)
        actual = dict(family=family, settings=parameters, selected_model=predictor.model_best,
                      selected_parameters=selected.get_trained_params(), ensemble_enabled=False,
                      feature_preprocessing_frozen=True)
    if config.get('preparation'):
        frozen = PreparedClassifier(config['preparation'], config['target'], config.get('id_columns', []), frozen)
    frozen.fit(x, y, deadline, output)
    _left(deadline)
    return frozen, actual


def _fresh_automl(model):
    if isinstance(model, PreparedClassifier):
        return PreparedClassifier(model.preparation, model.target, model.ids, _fresh_automl(model.estimator))
    if isinstance(model, FrozenFlamlClassifier):
        return FrozenFlamlClassifier(model.estimator, model.class_count)
    if isinstance(model, FrozenAutogluonClassifier):
        return FrozenAutogluonClassifier(model.feature_generator, model.template, model.class_count)
    raise ValueError('Frozen classifier is not a supported AutoML recipe')


def _submission_predictions(model, x, encoder, specification, metric):
    probability = specification.get('prediction_type', 'labels') == 'probability'
    predicted, probabilities = _predict(model, x, len(encoder.classes_), probability or metric in {'roc_auc', 'log_loss'})
    if probability:
        if len(encoder.classes_) != 2:
            raise ValueError('One-column probability submission supports binary classification only; multiclass requires a separately specified format')
        return probabilities[:, positive_index(encoder.classes_, specification)]
    return encoder.inverse_transform(predicted)


def run_job(job, time_budget):
    started = time.monotonic()
    deadline = started + float(time_budget)
    _left(deadline)
    input_path = Path(job['input_path']).resolve()
    config = json.loads(input_path.read_text(encoding='utf-8-sig'))
    if config.get('task_type') != 'classification':
        raise ValueError('Classification backend requires task_type=classification')
    output = Path(job['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = input_path.parent
    expected = job.get('expected_source_hashes', {})
    _check_source_files(config, root, expected)
    if config.get('preparation'):
        register_margo(config['preparation'])
    ids, target = config.get('id_columns', []), config['target']
    train = read_table(root/config['files']['train'], ids)
    test = read_table(root/config['files']['test'], ids)
    if target not in train or target in test:
        raise ValueError('Target must exist only in labelled training data, never hidden test data')
    if train[target].isna().any():
        raise ValueError('Classification target contains missing labels')
    encoder = LabelEncoder().fit(train[target])
    class_count = len(encoder.classes_)
    if class_count < 2:
        raise ValueError('Classification requires at least two training classes')
    y = pd.Series(encoder.transform(train[target]), index=train.index)
    specification = config.get('submission', {})
    positive = positive_index(encoder.classes_, specification)
    if specification.get('prediction_type', 'labels') == 'probability' and class_count != 2:
        raise ValueError('Binary probability submission requires exactly two classes')
    metric = config['primary_metric']
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f'Unsupported classification primary metric: {metric}')
    (x, xtest), numeric, categorical, excluded = _features(train, test, config)
    candidate = job['candidate']
    selection = config['selected_models'][int(candidate['model_index'])]
    family = selection['family']
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f'Unsupported classification family: {family}')
    if selection.get('target_transform', 'none') != 'none':
        raise ValueError('Classification does not permit regression target transformations')
    metadata = selection.get('selection_metadata', {})
    parameters = dict(selection.get('parameters', {}))
    native_loss = metadata.get('loss_native', {})
    locked = set(metadata.get('locked_parameters', [])) | set(native_loss)
    locked |= {'objective', 'loss', 'loss_function', 'criterion'} & set(parameters)
    if locked - set(parameters) or any(parameters.get(key) != value for key, value in native_loss.items()):
        raise ValueError('Upstream classification loss metadata disagrees with model parameters')
    overrides = candidate.get('parameters', {})
    changed = sorted(key for key in locked if key in overrides and overrides[key] != parameters[key])
    if changed:
        raise ValueError(f'Upstream classification loss parameters are locked: {changed}')
    parameters.update(overrides)
    phase, validation = job['phase'], job.get('validation', {})
    seed = int(validation.get('seed', 42))
    hashes = dict(candidate_hash=_hash(candidate), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256(Path(job['split_path']).read_bytes()).hexdigest())
    source_result, source = None, None
    if phase == 'evaluate':
        with np.load(job['split_path'], allow_pickle=False) as split:
            fit_idx, valid_idx = np.asarray(split['fit'], dtype=int), np.asarray(split['valid'], dtype=int)
        if not len(fit_idx) or not len(valid_idx) or min(fit_idx.min(), valid_idx.min()) < 0 or max(fit_idx.max(), valid_idx.max()) >= len(train):
            raise ValueError('Outer classification split is empty or out of bounds')
        if len(np.unique(fit_idx)) != len(fit_idx) or len(np.unique(valid_idx)) != len(valid_idx) or np.intersect1d(fit_idx, valid_idx).size:
            raise ValueError('Outer fit and validation rows must be unique and disjoint')
        if set(y.iloc[fit_idx]) != set(range(class_count)):
            raise ValueError('Outer training partition omits a class; use stratified splitting')
    elif phase == 'final':
        if not job.get('source_result_dir'):
            raise ValueError('Final refit requires the winning evaluated source_result_dir')
        source = Path(job['source_result_dir'])
        source_result = json.loads((source/'result.json').read_text(encoding='utf-8'))
        if source_result.get('status') != 'completed' or source_result.get('phase') != 'evaluate':
            raise ValueError('Final source must be a completed evaluated classification candidate')
        for key, value in {**hashes, 'primary_metric': metric, 'family': family}.items():
            if source_result.get(key) != value:
                raise ValueError(f'Final classifier frozen source mismatch: {key}')
        if hashlib.sha256((source/'model.joblib').read_bytes()).hexdigest() != source_result.get('model_sha256'):
            raise ValueError('Frozen classifier model hash mismatch')
        fit_idx, valid_idx = np.arange(len(train)), None
    else:
        raise ValueError(f'Unsupported classification phase: {phase}')
    xfit, yfit, raw_fit = x.iloc[fit_idx], y.iloc[fit_idx], train.iloc[fit_idx]
    inner_scores, score, training_score, diagnostics = [], None, None, None
    inner_warnings, requested_inner_folds, planned_inner_folds = [], 0, 0
    need_proba = metric in {'roc_auc', 'log_loss'} or specification.get('prediction_type') == 'probability'
    with threadpool_limits(limits=2):
        if phase == 'evaluate' and validation.get('inner_validation') == 'cv' and family not in {'flaml', 'autogluon'}:
            splits = list(_inner_splits(raw_fit, yfit, config, validation))
            requested_inner_folds, planned_inner_folds = int(validation.get('folds', 3)), len(splits)
            last_fold_seconds, last_fit_rows = None, None
            for inner_fit, inner_valid in splits:
                if last_fold_seconds is not None:
                    # Inner scores are advisory. Reserve the measured cost of a
                    # larger fit on ALL outer-fit rows and output production.
                    full_fit_reserve = last_fold_seconds * len(xfit) / last_fit_rows * 1.5 + 5.
                    next_fold_estimate = last_fold_seconds * len(inner_fit) / last_fit_rows * 1.25
                    if deadline - time.monotonic() <= full_fit_reserve + next_fold_estimate:
                        inner_warnings.append('Inner CV stopped early to reserve time for the common outer-fit model and predictions; '
                                              f'completed {len(inner_scores)} of {planned_inner_folds} planned folds')
                        break
                fold_started = time.monotonic()
                fitted, _ = _fit_plain(family, parameters, xfit.iloc[inner_fit], yfit.iloc[inner_fit], numeric, categorical, seed, deadline, config)
                prediction, probabilities = _predict(fitted, xfit.iloc[inner_valid], class_count, need_proba)
                inner_scores.append(metric_score(yfit.iloc[inner_valid], prediction, probabilities, metric, class_count, positive))
                last_fold_seconds, last_fit_rows = time.monotonic() - fold_started, len(inner_fit)
        if phase == 'final':
            bundle = joblib.load(source/'model.joblib')
            if not np.array_equal(bundle['label_encoder'].classes_, encoder.classes_):
                raise ValueError('Frozen classification label mapping changed before full refit')
            if family in {'flaml', 'autogluon'}:
                model = _fresh_automl(bundle['model']).fit(xfit, yfit, deadline, output)
                actual = source_result['actual_parameters']
            else:
                model, final_overrides = _freeze_full_fit(bundle['model'])
                model.fit(xfit, yfit)
                actual = {**source_result['actual_parameters'], 'full_refit_overrides': final_overrides}
        elif family in {'flaml', 'autogluon'}:
            model, actual = _fit_automl(family, parameters, xfit, yfit, raw_fit, config, validation, deadline, output, class_count, positive)
        else:
            model, actual = _fit_plain(family, parameters, xfit, yfit, numeric, categorical, seed, deadline, config)
        _left(deadline)
        if phase == 'evaluate':
            predicted_train, probabilities_train = _predict(model, xfit, class_count, need_proba)
            training_score = metric_score(yfit, predicted_train, probabilities_train, metric, class_count, positive)
            predicted, probabilities = _predict(model, x.iloc[valid_idx], class_count, need_proba)
            score = metric_score(y.iloc[valid_idx], predicted, probabilities, metric, class_count, positive)
            evidence = pd.DataFrame({'row_index': valid_idx, 'true_label': train[target].iloc[valid_idx].to_numpy(),
                                     'prediction': encoder.inverse_transform(predicted)})
            if probabilities is not None:
                for index in range(class_count):
                    evidence[f'probability_class_{index}'] = probabilities[:, index]
            evidence.to_csv(output/'validation_predictions.csv', index=False)
            diagnostics = {'class_labels': encoder.classes_.tolist(),
                           'confusion_matrix': confusion_matrix(y.iloc[valid_idx], predicted, labels=np.arange(class_count)).tolist(),
                           'per_class_report': classification_report(y.iloc[valid_idx], predicted, labels=np.arange(class_count),
                                                                    target_names=[str(label) for label in encoder.classes_],
                                                                    zero_division=0, output_dict=True)}
        prediction = _submission_predictions(model, xtest, encoder, specification, metric)
        submission = _submission(config, root, train, test, prediction)
        filename = specification.get('filename', 'submission.csv')
        if Path(filename).name != filename:
            raise ValueError('submission.filename must be a plain filename')
        submission.to_csv(output/filename, index=False)
        _left(deadline)
    _check_source_files(config, root, expected)
    model_path = output/'model.joblib'
    joblib.dump({'model': model, 'label_encoder': encoder}, model_path)
    upstream = model.features_ if isinstance(model, PreparedClassifier) else model.named_steps.get('upstream') if isinstance(model, Pipeline) else None
    result = {'status': 'completed', 'phase': phase, 'candidate': candidate['name'], 'family': family,
              'task_type': 'classification', 'target_transform': 'none', 'selection_metadata': metadata,
              'primary_metric': metric, 'metric': metric, 'direction': 'minimize' if metric == 'log_loss' else 'maximize',
              **hashes, 'score': score, 'score_scope': 'outer_validation' if phase == 'evaluate' else 'none_full_data_refit',
              'training_score': training_score, 'inner_cv_score': float(np.mean(inner_scores)) if inner_scores else None,
              'inner_fold_scores': inner_scores, 'actual_inner_folds': len(inner_scores),
              'requested_inner_folds': requested_inner_folds, 'planned_inner_folds': planned_inner_folds,
              'inner_cv_complete': bool(planned_inner_folds and len(inner_scores) == planned_inner_folds),
              'warnings': inner_warnings, 'duration_seconds': time.monotonic()-started,
              'actual_training_rows': len(fit_idx), 'total_labelled_rows': len(train),
              'outer_fit_rows': len(fit_idx) if phase == 'evaluate' else source_result.get('outer_fit_rows'),
              'outer_validation_rows': len(valid_idx) if valid_idx is not None else source_result.get('outer_validation_rows'),
              'full_training_rows': len(train) if phase == 'final' else None,
              'validation_rows': len(valid_idx) if valid_idx is not None else 0,
              'prediction_rows': len(prediction), 'parameters': parameters, 'actual_parameters': actual,
              'excluded_columns': excluded, 'feature_columns': upstream.feature_names_ if upstream else list(x.columns),
              'preparation_fit_rows': upstream.fit_rows_ if upstream else None,
              'class_labels': encoder.classes_.tolist(), 'positive_label': encoder.classes_[positive].item() if positive is not None and hasattr(encoder.classes_[positive], 'item') else encoder.classes_[positive] if positive is not None else None,
              'classification_diagnostics': diagnostics, 'model_path': str(model_path),
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'submission_path': str(output/filename), 'submission_scope': 'full_data_refit' if phase == 'final' else 'outer_fit_fallback'}
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    return result
