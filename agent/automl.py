"""Budgeted tabular AutoML adapters; inner selection never sees outer labels."""

from pathlib import Path
import json
import re
import time
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


_METRICS = {
    'rmse': ('regression', 'rmse', 'root_mean_squared_error'),
    'rootmeansquarederror': ('regression', 'rmse', 'root_mean_squared_error'),
    'mse': ('regression', 'mse', 'mean_squared_error'),
    'meansquarederror': ('regression', 'mse', 'mean_squared_error'),
    'mae': ('regression', 'mae', 'mean_absolute_error'),
    'meanabsoluteerror': ('regression', 'mae', 'mean_absolute_error'),
    'mape': ('regression', 'mape', 'mean_absolute_percentage_error'),
    'r2': ('regression', 'r2', 'r2'),
    'r2score': ('regression', 'r2', 'r2'),
    'accuracy': ('classification', 'accuracy', 'accuracy'),
    'accuracyscore': ('classification', 'accuracy', 'accuracy'),
    'macrof1': ('classification', 'macro_f1', 'f1_macro'),
    'f1macro': ('classification', 'macro_f1', 'f1_macro'),
    'f1': ('classification', 'f1', 'f1'),
    'rocauc': ('classification', 'roc_auc', 'roc_auc'),
    'auc': ('classification', 'roc_auc', 'roc_auc'),
    'logloss': ('classification', 'log_loss', 'log_loss'),
    'crossentropy': ('classification', 'log_loss', 'log_loss'),
}


def _remaining(runtime):
    deadline = float(runtime.get('deadline', time.monotonic() + runtime.get('time_budget_seconds', 60)))
    remaining = deadline - time.monotonic()
    if remaining < 1:
        raise TimeoutError('AutoML training budget exhausted before fit')
    return remaining


def _features(fit, validation, test, targets, ids):
    columns = [column for column in fit if column not in {*targets, *ids}]
    if not columns:
        raise ValueError('AutoML has no feature columns after excluding targets and identifiers')
    frames = [frame.loc[:, columns].reset_index(drop=True).copy() for frame in (fit, validation, test)]
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frames[0][column]):
            # Vocabulary is learned from fit only. Unknown categories stay explicit.
            known = set(frames[0][column].dropna().astype(str))
            categories = sorted(known | {'__automl_missing__', '__automl_unknown__'})
            for frame in frames:
                values = frame[column].astype('string').fillna('__automl_missing__')
                values = values.where(values.isin(known | {'__automl_missing__'}), '__automl_unknown__')
                frame[column] = pd.Categorical(values, categories=categories)
        else:
            for frame in frames:
                frame[column] = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return frames


def _splitter(y, classification, seed):
    if classification:
        smallest = int(y.value_counts().min())
        if smallest < 2:
            raise ValueError('AutoML inner validation requires at least two fit rows per class')
        return StratifiedKFold(n_splits=min(3, smallest), shuffle=True, random_state=seed)
    return KFold(n_splits=min(3, len(y)), shuffle=True, random_state=seed)


def _fit_flaml(x, y, metric, classification, runtime, seed):
    from flaml import AutoML

    model = AutoML()
    smoke = bool(runtime.get('is_smoke_check'))
    budget = min(5., _remaining(runtime)) if smoke else _remaining(runtime)
    model.fit(X_train=x, y_train=y, task='classification' if classification else 'regression',
              metric=metric, time_budget=budget,
              estimator_list=['lgbm'] if smoke else ['lgbm', 'catboost', 'xgboost', 'extra_tree'],
              n_jobs=int(runtime.get('num_cpus', 2)),
              mem_thres=int(float(runtime.get('memory_limit_gb', 6)) * 1024 ** 3),
              eval_method='cv', split_type=_splitter(y, classification, seed),
              retrain_full=True, model_history=False, seed=seed, verbose=1)
    return model, {'best_estimator': model.best_estimator, 'best_config': model.best_config,
                   'inner_best_loss': float(model.best_loss)}


def _fit_autogluon(x, y, metric, classification, runtime, seed):
    from autogluon.tabular import TabularPredictor

    label = '__automl_target__'
    while label in x:
        label += '_'
    data = x.copy()
    data[label] = y.to_numpy()
    if classification and int(y.value_counts().min()) < 2:
        raise ValueError('AutoGluon inner validation requires at least two fit rows per class')
    fraction = max(0.2, min(0.5, (y.nunique() + 1) / len(y))) if classification else 0.2
    train, tune = train_test_split(data, test_size=fraction, random_state=seed,
                                  stratify=y if classification else None)
    path = Path(runtime.get('work_directory', '.')) / f'autogluon_{uuid4().hex}'
    model = TabularPredictor(label=label, eval_metric=metric,
                            problem_type=('binary' if y.nunique() == 2 else 'multiclass')
                            if classification else 'regression', path=str(path), verbosity=1)
    smoke = bool(runtime.get('is_smoke_check'))
    # AutoGluon spends several seconds constructing its learner/feature pipeline.
    # A five-second fit budget can expire before the first tiny model starts.
    budget = min(20., _remaining(runtime)) if smoke else _remaining(runtime)
    model.fit(train_data=train, tuning_data=tune, time_limit=budget,
              hyperparameters={'GBM': {'num_boost_round': 20}} if smoke else {'GBM': {}, 'CAT': {}, 'XGB': {}, 'RF': {}, 'XT': {}},
              num_cpus=int(runtime.get('num_cpus', 2)), num_gpus=0,
              memory_limit=float(runtime.get('memory_limit_gb', 6)),
              fit_strategy='sequential', num_bag_folds=0, num_stack_levels=0,
              dynamic_stacking=False, calibrate_decision_threshold=False,
              fit_weighted_ensemble=not smoke)
    return model, {'best_model': model.model_best, 'model_directory': str(path),
                   'inner_fit_rows': len(train), 'inner_validation_rows': len(tune)}


def _probabilities(model, frame):
    values = model.predict_proba(frame)
    if isinstance(values, pd.DataFrame):
        return values.to_numpy(), list(values.columns)
    classes = getattr(model, 'classes_', None)
    if classes is None:
        classes = getattr(model, 'class_labels', None)
    if classes is None:
        raise ValueError('AutoML did not expose the class order of its probabilities')
    return np.asarray(values), list(classes)


def run_automl(state, engine='flaml', *, task_type=None, prediction_type=None, class_labels=None):
    """Return standard assembly updates for one supervised tabular target.

    task_type/prediction_type/class_labels should come from extracted requirements.
    Unknown metrics or ambiguous multi-class probability schemas fail explicitly.
    Group/time splits are excluded until their inner validation policy is provided.
    """
    preparation = state['preparation']
    requirements = preparation.get('requirements', {})
    problem = requirements.get('problem', {})
    task_type = task_type or problem.get('task_type')
    prediction_type = prediction_type or requirements.get('output', {}).get('prediction_type')
    class_labels = class_labels or problem.get('class_labels')
    runtime = preparation.get('runtime', {})
    binding = preparation['data']
    targets = binding.get('target_columns', [])
    if len(targets) != 1:
        raise ValueError('Built-in AutoML requires exactly one supervised target')
    validation_policy = preparation.get('validation', {})
    strategy = str(validation_policy.get('strategy', '')).lower()
    if validation_policy.get('group_columns') or any(word in strategy for word in ('time', 'temporal', 'group')):
        raise ValueError('Built-in AutoML is not enabled for temporal/group validation yet')
    name = re.sub(r'[^a-z0-9]', '', str(preparation['metric'].get('name', '')).lower())
    if name not in _METRICS:
        raise ValueError(f'Built-in AutoML does not have a verified mapping for metric {name!r}')
    family, flaml_metric, ag_metric = _METRICS[name]
    if task_type:
        requested = str(task_type).lower()
        if not any(word in requested for word in ('regress', 'classif', 'binary', 'multiclass')):
            raise ValueError(f'Unsupported built-in AutoML task type: {task_type}')
        if 'regress' in requested and family != 'regression':
            raise ValueError('AutoML task type conflicts with its metric')
        if any(word in requested for word in ('classif', 'binary', 'multiclass')) and family != 'classification':
            raise ValueError('AutoML task type conflicts with its metric')
    target = targets[0]
    ids = binding.get('id_columns', [])
    fit, validation, test = state['fit'], state['validation'], state['test']
    x, valid_x, test_x = _features(fit, validation, test, targets, ids)
    y = fit[target].reset_index(drop=True)
    if y.isna().any():
        raise ValueError('AutoML fit target contains missing values')
    classification = family == 'classification'
    if classification and y.nunique() < 2:
        raise ValueError('AutoML requires at least two classes in fit')
    if not classification:
        y = pd.to_numeric(y, errors='raise')
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError('AutoML regression target contains non-finite values')
    seed_value = validation_policy.get('seed')
    if seed_value is None:
        seed_value = validation_policy.get('random_state')
    seed = int(seed_value if seed_value is not None else 42)
    started = time.monotonic()
    if engine == 'flaml':
        model, diagnostics = _fit_flaml(x, y, flaml_metric, classification, runtime, seed)
    elif engine == 'autogluon':
        model, diagnostics = _fit_autogluon(x, y, ag_metric, classification, runtime, seed)
    else:
        raise ValueError(f'Unknown built-in AutoML engine: {engine}')
    columns = [column for column in preparation['output'].get('columns', []) if column not in ids]
    if not columns:
        columns = [target]
    probability = classification and (name in {'auc', 'rocauc', 'logloss', 'crossentropy'}
        or any(word in str(prediction_type).lower() for word in ('probab', 'вероят')))

    def predict(frame):
        if not probability:
            if len(columns) != 1:
                raise ValueError('Label/regression AutoML output requires exactly one prediction column')
            values = np.asarray(model.predict(frame)).reshape(-1)
            return pd.DataFrame({columns[0]: values})
        values, classes = _probabilities(model, frame)
        if values.ndim != 2 or values.shape[1] != len(classes):
            raise ValueError('Unexpected AutoML probability shape')
        if len(columns) == 1 and len(classes) == 2:
            # sklearn/FLAML and AutoGluon expose ascending class order; use
            # explicit requirement order if one is available.
            positive = class_labels[-1] if class_labels and len(class_labels) == 2 else classes[-1]
            match = [i for i, label in enumerate(classes) if str(label) == str(positive)]
            if len(match) != 1:
                raise ValueError('Positive class is absent from AutoML class labels')
            return pd.DataFrame({columns[0]: values[:, match[0]]})
        class_map = {str(label): i for i, label in enumerate(classes)}
        if len(columns) == len(classes) and all(str(column) in class_map for column in columns):
            order = [class_map[str(column)] for column in columns]
        elif class_labels and len(columns) == len(classes) == len(class_labels):
            order = [class_map[str(label)] for label in class_labels]
        else:
            raise ValueError('Multiclass probability columns must explicitly map to the class order')
        return pd.DataFrame(values[:, order], columns=columns)

    valid_predictions, test_predictions = predict(valid_x), predict(test_x)
    diagnostics.update(engine=engine, elapsed_seconds=time.monotonic() - started,
                       fit_rows=len(fit), feature_count=x.shape[1], target=target,
                       prediction_type='probabilities' if probability else 'values')
    phase = 'smoke' if runtime.get('is_smoke_check') else 'full'
    destination = Path(runtime.get('work_directory', '.')) / f'{engine}_{phase}_diagnostics.json'
    destination.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return {'validation_predictions': valid_predictions, 'test_predictions': test_predictions,
            'automl_diagnostics': diagnostics}
