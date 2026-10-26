"""Budgeted tabular AutoML adapters; inner selection never sees outer labels."""

from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .plans import AutoMLPlan
from .features import create_features
from .auto_settings import model_settings


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


def _fit_flaml(x, y, metric, classification, runtime, seed, plan, validation=None, weights=None,
               *, task=None, groups=None, groups_val=None):
    from flaml import AutoML
    from flaml.automl.model import CatBoostEstimator

    class DeadlineCallback:
        def __init__(self, deadline):
            self.deadline = deadline

        def after_iteration(self, info):
            return time.time() < self.deadline

    class BudgetedCatBoost(CatBoostEstimator):
        def fit(self, X_train, y_train, budget=None, free_mem_ratio=0, **kwargs):
            remaining = _remaining(runtime)
            deadline = time.time() + min(budget if budget is not None else remaining, remaining)
            estimator_class = self.estimator_class

            class TimedEstimator(estimator_class):
                def fit(self, *args, **fit_kwargs):
                    # FLAML 2.3.6 calls its base callback directly and skips CatBoost time checks.
                    callbacks = list(fit_kwargs.get('callbacks') or [])
                    fit_kwargs['callbacks'] = callbacks + [DeadlineCallback(deadline)]
                    return super().fit(*args, **fit_kwargs)

            self.estimator_class = TimedEstimator
            try:
                return super().fit(X_train, y_train, budget=budget, free_mem_ratio=free_mem_ratio, **kwargs)
            finally:
                self.estimator_class = estimator_class

    model = AutoML()
    model.add_learner(learner_name='catboost', learner_class=BudgetedCatBoost)
    smoke = bool(runtime.get('is_smoke_check'))
    # A final trial can finish after the search budget; reserve room for it and prediction.
    budget = min(5., _remaining(runtime)) if smoke else _remaining(runtime) * 0.75
    # Three folds plus an unconditional full refit can exceed the process deadline
    # even when the search itself respects its time budget.
    validation_options = {'eval_method': 'holdout',
                          'split_type': 'stratified' if classification else 'uniform',
                          'split_ratio': 0.2, 'retrain_full': 'budget',
                          'train_time_limit': min(30., budget / 4)}
    if plan.validation == 'cv' and not smoke:
        if task == 'rank':
            from sklearn.model_selection import GroupKFold
            splitter = GroupKFold(n_splits=plan.folds)
        else:
            factory = StratifiedKFold if classification else KFold
            splitter = factory(n_splits=plan.folds, shuffle=True, random_state=seed)
        validation_options = {'eval_method': 'cv', 'split_type': splitter, 'retrain_full': 'budget',
            'train_time_limit': min(30., budget / 4)}
    else:
        validation_options['split_ratio'] = plan.holdout_fraction
    if validation is not None:
        validation_options = {'eval_method': 'holdout', 'X_val': validation[0], 'y_val': validation[1],
                              'retrain_full': False, 'sample': False,
                              'train_time_limit': min(30., budget / 4)}
    models, parameters = model_settings(plan, smoke)
    if task == 'rank':
        validation_options.update(groups=groups, groups_val=groups_val)
        if validation_options['eval_method'] != 'cv':
            validation_options['split_type'] = 'group'
    model.fit(X_train=x, y_train=y, task=task or ('classification' if classification else 'regression'),
              metric=metric, time_budget=budget,
              estimator_list=models, custom_hp=parameters, max_iter=plan.num_trials,
              ensemble=False if smoke or plan.ensemble is None else plan.ensemble,
              n_jobs=int(runtime.get('num_cpus', 2)),
              mem_thres=int(float(runtime.get('memory_limit_gb', 6)) * 1024 ** 3),
              log_file_name=str(Path(runtime.get('work_directory', '.')) /
                                ('flaml_smoke_search.jsonl' if smoke else 'flaml_full_search.jsonl')),
              **validation_options, **({'sample_weight': weights} if weights is not None else {}),
              model_history=False, seed=seed, verbose=1)
    # A fixed one-trial replay fits directly, without an inner validation score.
    return model, {'best_estimator': model.best_estimator, 'best_config': model.best_config,
                   'inner_best_loss': float(model.best_loss) if np.isfinite(model.best_loss) else None}


def _fit_autogluon(x, y, metric, classification, runtime, seed, plan, validation=None, weights=None):
    from autogluon.tabular import TabularPredictor

    label = '__automl_target__'
    while label in x:
        label += '_'
    data = x.copy()
    data[label] = y.to_numpy()
    if classification and validation is None and int(y.value_counts().min()) < 2:
        raise ValueError('AutoGluon inner validation requires at least two fit rows per class')
    smoke = bool(runtime.get("is_smoke_check"))
    bags = 0 if smoke else (plan.bag_folds or (plan.folds if plan.validation == "cv" else 0))
    fraction = max(plan.holdout_fraction, min(0.5, (y.nunique() + 1) / len(y))) if classification else plan.holdout_fraction
    if bags:
        train, tune = data, None
    elif validation is not None:
        train, tune = data, validation[0].copy()
        tune[label] = np.asarray(validation[1])
    else:
        train, tune = train_test_split(data, test_size=fraction, random_state=seed,
                                      stratify=y if classification else None)
    path = Path(runtime.get('work_directory', '.')) / 'autogluon'
    weighting = {}
    if weights is not None:
        weight_column = label + '_weight'
        train[weight_column] = weights
        tune[weight_column] = 1.
        weighting = {'sample_weight': weight_column, 'weight_evaluation': False}
    model = TabularPredictor(label=label, eval_metric=metric, **weighting,
                            problem_type=('binary' if y.nunique() == 2 else 'multiclass')
                            if classification else 'regression', path=str(path), verbosity=1,
                            learner_kwargs={'label_count_threshold': 1})
    smoke = bool(runtime.get('is_smoke_check'))
    # AutoGluon spends several seconds constructing its learner/feature pipeline.
    # A five-second fit budget can expire before the first tiny model starts.
    budget = min(20., _remaining(runtime)) if smoke else _remaining(runtime)
    _, parameters = model_settings(plan, smoke)
    tuning = {'hyperparameter_tune_kwargs': {'num_trials': plan.num_trials or 10,
              'scheduler': 'local', 'searcher': 'random'}} if plan.search_space and not smoke else {}
    model.fit(train_data=train, tuning_data=tune, time_limit=budget,
              hyperparameters=parameters, **tuning,
              num_cpus=int(runtime.get('num_cpus', 2)), num_gpus=0,
              memory_limit=float(runtime.get('memory_limit_gb', 6)),
              fit_strategy='sequential', num_bag_folds=bags, num_bag_sets=1,
              num_stack_levels=0 if smoke else plan.stack_levels,
              dynamic_stacking=False, calibrate_decision_threshold=False,
              fit_weighted_ensemble=False if smoke else (True if plan.ensemble is None else plan.ensemble))
    # Keep only the winning predictor and the models it actually depends on.
    model.delete_models(models_to_keep='best', dry_run=False)
    # Retain the single native training cache needed by refit_full; no copies in child nodes.
    return model, {'best_model': model.model_best, 'model_directory': str(path),
                   'inner_fit_rows': len(train), 'inner_validation_rows': len(tune) if tune is not None else len(data),
                   'inner_validation': 'out_of_fold' if bags else 'holdout', 'bag_folds': bags}


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


def run_automl(state, engine='flaml', *, task_type=None, prediction_type=None, class_labels=None, config=None):
    from .task_types import family
    from .automl_tasks import run_targets, run_ranking, run_forecast
    plan = AutoMLPlan.model_validate(config or {'engine': engine})
    if plan.engine != engine:
        raise ValueError('Configuration engine differs from the requested engine')
    preparation = state['preparation']
    requirements = preparation.get('requirements', {})
    defaults = {'task_type': task_type or requirements.get('problem', {}).get('task_type'),
                'prediction_type': prediction_type or requirements.get('output', {}).get('prediction_type'),
                'class_labels': class_labels or requirements.get('problem', {}).get('class_labels')}
    kind = family(defaults['task_type'], preparation['data']['target_columns'], state['fit'])
    runtime = dict(preparation.get('runtime', {}))
    runtime['deadline'] = time.monotonic() + _remaining(runtime) * plan.budget_fraction
    state = {**state, 'preparation': {**preparation, 'runtime': runtime}}
    plan = plan.model_copy(update={'budget_fraction': 1.})
    if kind == 'clustering':
        raise ValueError('FLAML/AutoGluon Tabular do not search clustering models; use the clustering fit pipeline')
    if kind == 'ranking':
        return run_ranking(state, plan, defaults)
    if len(preparation['data']['target_columns']) > 1:
        return run_targets(state, plan, defaults)
    if kind == 'forecasting':
        return run_forecast(state, plan, defaults)
    return _run_single(state, engine, config=plan.model_dump(), **defaults)


def _run_single(state, engine='flaml', *, task_type=None, prediction_type=None, class_labels=None, config=None):
    """Return standard assembly updates for one supervised tabular target.

    task_type/prediction_type/class_labels should come from extracted requirements.
    Unknown metrics or ambiguous multi-class probability schemas fail explicitly.
    Feature-aware and group/time candidates use an explicit inner holdout, then frozen refit.
    """
    plan = AutoMLPlan.model_validate(config or {'engine': engine})
    if plan.engine != engine:
        raise ValueError('Configuration engine differs from the requested engine')
    preparation = state['preparation']
    requirements = preparation.get('requirements', {})
    problem = requirements.get('problem', {})
    task_type = task_type or problem.get('task_type')
    prediction_type = prediction_type or requirements.get('output', {}).get('prediction_type')
    class_labels = class_labels or problem.get('class_labels')
    runtime = dict(preparation.get('runtime', {}))
    runtime['deadline'] = time.monotonic() + _remaining(runtime) * plan.budget_fraction
    binding = preparation['data']
    targets = binding.get('target_columns', [])
    if len(targets) != 1:
        raise ValueError('Built-in AutoML requires exactly one supervised target')
    from .validation import metadata, strategy
    validation_policy = metadata(preparation.get('validation', {}))
    if plan.split_column:
        validation_policy.update(time_column=plan.split_column, strategy='time')
    if plan.group_by:
        validation_policy.update(group_columns=[plan.group_by], strategy='group')
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
    feature_targets = preparation.get('all_target_columns', targets)
    ids = binding.get('id_columns', [])
    inherited = state['fit'].attrs.get('feature_recipes', [])
    recipes = [*inherited, *(feature.model_dump() for feature in plan.features)]
    created = [recipe['name'] for recipe in inherited]
    raw = [state[scope].drop(columns=created) for scope in ('fit', 'validation', 'test')]
    fit, validation, test = create_features(raw, recipes, forbidden=feature_targets, policy=validation_policy)
    columns = plan.columns or [c for c in fit if c not in {*feature_targets, *ids}]
    columns = list(dict.fromkeys([*columns, *(feature.name for feature in plan.features)]))
    if set(columns) & set(feature_targets):
        raise ValueError('AutoML input columns cannot include targets')
    columns = [c for c in columns if c not in plan.drop_columns]
    x, valid_x, test_x = _features(fit[columns], validation[columns], test[columns], feature_targets, ids)
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
    explicit = bool(recipes or plan.row_weights or strategy(validation_policy) not in {'random', 'stratified'})
    selection_x, selection_y, selection_runtime = x, y, runtime
    tuning, weights, refit_weights = None, None, None
    if explicit:
        from .eda import _inner_split
        from .plans import AnalysisRequest
        from .supervised_features import row_weights
        if plan.validation != 'holdout' or plan.bag_folds or plan.stack_levels or (engine == 'flaml' and plan.ensemble):
            raise ValueError('Feature/group/time-aware AutoML requires holdout without bagging/stacking; combine completed nodes separately')
        request = AnalysisRequest(name='automl_selection', kind='predictive', seed=seed,
                                  validation_fraction=plan.holdout_fraction)
        train_rows, valid_rows = _inner_split(raw[0], y, request, validation_policy, classification)
        inner_fit, inner_valid = create_features([raw[0].iloc[train_rows], raw[0].iloc[valid_rows]],
                                                 recipes, forbidden=feature_targets, policy=validation_policy)
        selection_x, tune_x, _ = _features(inner_fit[columns], inner_valid[columns], inner_valid[columns], feature_targets, ids)
        selection_y = y.iloc[train_rows].reset_index(drop=True)
        tuning = (tune_x, y.iloc[valid_rows].reset_index(drop=True))
        selection_runtime = {**runtime, 'deadline': time.monotonic() + _remaining(runtime) * .6}
        if plan.row_weights:
            parameters = plan.row_weights.model_dump()
            weights = row_weights(raw[0].iloc[train_rows], parameters)
            refit_weights = row_weights(raw[0], parameters)
    if engine == 'flaml':
        model, diagnostics = _fit_flaml(selection_x, selection_y, flaml_metric, classification,
                                      selection_runtime, seed, plan, tuning, weights)
    elif engine == 'autogluon':
        model, diagnostics = _fit_autogluon(selection_x, selection_y, ag_metric, classification,
                                          selection_runtime, seed, plan, tuning, weights)
    else:
        raise ValueError(f'Unknown built-in AutoML engine: {engine}')
    if explicit:
        import cloudpickle
        from .selected_model import freeze
        from .storage import atomic_bytes
        model = freeze(model, engine, classification).refit(x, y, runtime, refit_weights)
        if engine == 'flaml' and len(model.models) == 1:
            from .model_evidence import estimator_facts
            diagnostics.update(estimator_facts(model.models[0].model))
        diagnostics['selected_model_count'] = len(model.models)
        selected_file = Path(runtime['work_directory']) / 'selected_model.pkl'
        atomic_bytes(selected_file, cloudpickle.dumps(model))
        diagnostics.update(selected_model_file=str(selected_file), refitted_on_fit=True,
                           inner_fit_rows=len(train_rows), inner_validation_rows=len(valid_rows),
                           inner_validation='explicit_holdout', recipe_statistics='inner_train_only',
                           full_fit_rows=len(fit))
    elif not runtime.get('is_smoke_check') and not plan.bag_folds and not plan.stack_levels:
        import cloudpickle
        from .selected_model import freeze
        from .storage import atomic_bytes
        frozen = freeze(model, engine, classification)
        selected_file = Path(runtime['work_directory']) / 'selected_model.pkl'
        atomic_bytes(selected_file, cloudpickle.dumps(frozen))
        diagnostics.update(selected_model_file=str(selected_file), selected_model_count=len(frozen.models))
        if engine == 'flaml' and len(frozen.models) == 1:
            from .model_evidence import estimator_facts
            diagnostics.update(estimator_facts(frozen.models[0].model))
    columns = [column for column in preparation['output'].get('columns', []) if column not in ids]
    if not columns:
        columns = [target]
    from .operations import _requires_probabilities
    probability = classification and _requires_probabilities(state, {'prediction_type': prediction_type})

    def predict(frame):
        if not probability:
            if len(columns) != 1:
                raise ValueError('Label/regression AutoML output requires exactly one prediction column')
            values = np.asarray(model.predict(frame)).reshape(-1)
            result = pd.DataFrame({columns[0]: values})
            if classification:
                from .predictions import attach_probabilities
                probabilities, classes = _probabilities(model, frame)
                attach_probabilities(result, probabilities, classes)
            return result
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
            from .predictions import attach_probabilities
            return attach_probabilities(pd.DataFrame({columns[0]: values[:, match[0]]}), values, classes)
        class_map = {str(label): i for i, label in enumerate(classes)}
        if len(columns) == len(classes) and all(str(column) in class_map for column in columns):
            order = [class_map[str(column)] for column in columns]
        elif class_labels and len(columns) == len(classes) == len(class_labels):
            order = [class_map[str(label)] for label in class_labels]
        else:
            raise ValueError('Multiclass probability columns must explicitly map to the class order')
        from .predictions import attach_probabilities
        return attach_probabilities(pd.DataFrame(values[:, order], columns=columns), values, classes)

    from .features import feature_warnings
    diagnostics['feature_warnings'] = feature_warnings(x, [recipe['name'] for recipe in recipes], 'fit')
    valid_predictions, test_predictions = predict(valid_x), predict(test_x)
    if not runtime.get('is_smoke_check') and runtime.get('fit_row_positions') is not None:
        import io
        from .storage import atomic_bytes
        buffer = io.BytesIO()
        np.savez(buffer, rows=runtime['fit_row_positions'])
        atomic_bytes(Path(runtime['work_directory']) / 'training_rows.npz', buffer.getvalue())
    if engine == 'flaml' and not explicit and not runtime.get('is_smoke_check'):
        import cloudpickle
        from .storage import atomic_bytes
        # Persist inference components, not AutoML search state and copies of training data.
        snapshot = {'model': model.model, 'feature_transformer': model.feature_transformer,
                    'label_transformer': model.label_transformer, 'columns': list(x.columns),
                    'categories': {column: x[column].cat.categories.tolist()
                                   for column in x.select_dtypes(include='category')}}
        model_file = Path(runtime['work_directory']) / 'model.pkl'
        atomic_bytes(model_file, cloudpickle.dumps(snapshot))
        diagnostics['model_file'] = str(model_file)
    diagnostics.update(configuration=plan.model_dump(), engine=engine, elapsed_seconds=time.monotonic() - started,
                       fit_rows=len(fit), feature_count=x.shape[1], target=target,
                       prediction_type='probabilities' if probability else 'values')
    phase = 'smoke' if runtime.get('is_smoke_check') else 'full'
    destination = Path(runtime.get('work_directory', '.')) / f'{engine}_{phase}_diagnostics.json'
    destination.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return {'validation_predictions': valid_predictions, 'test_predictions': test_predictions,
            'automl_diagnostics': diagnostics}
