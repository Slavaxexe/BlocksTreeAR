"""Budgeted Optuna search for ordinary boosting, with disjoint fit-only tuning."""

import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from max_automl import _METRICS


def _score(y, values, name, classification):
    if classification:
        labels = np.asarray(values).argmax(axis=1)
        if name in {'macrof1', 'f1macro'}:
            return float(f1_score(y, labels, labels=np.arange(values.shape[1]), average='macro', zero_division=0))
        if name == 'f1':
            if values.shape[1] != 2:
                raise ValueError('The f1 metric requires a binary task; specify macro_f1 for multiclass')
            return float(f1_score(y, labels, zero_division=0))
        if name in {'accuracy', 'accuracyscore'}:
            return float(accuracy_score(y, labels))
        if name in {'rocauc', 'auc'}:
            if values.shape[1] != 2:
                raise ValueError('roc_auc requires binary classification in the ordinary-model tuner')
            return float(roc_auc_score(y, values[:, 1]))
        return float(log_loss(y, values, labels=np.arange(values.shape[1])))
    predicted = np.asarray(values).reshape(-1)
    if name in {'mae', 'meanabsoluteerror'}:
        return float(mean_absolute_error(y, predicted))
    if name == 'mape':
        return float(mean_absolute_percentage_error(y, predicted))
    if name in {'r2', 'r2score'}:
        return float(r2_score(y, predicted))
    mse = float(mean_squared_error(y, predicted))
    return mse if name in {'mse', 'meansquarederror'} else float(np.sqrt(mse))


def _split(y, classification, seed):
    if len(y) < 12:
        raise ValueError('Optuna tuning requires at least 12 fit rows')
    if classification and int(pd.Series(y).value_counts().min()) < 3:
        raise ValueError('Disjoint tuning/calibration requires at least three fit rows per class')
    indexes = np.arange(len(y))
    calibration_size = max(int(np.ceil(len(y) * .2)), len(np.unique(y))) if classification else .2
    modeling, calibration = train_test_split(indexes, test_size=calibration_size, random_state=seed,
                                             stratify=y if classification else None)
    tune_size = max(int(np.ceil(len(modeling) * .2)), len(np.unique(y))) if classification else .2
    model_fit, tune = train_test_split(modeling, test_size=tune_size, random_state=seed + 1,
                                      stratify=y[modeling] if classification else None)
    return model_fit, tune, calibration, modeling


def _frames(state, targets, ids, learned_rows, engine):
    columns = [name for name in state['fit'] if name not in targets + ids]
    if not columns:
        raise ValueError('Ordinary model has no feature columns')
    frames = [state[key].loc[:, columns].copy().reset_index(drop=True) for key in ('fit', 'validation', 'test')]
    categories = []
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frames[0][column]):
            categories.append(column)
            known = set(frames[0].iloc[learned_rows][column].dropna().astype(str))
            vocabulary = sorted(known | {'__tuning_missing__', '__tuning_unknown__'})
            for frame in frames:
                values = frame[column].astype('string').fillna('__tuning_missing__')
                values = values.where(values.isin(known | {'__tuning_missing__'}), '__tuning_unknown__')
                frame[column] = pd.Categorical(values, categories=vocabulary) if engine == 'lightgbm' else values.astype(str)
        else:
            for frame in frames:
                frame[column] = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return frames, categories


def _cat_metric(name, classification, class_count, *, weighted=False):
    if classification:
        weight_option = ';use_weights=false' if weighted else ''
        if name in {'macrof1', 'f1macro'}:
            return 'TotalF1:average=Macro' + weight_option
        if name == 'f1':
            return 'F1:use_weights=false' if weighted else 'F1'
        if name in {'accuracy', 'accuracyscore'}:
            return 'Accuracy:use_weights=false' if weighted else 'Accuracy'
        if name in {'auc', 'rocauc'}:
            return 'AUC'
        return 'Logloss' if class_count == 2 else 'MultiClass'
    return {'mae': 'MAE', 'meanabsoluteerror': 'MAE', 'r2': 'R2', 'r2score': 'R2',
            'mape': 'MAPE'}.get(name, 'RMSE')


def _fit(engine, x, y, params, classification, class_count, categories, cpu, seed, deadline,
         name, *, evaluation=None, trial=None):
    import optuna

    if deadline - time.monotonic() < .5:
        raise TimeoutError('No training time remains for this Optuna fit')
    state = {'pruned': False, 'deadline_stop': False}
    if engine == 'lightgbm':
        import lightgbm as lgb

        def measured_metric(actual, predicted):
            values = np.asarray(predicted)
            if classification and values.ndim == 1:
                values = np.column_stack((1 - values, values))
            return 'primary', _score(actual, values, name, classification), name not in _LOSSES

        def stop_callback(env):
            if trial is not None and env.evaluation_result_list:
                value = next((entry[2] for entry in env.evaluation_result_list if entry[1] == 'primary'), None)
                if value is not None:
                    trial.report(float(value), step=env.iteration)
                    if trial.number and trial.should_prune():
                        state['pruned'] = True
                        raise optuna.TrialPruned()
            if time.monotonic() >= deadline:
                state['deadline_stop'] = True
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list or [])

        stop_callback.order = 25
        cls = lgb.LGBMClassifier if classification else lgb.LGBMRegressor
        settings = dict(params, n_jobs=cpu, verbosity=-1, random_state=seed, metric='None')
        model = cls(**settings)
        kwargs = {'callbacks': [stop_callback]}
        if evaluation is not None:
            kwargs.update(eval_set=[evaluation], eval_metric=measured_metric)
            kwargs['callbacks'].append(lgb.early_stopping(25, first_metric_only=True, verbose=False))
        model.fit(x, y, **kwargs)
        iterations = int(getattr(model, 'best_iteration_', 0) or model.booster_.current_iteration())
    else:
        from catboost import CatBoostClassifier, CatBoostRegressor

        weighted = params.get('auto_class_weights') not in (None, 'None') or params.get('class_weights') is not None
        metric = _cat_metric(name, classification, class_count, weighted=weighted)

        class StopCallback:
            def after_iteration(self, info):
                # CatBoost MAPE uses a different denominator from sklearn MAPE;
                # do not prune that objective with an incomparable intermediate score.
                if trial is not None and name != 'mape':
                    validation = next((value for key, value in info.metrics.items() if key != 'learn'), {})
                    values = next((value for key, value in validation.items() if key.split(':')[0] == metric.split(':')[0]), None)
                    if values:
                        score = float(values[-1])
                        if name in {'mse', 'meansquarederror'}:
                            score *= score
                        trial.report(score, step=info.iteration)
                        if trial.number and trial.should_prune():
                            state['pruned'] = True
                            return False
                if time.monotonic() >= deadline:
                    state['deadline_stop'] = True
                    return False
                return True

        cls = CatBoostClassifier if classification else CatBoostRegressor
        settings = dict(params, thread_count=cpu, random_seed=seed, allow_writing_files=False,
                        verbose=False, loss_function=('Logloss' if class_count == 2 else 'MultiClass')
                        if classification else ('MAE' if name in {'mae', 'meanabsoluteerror'} else 'RMSE'))
        model = cls(**settings)
        kwargs = {'cat_features': categories, 'callbacks': [StopCallback()]}
        if evaluation is not None:
            model.set_params(eval_metric=metric)
            kwargs.update(eval_set=evaluation, early_stopping_rounds=25, use_best_model=True)
        model.fit(x, y, **kwargs)
        if state['pruned']:
            raise optuna.TrialPruned()
        iterations = int(model.tree_count_)
    return model, max(1, iterations), state


_LOSSES = {'rmse', 'rootmeansquarederror', 'mse', 'meansquarederror', 'mae', 'meanabsoluteerror',
           'mape', 'logloss', 'crossentropy'}


def run_tuned_model(state, engine='lightgbm', *, task_type=None, prediction_type=None, class_labels=None):
    """Fit one ordinary model family; external validation/test are prediction-only."""
    import optuna

    if engine not in {'lightgbm', 'catboost'}:
        raise ValueError(f'Unsupported Optuna model family: {engine}')
    preparation = state['preparation']
    requirements = preparation.get('requirements', {})
    runtime = preparation.get('runtime', {})
    task = str(task_type or requirements.get('problem', {}).get('task_type') or '').lower()
    policy = preparation.get('validation', {})
    description = ' '.join(str(policy.get(key) or '') for key in ('strategy', 'method', 'scheme', 'split_type')).lower()
    if (policy.get('group_columns') or policy.get('time_column') or policy.get('time_columns')
            or any(word in description for word in ('time', 'temporal', 'chronological', 'group'))):
        raise ValueError('Ordinary-model tuning requires a mapped inner policy for temporal/group tasks')
    if any(word in task for word in ('multilabel', 'multioutput', 'forecast', 'rank')):
        raise ValueError('Ordinary-model tuner supports single-target classification/regression only')
    binding = preparation['data']
    targets, ids = binding.get('target_columns', []), binding.get('id_columns', [])
    if len(targets) != 1:
        raise ValueError('Ordinary-model tuner requires exactly one target')
    name = re.sub(r'[^a-z0-9]', '', str(preparation['metric'].get('name') or '').lower())
    if name not in _METRICS:
        raise ValueError(f'No verified metric mapping for Optuna: {name}')
    classification = _METRICS[name][0] == 'classification'
    if ('regress' in task and classification) or (('class' in task or 'binary' in task) and not classification):
        raise ValueError('Optuna task type conflicts with its metric')
    raw_y = state['fit'][targets[0]].reset_index(drop=True)
    if raw_y.isna().any():
        raise ValueError('Fit target contains missing values')
    encoder = LabelEncoder().fit(raw_y) if classification else None
    y = encoder.transform(raw_y) if classification else pd.to_numeric(raw_y, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Fit target contains non-finite values')
    class_count = len(encoder.classes_) if classification else 0
    if classification and class_count < 2:
        raise ValueError('Classification requires at least two fit classes')
    seed = int(policy.get('seed') or policy.get('random_state') or 42)
    model_rows, tune_rows, calibration_rows, final_rows = _split(y, classification, seed)
    (x, valid_x, test_x), categories = _frames(state, targets, ids, model_rows, engine)
    started = time.monotonic()
    available = float(runtime.get('deadline', started + runtime.get('time_budget_seconds', 120))) - started
    smoke = bool(runtime.get('is_smoke_check'))
    if available < 2:
        raise TimeoutError('Optuna budget exhausted before model search')
    available = min(available, 8.) if smoke else available
    final_deadline = started + available
    search_deadline = started + available * .48
    baseline_deadline = started + available * .65
    refit_deadline = started + available * .86
    cpu = max(1, int(runtime.get('num_cpus', 2)))
    baseline = ({'n_estimators': 12 if smoke else 160, 'learning_rate': .05, 'num_leaves': 31,
                 'min_child_samples': 20, 'reg_lambda': 1., 'class_weight': None}
                if engine == 'lightgbm' else
                {'iterations': 12 if smoke else 100, 'learning_rate': .05, 'depth': 6,
                 'l2_leaf_reg': 3., 'border_count': 64, 'auto_class_weights': None})
    if not classification:
        baseline.pop('class_weight' if engine == 'lightgbm' else 'auto_class_weights', None)
    study = optuna.create_study(direction='minimize' if name in _LOSSES else 'maximize',
                               sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=20, interval_steps=10))
    best = {}
    baseline_search = {}

    def objective(trial):
        if trial.number == 0:
            params = dict(baseline)
        elif engine == 'lightgbm':
            params = {'n_estimators': 20 if smoke else trial.suggest_int('n_estimators', 80, 450),
                      'learning_rate': trial.suggest_float('learning_rate', .025, .16, log=True),
                      'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                      'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
                      'reg_lambda': trial.suggest_float('reg_lambda', .01, 20., log=True)}
            if classification:
                params['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])
        else:
            params = {'iterations': 20 if smoke else trial.suggest_int('iterations', 60, 260),
                      'learning_rate': trial.suggest_float('learning_rate', .025, .16, log=True),
                      'depth': trial.suggest_int('depth', 4, 7),
                      'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', .5, 15., log=True), 'border_count': 64}
            if classification:
                params['auto_class_weights'] = trial.suggest_categorical('auto_class_weights', [None, 'Balanced', 'SqrtBalanced'])
        remaining = search_deadline - time.monotonic()
        if remaining < 1:
            raise optuna.TrialPruned('Search deadline reached')
        trial_deadline = min(search_deadline, time.monotonic() + max(2., min(25., available * .18)))
        try:
            model, iterations, fit_state = _fit(engine, x.iloc[model_rows], y[model_rows], params,
                classification, class_count, categories, cpu, seed, trial_deadline, name,
                evaluation=(x.iloc[tune_rows], y[tune_rows]), trial=trial)
        except TimeoutError as exc:
            trial.set_user_attr('deadline_stop', True)
            raise optuna.TrialPruned(str(exc)) from exc
        values = np.asarray(model.predict_proba(x.iloc[tune_rows])) if classification else model.predict(x.iloc[tune_rows])
        score = _score(y[tune_rows], values, name, classification)
        if not np.isfinite(score):
            raise ValueError('Optuna trial produced a non-finite inner metric')
        trial.set_user_attr('actual_iterations', iterations)
        trial.set_user_attr('model_parameters', params)
        trial.set_user_attr('deadline_stop', fit_state['deadline_stop'])
        if trial.number == 0:
            baseline_search.update(model=model, score=score, iterations=iterations)
        if not best or (score < best['score'] if name in _LOSSES else score > best['score']):
            best.update(model=model, params=dict(params), score=score, iterations=iterations, trial=trial.number)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=2 if smoke else 12, timeout=max(.1, search_deadline-time.monotonic()),
                   catch=(ValueError,), show_progress_bar=False)
    if not best or not baseline_search:
        raise ValueError('No ordinary-model baseline completed within the search budget')
    count_key = 'n_estimators' if engine == 'lightgbm' else 'iterations'
    final_baseline = baseline_search['model']
    baseline_iterations = baseline_search['iterations']
    baseline_refit = False
    if not smoke and baseline_deadline - time.monotonic() > 2:
        params = dict(baseline)
        params[count_key] = baseline_search['iterations']
        try:
            final_baseline, baseline_iterations, _ = _fit(engine, x.iloc[final_rows], y[final_rows], params,
                classification, class_count, categories, cpu, seed, baseline_deadline, name)
            baseline_refit = True
        except TimeoutError:
            pass  # Retain the completed inner model if setup exhausted the refit allocation.
    final_model = best['model']
    tuned_iterations = best['iterations']
    tuned_refit = False
    if not smoke and refit_deadline - time.monotonic() > 2:
        params = dict(best['params'])
        params[count_key] = best['iterations']
        try:
            final_model, tuned_iterations, _ = _fit(engine, x.iloc[final_rows], y[final_rows], params,
                classification, class_count, categories, cpu, seed, refit_deadline, name)
            tuned_refit = True
        except TimeoutError:
            pass

    def values(model, frame):
        return np.asarray(model.predict_proba(frame)) if classification else np.asarray(model.predict(frame)).reshape(-1)

    baseline_valid, baseline_test = values(final_baseline, valid_x), values(final_baseline, test_x)
    tuned_valid, tuned_test = values(final_model, valid_x), values(final_model, test_x)
    recipe = None
    post_report = {'selected': 'identity', 'reason': 'Small smoke invocation; no postprocessing search'}
    if not smoke and final_deadline - time.monotonic() > 1:
        from max_postprocessing import fit_postprocessing
        try:
            recipe = fit_postprocessing(raw_y.iloc[calibration_rows].to_numpy(),
                values(final_model, x.iloc[calibration_rows]), values(final_baseline, x.iloc[calibration_rows]),
                classification=classification, metric_name=name, classes=list(encoder.classes_) if classification else None,
                seed=seed, deadline=final_deadline,
                positive_class=encoder.classes_[-1] if classification and class_count == 2 else None)
            post_report = recipe.report
        except (ValueError, TypeError, FloatingPointError, TimeoutError) as exc:
            recipe = None
            post_report = {'selected': 'identity', 'reason': f'Optional postprocessing failed: {exc}'}
    output_columns = [column for column in preparation['output'].get('columns', []) if column not in ids] or targets
    requested_type = str(prediction_type or requirements.get('output', {}).get('prediction_type') or '').lower()
    probability = classification and (name in {'auc', 'rocauc', 'logloss', 'crossentropy'}
                                       or any(word in requested_type for word in ('probab', 'вероят')))

    def output(predicted, *, processed=False):
        if not classification:
            if len(output_columns) != 1:
                raise ValueError('Regression tuner requires one prediction output column')
            return pd.DataFrame({output_columns[0]: np.asarray(predicted).reshape(-1)})
        if not probability:
            if len(output_columns) != 1:
                raise ValueError('Classification labels require one prediction output column')
            labels = recipe.predict_labels(predicted) if processed and recipe else encoder.inverse_transform(np.asarray(predicted).argmax(axis=1))
            return pd.DataFrame({output_columns[0]: np.asarray(labels).reshape(-1)})
        classes = list(encoder.classes_)
        if len(output_columns) == 1 and class_count == 2:
            positive = class_labels[-1] if class_labels and len(class_labels) == 2 else classes[-1]
            indexes = [i for i, label in enumerate(classes) if str(label) == str(positive)]
            if len(indexes) != 1:
                raise ValueError('Positive class does not map to model classes')
            return pd.DataFrame({output_columns[0]: predicted[:, indexes[0]]})
        mapping = {str(label): i for i, label in enumerate(classes)}
        if len(output_columns) == class_count and all(str(column) in mapping for column in output_columns):
            order = [mapping[str(column)] for column in output_columns]
        elif class_labels and len(output_columns) == class_count == len(class_labels):
            order = [mapping[str(label)] for label in class_labels]
        else:
            raise ValueError('Multiclass probability output columns do not map to classes')
        return pd.DataFrame(np.asarray(predicted)[:, order], columns=output_columns)

    final_valid = recipe.transform(tuned_valid, baseline_valid) if recipe else tuned_valid
    final_test = recipe.transform(tuned_test, baseline_test) if recipe else tuned_test
    validation_predictions, test_predictions = output(final_valid, processed=True), output(final_test, processed=True)
    diagnostics = {'engine': engine, 'metric': name, 'direction': study.direction.name.lower(),
        'inner_baseline_score': baseline_search['score'], 'inner_best_score': best['score'],
        'selected_trial': best['trial'], 'best_parameters': best['params'], 'baseline_parameters': baseline,
        'baseline_iterations': baseline_iterations, 'tuned_iterations': tuned_iterations,
        'baseline_refit': baseline_refit, 'tuned_refit': tuned_refit,
        'split_rows': {'model_fit': len(model_rows), 'tune': len(tune_rows), 'calibration': len(calibration_rows)},
        'calibration_excluded_from_all_model_fits': True,
        'preprocessing_vocabulary_scope': 'model_fit only',
        'completed_trials': sum(trial.state.name == 'COMPLETE' for trial in study.trials),
        'pruned_trials': sum(trial.state.name == 'PRUNED' for trial in study.trials),
        'budget_stop': time.monotonic() >= search_deadline or any(
            trial.user_attrs.get('deadline_stop') for trial in study.trials),
        'pruning_metric': 'disabled: CatBoost MAPE differs from sklearn MAPE' if engine == 'catboost' and name == 'mape' else name,
        'postprocessing': post_report, 'elapsed_seconds': time.monotonic()-started,
        'trials': [{'number': trial.number, 'state': trial.state.name, 'value': trial.value,
                    'parameters': trial.user_attrs.get('model_parameters', trial.params),
                    'actual_iterations': trial.user_attrs.get('actual_iterations'),
                    'deadline_stop': trial.user_attrs.get('deadline_stop')} for trial in study.trials]}
    directory = Path(runtime.get('work_directory', '.'))
    phase = 'smoke' if smoke else 'full'
    (directory/f'optuna_{engine}_{phase}_diagnostics.json').write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    if not smoke:
        output(baseline_valid).to_csv(directory/'baseline_validation_predictions.csv', index=False)
        output(baseline_test).to_csv(directory/'baseline_test_predictions.csv', index=False)
    return {'validation_predictions': validation_predictions, 'test_predictions': test_predictions,
            'tuning_diagnostics': diagnostics}
