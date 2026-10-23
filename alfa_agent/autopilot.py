"""Controller-written first experiment: a strong generic GBDT instead of a constant LLM baseline.

The source uses the ordinary fit_model interface, so checks, execution, repair, Baker parents and the
final refit treat it like any generated node. It applies only to one-target row-wise tables.
"""

import json
import re

TEMPLATE = '''"""Controller baseline: LightGBM, inner early stopping, loss / class weights chosen by the task scorer."""

import json

import numpy as np
import pandas as pd
import lightgbm as lgb

TASK = {task!r}
PROBA = {proba!r}
TEMPORAL = {temporal!r}
THREADS = {threads!r}


def ordinal(series):
    values = pd.to_numeric(series, errors='coerce')
    if values.notna().sum() < .5 * series.notna().sum():
        parsed = pd.to_datetime(series, errors='coerce', format='mixed')
        values = pd.Series(parsed.to_numpy().astype('datetime64[ns]').astype(np.int64), index=series.index).where(parsed.notna())
    return values.astype(float)


def schema(fit, preparation):
    data = preparation['data']
    drop = set(data['target_columns']) | set(data.get('id_columns') or [])
    columns, vocab = [], {{}}
    for column in fit.columns:
        series = fit[column]
        if column in drop:
            continue
        if not (pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series)):
            present = series.dropna().astype(str)
            if present.nunique() > max(1000, .5 * len(present)):
                continue  # free text or an undeclared identifier
            vocab[column] = sorted(present.unique())
        columns.append(column)
    return columns, vocab


def encode(frame, columns, vocab):
    result = {{}}
    for column in columns:
        values = frame[column].reset_index(drop=True)
        if column in vocab:
            result[column] = pd.Categorical(values.astype(str).where(values.notna()), categories=vocab[column])
        else:
            result[column] = pd.to_numeric(values, errors='coerce').astype(float)
    return pd.DataFrame(result)


def inner_split(fit, labels):
    rows = np.arange(len(fit))
    if TEMPORAL is not None and TEMPORAL in fit:
        values = ordinal(fit[TEMPORAL]).to_numpy()
        known = ~np.isnan(values)
        if known.any():
            later = known & (values >= np.quantile(values[known], .85, method='higher'))
            if later.any() and (~later).sum() >= 20:
                return rows[~later], rows[later]
    shuffled = np.random.RandomState(0).permutation(len(fit))
    if labels is not None:  # stratify so rare classes appear on both sides when possible
        order = np.argsort(labels[shuffled], kind='stable')
        held = np.zeros(len(fit), bool)
        held[shuffled[order][np.arange(len(order)) % 7 == 3]] = True
        return rows[~held], rows[held]
    cut = max(1, int(.15 * len(fit)))
    return np.sort(shuffled[cut:]), np.sort(shuffled[:cut])


def fit_model(fit, preparation):
    target = preparation['data']['target_columns'][0]
    scorer = preparation.get('scorer')
    fit = fit[fit[target].notna()].reset_index(drop=True)
    columns, vocab = schema(fit, preparation)
    X = encode(fit, columns, vocab)
    params = dict(n_estimators=5000, learning_rate=.05, num_leaves=31, min_child_samples=20, subsample=.8,
                  subsample_freq=1, colsample_bytree=.8, reg_lambda=1., n_jobs=THREADS, verbose=-1, random_state=0)
    if TASK == 'classification':
        classes, y = np.unique(fit[target].to_numpy(), return_inverse=True)
        counts = np.bincount(y, minlength=len(classes))
        balanced = len(y) / (len(classes) * counts)
        candidates = [('plain', {{}}, None), ('balanced', {{}}, balanced[y]), ('sqrt_balanced', {{}}, np.sqrt(balanced)[y])]
        train, valid = inner_split(fit, y)
    else:
        y = fit[target].to_numpy(dtype=float)
        spread = float(np.median(np.abs(y - np.median(y)))) or float(np.std(y)) or 1.
        candidates = [('l2', {{}}, None), ('huber', {{'objective': 'huber', 'alpha': 1.5 * spread}}, None)]
        if y.min() >= 0:
            candidates.append(('log_target', {{}}, 'log'))
        train, valid = inner_split(fit, None)
    reference = fit.iloc[valid].reset_index(drop=True)

    def train_one(extra, option, rows, eval_rows, trees=None):
        model_params = {{**params, **extra, **({{'n_estimators': trees}} if trees else {{}})}}
        if TASK == 'classification':
            model = lgb.LGBMClassifier(**model_params)
            weights = option[rows] if option is not None else None
            if eval_rows is None:
                return model.fit(X.iloc[rows], y[rows], sample_weight=weights)
            return model.fit(X.iloc[rows], y[rows], sample_weight=weights, eval_set=[(X.iloc[eval_rows], y[eval_rows])],
                             eval_sample_weight=[option[eval_rows]] if option is not None else None,
                             callbacks=[lgb.early_stopping(100, verbose=False)])
        model = lgb.LGBMRegressor(**model_params)
        labels = np.log1p(y) if option == 'log' else y
        if eval_rows is None:
            return model.fit(X.iloc[rows], labels[rows])
        return model.fit(X.iloc[rows], labels[rows], eval_set=[(X.iloc[eval_rows], labels[eval_rows])],
                         callbacks=[lgb.early_stopping(100, verbose=False)])

    def probabilities(model, features):  # a class absent from the inner train rows keeps its column
        result = np.zeros((len(features), len(classes)))
        result[:, model.classes_] = model.predict_proba(features)
        return result

    def output(model, option, features, scale=None):
        if TASK == 'classification':
            probability = probabilities(model, features)
            if PROBA:
                return probability[:, 1]
            return classes[np.argmax(probability * (scale if scale is not None else 1.), axis=1)]
        values = model.predict(features)
        return np.expm1(values) if option == 'log' else values

    def measure(values):
        if scorer is None:
            return None
        score = float(scorer(reference, pd.DataFrame({{target: values}}), preparation))
        return score if preparation['metric'].get('direction') != 'minimize' else -score

    results = []
    for name, extra, option in candidates:
        model = train_one(extra, option, train, valid)
        trees = max(50, int((model.best_iteration_ or params['n_estimators']) * 1.15))
        value = measure(output(model, option, X.iloc[valid]))
        results.append((value if value is not None else -len(results), name, extra, option, trees, model))
        if scorer is None:
            break
    value, name, extra, option, trees, model = max(results, key=lambda item: item[0])
    scale = None
    if TASK == 'classification' and not PROBA and scorer is not None and len(classes) <= 50:
        probability, scale = probabilities(model, X.iloc[valid]), np.ones(len(classes))
        best = measure(classes[np.argmax(probability, axis=1)])
        for _ in range(2):  # coordinate search of per-class multipliers for the task's own label metric
            for k in range(len(classes)):
                for factor in (.5, .7, .85, 1.2, 1.5, 2., 3.):
                    trial = scale.copy()
                    trial[k] = scale[k] * factor
                    score = measure(classes[np.argmax(probability * trial, axis=1)])
                    if score > best + 1e-9:
                        best, scale = score, trial
    final = train_one(extra, option, np.arange(len(fit)), None, trees)
    print(json.dumps({{'selected': name, 'trees': trees, 'inner_scores': {{r[1]: r[0] for r in results}},
                      'class_scale': None if scale is None else [round(float(s), 3) for s in scale]}}))

    ids = [c for c in preparation['data'].get('id_columns') or [] if c in fit]

    def predict(features):  # identifiers too: some generated writers read them from the predictions
        values = {{c: features[c].to_numpy() for c in ids if c in features}}
        return pd.DataFrame({{**values, target: output(final, option, encode(features, columns, vocab), scale)}})
    return predict
'''


AUTOML = '''"""Controller AutoML baseline: one library under a fixed time limit, the task metric where it maps."""

import tempfile

import numpy as np
import pandas as pd

ENGINE = {engine!r}
TASK = {task!r}
PROBA = {proba!r}
METRIC = {metric!r}
TIME_LIMIT = {time_limit!r}
THREADS = {threads!r}


def features(frame, preparation, vocab=None):
    data = preparation['data']
    drop = set(data['target_columns']) | set(data.get('id_columns') or [])
    result = frame[[c for c in frame.columns if c not in drop]].reset_index(drop=True).copy()
    for column, categories in (vocab or {{}}).items():
        result[column] = pd.Categorical(result[column].astype(str).where(result[column].notna()), categories=categories)
    return result


def fit_model(fit, preparation):
    target = preparation['data']['target_columns'][0]
    fit = fit[fit[target].notna()].reset_index(drop=True)
    X, y = features(fit, preparation), fit[target].reset_index(drop=True)
    limit = TIME_LIMIT if len(fit) >= 2000 else 8  # smoke checks use a few hundred rows
    classes = np.unique(y.to_numpy()) if TASK == 'classification' else None
    workspace = tempfile.TemporaryDirectory()  # kept alive by the returned predictor
    if ENGINE == 'autogluon':
        from autogluon.tabular import TabularPredictor
        kind = 'regression' if TASK == 'regression' else 'binary' if len(classes) == 2 else 'multiclass'
        model = TabularPredictor(label=target, problem_type=kind, eval_metric=METRIC, path=workspace.name, verbosity=0)
        model.fit(X.assign(**{{target: y}}), time_limit=limit, presets='best_quality' if len(fit) >= 2000 else 'medium_quality',
                  num_cpus=THREADS, num_gpus=0, dynamic_stacking=False)

        def raw(frame):
            if PROBA:
                return model.predict_proba(frame)[classes[1]].to_numpy()
            return model.predict(frame).to_numpy()
    elif ENGINE == 'flaml':
        from flaml import AutoML
        vocab = {{c: sorted(X[c].dropna().astype(str).unique()) for c in X.columns
                 if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))}}
        X = features(fit, preparation, vocab)
        model = AutoML()
        model.fit(X_train=X, y_train=y, task=TASK, metric=METRIC or 'auto', time_budget=limit, n_jobs=THREADS,
                  verbose=0, seed=0, log_file_name='')

        def raw(frame):
            frame = features(frame, preparation, vocab)
            return model.predict_proba(frame)[:, 1] if PROBA else np.asarray(model.predict(frame))
    else:
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task
        kind = 'reg' if TASK == 'regression' else 'binary' if len(classes) == 2 else 'multiclass'
        # Boosting only: LightAutoML's linear model indexes past its embedding table on unseen categories.
        model = TabularAutoML(task=Task(kind), timeout=limit, cpu_limit=THREADS,
                              general_params={{'use_algos': [['lgb', 'lgb_tuned', 'cb_tuned']]}},
                              reader_params={{'n_jobs': THREADS, 'cv': 5, 'random_state': 0}})
        ids = [c for c in preparation['data'].get('id_columns') or [] if c in fit]
        if TASK == 'classification':  # stratified 5-fold CV needs at least five rows per class
            counts = y.value_counts()
            fit = pd.concat([fit] + [fit[y == label].sample(5 - count, replace=True, random_state=0)
                                     for label, count in counts[counts < 5].items()], ignore_index=True)
        model.fit_predict(fit, roles={{'target': target, **({{'drop': ids}} if ids else {{}})}}, verbose=0)
        mapping = getattr(model.reader, 'class_mapping', None)
        order = classes if mapping is None else np.array(sorted(mapping, key=mapping.get), dtype=object)

        def raw(frame):
            values = model.predict(frame.reset_index(drop=True)).data
            if TASK == 'regression':
                return values[:, 0]
            if len(order) == 2:
                return values[:, 0] if PROBA else order[(values[:, 0] > .5).astype(int)]
            return order[np.argmax(values, axis=1)]

    ids = [c for c in preparation['data'].get('id_columns') or [] if c in fit]

    def predict(frame):
        values = raw(features(frame, preparation) if ENGINE != 'flaml' and ENGINE != 'lightautoml' else frame)
        if TASK == 'classification' and not PROBA:
            values = pd.Series(values).astype(y.dtype, errors='ignore').to_numpy()
        return pd.DataFrame({{**{{c: frame[c].to_numpy() for c in ids if c in frame}}, target: values}})
    predict.workspace = workspace
    return predict
'''

METRICS = (  # task metric → AutoGluon / FLAML names; unknown metrics keep library defaults
    (r'macro', {'autogluon': 'f1_macro', 'flaml': 'macro_f1'}),
    (r'weighted', {'autogluon': 'f1_weighted', 'flaml': None}),
    (r'rmse|root.?mean.?squared|среднеквадрат', {'autogluon': 'root_mean_squared_error', 'flaml': 'rmse'}),
    (r'\bmae\b|mean.?absolute|абсолютн', {'autogluon': 'mean_absolute_error', 'flaml': 'mae'}),
    (r'\bmse\b|mean.?squared', {'autogluon': 'mean_squared_error', 'flaml': 'mse'}),
    (r'\br2\b|r\^2|determination', {'autogluon': 'r2', 'flaml': 'r2'}),
    (r'auc|roc', {'autogluon': 'roc_auc', 'flaml': 'roc_auc'}),
    (r'log.?loss|cross.?entropy', {'autogluon': 'log_loss', 'flaml': 'log_loss'}),
    (r'accuracy|точност', {'autogluon': 'accuracy', 'flaml': 'accuracy'}),
    (r'\bf1\b', {'autogluon': 'f1', 'flaml': 'f1'}),
)


def automl(preparation: dict, threads: int, time_limit: float) -> list[tuple[str, dict]]:
    """AutoGluon, LightAutoML and FLAML sources under the GBDT template's applicability rules."""
    base = template(preparation, threads)
    if base is None:
        return []
    model = base[1]['pipeline']['model']
    task, proba = model['task'], model['probability_output']
    described = json.dumps({k: preparation['metric'].get(k) for k in ('name', 'definition')}, ensure_ascii=False).casefold()
    names = next((names for pattern, names in METRICS if re.search(pattern, described)), {})
    result = []
    for engine in ('autogluon', 'lightautoml', 'flaml'):
        metric = names.get(engine)
        code = AUTOML.format(engine=engine, task=task, proba=proba, metric=metric, time_limit=int(time_limit), threads=threads)
        result.append((code, {
            'rationale': f'Controller AutoML reference ({engine}): a strong library search the agent must beat or blend with.',
            'expected_effect': 'Competitive measured score from automated model search and ensembling under a time limit.',
            'pipeline': {'data': {'features': 'all columns except targets and ids'},
                         'preprocessing': {'automl': f'{engine} internal preprocessing'},
                         'model': {'family': engine, 'task': task, 'probability_output': proba, 'metric': metric,
                                   'preset': 'best_quality without dynamic stacking' if engine == 'autogluon' else 'default'},
                         'training': {'time_limit_seconds': int(time_limit), 'threads': threads}}}))
    return result


def template(preparation: dict, threads: int) -> tuple[str, dict] | None:
    """Return source and its plan for a one-target row task, or None to keep the LLM baseline."""
    data, metric = preparation['data'], preparation['metric']
    targets = data['target_columns']
    summary = preparation.get('profile', {}).get('target', {}).get(targets[0]) if len(targets) == 1 else None
    if summary is None or preparation.get('prediction_scope') != 'row':
        return None
    kind = (preparation.get('task_type') or '').casefold()
    if 'regress' in kind or 'регресс' in kind:
        task = 'regression'
    elif 'classif' in kind or 'классиф' in kind:
        task = 'classification'
    else:
        task = 'regression' if 'mean' in summary and summary.get('distinct', 0) > 20 else 'classification'
    if task == 'regression' and 'mean' not in summary:
        return None
    described = json.dumps({k: metric.get(k) for k in ('name', 'definition')}, ensure_ascii=False).casefold()
    proba = task == 'classification' and bool(re.search(r'auc|roc|log.?loss|cross.?entropy|brier|probab|вероятн', described))
    if proba and set(summary.get('counts', {})) - {'0', '1', '0.0', '1.0', 'True', 'False'}:
        return None  # probability columns for named or multiple classes need a task-specific writer
    temporal = preparation.get('validation', {}).get('shift', {}).get('temporal_column')
    code = TEMPLATE.format(task=task, proba=proba, temporal=temporal, threads=threads)
    choices = ('L2, Huber and log-target' if task == 'regression' else 'unweighted, balanced and sqrt-balanced class weights')
    hypothesis = {
        'rationale': 'Controller baseline: a robust gradient-boosting model gives the search a strong measured leader '
                     'before any LLM-generated experiment.',
        'expected_effect': 'Much better than a constant prediction; later nodes improve features, loss and ensembling.',
        'pipeline': {
            'data': {'features': 'all train/test columns except targets and ids; high-cardinality text dropped'},
            'preprocessing': {'numeric': 'as is, NaN kept', 'categorical': 'pandas category with fit vocabulary'},
            'model': {'family': 'LightGBM', 'task': task, 'probability_output': proba,
                      'candidates': choices, 'num_leaves': 31, 'learning_rate': .05},
            'training': {'inner_split': f'latest 15% of {temporal}' if temporal else 'stratified/random 15% of fit',
                         'early_stopping': 100, 'selection': 'task scorer on the inner split',
                         'post_processing': 'per-class probability multipliers tuned by the scorer' if task == 'classification' and not proba else None,
                         'final': 'retrain on all fit rows with 1.15x best iterations'}}}
    return code, hypothesis
