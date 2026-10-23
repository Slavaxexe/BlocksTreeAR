"""Importable building blocks for generated experiments.

gbdt() is a shift-aware LightGBM pipeline whose loss / class weights are chosen by the task scorer on an
inner split; automl() wraps AutoGluon, LightAutoML and FLAML; target_stats() builds leakage-safe statistics
of the target for repeated entities. Everything reads the task from `preparation`, so one call works on a
new task, and a generated node only has to describe its hypothesis (features, parameters, engine).
"""

import json
import os
import re
import tempfile
import warnings

import numpy as np
import pandas as pd

THREADS = int(os.environ.get('ALFA_THREADS', '4'))

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


def task_kind(preparation):
    """{'task': 'regression'|'classification', 'proba': bool} for one-target row tasks, else None."""
    data, metric = preparation['data'], preparation['metric']
    targets = data['target_columns']
    summary = preparation.get('profile', {}).get('target', {}).get(targets[0]) if len(targets) == 1 else None
    if summary is None or preparation.get('prediction_scope', 'row') != 'row':
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
    return {'task': task, 'proba': proba}


def metric_names(preparation):
    described = json.dumps({k: preparation['metric'].get(k) for k in ('name', 'definition')}, ensure_ascii=False).casefold()
    return next((names for pattern, names in METRICS if re.search(pattern, described)), {})


def temporal_column(preparation):
    return preparation.get('validation', {}).get('shift', {}).get('temporal_column')


def ordinal(series):
    values = pd.to_numeric(series, errors='coerce')
    if values.notna().sum() < .5 * series.notna().sum():
        parsed = pd.to_datetime(series, errors='coerce', format='mixed')
        values = pd.Series(parsed.to_numpy().astype('datetime64[ns]').astype(np.int64), index=series.index).where(parsed.notna())
    return values.astype(float)


def dates(series):
    present = series.dropna().astype(str)
    if not len(present) or present.str.fullmatch(r'[\d\s:/.T+-]{6,}').mean() < .95:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parsed = pd.to_datetime(series, errors='coerce', format='mixed')
    return parsed if parsed.notna().sum() >= .95 * len(present) else None


def schema(frame, preparation):
    """Feature columns, category vocabularies fixed on fit, and date-like columns."""
    data = preparation['data']
    drop = set(data['target_columns']) | set(data.get('id_columns') or [])
    columns, vocab, stamps = [], {}, []
    for column in frame.columns:
        series = frame[column]
        if column in drop:
            continue
        if not (pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series)):
            if pd.api.types.is_datetime64_any_dtype(series) or dates(series) is not None:
                stamps.append(column)  # calendar parts instead of thousands of string categories
                continue
            present = series.dropna().astype(str)
            if present.nunique() > max(1000, .5 * len(present)):
                continue  # free text or an undeclared identifier
            vocab[column] = sorted(present.unique())
        columns.append(column)
    return columns, vocab, stamps


def encode(frame, columns, vocab, stamps):
    result = {}
    for column in columns:
        values = frame[column].reset_index(drop=True) if column in frame else pd.Series(np.nan, index=range(len(frame)))
        if column in vocab:
            result[column] = pd.Categorical(values.astype(str).where(values.notna()), categories=vocab[column])
        else:
            result[column] = pd.to_numeric(values, errors='coerce').astype(float)
    for column in stamps:
        moment = pd.to_datetime(frame[column].reset_index(drop=True), errors='coerce', format='mixed')
        for part in ('year', 'month', 'day', 'dayofweek', 'hour'):
            result[column + '_' + part] = getattr(moment.dt, part).astype(float)
        result[column + '_epoch_days'] = (moment - pd.Timestamp(0)).dt.total_seconds() / 86400.
    return pd.DataFrame(result)


def inner_split(fit, labels, temporal):
    """Latest 15% of the temporal column when the test period is later; otherwise a (stratified) 15% holdout."""
    rows = np.arange(len(fit))
    if temporal is not None and temporal in fit:
        values = ordinal(fit[temporal]).to_numpy()
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


class Features:
    """Encode fit and later batches identically, in the format a library expects; no model inside.

    kind='category': pandas categories with the fit vocabulary (LightGBM, XGBoost enable_categorical,
    HistGradientBoosting); 'string': categories as str with NaN -> 'nan' (CatBoost; pass cat_features=
    features.categorical); 'codes': integer codes, -1 for missing/unseen (sklearn, torch). Dates become calendar
    parts; targets, ids and free text are dropped. transform(frame, fit, training) adds features first.
    features.train is the encoded fit; features(frame) encodes a prediction batch.
    """
    def __init__(self, fit, preparation, *, transform=None, kind='category'):
        if kind not in ('category', 'string', 'codes'):
            raise ValueError("kind must be 'category', 'string' or 'codes'")
        self.fit, self.transform, self.kind = fit, transform, kind
        data = transform(fit.copy(), fit, True) if transform else fit
        kept = len(set(fit.columns) & set(data.columns)) / max(1, len(fit.columns))
        if kept < .5 or len(data) != len(fit):  # e.g. `frame = target_stats(...)` instead of frame.join(...)
            raise ValueError(f'transform must return the input frame (same rows) with added columns; it kept {kept:.0%} '
                             f'of the original columns and {len(data)} of {len(fit)} rows. Join new features: frame.join(stats).')
        self.columns, self.vocab, self.stamps = schema(data, preparation)
        self.categorical = list(self.vocab)
        self.train = self._encode(data)

    def __call__(self, frame):
        return self._encode(self.transform(frame.copy(), self.fit, False) if self.transform else frame)

    def _encode(self, data):
        X = encode(data, self.columns, self.vocab, self.stamps)
        for column in self.vocab:
            X[column] = X[column].astype(str).where(X[column].notna(), 'nan') if self.kind == 'string' else \
                X[column].cat.codes if self.kind == 'codes' else X[column]
        return X


def labels(fit, preparation):
    """(target name, label array, classes or None); class labels become 0..K-1 codes, classes maps them back."""
    target = preparation['data']['target_columns'][0]
    if (task_kind(preparation) or {}).get('task') == 'classification':
        classes, codes = np.unique(fit[target].to_numpy(), return_inverse=True)
        return target, codes, classes
    return target, fit[target].to_numpy(dtype=float), None


def split(fit, preparation):
    """Inner (train_rows, valid_rows) positions of fit: latest periods of the temporal column, else stratified."""
    _, y, classes = labels(fit, preparation)
    return inner_split(fit, y if classes is not None else None, temporal_column(preparation))


def inner_score(values, fit, rows, preparation):
    """Task metric of predictions for fit.iloc[rows], oriented so that higher is better (None without a scorer)."""
    scorer = preparation.get('scorer')
    if scorer is None:
        return None
    target = preparation['data']['target_columns'][0]
    score = float(scorer(fit.iloc[rows].reset_index(drop=True), pd.DataFrame({target: np.asarray(values)}), preparation))
    return -score if preparation['metric'].get('direction') == 'minimize' else score


def class_scale(probability, classes, fit, rows, preparation):
    """Per-class probability multipliers maximizing the task metric on fit.iloc[rows]; use argmax(probability * scale)."""
    scale = np.ones(len(classes))
    best = inner_score(classes[np.argmax(probability, axis=1)], fit, rows, preparation)
    if best is None or len(classes) > 50:
        return scale
    for _ in range(2):
        for k in range(len(classes)):
            for factor in (.5, .7, .85, 1.2, 1.5, 2., 3.):
                trial = scale.copy()
                trial[k] *= factor
                score = inner_score(classes[np.argmax(probability * trial, axis=1)], fit, rows, preparation)
                if score > best + 1e-9:
                    best, scale = score, trial
    return scale


def output(frame, preparation, values):
    """Prediction DataFrame: identifier columns of frame plus the target column."""
    ids = [c for c in preparation['data'].get('id_columns') or [] if c in frame]
    target = preparation['data']['target_columns'][0]
    return pd.DataFrame({**{c: frame[c].to_numpy() for c in ids}, target: np.asarray(values)})


def target_stats(reference, frame, keys, target, *, training=False, time=None, folds=5, prefix=None):
    """Target statistics of rows sharing the same `keys` values (a repeated entity), without target leakage.

    reference: labeled rows (usually fit). frame: rows to describe, same index order returned.
    training=True means frame IS reference: each row sees only earlier periods of `time`, or other folds.
    Columns: <prefix>_mean (one per class for labels), <prefix>_count and <prefix>_last (latest earlier value).
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    prefix = prefix or 'ts_' + '_'.join(map(str, keys))[:40]
    labels = reference[target]
    numeric = pd.api.types.is_numeric_dtype(labels) and labels.nunique() > 20
    if numeric:
        values = pd.DataFrame({prefix + '_mean': labels.astype(float).to_numpy()})
    else:
        top = labels.astype(str).value_counts().index[:20]
        values = pd.DataFrame({f'{prefix}_mean_{c}': (labels.astype(str) == c).astype(float).to_numpy() for c in top})
    key = lambda d: d[keys].astype(str).agg('|'.join, axis=1).to_numpy()
    reference_keys = key(reference)
    moments = ordinal(reference[time]).to_numpy() if time is not None else None

    def describe(source, target_frame):
        grouped = values.iloc[source].assign(_key=reference_keys[source]).groupby('_key')
        wanted = key(target_frame)
        out = grouped.mean().reindex(wanted)
        out[prefix + '_count'] = grouped.size().reindex(wanted).fillna(0).to_numpy()
        if numeric and moments is not None:
            latest = values.iloc[source].assign(_key=reference_keys[source], _t=moments[source]).sort_values('_t', kind='stable')
            out[prefix + '_last'] = latest.groupby('_key')[prefix + '_mean'].last().reindex(wanted).to_numpy()
        return out.reset_index(drop=True)

    if not training:
        return describe(np.arange(len(reference)), frame).set_axis(frame.index)
    parts = []
    if moments is not None:
        for moment in np.unique(moments[~np.isnan(moments)]):
            rows = np.flatnonzero(moments == moment)
            parts.append((rows, describe(np.flatnonzero(moments < moment), reference.iloc[rows])))
        rows = np.flatnonzero(np.isnan(moments))
        if len(rows):
            parts.append((rows, describe(np.flatnonzero(~np.isnan(moments)), reference.iloc[rows])))
    else:
        order = np.random.RandomState(0).permutation(len(reference))
        for fold in range(folds):
            rows = np.sort(order[fold::folds])
            parts.append((rows, describe(np.setdiff1d(np.arange(len(reference)), rows), reference.iloc[rows])))
    result = pd.concat([part.set_axis(rows) for rows, part in parts]).sort_index()
    return result.set_axis(frame.index)


def gbdt(fit, preparation, *, transform=None, params=None, candidates=None, tune_classes=True):
    """Shift-aware LightGBM predictor for one-target row tasks.

    transform(frame, fit, training) -> frame adds/changes features; fit is the labeled table, training=True only when
    frame is fit itself (use target_stats(..., training=training) for target-derived features). params override
    LightGBM parameters. candidates: list of (name, extra_params, option) where option is None, 'log' (regression
    log1p target) or 'balanced'/'sqrt_balanced' (class weights); default tries L2/Huber/log or plain/balanced/sqrt.
    Selection, early stopping and per-class probability multipliers use preparation['scorer'] on an inner split
    that follows the temporal column; the final model is retrained on all fit rows.
    """
    import lightgbm as lgb
    kind = task_kind(preparation)
    if kind is None:
        raise ValueError('gbdt supports one target with row-wise predictions; write a task-specific model instead')
    task, proba = kind['task'], kind['proba']
    target = preparation['data']['target_columns'][0]
    scorer = preparation.get('scorer')
    fit = fit[fit[target].notna()].reset_index(drop=True)
    data = transform(fit.copy(), fit, True) if transform else fit
    columns, vocab, stamps = schema(data, preparation)
    X = encode(data, columns, vocab, stamps)
    settings = {**dict(n_estimators=5000, learning_rate=.05, num_leaves=31, min_child_samples=20, subsample=.8,
                       subsample_freq=1, colsample_bytree=.8, reg_lambda=1., n_jobs=THREADS, verbose=-1, random_state=0),
                **(params or {})}  # node parameters override the defaults
    if task == 'classification':
        classes, y = np.unique(fit[target].to_numpy(), return_inverse=True)
        balanced = len(y) / (len(classes) * np.bincount(y, minlength=len(classes)))
        weights = {None: None, 'balanced': balanced[y], 'sqrt_balanced': np.sqrt(balanced)[y]}
        defaults = ([('plain', {}, None), ('balanced', {}, 'balanced'), ('sqrt_balanced', {}, 'sqrt_balanced')]
                    if len(classes) <= 100 else [('sqrt_balanced', {}, 'sqrt_balanced')])
        train, valid = inner_split(fit, y, temporal_column(preparation))
    else:
        y = fit[target].to_numpy(dtype=float)
        spread = float(np.median(np.abs(y - np.median(y)))) or float(np.std(y)) or 1.
        defaults = ([('l2', {}, None), ('huber', {'objective': 'huber', 'alpha': 1.5 * spread}, None)]
                    + ([('log_target', {}, 'log')] if y.min() >= 0 else []))
        # LightGBM's own Huber alpha (0.9) is tiny on most target scales; derive it from the data unless given.
        candidates = [(name, {**extra, 'alpha': 1.5 * spread} if extra.get('objective') == 'huber' and 'alpha' not in extra
                       else extra, option) for name, extra, option in candidates or []]
        train, valid = inner_split(fit, None, temporal_column(preparation))
    # Node candidates add to the defaults: a plan never loses the measured baseline choices.
    seen = {(json.dumps(extra, sort_keys=True), option) for _, extra, option in candidates or []}
    candidates = list(candidates or []) + [c for c in defaults if (json.dumps(c[1], sort_keys=True), c[2]) not in seen]
    if len(train) > 400000:  # candidate selection on a sample keeps large tables inside the process limit
        train = np.sort(np.random.RandomState(1).choice(train, 400000, replace=False))
    reference = fit.iloc[valid].reset_index(drop=True)

    def train_one(extra, option, rows, eval_rows, trees=None):
        model_params = {**settings, **extra, **({'n_estimators': trees} if trees else {})}
        stop = [lgb.early_stopping(100, verbose=False)]
        if task == 'classification':
            weight = weights.get(option)  # unknown options train unweighted
            model = lgb.LGBMClassifier(**model_params)
            if eval_rows is None:
                return model.fit(X.iloc[rows], y[rows], sample_weight=None if weight is None else weight[rows])
            return model.fit(X.iloc[rows], y[rows], sample_weight=None if weight is None else weight[rows],
                             eval_set=[(X.iloc[eval_rows], y[eval_rows])], callbacks=stop,
                             eval_sample_weight=None if weight is None else [weight[eval_rows]])
        labels = np.log1p(y) if option == 'log' else y
        model = lgb.LGBMRegressor(**model_params)
        if eval_rows is None:
            return model.fit(X.iloc[rows], labels[rows])
        return model.fit(X.iloc[rows], labels[rows], eval_set=[(X.iloc[eval_rows], labels[eval_rows])], callbacks=stop)

    def probabilities(model, features):  # a class absent from the inner train rows keeps its column
        result = np.zeros((len(features), len(classes)))
        result[:, model.classes_] = model.predict_proba(features)
        return result

    def output(model, option, features, scale=None):
        if task == 'classification':
            probability = probabilities(model, features)
            if proba:
                return probability[:, 1]
            return classes[np.argmax(probability * (scale if scale is not None else 1.), axis=1)]
        values = model.predict(features)
        return np.expm1(values) if option == 'log' else values

    def measure(values):
        if scorer is None:
            return None
        score = float(scorer(reference, pd.DataFrame({target: values}), preparation))
        return score if preparation['metric'].get('direction') != 'minimize' else -score

    results = []
    for name, extra, option in candidates:
        model = train_one(extra, option, train, valid)
        trees = max(50, int((model.best_iteration_ or settings['n_estimators']) * 1.15))
        value = measure(output(model, option, X.iloc[valid]))
        results.append((value if value is not None else -len(results), name, extra, option, trees, model))
        if scorer is None:
            break
    value, name, extra, option, trees, model = max(results, key=lambda item: item[0])
    scale = None
    if task == 'classification' and not proba and tune_classes and scorer is not None and len(classes) <= 50:
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
    print(json.dumps({'selected': name, 'trees': trees, 'inner_scores': {r[1]: r[0] for r in results},
                      'class_scale': None if scale is None else [round(float(s), 3) for s in scale]}))
    ids = [c for c in preparation['data'].get('id_columns') or [] if c in fit]

    def predict(features):  # identifiers too: some generated writers read them from the predictions
        frame = transform(features.copy(), fit, False) if transform else features
        values = {c: features[c].to_numpy() for c in ids if c in features}
        return pd.DataFrame({**values, target: output(final, option, encode(frame, columns, vocab, stamps), scale)})
    if task == 'classification':  # probabilities and class order let other code blend this model
        predict.classes = classes
        predict.proba = lambda features: probabilities(final, encode(
            transform(features.copy(), fit, False) if transform else features, columns, vocab, stamps))
    return predict


def automl(fit, preparation, engine, time_limit):
    """AutoGluon ('autogluon'), LightAutoML ('lightautoml') or FLAML ('flaml') under time_limit seconds."""
    kind = task_kind(preparation)
    if kind is None:
        raise ValueError('automl supports one target with row-wise predictions')
    task, proba = kind['task'], kind['proba']
    target = preparation['data']['target_columns'][0]
    ids = [c for c in preparation['data'].get('id_columns') or [] if c in fit]
    metric = metric_names(preparation).get(engine)
    fit = fit[fit[target].notna()].reset_index(drop=True)
    drop = set(preparation['data']['target_columns']) | set(ids)
    features = lambda frame, vocab=None: frame[[c for c in frame.columns if c not in drop]].reset_index(drop=True).copy().assign(
        **{c: pd.Categorical(frame[c].reset_index(drop=True).astype(str).where(frame[c].reset_index(drop=True).notna()),
                             categories=v) for c, v in (vocab or {}).items()})
    X, y = features(fit), fit[target].reset_index(drop=True)
    limit = time_limit if len(fit) >= 2000 else 8  # smoke checks use a few hundred rows
    classes = np.unique(y.to_numpy()) if task == 'classification' else None
    workspace = tempfile.TemporaryDirectory()  # kept alive by the returned predictor
    if engine == 'autogluon':
        from autogluon.tabular import TabularPredictor
        problem = 'regression' if task == 'regression' else 'binary' if len(classes) == 2 else 'multiclass'
        model = TabularPredictor(label=target, problem_type=problem, eval_metric=metric, path=workspace.name, verbosity=0)
        model.fit(X.assign(**{target: y}), time_limit=limit, presets='best_quality' if len(fit) >= 2000 else 'medium_quality',
                  num_cpus=THREADS, num_gpus=0, dynamic_stacking=False)
        raw = lambda frame: model.predict_proba(features(frame))[classes[1]].to_numpy() if proba else model.predict(features(frame)).to_numpy()
    elif engine == 'flaml':
        from flaml import AutoML
        vocab = {c: sorted(X[c].dropna().astype(str).unique()) for c in X.columns
                 if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))}
        model = AutoML()
        model.fit(X_train=features(fit, vocab), y_train=y, task=task, metric=metric or 'auto', time_budget=limit,
                  n_jobs=THREADS, verbose=0, seed=0, log_file_name='')
        raw = lambda frame: model.predict_proba(features(frame, vocab))[:, 1] if proba else np.asarray(model.predict(features(frame, vocab)))
    elif engine == 'lightautoml':
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task
        problem = 'reg' if task == 'regression' else 'binary' if len(classes) == 2 else 'multiclass'
        if task == 'classification':  # stratified 5-fold CV needs at least five rows per class
            counts = y.value_counts()
            fit = pd.concat([fit] + [fit[y == label].sample(5 - count, replace=True, random_state=0)
                                     for label, count in counts[counts < 5].items()], ignore_index=True)
        # Boosting only: LightAutoML's linear model indexes past its embedding table on unseen categories.
        model = TabularAutoML(task=Task(problem), timeout=limit, cpu_limit=THREADS,
                              general_params={'use_algos': [['lgb', 'lgb_tuned', 'cb_tuned']]},
                              reader_params={'n_jobs': THREADS, 'cv': 5, 'random_state': 0})
        model.fit_predict(fit, roles={'target': target, **({'drop': ids} if ids else {})}, verbose=0)
        mapping = getattr(model.reader, 'class_mapping', None)
        order = classes if mapping is None else np.array(sorted(mapping, key=mapping.get), dtype=object)

        def raw(frame):
            values = model.predict(frame.reset_index(drop=True)).data
            if task == 'regression':
                return values[:, 0]
            if len(order) == 2:
                return values[:, 0] if proba else order[(values[:, 0] > .5).astype(int)]
            return order[np.argmax(values, axis=1)]
    else:
        raise ValueError(f'Unknown AutoML engine: {engine}')

    def predict(frame):
        values = raw(frame)
        if task == 'classification' and not proba:
            values = pd.Series(values).astype(y.dtype, errors='ignore').to_numpy()
        return pd.DataFrame({**{c: frame[c].to_numpy() for c in ids if c in frame}, target: values})
    predict.workspace = workspace
    return predict
