"""Classifier templates, probability alignment, and primary task metrics."""
import inspect
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score


PARAM_SPACE = {
    'lightgbm': {'learning_rate': (.005, .3), 'num_leaves': (7, 255), 'max_depth': (-1, 16),
        'min_child_samples': (2, 500), 'subsample': (.3, 1.), 'subsample_freq': (0, 10),
        'colsample_bytree': (.2, 1.), 'reg_alpha': (0., 100.), 'reg_lambda': (0., 100.),
        'class_weight': ('balanced', None)},
    'catboost': {'learning_rate': (.005, .3), 'depth': (3, 10), 'l2_leaf_reg': (0., 100.),
        'random_strength': (0., 20.), 'auto_class_weights': ('Balanced', 'SqrtBalanced', None)},
    'xgboost': {'learning_rate': (.005, .3), 'max_depth': (2, 12), 'min_child_weight': (0., 100.),
        'subsample': (.3, 1.), 'colsample_bytree': (.2, 1.), 'reg_alpha': (0., 100.),
        'reg_lambda': (0., 100.), 'class_weight': ('balanced', None)},
    'extra_trees': {'n_estimators': (40, 500), 'max_features': (.1, 1.), 'min_samples_leaf': (1, 100),
        'max_depth': (3, 60), 'class_weight': ('balanced', None)},
    'random_forest': {'n_estimators': (40, 500), 'max_features': (.1, 1.), 'min_samples_leaf': (1, 100),
        'max_depth': (3, 60), 'class_weight': ('balanced', None)},
    'logistic_regression': {'C': (.0001, 10000.), 'max_iter': (100, 2000), 'class_weight': ('balanced', None)},
    'sgd': {'alpha': (.000001, 1.), 'l1_ratio': (0., 1.), 'class_weight': ('balanced', None)},
}
DEFAULTS = {
    'lightgbm': {'learning_rate': .05, 'num_leaves': 31, 'min_child_samples': 30,
                 'subsample': .8, 'subsample_freq': 1, 'colsample_bytree': .8, 'reg_lambda': 1.},
    'catboost': {'learning_rate': .05, 'depth': 6, 'l2_leaf_reg': 3.},
    'xgboost': {'learning_rate': .05, 'max_depth': 6, 'min_child_weight': 3., 'subsample': .8, 'colsample_bytree': .8},
    'extra_trees': {'n_estimators': 150, 'min_samples_leaf': 2, 'max_features': .8},
    'random_forest': {'n_estimators': 150, 'min_samples_leaf': 2, 'max_features': .8},
    'logistic_regression': {'C': 1., 'max_iter': 500}, 'sgd': {'alpha': .0001, 'l1_ratio': .15},
}
MAX_ROUNDS = 2000


def positive_index(classes, positive=None):
    if positive is None:
        return len(classes)-1
    matches = [i for i, label in enumerate(classes) if str(label) == str(positive)]
    if not matches and all(isinstance(label, (int, float, np.number)) for label in classes):
        try:
            matches = [i for i, label in enumerate(classes) if float(label) == float(positive)]
        except ValueError:
            pass
    if len(matches) != 1:
        raise ValueError('Declared positive class must match exactly one observed training label')
    return matches[0]


def normalize(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < -1e-8):
        raise ValueError('Expected finite nonnegative probability matrix')
    values = np.maximum(values, 1e-12)
    return values/values.sum(axis=1, keepdims=True)


def labels_from_probabilities(probabilities, threshold=.5, positive=1):
    p = normalize(probabilities)
    if p.shape[1] == 2:
        return np.where(p[:, positive] >= threshold, positive, 1-positive)
    return p.argmax(axis=1)


def score(metric, y, probabilities, threshold=.5, positive=1):
    p = normalize(probabilities)
    y = np.asarray(y, dtype=int)
    if len(y) != len(p):
        raise ValueError('Labels and predictions must have the same length')
    predicted = labels_from_probabilities(p, threshold, positive)
    if metric == 'accuracy':
        value = accuracy_score(y, predicted)
    elif metric == 'balanced_accuracy':
        value = balanced_accuracy_score(y, predicted)
    elif metric in {'f1_macro', 'f1_weighted', 'f1_micro'}:
        value = f1_score(y, predicted, average=metric.split('_')[1], labels=np.arange(p.shape[1]), zero_division=0)
    elif metric == 'f1':
        if p.shape[1] != 2:
            raise ValueError('Binary F1 requires exactly two classes; use f1_macro for multiclass')
        value = f1_score(y, predicted, pos_label=positive, zero_division=0)
    elif metric == 'log_loss':
        value = log_loss(y, p, labels=np.arange(p.shape[1]))
    elif metric == 'roc_auc':
        if p.shape[1] == 2:
            value = roc_auc_score((y == positive).astype(int), p[:, positive])
        else:
            value = roc_auc_score(y, p, multi_class='ovr', average='macro', labels=np.arange(p.shape[1]))
    else:
        raise ValueError(f'Unsupported classification primary metric: {metric}')
    if not np.isfinite(value):
        raise ValueError('Primary classification score is not finite')
    return float(value)


def validate_spec(model_id, loss, params):
    if model_id not in PARAM_SPACE:
        return [f'Unknown classifier {model_id}; choose from {list(PARAM_SPACE)}']
    if loss not in {'cross_entropy', 'log_loss'}:
        return ['Classification loss must be cross_entropy/log_loss, never a regression loss']
    errors = []
    for name, value in params.items():
        if name not in PARAM_SPACE[model_id]:
            errors.append(f'Unsupported {model_id} parameter {name}')
            continue
        bounds = PARAM_SPACE[model_id][name]
        if isinstance(bounds[0], str):
            if value not in bounds:
                errors.append(f'{name} must be one of {bounds}')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            errors.append(f'{name} must be numeric within {bounds}')
    return errors


def _numeric(frame):
    result = frame.copy()
    for name in result:
        if isinstance(result[name].dtype, pd.CategoricalDtype):
            result[name] = result[name].cat.codes
    return result.astype(float).replace([np.inf, -np.inf], np.nan).fillna(-999.)


def fit_predict(model_id, loss, params, x, y, early_x, early_y, predict, n_classes,
                n_jobs=2, seed=42, n_rounds=None, sample_weight=None):
    errors = validate_spec(model_id, loss, params)
    if errors:
        raise ValueError('; '.join(errors))
    parameters = {**DEFAULTS[model_id], **params}
    observed = np.unique(y)
    if len(observed) == 1:
        probabilities = []
        for frame in predict:
            p = np.zeros((len(frame), n_classes));p[:, observed[0]] = 1.
            probabilities.append(normalize(p))
        return probabilities, 1, None
    # XGBoost requires locally contiguous classes when a temporal fold omits classes.
    local_y = np.searchsorted(observed, y)
    can_stop = (early_x is not None and len(early_y) and n_rounds is None
                and set(np.unique(early_y)).issubset(set(observed)))
    iterations = int(n_rounds or MAX_ROUNDS)
    importance = None
    if model_id == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=iterations, n_jobs=n_jobs, random_state=seed, verbosity=-1, **parameters)
        options = dict(eval_set=[(early_x, early_y)], callbacks=[lgb.early_stopping(60, verbose=False)]) if can_stop else {}
        model.fit(x, y, sample_weight=sample_weight, **options)
        best = int(model.best_iteration_ or model.n_estimators)
        importance = pd.Series(model.booster_.feature_importance('gain'), index=x.columns)
        probabilities = [model.predict_proba(frame) for frame in predict]
        classes = model.classes_
    elif model_id == 'catboost':
        from catboost import CatBoostClassifier, Pool
        categorical = [name for name in x if isinstance(x[name].dtype, pd.CategoricalDtype)]
        def pool(frame, labels=None, weights=None):
            frame = frame.copy()
            for name in categorical:frame[name] = frame[name].astype(str)
            return Pool(frame, labels, cat_features=categorical, weight=weights)
        if parameters.get('auto_class_weights') is None:parameters.pop('auto_class_weights', None)
        model = CatBoostClassifier(iterations=iterations, loss_function='Logloss' if len(observed) == 2 else 'MultiClass',
            thread_count=n_jobs, random_seed=seed, verbose=False, allow_writing_files=False, **parameters)
        options = dict(eval_set=pool(early_x, early_y), early_stopping_rounds=60, use_best_model=True) if can_stop else {}
        model.fit(pool(x, y, sample_weight), **options)
        best = int(model.tree_count_)
        importance = pd.Series(model.get_feature_importance(), index=x.columns)
        probabilities = [model.predict_proba(pool(frame)) for frame in predict]
        classes = model.classes_
    elif model_id == 'xgboost':
        from xgboost import XGBClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        balancing = parameters.pop('class_weight', None)
        weight = compute_sample_weight('balanced', y) if balancing == 'balanced' else np.ones(len(y))
        if sample_weight is not None:weight *= sample_weight
        model = XGBClassifier(n_estimators=iterations, n_jobs=n_jobs, random_state=seed, tree_method='hist',
            objective='binary:logistic' if len(observed) == 2 else 'multi:softprob',
            early_stopping_rounds=60 if can_stop else None, **parameters)
        options = dict(eval_set=[(_numeric(early_x), np.searchsorted(observed, early_y))]) if can_stop else {}
        model.fit(_numeric(x), local_y, sample_weight=weight, verbose=False, **options)
        best = int(model.best_iteration+1) if can_stop else iterations
        importance = pd.Series(model.feature_importances_, index=x.columns)
        probabilities = [model.predict_proba(_numeric(frame)) for frame in predict]
        classes = observed
    elif model_id in {'extra_trees', 'random_forest'}:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        factory = ExtraTreesClassifier if model_id == 'extra_trees' else RandomForestClassifier
        model = factory(n_jobs=n_jobs, random_state=seed, criterion='log_loss', **parameters)
        model.fit(_numeric(x), y, sample_weight=sample_weight)
        best = model.n_estimators
        importance = pd.Series(model.feature_importances_, index=x.columns)
        probabilities = [model.predict_proba(_numeric(frame)) for frame in predict]
        classes = model.classes_
    else:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        cats = [name for name in x if isinstance(x[name].dtype, pd.CategoricalDtype)]
        nums = [name for name in x if name not in cats]
        preprocess = ColumnTransformer([
            ('num', Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), nums),
            ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=100), cats)])
        estimator = (LogisticRegression(solver='lbfgs', n_jobs=n_jobs, random_state=seed, **parameters)
                     if model_id == 'logistic_regression' else SGDClassifier(loss='log_loss', penalty='elasticnet', random_state=seed, **parameters))
        model = Pipeline([('preprocess', preprocess), ('model', estimator)])
        model.fit(x, y, model__sample_weight=sample_weight)
        best = int(np.max(model.named_steps['model'].n_iter_))
        probabilities = [model.predict_proba(frame) for frame in predict]
        classes = model.classes_
    aligned = []
    for values in probabilities:
        p = np.zeros((len(values), n_classes))
        p[:, np.asarray(classes, dtype=int)] = values
        aligned.append(normalize(p))
    return aligned, best, importance
