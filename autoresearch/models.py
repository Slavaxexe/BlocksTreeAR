"""
Модели исследования: единый fit/predict с ранней остановкой, лоссом из реестра и белым списком параметров.

Белый список с границами — то, что Qwen может менять. Всё вне списка отклоняется валидатором с понятной
ошибкой (её Qwen получает в следующем вызове). n_estimators не задаётся: его определяет ранняя остановка.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model_selection.builders import loss_kwargs
from model_selection.registry import MODELS as REGISTRY
from model_selection.registry import supports

# параметр: (тип, min, max)
PARAM_SPACE = {
    "lightgbm": {"learning_rate": (float, 0.005, 0.3), "num_leaves": (int, 7, 1023), "max_depth": (int, -1, 20),
                 "min_child_samples": (int, 2, 1000), "min_child_weight": (float, 0.0, 100.0),
                 "subsample": (float, 0.3, 1.0), "subsample_freq": (int, 0, 10), "colsample_bytree": (float, 0.1, 1.0),
                 "reg_alpha": (float, 0.0, 100.0), "reg_lambda": (float, 0.0, 100.0), "min_split_gain": (float, 0.0, 10.0),
                 "max_bin": (int, 31, 1023), "cat_smooth": (float, 0.0, 200.0), "path_smooth": (float, 0.0, 100.0),
                 "extra_trees": (bool, 0, 1)},
    "catboost": {"learning_rate": (float, 0.005, 0.3), "depth": (int, 3, 10), "l2_leaf_reg": (float, 0.0, 100.0),
                 "random_strength": (float, 0.0, 20.0), "bagging_temperature": (float, 0.0, 10.0),
                 "border_count": (int, 32, 255), "min_data_in_leaf": (int, 1, 500), "one_hot_max_size": (int, 2, 255)},
    "xgboost": {"learning_rate": (float, 0.005, 0.3), "max_depth": (int, 2, 14), "min_child_weight": (float, 0.0, 200.0),
                "subsample": (float, 0.3, 1.0), "colsample_bytree": (float, 0.1, 1.0), "reg_alpha": (float, 0.0, 100.0),
                "reg_lambda": (float, 0.0, 100.0), "gamma": (float, 0.0, 20.0), "max_bin": (int, 32, 1024),
                "max_cat_to_onehot": (int, 1, 64)},
    "extra_trees": {"n_estimators": (int, 50, 800), "max_features": (float, 0.1, 1.0), "min_samples_leaf": (int, 1, 100),
                    "max_depth": (int, 3, 60)},
    "random_forest": {"n_estimators": (int, 50, 800), "max_features": (float, 0.1, 1.0), "min_samples_leaf": (int, 1, 100),
                      "max_depth": (int, 3, 60)},
    "ridge": {"alpha": (float, 1e-4, 1e4)},
    "elasticnet": {"alpha": (float, 1e-5, 100.0), "l1_ratio": (float, 0.0, 1.0)},
}
DEFAULTS = {
    "lightgbm": {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 30, "subsample": 0.8,
                 "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0},
    "catboost": {"learning_rate": 0.06, "depth": 8, "l2_leaf_reg": 3.0},
    "xgboost": {"learning_rate": 0.05, "max_depth": 8, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.7},
    "extra_trees": {"n_estimators": 300, "max_features": 0.5, "min_samples_leaf": 2},
    "random_forest": {"n_estimators": 300, "max_features": 0.5, "min_samples_leaf": 2},
    "ridge": {"alpha": 1.0},
    "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5},
}
MODEL_IDS = list(PARAM_SPACE)
MAX_ROUNDS = 20000
EARLY_STOP = 200


def validate_params(model_id: str, params: dict) -> list[str]:
    errs = []
    space = PARAM_SPACE.get(model_id)
    if space is None:
        return [f"model_id '{model_id}' не из {MODEL_IDS}"]
    for k, v in (params or {}).items():
        if k not in space:
            errs.append(f"{model_id}: параметр '{k}' не разрешён; разрешены {sorted(space)}")
            continue
        typ, lo, hi = space[k]
        try:
            vv = typ(v)
        except (TypeError, ValueError):
            errs.append(f"{model_id}.{k}={v!r}: ожидается {typ.__name__}")
            continue
        if typ is not bool and not (lo <= vv <= hi):
            errs.append(f"{model_id}.{k}={v} вне [{lo}, {hi}]")
    return errs


def _encode_for_sklearn(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes.astype("int32")
    return X.astype(float).fillna(-999.0)


def fit_predict(model_id: str, loss: str, transform: str, params: dict, X_fit: pd.DataFrame, y_fit: np.ndarray,
                X_es: pd.DataFrame | None, y_es: np.ndarray | None, X_pred: list[pd.DataFrame], n_jobs: int,
                seed: int, sample_weight: np.ndarray | None = None, n_rounds: int | None = None):
    """Возвращает (предсказания для каждого X_pred, число итераций, важности признаков)."""
    yt = np.log1p(y_fit) if transform == "log1p" else y_fit
    ye = None if y_es is None else (np.log1p(y_es) if transform == "log1p" else y_es)
    scale = float(np.median(np.abs(yt - np.median(yt))) * 1.4826) or float(np.std(yt)) or 1.0
    lk = loss_kwargs(model_id, loss, y_scale=scale)
    p = {**DEFAULTS.get(model_id, {}), **(params or {})}
    cats = [c for c in X_fit.columns if str(X_fit[c].dtype) == "category"]
    use_es = X_es is not None and n_rounds is None
    best_iter, imp = None, None

    if model_id == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=n_rounds or MAX_ROUNDS, random_state=seed, n_jobs=n_jobs, verbose=-1, **lk, **p)
        kw = {}
        if use_es:
            kw = dict(eval_set=[(X_es, ye)], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        m.fit(X_fit, yt, sample_weight=sample_weight, **kw)
        best_iter = int(m.best_iteration_ or m.n_estimators) if use_es else int(m.n_estimators)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=X_fit.columns)
        preds = [m.predict(X, num_iteration=best_iter) for X in X_pred]
    elif model_id == "catboost":
        from catboost import CatBoostRegressor, Pool
        def pool(X, y=None, w=None):
            X = X.copy()
            for c in cats:
                X[c] = X[c].astype(str)
            return Pool(X, y, cat_features=cats, weight=w)
        m = CatBoostRegressor(iterations=n_rounds or MAX_ROUNDS, random_seed=seed, thread_count=n_jobs, verbose=0,
                              allow_writing_files=False, **lk, **p)
        if use_es:
            m.fit(pool(X_fit, yt, sample_weight), eval_set=pool(X_es, ye), early_stopping_rounds=EARLY_STOP, use_best_model=True)
            best_iter = int(m.get_best_iteration() or 0) + 1
        else:
            m.fit(pool(X_fit, yt, sample_weight))
            best_iter = int(m.tree_count_)
        imp = pd.Series(m.get_feature_importance(), index=X_fit.columns)
        preds = [m.predict(pool(X)) for X in X_pred]
    elif model_id == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=n_rounds or MAX_ROUNDS, random_state=seed, n_jobs=n_jobs, tree_method="hist",
                             enable_categorical=True, early_stopping_rounds=EARLY_STOP if use_es else None, **lk, **p)
        m.fit(X_fit, yt, sample_weight=sample_weight, eval_set=[(X_es, ye)] if use_es else None, verbose=False)
        best_iter = int(m.best_iteration + 1) if use_es else int(m.n_estimators)
        imp = pd.Series(m.feature_importances_, index=X_fit.columns)
        preds = [m.predict(X, iteration_range=(0, best_iter)) for X in X_pred]
    elif model_id in ("extra_trees", "random_forest"):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        cls = ExtraTreesRegressor if model_id == "extra_trees" else RandomForestRegressor
        crit = {"l2": "squared_error", "poisson": "poisson"}.get(loss, "squared_error")
        m = cls(criterion=crit, random_state=seed, n_jobs=n_jobs, **p)
        m.fit(_encode_for_sklearn(X_fit), yt, sample_weight=sample_weight)
        imp = pd.Series(m.feature_importances_, index=X_fit.columns)
        preds = [m.predict(_encode_for_sklearn(X)) for X in X_pred]
    elif model_id in ("ridge", "elasticnet"):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        nums = [c for c in X_fit.columns if c not in cats]
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), nums),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, max_categories=40), cats)])
        est = Ridge(**p) if model_id == "ridge" else ElasticNet(max_iter=5000, **p)
        m = Pipeline([("pre", pre), ("est", est)])
        conv = lambda X: X.assign(**{c: X[c].astype(str) for c in cats})
        m.fit(conv(X_fit), yt, est__sample_weight=sample_weight)
        preds = [m.predict(conv(X)) for X in X_pred]
    else:
        raise ValueError(f"неизвестная модель {model_id}")

    if transform == "log1p":
        preds = [np.expm1(np.clip(pr, -20, 30)) for pr in preds]
    return preds, best_iter, imp


def check_model_spec(model_id: str, loss: str, transform: str) -> list[str]:
    errs = []
    if model_id not in PARAM_SPACE:
        return [f"model_id '{model_id}' не из {MODEL_IDS}"]
    if model_id in ("extra_trees", "random_forest"):
        if loss not in ("l2", "poisson"):
            errs.append(f"{model_id}: лосс только l2|poisson")
    elif model_id in ("ridge", "elasticnet"):
        if loss != "l2":
            errs.append(f"{model_id}: лосс только l2")
    elif not supports(model_id, loss):
        errs.append(f"{model_id}: лосс '{loss}' не поддержан; есть {list(REGISTRY[model_id]['losses'])}")
    if transform not in ("none", "log1p"):
        errs.append("target_transform: none|log1p")
    return errs
