"""
Сборка эстиматора по (model_id, loss, transform) из реестра.
Используется в probe; Макс может импортировать это же, чтобы его шаблоны и наш выбор
не разъезжались (одна точка правды по нативным именам лоссов).
"""
from __future__ import annotations
import importlib
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .registry import MODELS

N_JOBS = os.cpu_count() or 1

# Нейтральные параметры для probe: цель — сравнить пары (модель, лосс), а не выжать максимум.
# Тюнинг — зона этапа Макса.
PROBE_PARAMS = {
    "lightgbm": dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=N_JOBS),
    "xgboost": dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, tree_method="hist", enable_categorical=True, n_jobs=N_JOBS),
    "catboost": dict(iterations=400, learning_rate=0.08, depth=6, verbose=0, thread_count=N_JOBS,
                     allow_writing_files=False),
    "hist_gb": dict(max_iter=400, learning_rate=0.05, categorical_features="from_dtype"),
    "ridge": dict(alpha=1.0),
    "elasticnet": dict(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    "huber_linear": dict(max_iter=500),
    "sgd_linear": dict(average=True, max_iter=2000, tol=1e-4, random_state=0),
    "mlp": dict(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200, random_state=0),
    "knn": dict(n_neighbors=15, weights="distance", n_jobs=N_JOBS),
    "random_forest": dict(n_estimators=200, min_samples_leaf=3, n_jobs=N_JOBS, random_state=0),
    "extra_trees": dict(n_estimators=300, min_samples_leaf=2, n_jobs=N_JOBS, random_state=0),
    "dummy_mean": dict(),
}


def _import(path: str):
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def loss_kwargs(model_id: str, loss: str, y_scale: float = 1.0, alpha: float = 0.5,
                tweedie_power: float = 1.5) -> dict:
    """Нативные параметры лосса. y_scale — робастная сигма таргета (в той шкале, где учимся):
    дельта Хьюбера масштаб-зависима, её нельзя оставлять дефолтной."""
    spec = dict(MODELS[model_id]["losses"][loss])
    delta = 1.35 * max(y_scale, 1e-9)
    out = {}
    for k, v in spec.items():
        if isinstance(v, str) and "{" in v:
            v = v.format(delta=round(delta, 6), p=tweedie_power, alpha=alpha)
        out[k] = v
    if model_id == "lightgbm":
        if loss == "huber":
            out["alpha"] = delta
        elif loss == "quantile":
            out["alpha"] = alpha
        elif loss == "tweedie":
            out["tweedie_variance_power"] = tweedie_power
    if model_id == "xgboost":
        if loss == "pseudo_huber":
            out["huber_slope"] = delta
        elif loss == "quantile":
            out["quantile_alpha"] = alpha
        elif loss == "tweedie":
            out["tweedie_variance_power"] = tweedie_power
    if model_id == "hist_gb" and loss == "quantile":
        out["quantile"] = alpha
    if model_id == "huber_linear":
        out = {"epsilon": 1.35}  # у HuberRegressor эпсилон уже в единицах масштаба
    return out


def make_preprocessor(model_id: str, num_cols: list[str], cat_cols: list[str]):
    m = MODELS[model_id]
    if m["needs_scaling"]:
        return ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=20, max_categories=30))]), cat_cols),
        ], remainder="drop", sparse_threshold=0.0 if model_id in ("mlp", "knn") else 0.3)
    if model_id in ("random_forest", "extra_trees"):
        # sklearn-деревья не умеют строки: ordinal-кодирование
        from sklearn.preprocessing import OrdinalEncoder
        return ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), cat_cols),
        ])
    return None  # бустинги получают DataFrame с category-dtype


def build(model_id: str, loss: str, transform: str = "none", num_cols=None, cat_cols=None,
          y_scale: float = 1.0, params: dict | None = None, **loss_kw):
    m = MODELS[model_id]
    est_cls = _import(m["estimator"])
    kw = dict(PROBE_PARAMS.get(model_id, {}))
    kw.update(loss_kwargs(model_id, loss, y_scale=y_scale, **loss_kw))
    if model_id == "catboost" and cat_cols:
        kw["cat_features"] = list(cat_cols)
    if params:
        kw.update(params)
    est = est_cls(**kw)
    pre = make_preprocessor(model_id, num_cols or [], cat_cols or [])
    model = Pipeline([("pre", pre), ("est", est)]) if pre is not None else est
    if transform == "log1p":
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1,
                                           check_inverse=False)
    return model


def prepare_frame(X: pd.DataFrame, cat_cols: list[str], for_model: str) -> pd.DataFrame:
    """Приведение типов под нативную работу с категориями."""
    X = X.copy()
    if for_model == "catboost":
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna("__nan__")
    elif for_model in ("lightgbm", "xgboost", "hist_gb"):
        for c in cat_cols:
            X[c] = X[c].astype("category")
    return X
