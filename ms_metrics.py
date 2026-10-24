"""Реализации метрик на исходной шкале таргета. Используются в probe и могут переиспользоваться Максом."""
from __future__ import annotations
import numpy as np


def _rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def _mse(y, p): return float(np.mean((y - p) ** 2))
def _mae(y, p): return float(np.mean(np.abs(y - p)))
def _medae(y, p): return float(np.median(np.abs(y - p)))
def _r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def _msle(y, p):
    p = np.clip(p, -1 + 1e-9, None)
    return float(np.mean((np.log1p(y) - np.log1p(p)) ** 2))


def _rmsle(y, p): return float(np.sqrt(_msle(y, p)))


def _mape(y, p):
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])))


def _smape(y, p):
    d = np.abs(y) + np.abs(p)
    m = d != 0
    return float(np.mean(2 * np.abs(y[m] - p[m]) / d[m]))


def _pinball(y, p, alpha=0.5):
    e = y - p
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


METRIC_FUNCS = {"rmse": _rmse, "mse": _mse, "mae": _mae, "medae": _medae, "r2": _r2,
                "rmsle": _rmsle, "msle": _msle, "mape": _mape, "smape": _smape, "pinball": _pinball}


def score(metric: str, y, p, **kw) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return METRIC_FUNCS[metric](y, p, **kw) if kw else METRIC_FUNCS[metric](y, p)
