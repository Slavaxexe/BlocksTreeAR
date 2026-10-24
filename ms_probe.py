"""
Probe: быстрый эмпирический прогон пар (модель, лосс, трансформ) на подвыборке.
Задача — ранжировать гипотезы, а не получить финальный скор. Бюджет по времени жёсткий.
"""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from ms_builders import build, prepare_frame
from ms_metrics import score

warnings.filterwarnings("ignore")


def make_folds(n: int, k: int, groups=None, time_order=None, seed: int = 0):
    idx = np.arange(n)
    if time_order is not None:
        order = np.argsort(np.asarray(time_order), kind="stable")
        return [(order[tr], order[va]) for tr, va in TimeSeriesSplit(n_splits=k).split(order)], "time_series"
    if groups is not None:
        return list(GroupKFold(n_splits=k).split(idx, groups=groups)), "group_kfold"
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(idx)), "kfold"


def run_probe(X: pd.DataFrame, y: pd.Series, configs: list[dict], metric: str, cat_cols: list[str],
              budget_s: float = 300, max_rows: int = 30000, k: int = 3, group_col: str | None = None,
              time_col: str | None = None, metric_kw: dict | None = None, seed: int = 0) -> dict:
    """configs: [{"id": "...", "model_id": "...", "loss": "...", "transform": "none"|"log1p"}]"""
    t0 = time.time()
    metric_kw = metric_kw or {}
    if len(X) > max_rows:
        if time_col:  # для времени берём хвост, чтобы не ломать порядок
            keep = X[time_col].sort_values(kind="stable").index[-max_rows:]
        else:
            keep = X.sample(max_rows, random_state=seed).index
        X, y = X.loc[keep], y.loc[keep]
    X = X.reset_index(drop=True)
    y = pd.to_numeric(y, errors="coerce").reset_index(drop=True).astype(float)

    groups = X[group_col].to_numpy() if group_col else None
    torder = X[time_col].to_numpy() if time_col else None
    folds, split_kind = make_folds(len(X), k, groups, torder, seed)
    num_cols = [c for c in X.columns if c not in cat_cols]

    results, skipped = [], []
    for cfg in configs:
        if time.time() - t0 > budget_s:
            skipped.append({"id": cfg["id"], "reason": "probe budget exhausted"})
            continue
        fold_scores, ts = [], time.time()
        try:
            Xm = prepare_frame(X, cat_cols, cfg["model_id"])
            for tr, va in folds:
                ytr = y.iloc[tr]
                # масштаб для дельты Хьюбера — в той шкале, где модель учится
                ys = np.log1p(ytr) if cfg.get("transform") == "log1p" else ytr
                mad = float(np.median(np.abs(ys - np.median(ys))) * 1.4826) or float(np.std(ys)) or 1.0
                model = build(cfg["model_id"], cfg["loss"], cfg.get("transform", "none"),
                              num_cols=num_cols, cat_cols=cat_cols, y_scale=mad,
                              **cfg.get("loss_kw", {}))
                model.fit(Xm.iloc[tr], ytr)
                p = model.predict(Xm.iloc[va])
                fold_scores.append(score(metric, y.iloc[va].to_numpy(), p, **metric_kw))
            results.append({"id": cfg["id"], "model_id": cfg["model_id"], "loss": cfg["loss"],
                            "transform": cfg.get("transform", "none"),
                            "metric_mean": round(float(np.mean(fold_scores)), 6),
                            "metric_std": round(float(np.std(fold_scores)), 6),
                            "fold_scores": [round(s, 6) for s in fold_scores],
                            "fit_seconds": round(time.time() - ts, 1)})
        except Exception as e:  # probe не должен ронять этап
            skipped.append({"id": cfg["id"], "reason": f"error: {type(e).__name__}: {str(e)[:160]}"})

    return {"metric": metric, "split": split_kind, "k": k, "rows_used": int(len(X)),
            "seconds": round(time.time() - t0, 1), "results": results, "skipped": skipped}


def paired_gain(a: dict, b: dict, greater_is_better: bool) -> dict:
    """Насколько a лучше b по фолдам (парное сравнение на одних и тех же фолдах).
    rel_gain > 0 — a лучше. significant: средний выигрыш больше 1 стандартной ошибки
    парной разницы и больше 0.5% относительного."""
    fa, fb = np.array(a["fold_scores"]), np.array(b["fold_scores"])
    d = (fa - fb) if greater_is_better else (fb - fa)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    rel = float(d.mean() / abs(np.mean(fb))) if np.mean(fb) != 0 else 0.0
    return {"a": a["id"], "b": b["id"], "rel_gain": round(rel, 4), "abs_gain": round(float(d.mean()), 6),
            "se": round(float(se), 6), "significant": bool(d.mean() > se and rel > 0.005)}
