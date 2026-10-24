"""
Детерминированные диагностики. Qwen не считает сама — она читает эти числа
и обязана ссылаться на них по имени поля (валидатор это проверяет).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def diagnose_target(y: pd.Series) -> dict:
    y = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826
    robust_z = np.abs(y - med) / mad if mad > 0 else np.zeros_like(y)
    q = np.quantile(y, [0.01, 0.5, 0.99])
    out = {
        "y_n": int(len(y)),
        "y_mean": _r(y.mean()), "y_std": _r(y.std()),
        "y_min": _r(y.min()), "y_max": _r(y.max()),
        "y_p01": _r(q[0]), "y_p50": _r(q[1]), "y_p99": _r(q[2]),
        "y_skew": _r(stats.skew(y)),
        "y_kurtosis": _r(stats.kurtosis(y)),
        "y_share_zero": _r((y == 0).mean()),
        "y_share_negative": _r((y < 0).mean()),
        "y_is_integer": bool(np.all(np.mod(y, 1) == 0)),
        "y_n_unique": int(len(np.unique(y))),
        # доля точек дальше 3.5 робастных сигм от медианы
        "y_outlier_share_mad": _r((robust_z > 3.5).mean()),
        # вклад топ-1% самых больших квадратов отклонений в общую сумму квадратов:
        # показывает, насколько RMSE/MSE будет доминироваться хвостом
        "y_top1pct_sq_share": None,
        "y_tail_ratio_p99_p50": _r(q[2] / q[1]) if q[1] not in (0,) else None,
        "y_log1p_skew": _r(stats.skew(np.log1p(y))) if y.min() > -1 else None,
    }
    sq = (y - y.mean()) ** 2
    k = max(1, int(0.01 * len(sq)))
    out["y_top1pct_sq_share"] = _r(np.sort(sq)[-k:].sum() / sq.sum()) if sq.sum() > 0 else 0.0
    return out


def diagnose_features(X: pd.DataFrame, y: pd.Series, cat_cols: list[str] | None = None,
                      max_corr_cols: int = 200) -> dict:
    cat_cols = cat_cols or [c for c in X.columns
                            if X[c].dtype == "object" or str(X[c].dtype).startswith("category")
                            or X[c].dtype == "bool"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    card = {c: int(X[c].nunique(dropna=True)) for c in cat_cols}
    miss = X.isna().mean()
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]

    # сильно коррелированные пары среди числовых
    n_high_corr = 0
    if len(num_cols) > 1:
        cols = num_cols[:max_corr_cols]
        sample = X[cols].sample(min(len(X), 20000), random_state=0)
        c = sample.corr(method="pearson").abs().to_numpy()
        iu = np.triu_indices_from(c, k=1)
        n_high_corr = int(np.nansum(c[iu] > 0.95))

    # связь признаков с таргетом: ранговая (ловит монотонную нелинейность) vs линейная
    yv = pd.to_numeric(y, errors="coerce")
    top_signal, lin_vs_rank_gap = [], None
    if num_cols:
        s = X[num_cols].sample(min(len(X), 20000), random_state=0)
        ys = yv.loc[s.index]
        sp = s.corrwith(ys, method="spearman").abs()
        pe = s.corrwith(ys, method="pearson").abs()
        top = sp.sort_values(ascending=False).head(10)
        top_signal = [{"feature": f, "spearman_abs": _r(v, 3), "pearson_abs": _r(pe[f], 3)}
                      for f, v in top.items() if pd.notna(v)]
        # если ранговая связь заметно сильнее линейной — есть нелинейность/выбросы
        if len(top) and pe[top.index].notna().any():
            lin_vs_rank_gap = _r(float((top - pe[top.index]).mean()), 3)

    return {
        "n_rows": int(len(X)), "n_features": int(X.shape[1]),
        "n_numeric": len(num_cols), "n_categorical": len(cat_cols),
        "categorical_cols": cat_cols,
        "max_cardinality": max(card.values()) if card else 0,
        "high_card_cols": [c for c, v in card.items() if v > 50],
        "missing_share_total": _r(float(X.isna().to_numpy().mean())),
        "missing_share_max_col": _r(float(miss.max())) if len(miss) else 0.0,
        "n_cols_missing_gt_50pct": int((miss > 0.5).sum()),
        "n_constant_cols": len(const),
        "n_high_corr_pairs_095": n_high_corr,
        "rows_per_feature": _r(len(X) / max(1, X.shape[1]), 1),
        "top_signal_features": top_signal,
        "spearman_minus_pearson_top10": lin_vs_rank_gap,
    }


def diagnose(train: pd.DataFrame, target: str, id_col: str | None = None,
             cat_cols: list[str] | None = None) -> dict:
    X = train.drop(columns=[c for c in [target, id_col] if c and c in train.columns])
    return {"target": diagnose_target(train[target]),
            "features": diagnose_features(X, train[target], cat_cols)}
