"""
Этап 3. Библиотека блоков признаков.

Каждый блок — детерминированная функция над объединённой таблицей ALL = train (с таргетом) + test (таргет NaN).
Блоки, использующие таргет, берут его ТОЛЬКО из строго более ранних периодов (есть колонка времени)
или out-of-fold (нет колонки времени). Поэтому один и тот же расчёт без утечки годится и для любого
фолда валидации по времени, и для test.

Блоки кэшируются на диск (parquet) — эксперименты только выбирают, какие блоки склеить.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

BLOCKS = {
    "base": "Исходные признаки. Категории — как category dtype (бустинги работают с ними нативно). Константы удалены.",
    "missing": "Индикаторы пропусков для колонок с 1–95% пропусков + число пропусков в строке.",
    "families": "Агрегаты по семействам однотипных колонок (например, помесячные температуры): mean/min/max/std/range.",
    "entity_history": "История сущности (здание и т.п.) по таргету СТРОГО за прошлые периоды: n, mean, last, std, тренд, "
                      "давность, отношение к медиане группы. Нужны колонка времени и ключ сущности.",
    "group_history": "Средний таргет по категориальным группам (и их паре) СТРОГО за прошлые периоды — "
                     "покрывает строки без истории сущности.",
    "similar_history": "Средний таргет ПОХОЖИХ объектов в прошлом: та же категория × та же децильная корзина каждой "
                       "числовой части ключа объекта (например, тип здания × дециль площади). Для строк без своей истории.",
    "oof_target_encoding": "Out-of-fold target encoding категорий (используется, когда нет колонки времени).",
    "log_numeric": "log1p сильно скошенных неотрицательных числовых колонок (полезно линейным моделям).",
}


# --------------------------------------------------------------------------- история по прошлым периодам
def past_target_stats(keys: pd.DataFrame, period: pd.Series, y: pd.Series, prefix: str,
                      full: bool = True) -> pd.DataFrame:
    """Для каждой строки — статистики y той же группы (keys) по строго более ранним периодам.
    Работает векторно: агрегат по (ключ, период) → накопленные суммы со сдвигом на один период."""
    df = keys.copy()
    df["__key"] = pd.util.hash_pandas_object(keys.astype(str), index=False).to_numpy()
    df["__p"] = period.to_numpy()
    df["__y"] = y.to_numpy()
    lab = df[df["__y"].notna()]
    agg = lab.groupby(["__key", "__p"])["__y"].agg(["sum", "count", "mean"]).reset_index()
    agg["sq"] = lab.assign(sq=lab["__y"] ** 2).groupby(["__key", "__p"])["sq"].sum().to_numpy()
    agg = agg.sort_values(["__key", "__p"])
    g = agg.groupby("__key")
    agg["c_sum"] = g["sum"].cumsum()
    agg["c_cnt"] = g["count"].cumsum()
    agg["c_sq"] = g["sq"].cumsum()
    agg["n_periods"] = g.cumcount() + 1
    # состояние ПОСЛЕ периода p; строкам периода q нужно состояние последнего периода < q
    state = agg[["__key", "__p", "c_sum", "c_cnt", "c_sq", "mean", "n_periods"]].rename(
        columns={"mean": "last_mean", "__p": "__p_state"})
    rows = df[["__key", "__p"]].reset_index().rename(columns={"index": "__row"})
    rows = rows.sort_values("__p")
    state = state.sort_values("__p_state")
    # merge_asof: для каждой строки берём последнее состояние с __p_state < __p (allow_exact_matches=False)
    m = pd.merge_asof(rows, state, left_on="__p", right_on="__p_state", by="__key",
                      allow_exact_matches=False, direction="backward")
    m = m.set_index("__row").sort_index()
    n = m["c_cnt"]
    mean = m["c_sum"] / n
    out = pd.DataFrame(index=keys.index)
    out[f"{prefix}_n"] = n.fillna(0).to_numpy()
    out[f"{prefix}_mean"] = mean.to_numpy()
    if full:
        var = (m["c_sq"] / n - mean ** 2).clip(lower=0)
        out[f"{prefix}_std"] = np.sqrt(var).where(n > 1).to_numpy()
        out[f"{prefix}_last"] = m["last_mean"].to_numpy()
        out[f"{prefix}_since"] = (m["__p"] - m["__p_state"]).to_numpy()
        out[f"{prefix}_trend"] = (m["last_mean"] - mean).to_numpy()
        out[f"{prefix}_periods"] = m["n_periods"].fillna(0).to_numpy()
    return out


def oof_target_mean(keys: pd.DataFrame, y: pd.Series, n_splits: int, seed: int, smoothing: float = 20) -> np.ndarray:
    """Out-of-fold среднее таргета по группе со сглаживанием к глобальному среднему. Для test — по всему train."""
    from sklearn.model_selection import KFold
    key = pd.util.hash_pandas_object(keys.astype(str), index=False).to_numpy()
    lab = np.where(y.notna().to_numpy())[0]
    res = np.full(len(keys), np.nan)
    yv = y.to_numpy()
    glob = np.nanmean(yv)

    def enc(train_idx, apply_idx):
        s = pd.DataFrame({"k": key[train_idx], "y": yv[train_idx]}).groupby("k")["y"].agg(["sum", "count"])
        st = (s["sum"] + smoothing * glob) / (s["count"] + smoothing)
        return pd.Series(key[apply_idx]).map(st).fillna(glob).to_numpy()
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(lab):
        res[lab[va]] = enc(lab[tr], lab[va])
    test_idx = np.where(y.isna().to_numpy())[0]
    if len(test_idx):
        res[test_idx] = enc(lab, test_idx)
    return res


# --------------------------------------------------------------------------- семейства колонок
def find_families(cols: list[str], min_size: int = 3) -> dict[str, list[str]]:
    """Колонки вида <префикс>_<суффикс>, у которых общий суффикс (например, january_min_temp_c,
    february_min_temp_c, …) → семейство 'min_temp_c'."""
    fam: dict[str, list[str]] = {}
    for c in cols:
        parts = c.split("_", 1)
        if len(parts) == 2 and re.match(r"^[a-zA-Zа-яА-Я]+$", parts[0]):
            fam.setdefault(parts[1], []).append(c)
    return {k: v for k, v in fam.items() if len(v) >= min_size}


# --------------------------------------------------------------------------- построение блоков
class FeatureStore:
    """Строит и кэширует блоки на объединённой таблице. Строки: сначала train (в исходном порядке), потом test."""

    def __init__(self, cache_dir: Path, all_df: pd.DataFrame, n_train: int, target: str, id_cols: list[str],
                 cat_cols: list[str], num_cols: list[str], time_col: str | None, entity_key: list[str] | None,
                 group_cols: list[str], seed: int = 42):
        self.dir = Path(cache_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.all, self.n_train, self.target, self.id_cols = all_df, n_train, target, id_cols
        self.cat_cols, self.num_cols, self.time_col = cat_cols, num_cols, time_col
        self.entity_key, self.group_cols, self.seed = entity_key, group_cols, seed

    def available(self) -> dict[str, str]:
        out = {}
        for b, d in BLOCKS.items():
            if b in ("entity_history",) and not (self.time_col and self.entity_key):
                continue
            if b == "group_history" and not (self.time_col and self.group_cols):
                continue
            if b == "similar_history" and not (self.time_col and self.group_cols and self._key_numeric()):
                continue
            if b == "oof_target_encoding" and (self.time_col or not self.group_cols):
                continue
            out[b] = d
        for p in sorted(self.dir.glob("custom_*.parquet")):
            out[p.stem] = "Признаки из кода Qwen (см. журнал)."
        return out

    def path(self, block: str) -> Path:
        return self.dir / f"{block}.parquet"

    def get(self, block: str) -> pd.DataFrame:
        p = self.path(block)
        if p.exists():
            return pd.read_parquet(p)
        df = getattr(self, f"_b_{block}")()
        df.to_parquet(p, index=False)
        return df

    def matrix(self, blocks: list[str]) -> pd.DataFrame:
        parts = [self.get(b) for b in dict.fromkeys(["base", *blocks])]
        X = pd.concat(parts, axis=1)
        return X.loc[:, ~X.columns.duplicated()]

    # ---- блоки
    def _b_base(self) -> pd.DataFrame:
        cols = [c for c in self.cat_cols + self.num_cols if self.all[c].nunique(dropna=False) > 1]
        X = self.all[cols].copy()
        for c in self.cat_cols:
            if c in X:
                X[c] = X[c].astype(str).replace({"nan": "__NA__"}).astype("category")
        return X.reset_index(drop=True)

    def _b_missing(self) -> pd.DataFrame:
        miss = self.all[self.cat_cols + self.num_cols].isna()
        share = miss.mean()
        cols = share[(share > 0.01) & (share < 0.95)].index
        out = pd.DataFrame({f"isna__{c}": miss[c].astype("int8") for c in cols})
        out["n_missing"] = miss.sum(axis=1).astype("int16")
        return out.reset_index(drop=True)

    def _b_families(self) -> pd.DataFrame:
        out = {}
        for name, cols in find_families(self.num_cols).items():
            v = self.all[cols].astype(float)
            out[f"fam__{name}__mean"] = v.mean(axis=1)
            out[f"fam__{name}__min"] = v.min(axis=1)
            out[f"fam__{name}__max"] = v.max(axis=1)
            out[f"fam__{name}__std"] = v.std(axis=1)
            out[f"fam__{name}__range"] = out[f"fam__{name}__max"] - out[f"fam__{name}__min"]
        return pd.DataFrame(out).reset_index(drop=True)

    def _b_entity_history(self) -> pd.DataFrame:
        a = self.all
        h = past_target_stats(a[self.entity_key], a[self.time_col], a[self.target], "ent")
        # отношение последнего значения к медиане по первой категориальной группе (масштаб «типичного» объекта)
        if self.group_cols:
            gmed = past_target_stats(a[[self.group_cols[0]]], a[self.time_col], a[self.target], "g0", full=False)
            h["ent_last_vs_group"] = h["ent_last"] / gmed["g0_mean"]
        return h.reset_index(drop=True)

    def _b_group_history(self) -> pd.DataFrame:
        a = self.all
        parts = []
        for c in self.group_cols:
            parts.append(past_target_stats(a[[c]], a[self.time_col], a[self.target], f"grp__{c}", full=False))
        if len(self.group_cols) >= 2:
            c1, c2 = self.group_cols[:2]
            parts.append(past_target_stats(a[[c1, c2]], a[self.time_col], a[self.target], f"grp__{c1}__{c2}", full=False))
        return pd.concat(parts, axis=1).reset_index(drop=True)

    def _key_numeric(self) -> list[str]:
        return [c for c in (self.entity_key or []) if c in self.num_cols and self.all[c].nunique() > 20]

    def _b_similar_history(self) -> pd.DataFrame:
        a = self.all
        g = self.group_cols[0]
        parts = []
        for c in self._key_numeric():
            bins = pd.qcut(a[c].rank(method="first"), 10, labels=False).fillna(-1).astype(int)
            keys = pd.DataFrame({g: a[g].astype(str), f"{c}_q10": bins})
            parts.append(past_target_stats(keys, a[self.time_col], a[self.target], f"sim__{g}__{c}", full=False))
            keys2 = keys.assign(**{f"{c}_q10": (bins // 2)})  # грубее: квинтиль — больше покрытие
            parts.append(past_target_stats(keys2, a[self.time_col], a[self.target], f"sim5__{g}__{c}", full=False))
        return pd.concat(parts, axis=1).reset_index(drop=True)

    def _b_oof_target_encoding(self) -> pd.DataFrame:
        a = self.all
        out = {}
        for c in self.group_cols:
            out[f"te__{c}"] = oof_target_mean(a[[c]], a[self.target], 5, self.seed)
        return pd.DataFrame(out).reset_index(drop=True)

    def _b_log_numeric(self) -> pd.DataFrame:
        out = {}
        for c in self.num_cols:
            v = self.all[c]
            if v.min() >= 0 and v.nunique() > 20 and v.skew() > 2:
                out[f"log1p__{c}"] = np.log1p(v.astype(float))
        return pd.DataFrame(out).reset_index(drop=True)
