"""
Этап 1. Аудит данных: всё, что меняет стратегию, считается кодом и пишется в память фактами.

  - типы колонок, пропуски, константы;
  - таргет: асимметрия, хвост (какая доля суммы квадратов приходится на 1% строк);
  - колонка времени: есть ли признак, по которому test строго «в будущем» относительно train;
  - ключ сущности: повторяется ли один и тот же объект в разных периодах и есть ли он в test;
  - сдвиг train→test: adversarial validation (насколько легко отличить test от train и чем);
  - группы для статистик.
Qwen (если доступна) интерпретирует факты и может предложить свой ключ сущности — код его оценит.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .features import find_families, past_target_stats
from .llm import LLM
from .memory import Memory


def _r(x, n=4):
    try:
        return None if x is None or not np.isfinite(x) else round(float(x), n)
    except TypeError:
        return x


def target_profile(y: pd.Series) -> dict:
    y = y.astype(float)
    sq = (y - y.mean()) ** 2
    k = max(1, int(0.01 * len(y)))
    return {"n": int(len(y)), "mean": _r(y.mean()), "std": _r(y.std()), "min": _r(y.min()), "p50": _r(y.median()),
            "p99": _r(y.quantile(0.99)), "max": _r(y.max()), "skew": _r(y.skew()),
            "top1pct_share_of_sq_dev": _r(np.sort(sq.to_numpy())[-k:].sum() / sq.sum()),
            "share_zero": _r((y == 0).mean()), "min_positive": bool(y.min() > 0)}


def detect_time_col(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict | None:
    """Колонка, по которой test целиком позже train (test.min > train.max) — разбиение по времени."""
    best = None
    for c in cols:
        if not pd.api.types.is_numeric_dtype(train[c]):
            continue
        a, b = train[c].dropna(), test[c].dropna()
        if len(b) == 0 or a.nunique() < 2 or a.nunique() > 400:
            continue
        if b.min() > a.max():
            info = {"col": c, "train_periods": sorted(a.unique().tolist())[-12:], "test_periods": sorted(b.unique().tolist())[:12],
                    "n_train_periods": int(a.nunique())}
            if best is None or info["n_train_periods"] > best["n_train_periods"]:
                best = info
    return best


def score_entity_key(all_df: pd.DataFrame, n_train: int, key: list[str], time_col: str, target: str) -> dict:
    """Насколько набор колонок похож на идентификатор объекта, повторяющегося во времени."""
    k = all_df[key].astype(str).agg("|".join, axis=1)
    tr, te = k.iloc[:n_train], k.iloc[n_train:]
    per_kp = all_df.groupby([k, all_df[time_col]]).size()
    rows_per_key_period = float(per_kp.mean())
    hist = past_target_stats(all_df[key], all_df[time_col], all_df[target], "h", full=False)
    lab = all_df[target].notna().to_numpy()
    has = (hist["h_n"].to_numpy() > 0) & lab
    corr = float(np.corrcoef(all_df[target].to_numpy()[has], hist["h_mean"].to_numpy()[has])[0, 1]) if has.sum() > 50 else 0.0
    test_cov = float(te.isin(set(tr)).mean()) if len(te) else 0.0
    test_hist = float((hist["h_n"].to_numpy()[n_train:] > 0).mean()) if len(te) else 0.0
    purity = 1.0 / max(1.0, rows_per_key_period)
    return {"key": key, "n_entities": int(k.nunique()), "rows_per_key_period": _r(rows_per_key_period, 3),
            "test_rows_with_history": _r(test_hist, 3), "test_coverage": _r(test_cov, 3),
            "history_corr": _r(corr, 3), "train_rows_with_history": _r(float(has[:n_train].mean()), 3),
            "score": _r(test_hist * max(corr, 0) * purity, 4)}


def search_entity_key(all_df: pd.DataFrame, n_train: int, cols: list[str], time_col: str, target: str,
                      cat_cols: list[str], max_seconds: float = 90) -> tuple[dict | None, list[dict]]:
    """Жадный поиск: кандидаты — колонки, значения которых повторяются в разных периодах
    (свойства объекта, а не периода); добавляем, пока растёт score."""
    import time as _t
    t0 = _t.monotonic()
    tried = []
    cand = []
    for c in cols:
        if c == time_col or c == target:
            continue
        nun = all_df[c].nunique()
        if nun < 2:
            continue
        per_val_periods = all_df.groupby(c)[time_col].transform("nunique")
        recur = float((per_val_periods >= 2).mean())
        if recur >= 0.5:
            cand.append((c, nun, recur))
    if not cand:
        return None, tried
    cand.sort(key=lambda t: -t[1])
    best = None
    pool = [c for c, _, _ in cand]
    current: list[str] = []
    for _ in range(5):
        step_best = None
        for c in pool[:25]:
            if c in current:
                continue
            if _t.monotonic() - t0 > max_seconds:
                break
            s = score_entity_key(all_df, n_train, current + [c], time_col, target)
            tried.append(s)
            if step_best is None or s["score"] > step_best["score"]:
                step_best = s
        if step_best is None or (best is not None and step_best["score"] <= best["score"] * 1.01):
            break
        best, current = step_best, list(step_best["key"])
    return best, tried


def adversarial(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], cat_cols: list[str], seed: int) -> dict:
    """AUC классификатора train-vs-test. ~0.5 — распределения похожи; ~1 — сильный сдвиг."""
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    n = min(20000, len(train), len(test))
    a = pd.concat([train[cols].sample(n, random_state=seed), test[cols].sample(n, random_state=seed)], ignore_index=True)
    for c in cat_cols:
        if c in a:
            a[c] = a[c].astype(str).astype("category")
    yb = np.r_[np.zeros(n), np.ones(n)]
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, verbose=-1, n_jobs=-1)
    p = cross_val_predict(m, a, yb, cv=3, method="predict_proba")[:, 1]
    m.fit(a, yb)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=a.columns).sort_values(ascending=False)
    return {"auc": _r(roc_auc_score(yb, p), 3),
            "top_shift_features": [{"feature": f, "gain_share": _r(v / imp.sum(), 3)} for f, v in imp.head(8).items()]}


class AuditReading(BaseModel):
    key_findings: list[str] = Field(..., description="3–6 выводов из фактов аудита, каждый со ссылкой на число")
    strategy: list[str] = Field(..., description="3–6 стратегических следствий для признаков/валидации/моделей")
    entity_key_alternatives: list[list[str]] = Field(default_factory=list, description="до 3 альтернативных ключей сущности")


def run_audit(train: pd.DataFrame, test: pd.DataFrame, target: str, id_cols: list[str], dictionary: list[dict],
              mem: Memory, llm: LLM, deadline: float, seed: int = 42) -> dict:
    feat_cols = [c for c in test.columns if c not in id_cols and c != target]
    dd_types = {str(d.get("feature")): str(d.get("dtype", "")).lower() for d in dictionary}
    cat_cols = [c for c in feat_cols if train[c].dtype == object or str(train[c].dtype) in ("category", "bool")
                or dd_types.get(c) in ("category", "categorical", "string")]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    all_df = pd.concat([train, test], ignore_index=True)
    n_train = len(train)

    out: dict = {"n_train": n_train, "n_test": len(test), "n_features": len(feat_cols),
                 "cat_cols": cat_cols, "num_cols": num_cols}
    out["target"] = target_profile(train[target])
    miss = train[feat_cols].isna().mean()
    out["missing"] = {"cols_with_missing": int((miss > 0).sum()), "max_share": _r(miss.max(), 3),
                      "top": {k: _r(v, 3) for k, v in miss.sort_values(ascending=False).head(6).items() if v > 0}}
    out["cardinality"] = {c: int(train[c].nunique()) for c in cat_cols}
    out["families"] = {k: len(v) for k, v in find_families(num_cols).items()}

    tc = detect_time_col(train, test, feat_cols)
    out["time"] = tc
    time_col = tc["col"] if tc else None
    entity, tried = (None, [])
    if time_col:
        entity, tried = search_entity_key(all_df, n_train, feat_cols, time_col, target, cat_cols)
    out["entity"] = entity
    out["entity_candidates_tried"] = len(tried)

    # группы для статистик: категориальные средней кардинальности, по убыванию связи с таргетом
    groups = []
    for c in cat_cols:
        if 2 <= train[c].nunique() <= 300:
            m = train.groupby(c)[target].mean()
            eta = train[c].map(m).var() / train[target].var()
            groups.append((c, float(eta)))
    groups.sort(key=lambda t: -t[1])
    out["group_cols"] = [c for c, _ in groups[:4]]
    out["group_eta2"] = {c: _r(e, 3) for c, e in groups[:6]}

    adv_cols = [c for c in feat_cols if c != time_col]
    try:
        out["adversarial"] = adversarial(train, test, adv_cols, cat_cols, seed)
    except Exception as e:  # аудит не должен ронять пайплайн
        out["adversarial"] = {"error": str(e)[:200]}

    for k in ("target", "time", "entity", "group_cols", "adversarial", "missing", "families"):
        mem.fact(f"audit.{k}", out[k])

    if llm.enabled:
        def validate(r: AuditReading) -> list[str]:
            bad = [k for k in r.entity_key_alternatives for c in k if c not in feat_cols]
            return [f"колонки ключа не существуют: {bad[:5]}"] if bad else []
        facts = {k: out[k] for k in ("target", "time", "entity", "group_eta2", "adversarial", "missing", "families", "cardinality")}
        r = llm.ask("auditor",
                    "Ты ведущий ML-исследователь. По фактам аудита данных сформулируй выводы и стратегию. "
                    "Каждый вывод опирай на конкретное число из FACTS. Ответ — один JSON: "
                    + str(AuditReading.model_json_schema()["properties"]),
                    f"DOMAIN: {mem.get('domain_summary', '')}\nFACTS: {facts}\nCOLUMNS: {feat_cols[:120]}",
                    AuditReading, deadline=deadline, validator=validate)
        if r is not None:
            for t in r.key_findings[:6]:
                mem.add_insight(t, "qwen:audit")
            mem.fact("audit.strategy_by_qwen", r.strategy[:6], "qwen")
            if time_col:
                for alt in r.entity_key_alternatives[:3]:
                    s = score_entity_key(all_df, n_train, list(alt), time_col, target)
                    mem.log("entity_key_from_qwen", **s)
                    if entity is None or s["score"] > entity["score"] * 1.02:
                        entity = s
                        out["entity"] = s
                        mem.fact("audit.entity", s, "qwen+code_score")
    return out
