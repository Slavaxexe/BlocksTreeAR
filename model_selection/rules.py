"""
Правила: (1) метрика -> политика лосса, (2) план probe, (3) дефолтный набор кандидатов.
Дефолт правил = фолбэк, если Qwen ответила невалидно. Пороги — в THRESHOLDS, их можно
настраивать, это стартовые эвристики, а не истины.
"""
from __future__ import annotations
from .registry import METRICS, MODELS, is_metric_consistent, supports
from .probe import paired_gain

THRESHOLDS = {
    "skew_for_log": 1.0,            # |skew| выше — пробуем log1p / log-link лоссы
    "outlier_share_for_robust": 0.03,  # доля MAD-выбросов, при которой пробуем робастный лосс
    "zero_share_for_tweedie": 0.10,
    "linear_gap_primary": 1.10,     # linear/gbm <= 1.10 -> линейная модель полноценный кандидат
    "linear_gap_ensemble": 1.50,    # <= 1.50 -> член ансамбля, иначе отбрасываем
    "small_data_rows": 5000,
    "sgd_min_rows": 1_000_000,
    "mlp_min_rows": 10_000,
    "knn_max_features": 20,
    "max_candidates": 6,
}


def loss_policy(metric: str, td: dict, alpha: float = 0.5) -> dict:
    """Базовая пара + альтернативы, которые разрешено проверить в probe.
    consistent=True — пара оценивает ту же статистику, что вознаграждает метрика."""
    st = METRICS[metric]["statistic"]
    y_min, skew = td["y_min"], abs(td["y_skew"] or 0)
    nonneg, pos = y_min is not None and y_min >= 0, y_min is not None and y_min > 0
    alts, notes = [], []

    if st == "mean":
        base = ("l2", "none")
        if nonneg and skew > THRESHOLDS["skew_for_log"]:
            if (td["y_share_zero"] or 0) > THRESHOLDS["zero_share_for_tweedie"]:
                alts.append(("tweedie", "none"))
            elif pos:
                alts.append(("gamma", "none"))
            else:
                alts.append(("poisson", "none"))
            notes.append("y>=0 и скошен: log-link лосс оценивает СРЕДНЕЕ -> согласован с метрикой")
        if y_min is not None and y_min > -1 and skew > THRESHOLDS["skew_for_log"]:
            alts.append(("l2", "log1p"))
            notes.append("log1p+l2 оценивает среднее в log-шкале -> смещение вниз; только через probe")
        if (td["y_outlier_share_mad"] or 0) > THRESHOLDS["outlier_share_for_robust"]:
            alts.append(("huber", "none"))
            notes.append("много выбросов: Хьюбер тянет к медиане -> для mean-метрики только через probe")
    elif st == "median":
        base = ("l1", "none")
        alts.append(("huber", "none"))
        if y_min is not None and y_min > -1 and skew > THRESHOLDS["skew_for_log"]:
            alts.append(("l1", "log1p"))  # медиана инвариантна к монотонным преобразованиям
            notes.append("медиана инвариантна к log1p -> l1+log1p согласован с MAE")
    elif st == "log_mean":
        base = ("l2", "log1p")
        notes.append("RMSLE == RMSE в log1p-шкале: l2+log1p точно согласован")
    elif st == "relative":
        base = ("l2", "log1p")
        alts.append(("mape", "none"))
        notes.append("MAPE/SMAPE ~ относительная ошибка: log1p+l2 — устойчивое приближение")
    elif st == "quantile":
        base = ("quantile", "none")
    else:
        base = ("l2", "none")

    def cons(l, t):
        if st == "relative":
            return (l, t) in (("l2", "log1p"), ("mape", "none"))
        if st == "median" and (l, t) == ("l1", "log1p"):
            return True
        return is_metric_consistent(l, t, metric)

    return {"metric": metric, "metric_statistic": st,
            "base": {"loss": base[0], "transform": base[1], "consistent": True},
            "alternatives": [{"loss": l, "transform": t, "consistent": cons(l, t)} for l, t in alts],
            "loss_kw": {"alpha": alpha} if st == "quantile" else {},
            "notes": notes}


def probe_plan(policy: dict, fd: dict) -> list[dict]:
    """Что прогнать в probe. Минимум: пол качества, линейный эталон, бустинг с базовой парой,
    бустинг с каждой альтернативой. CatBoost — если есть категориальные."""
    b = policy["base"]
    floor_loss = "l1" if policy["metric_statistic"] == "median" else "l2"
    cfgs = [{"id": "dummy", "model_id": "dummy_mean", "loss": floor_loss, "transform": "none"},
            {"id": f"ridge_l2_{b['transform']}", "model_id": "ridge", "loss": "l2", "transform": b["transform"]}]
    gbm_loss = b["loss"]
    cfgs.append({"id": f"lightgbm_{gbm_loss}_{b['transform']}", "model_id": "lightgbm", "loss": gbm_loss,
                 "transform": b["transform"], "loss_kw": policy["loss_kw"]})
    for a in policy["alternatives"]:
        if supports("lightgbm", a["loss"]):
            cfgs.append({"id": f"lightgbm_{a['loss']}_{a['transform']}", "model_id": "lightgbm",
                         "loss": a["loss"], "transform": a["transform"], "loss_kw": policy["loss_kw"]})
    if fd["n_categorical"] > 0 and supports("catboost", gbm_loss):
        cfgs.append({"id": f"catboost_{gbm_loss}_{b['transform']}", "model_id": "catboost",
                     "loss": gbm_loss, "transform": b["transform"], "loss_kw": policy["loss_kw"]})
    return cfgs


def _by_id(probe: dict) -> dict:
    return {r["id"]: r for r in probe["results"]}


def default_candidates(policy: dict, diag: dict, probe: dict) -> dict:
    """Детерминированный выбор кандидатов. Это фолбэк и стартовая точка для Qwen."""
    td, fd = diag["target"], diag["features"]
    gib = METRICS[policy["metric"]]["greater_is_better"]
    res = _by_id(probe)
    b = policy["base"]
    base_id = f"lightgbm_{b['loss']}_{b['transform']}"
    evidence, cands, rejected = {}, [], []

    # 1) выбор пары (лосс, трансформ) для бустингов: базовая, если альтернатива значимо не лучше
    best = (b["loss"], b["transform"])
    if base_id in res:
        for a in policy["alternatives"]:
            aid = f"lightgbm_{a['loss']}_{a['transform']}"
            if aid in res:
                g = paired_gain(res[aid], res[base_id], gib)
                evidence[f"gain_{aid}_vs_base"] = g
                if g["significant"] and (best == (b["loss"], b["transform"]) or
                                         g["rel_gain"] > evidence.get("_best_gain", 0)):
                    best = (a["loss"], a["transform"])
                    evidence["_best_gain"] = g["rel_gain"]
    evidence.pop("_best_gain", None)
    loss, tr = best
    best_gbm = res.get(f"lightgbm_{loss}_{tr}")

    def add(model_id, role, prio, why, l=loss, t=tr):
        if not supports(model_id, l):
            if supports(model_id, b["loss"]):  # откат к базовой паре — она согласована с метрикой
                why += f"; лосс {l} не поддерживается — базовая пара"
                l, t = b["loss"], b["transform"]
            else:
                rejected.append({"model_id": model_id, "reason": f"не поддерживает лосс {l}"})
                return
        cands.append({"model_id": model_id, "loss": l, "target_transform": t, "priority": prio,
                      "role": role, "rationale": why})

    add("lightgbm", "primary", 1, "основной бустинг; пара (лосс, трансформ) выбрана по probe")
    cat_first = fd["n_categorical"] > 0 and len(fd["high_card_cols"]) > 0
    add("catboost", "primary", 2 if not cat_first else 1,
        "категориальные высокой кардинальности" if cat_first else "второй бустинг: другая схема роста деревьев")
    add("xgboost", "ensemble_member", 4, "третий бустинг для разнообразия ансамбля")

    # 2) линейные: по разрыву с бустингом на probe
    rid = f"ridge_l2_{b['transform']}"
    if rid in res and best_gbm and policy["metric_statistic"] in ("mean", "log_mean", "relative"):
        lin, gbm = res[rid]["metric_mean"], best_gbm["metric_mean"]
        gap = (gbm / lin) if gib else (lin / gbm if gbm else 99)
        evidence["linear_gap"] = round(gap, 4)
        lin_model = "elasticnet" if (fd["n_high_corr_pairs_095"] >= 5 or fd["n_features"] >= 100) else "ridge"
        if gap <= THRESHOLDS["linear_gap_primary"] or fd["n_rows"] < THRESHOLDS["small_data_rows"]:
            add(lin_model, "primary", 2, f"linear_gap={gap:.3f}: линейный сигнал почти полный", l="l2", t=b["transform"])
        elif gap <= THRESHOLDS["linear_gap_ensemble"]:
            add(lin_model, "ensemble_member", 5, f"linear_gap={gap:.3f}: слабее, но некоррелированные ошибки", l="l2", t=b["transform"])
        else:
            rejected.append({"model_id": lin_model, "reason": f"linear_gap={gap:.3f} > {THRESHOLDS['linear_gap_ensemble']}"})

    # 3) SGD-линейная: только на очень больших данных
    if fd["n_rows"] >= THRESHOLDS["sgd_min_rows"]:
        add("sgd_linear", "ensemble_member", 5, "n_rows >= 1M: закрытое решение дорого", l="l2", t=b["transform"])
    else:
        rejected.append({"model_id": "sgd_linear",
                         "reason": f"n_rows={fd['n_rows']} < 1M: дублирует ridge без выигрыша"})

    # 4) MLP — член ансамбля и полигон для ветки оптимизаторов Макса
    if (fd["n_rows"] >= THRESHOLDS["mlp_min_rows"] and fd["n_numeric"] >= fd["n_features"] / 2
            and policy["metric_statistic"] in ("mean", "log_mean")):
        add("mlp", "ensemble_member", 6, "много строк, преимущественно числовые признаки", l="l2", t=b["transform"])
    else:
        rejected.append({"model_id": "mlp", "reason": "мало строк / мало числовых / метрика не mean"})

    # 5) деревья-бэггинг: ExtraTrees как разнообразие; RF дублирует
    if supports("extra_trees", loss if loss in ("l2", "poisson") else "l2"):
        add("extra_trees", "ensemble_member", 5, "бэггинг: декоррелированные ошибки к бустингу",
            l=loss if loss in ("l2", "poisson") else "l2", t=tr)
    rejected.append({"model_id": "random_forest", "reason": "дублирует extra_trees и медленнее"})

    if fd["n_features"] <= THRESHOLDS["knn_max_features"] and fd["n_rows"] <= 200_000:
        add("knn", "ensemble_member", 6, "мало признаков: локальная структура", l="l2", t=b["transform"])
    else:
        rejected.append({"model_id": "knn", "reason": f"n_features={fd['n_features']}: проклятие размерности"})

    cands.sort(key=lambda c: c["priority"])
    cands = cands[:THRESHOLDS["max_candidates"]]
    return {"candidates": cands, "rejected": rejected, "evidence": evidence,
            "chosen_pair": {"loss": loss, "target_transform": tr}}
