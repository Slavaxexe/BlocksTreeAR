"""Прогон этапа на синтетике, похожей на задачу site_eui: скошенный положительный таргет,
категориальные признаки (в т.ч. высокой кардинальности), пропуски.
Проверяет: фолбэк правил, валидный ответ Qwen (мок), отказ валидатора на галлюцинации, другие метрики."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from ..stage import run_stage

OUT = ROOT / "tests" / "out"
OUT.mkdir(exist_ok=True)


def make_data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 40)), columns=[f"num_{i}" for i in range(40)])
    X["facility_type"] = rng.choice([f"type_{i}" for i in range(60)], n)
    X["state"] = rng.choice([f"S{i}" for i in range(7)], n)
    X["year_built"] = rng.integers(1900, 2020, n).astype(float)
    eff = X["facility_type"].str[5:].astype(int) / 20
    mu = np.exp(3.5 + 0.4 * X["num_0"] + 0.3 * np.sin(2 * X["num_1"]) + 0.2 * X["num_2"] * X["num_3"]
                + eff - 0.004 * (X["year_built"] - 1960))
    y = rng.gamma(shape=2.0, scale=mu / 2.0)  # мультипликативный шум, y>0, скошен
    X.loc[rng.random(n) < 0.15, "num_5"] = np.nan
    X.loc[rng.random(n) < 0.6, "num_6"] = np.nan
    df = X.copy()
    df.insert(0, "id", np.arange(n))
    df["site_eui"] = y
    return df


def stage_input(path, metric="RMSE"):
    return {"task": {"task_type": "regression", "target": "site_eui", "id_col": "id", "target_metric": metric},
            "paths": {"train": str(path)},
            "hints": {"categorical_cols": None, "time_col": None, "group_col": None},
            "budget": {"probe_seconds": 240, "probe_max_rows": 15000, "probe_folds": 3},
            "eda": {"note": "проброс от Марго"}}


def good_mock(messages):
    """Имитация корректного ответа Qwen: берёт дефолт правил и цитирует FACTS."""
    u = json.loads(messages[1]["content"])
    facts = u["FACTS"]
    base = u["LOSS_POLICY"]["base"]
    bid = f"probe.lightgbm_{base['loss']}_{base['transform']}.metric_mean"
    cands = []
    for c in u["RULES_DEFAULT"]:
        ev = {bid: facts[bid]}
        if c["model_id"] in ("ridge", "elasticnet") and "linear_gap" in facts:
            ev = {"linear_gap": facts["linear_gap"]}
        cands.append({**c, "evidence": ev})
    return json.dumps({"reasoning": [{"claim": "метрика вознаграждает среднее", "evidence": {bid: facts[bid]}},
                                     {"claim": "таргет скошен", "evidence": {"y_skew": facts["y_skew"]}}],
                       "candidates": cands, "rejected": [{"model_id": "sgd_linear", "reason": "n_rows < 1M"}]},
                      ensure_ascii=False)


def bad_mock(messages):
    """Галлюцинации: выдуманное число, неподдерживаемый лосс, несогласованная primary-пара, модель вне реестра."""
    return "<think>...</think>```json\n" + json.dumps({
        "reasoning": [{"claim": "y сильно скошен", "evidence": {"y_skew": 99.0}}],
        "candidates": [
            {"model_id": "ridge", "loss": "huber", "target_transform": "none", "priority": 1, "role": "primary",
             "rationale": "x", "evidence": {"y_skew": 99.0}},
            {"model_id": "lightgbm", "loss": "huber", "target_transform": "none", "priority": 1, "role": "primary",
             "rationale": "выбросы", "evidence": {"made_up_key": 1}},
            {"model_id": "tabnet", "loss": "l2", "target_transform": "none", "priority": 2, "role": "primary",
             "rationale": "x", "evidence": {}}]}) + "\n```"


if __name__ == "__main__":
    df = make_data()
    p = OUT / "train.csv"
    df.to_csv(p, index=False)
    report = {}

    t = time.time()
    r1 = run_stage(stage_input(p), use_llm=False, out_path=str(OUT / "out_rules_rmse.json"))
    report["rules_rmse"] = {"source": r1["source"], "time_s": round(time.time() - t, 1),
                            "probe": [(x["id"], x["metric_mean"]) for x in r1["probe"]["results"]],
                            "skipped": r1["probe"]["skipped"],
                            "candidates": [(c["model_id"], c["loss"], c["target_transform"], c["role"], c["priority"])
                                           for c in r1["candidates"]],
                            "automl": r1["automl_hint"]}

    r2 = run_stage(stage_input(p), llm_fn=good_mock, out_path=str(OUT / "out_qwen_mock.json"))
    report["good_mock"] = {"source": r2["source"], "errors": r2["validation_errors"]}

    r3 = run_stage(stage_input(p), llm_fn=bad_mock, out_path=str(OUT / "out_bad_mock.json"))
    report["bad_mock"] = {"source": r3["source"], "n_errors": len(r3["validation_errors"]),
                          "errors_first_attempt": [e for e in r3["validation_errors"] if e.startswith("attempt 1")]}

    for m in ("MAE", "RMSLE", "unknown_metric"):
        r = run_stage(stage_input(p, m), use_llm=False)
        report[f"rules_{m}"] = {"metric": r["target_metric"], "metric_source": r["metric_source"],
                                "policy_base": r["loss_policy"]["base"],
                                "alts": r["loss_policy"]["alternatives"],
                                "top3": [(c["model_id"], c["loss"], c["target_transform"]) for c in r["candidates"][:3]]}

    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
