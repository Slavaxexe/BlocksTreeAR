"""
Этап пайплайна: выбор семейств моделей и функции потерь для регрессии.

    вход  — JSON от этапа Марго (schemas.StageInput)
    выход — JSON для этапа Макса (schemas.StageOutput)

Поток: метрика -> диагностики -> политика лосса -> probe -> дефолт правил -> Qwen -> валидатор -> выход.
Если Qwen недоступна или дважды ответила невалидно — выход строится из дефолта правил
(source="rules_fallback"), пайплайн не останавливается.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .builders import loss_kwargs                       
from .diagnostics import diagnose                       
from .llm_step import decide                            
from .probe import run_probe                            
from .registry import METRICS, MODELS, is_metric_consistent, normalize_metric
from .rules import default_candidates, loss_policy, probe_plan
from .schemas import AutoMLHint, OutCandidate, StageInput, StageOutput


def _automl_hint(cands: list[dict], metric: str, pair: dict) -> AutoMLHint:
    notes = []
    tr = pair["target_transform"]
    flaml_list = sorted({MODELS[c["model_id"]]["flaml"] for c in cands if MODELS[c["model_id"]]["flaml"]})
    ag_list = sorted({MODELS[c["model_id"]]["autogluon"] for c in cands if MODELS[c["model_id"]]["autogluon"]})
    flaml_metric = METRICS[metric]["flaml"]
    ag_metric = METRICS[metric]["autogluon"]
    if metric in ("rmsle", "msle"):
        flaml_metric, ag_metric = "rmse", "root_mean_squared_error"
        notes.append("RMSLE: обучать AutoML на log1p(y) с метрикой rmse, предсказания через expm1 — точная эквивалентность")
    if tr == "log1p" and metric not in ("rmsle", "msle"):
        notes.append("выбран log1p: подавать в AutoML log1p(y) и делать expm1, но сравнивать по метрике на исходной шкале")
    if pair["loss"] != "l2":
        notes.append(f"выбран лосс {pair['loss']}: FLAML/AutoGluon оптимизируют свой objective, "
                     f"выбор лосса в AutoML не переносится — для этой пары использовать обычные шаблоны")
    if metric == "pinball":
        notes.append("квантиль: AutoGluon только через problem_type='quantile'")
    return AutoMLHint(flaml_metric=flaml_metric, flaml_estimator_list=flaml_list,
                      autogluon_eval_metric=ag_metric, autogluon_models=ag_list,
                      label_transform=tr, notes=notes)


def run_stage(stage_input: dict, llm_fn: Callable | None = None, out_path: str | None = None,
              use_llm: bool = True) -> dict:
    t0 = time.time()
    timings, warnings = {}, []
    inp = StageInput.model_validate(stage_input)

    # 1) метрика: из описания задачи; если не распарсилась — фолбэк с предупреждением
    metric = normalize_metric(inp.task.target_metric)
    metric_source = "task_description"
    if metric is None:
        metric, metric_source = "rmse", "fallback"
        warnings.append(f"метрика '{inp.task.target_metric}' не распознана -> фолбэк rmse")

    # 2) данные
    train = pd.read_csv(inp.paths.train)
    drop = [c for c in [inp.task.target, inp.task.id_col, *inp.hints.drop_cols] if c and c in train.columns]
    X, y = train.drop(columns=drop), train[inp.task.target]
    cat_cols = inp.hints.categorical_cols
    if cat_cols is None:
        cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) in ("category", "bool")]
    cat_cols = [c for c in cat_cols if c in X.columns]

    # 3) диагностики
    ts = time.time()
    diag = diagnose(train.drop(columns=[c for c in inp.hints.drop_cols if c in train.columns]),
                    inp.task.target, inp.task.id_col, cat_cols)
    timings["diagnostics_s"] = round(time.time() - ts, 1)

    # 4) политика лосса + probe
    policy = loss_policy(metric, diag["target"], alpha=inp.task.metric_params.get("alpha", 0.5))
    plan = probe_plan(policy, diag["features"])
    ts = time.time()
    probe = run_probe(X, y, plan, metric, cat_cols, budget_s=inp.budget.probe_seconds,
                      max_rows=inp.budget.probe_max_rows, k=inp.budget.probe_folds,
                      group_col=inp.hints.group_col, time_col=inp.hints.time_col,
                      metric_kw={"alpha": policy["loss_kw"]["alpha"]} if metric == "pinball" else None)
    timings["probe_s"] = round(time.time() - ts, 1)
    if probe["skipped"]:
        warnings.append(f"probe пропустил: {probe['skipped']}")

    # 5) дефолт правил
    rule_default = default_candidates(policy, diag, probe)

    # 6) Qwen
    source, reasoning, errors = "rules_fallback", [], []
    chosen, rejected = rule_default["candidates"], rule_default["rejected"]
    for c in chosen:
        c.setdefault("evidence", {})
    if use_llm:
        ts = time.time()
        task_view = {"target": inp.task.target, "metric": metric, "metric_source": metric_source,
                     **(inp.task.model_extra or {})}
        res = decide(task_view, policy, diag, probe, rule_default, llm_fn=llm_fn)
        timings["llm_s"] = round(time.time() - ts, 1)
        errors = res["errors"]
        if res["decision"] is not None:
            source = "qwen"
            d = res["decision"]
            chosen = [c.model_dump() for c in d.candidates]
            rejected = [r.model_dump() for r in d.rejected]
            reasoning = [s.model_dump() for s in d.reasoning]
        else:
            warnings.append("Qwen не дала валидного ответа -> дефолт правил")

    # 7) обогащение под шаблоны Макса
    ys_full = pd.to_numeric(y, errors="coerce").dropna()
    out_cands = []
    for c in sorted(chosen, key=lambda c: c["priority"]):
        m = MODELS[c["model_id"]]
        ys = np.log1p(ys_full) if c["target_transform"] == "log1p" else ys_full
        scale = float(np.median(np.abs(ys - np.median(ys))) * 1.4826) or float(ys.std()) or 1.0
        out_cands.append(OutCandidate(
            model_id=c["model_id"], library=m["library"], estimator=m["estimator"],
            loss=c["loss"], loss_native=loss_kwargs(c["model_id"], c["loss"], y_scale=scale,
                                                    **policy["loss_kw"]),
            target_transform=c["target_transform"],
            metric_consistent=is_metric_consistent(c["loss"], c["target_transform"], metric),
            priority=c["priority"], role=c["role"], needs_scaling=m["needs_scaling"],
            native_categorical=m["native_categorical"], optimizer_applicable=m["optimizer_applicable"],
            rationale=c["rationale"], evidence=c.get("evidence") or {}))

    out = StageOutput(
        task=inp.task.model_dump(), target_metric=metric, metric_source=metric_source,
        metric_statistic=policy["metric_statistic"], loss_policy=policy, diagnostics=diag, probe=probe,
        candidates=out_cands, rejected=rejected, reasoning=reasoning,
        automl_hint=_automl_hint(chosen, metric, {"loss": out_cands[0].loss,
                                                  "target_transform": out_cands[0].target_transform}),
        source=source, validation_errors=errors, warnings=warnings,
        hints=inp.hints.model_dump(), eda=inp.eda, timings={**timings, "total_s": round(time.time() - t0, 1)},
    ).model_dump()
    if out_path:
        Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    # python -m model_selection.stage input.json output.json [--no-llm]
    src, dst = sys.argv[1], sys.argv[2]
    run_stage(json.loads(Path(src).read_text(encoding="utf-8")), out_path=dst,
              use_llm="--no-llm" not in sys.argv)
    print(f"written {dst}")
