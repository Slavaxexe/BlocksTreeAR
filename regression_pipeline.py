#!/usr/bin/env python3
"""
Сквозной пайплайн регрессии: задача → подготовка данных (Марго) → выбор модели и лосса
(model_selection) → обучение и валидация (Макс) → submission.csv.

    python regression_pipeline.py data/regression                 # с Qwen (ключ из .env)
    python regression_pipeline.py data/regression --no-llm        # без Qwen: правила + офлайн-планировщик

Каталог задачи: task.md, train.csv, test.csv, sample_submission.csv, data_dictionary.csv.
Ключ Qwen: .env в корне репозитория (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL).

Каждый этап пишет свой JSON в <output>/<NN_stage>/, стыки — только через эти JSON и файлы.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Поддерживаемые этапом Макса метрики (agent/training_contracts.py)
MAX_METRICS = {"rmse", "mae", "mse", "r2", "mape"}

# model_selection.model_id -> family этапа Макса. None = у Макса нет шаблона.
FAMILY_MAP = {
    "ridge": "ridge", "elasticnet": "elasticnet", "sgd_linear": "sgd", "mlp": "mlp",
    "random_forest": "random_forest", "extra_trees": "extra_trees", "hist_gb": "hist_gradient_boosting",
    "lightgbm": "lightgbm", "catboost": "catboost", "xgboost": "xgboost",
    "knn": None, "huber_linear": None,
}

METRIC_PATTERNS = [  # порядок важен: специфичные раньше общих
    ("rmsle", r"\bRMSLE\b|root mean squared log"), ("msle", r"\bMSLE\b"),
    ("rmse", r"\bRMSE\b|root mean squared error|корень из среднеквадратич"),
    ("mse", r"\bMSE\b|mean squared error|среднеквадратичн"),
    ("mae", r"\bMAE\b|mean absolute error|средняя абсолютная"),
    ("mape", r"\bMAPE\b|mean absolute percentage"), ("smape", r"\bSMAPE\b"),
    ("r2", r"\bR\^?2\b|R²|коэффициент детерминации"),
]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ============================================================================ 0. Задача
def read_task(task_dir: Path) -> dict:
    """Детерминированное понимание задачи: таргет и id — из sample_submission,
    метрика — из task.md (ищем строку про метрику, затем весь текст)."""
    text = (task_dir / "task.md").read_text(encoding="utf-8") if (task_dir / "task.md").exists() else ""
    sample = pd.read_csv(task_dir / "sample_submission.csv", nrows=5)
    test_cols = pd.read_csv(task_dir / "test.csv", nrows=1).columns
    ids = [c for c in sample.columns if c in test_cols]
    targets = [c for c in sample.columns if c not in ids]
    metric_lines = [ln for ln in text.splitlines() if re.search(r"метрик|metric", ln, re.I)]
    metric, source = None, "fallback"
    for scope in (metric_lines, [text]):
        for name, pat in METRIC_PATTERNS:
            if any(re.search(pat, s, re.I) for s in scope):
                metric, source = name, "task_description"
                break
        if metric:
            break
    return {"target": targets[0], "id_columns": ids, "metric": metric or "rmse", "metric_source": source,
            "metric_evidence": metric_lines[:3]}


# ============================================================================ 1. Марго
def run_preparation(task_dir: Path, out: Path, task: dict, use_llm: bool, budget_s: int) -> dict:
    from data_preparation.config import PipelineConfig
    from data_preparation.pipeline import DataPreparationPipeline

    cfg = PipelineConfig(task_dir=str(task_dir), target_columns=[task["target"]], metric_name=task["metric"],
                         output_dir=str(out), time_budget_seconds=budget_s, random_state=42)
    if not use_llm:
        cfg.llm.api_key, cfg.llm.base_url = "", ""
    errors = cfg.validate()
    if errors:
        raise RuntimeError(f"Конфиг подготовки данных: {errors}")
    res = DataPreparationPipeline(cfg).run()
    if res.errors:
        raise RuntimeError(f"Подготовка данных: {[e.message for e in res.errors]}")
    return {"status": res.status.value, "n_features": len(res.feature_names),
            "warnings": [w.message for w in (res.warnings or [])][:20]}


def assemble_tables(prep: Path, data_out: Path, task: dict, task_dir: Path) -> dict:
    """Стык Марго → дальше. У Марго train разбит на train/val (80/20) — склеиваем обратно
    в одну размеченную таблицу: внешнюю валидацию делает этап Макса, один раз и для всех."""
    tgt, ids = task["target"], task["id_columns"]
    parts = []
    for p in ("train", "val"):
        X = pd.read_parquet(prep / f"{p}_features.parquet")
        i = pd.read_parquet(prep / f"{p}_ids.parquet")
        y = pd.read_parquet(prep / f"{p}_target.parquet")
        parts.append(pd.concat([i.reset_index(drop=True), X.reset_index(drop=True), y.reset_index(drop=True)], axis=1))
    train = pd.concat(parts, ignore_index=True)
    test = pd.concat([pd.read_parquet(prep / "test_ids.parquet").reset_index(drop=True),
                      pd.read_parquet(prep / "test_features.parquet").reset_index(drop=True)], axis=1)
    # порядок строк test — как в исходном test.csv (по id)
    raw_test_ids = pd.read_csv(task_dir / "test.csv", usecols=ids)
    test = raw_test_ids.merge(test, on=ids, how="left", validate="one_to_one")
    data_out.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_out / "train.csv", index=False)
    test.to_csv(data_out / "test.csv", index=False)
    shutil.copy(task_dir / "sample_submission.csv", data_out / "sample_submission.csv")
    info = {"train_rows": len(train), "test_rows": len(test), "n_features": train.shape[1] - len(ids) - 1,
            "test_nan_share": float(test.drop(columns=ids).isna().to_numpy().mean())}
    if info["test_rows"] != len(raw_test_ids):
        raise RuntimeError("test после подготовки не совпадает по строкам с исходным test.csv")
    return info


# ============================================================================ 2. Выбор модели и лосса
def run_selection(data_dir: Path, out: Path, task: dict, prep_dir: Path, use_llm: bool, probe_s: int) -> dict:
    from model_selection.stage import run_stage
    eda = {}
    rep = prep_dir / "preparation_report.json"
    if rep.exists():
        payload = json.loads(rep.read_text(encoding="utf-8")).get("payload", {})
        eda = {k: payload.get(k) for k in ("new_features_created", "target_correlations", "high_correlations")}
    stage_input = {
        "task": {"task_type": "regression", "target": task["target"], "id_col": task["id_columns"][0],
                 "target_metric": task["metric"]},
        "paths": {"train": str(data_dir / "train.csv"), "test": str(data_dir / "test.csv")},
        "hints": {"categorical_cols": [], "drop_cols": task["id_columns"][1:]},
        "budget": {"probe_seconds": probe_s, "probe_max_rows": 30000, "probe_folds": 3},
        "eda": {"from_preparation": eda},
    }
    save(out / "input.json", stage_input)
    return run_stage(stage_input, out_path=str(out / "output.json"), use_llm=use_llm)


# ============================================================================ 3. Адаптер → Макс
def selection_to_training_input(sel: dict, task: dict, data_dir: Path) -> tuple[dict, list[str]]:
    warnings, models = [], []
    metric = sel["target_metric"]
    if metric not in MAX_METRICS:
        raise RuntimeError(f"Метрика {metric} не поддержана этапом обучения (есть {sorted(MAX_METRICS)}). "
                           f"Нужно расширить agent/training_backend.metric_score.")
    base = sel["loss_policy"]["base"]
    for c in sel["candidates"]:
        fam = FAMILY_MAP.get(c["model_id"])
        if fam is None:
            warnings.append(f"{c['model_id']}: у этапа обучения нет шаблона — пропущен")
            continue
        params = dict(c["loss_native"])
        note = ""
        if c["target_transform"] != "none":
            # Бэкенд Макса не умеет трансформ таргета — откатываемся к базовой паре метрики.
            warnings.append(f"{c['model_id']}: трансформ {c['target_transform']} не поддержан бэкендом — "
                            f"обучаю с базовым лоссом {base['loss']}")
            from model_selection.builders import loss_kwargs
            from model_selection.registry import supports
            if base["transform"] != "none" or not supports(c["model_id"], base["loss"]):
                warnings.append(f"{c['model_id']}: базовая пара тоже требует трансформ — пропущен")
                continue
            params = loss_kwargs(c["model_id"], base["loss"])
            note = " (трансформ таргета снят)"
        models.append({"family": fam, "parameters": params,
                       "rationale": f"[{c['role']}, priority {c['priority']}] {c['rationale']}{note}"})
    if not models:
        raise RuntimeError("После адаптации не осталось ни одной модели")
    diag = sel["diagnostics"]
    eda = {"target": diag["target"], "features": {k: v for k, v in diag["features"].items()
                                                  if k not in ("categorical_cols",)},
           "probe": [{k: r[k] for k in ("id", "metric_mean", "metric_std")} for r in sel["probe"]["results"]],
           "selection_reasoning": sel["reasoning"], "selection_source": sel["source"]}
    ti = {"schema_version": 1, "task_type": "regression",
          "files": {"train": "train.csv", "test": "test.csv", "sample_submission": "sample_submission.csv"},
          "target": task["target"], "id_columns": task["id_columns"], "primary_metric": metric,
          "selected_models": models[:12], "eda": eda,
          "validation_constraints": {"strategy": "random", "group_columns": [], "time_column": None},
          "submission": {"filename": "submission.csv", "prediction_column": task["target"]}}
    return ti, warnings


# ============================================================================ 4. Макс
# Разумные стартовые параметры для офлайн-режима (без Qwen). С Qwen их выбирает её планировщик.
OFFLINE_PARAMS = {
    "lightgbm": {"n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 30,
                 "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0},
    "catboost": {"iterations": 2000, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0},
    "xgboost": {"n_estimators": 1500, "learning_rate": 0.03, "max_depth": 8, "subsample": 0.8,
                "colsample_bytree": 0.7, "min_child_weight": 5},
    "extra_trees": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.5},
    "random_forest": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.5},
    "ridge": {"alpha": 1.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5},
    "mlp": {"hidden_layer_sizes": [128, 64], "early_stopping": True, "max_iter": 200, "alpha": 1e-4},
    "hist_gradient_boosting": {"max_iter": 1000, "learning_rate": 0.05},
    "sgd": {"average": True, "max_iter": 2000},
}
OFFLINE_SECONDS = {"lightgbm": 120, "catboost": 240, "xgboost": 180, "extra_trees": 120, "random_forest": 180,
                   "ridge": 15, "elasticnet": 30, "mlp": 200, "hist_gradient_boosting": 60, "sgd": 20}


class OfflineAgents:
    """Детерминированная замена Qwen-ролей Макса для прогона без API.
    Тот же интерфейс, что у agent.agents.JSONAgents: ask(schema, prompt, role) -> экземпляр схемы."""

    def __init__(self, selected_models: list[dict]):
        self.selected = selected_models
        self.deadline = None
        self.root = self
        self.log_directory = None
        self.calls = []

    async def ask(self, schema, prompt, role, **_):
        from agent.training_contracts import (TrainingAnalysis, TrainingCandidate, TrainingNextPlan,
                                              TrainingPlan, ValidationDecision)
        self.calls.append(role)
        name = schema.__name__
        if name == "ValidationDecision":
            return ValidationDecision(validation_fraction=0.2, inner_validation="holdout", folds=3, seed=42,
                                      rationale="offline: 20% внешний holdout; строк достаточно для устойчивой оценки")
        if name == "TrainingPlan":
            cands = [TrainingCandidate(name=f"{m['family']}_{i}", model_index=i,
                                       parameters=OFFLINE_PARAMS.get(m["family"], {}),
                                       rationale="offline: стартовая конфигурация семейства",
                                       estimated_seconds=OFFLINE_SECONDS.get(m["family"], 120))
                     for i, m in enumerate(self.selected)]
            return TrainingPlan(candidates=cands, rationale="offline: по одной конфигурации на модель")
        if name == "TrainingAnalysis":
            return TrainingAnalysis(recommendation="stop", rationale="offline: без LLM поиск не продолжаем")
        if name == "TrainingNextPlan":
            return TrainingNextPlan(stop=True, rationale="offline: без LLM поиск не продолжаем")
        raise RuntimeError(f"OfflineAgents: неизвестная схема {name}")


def run_training_stage(input_path: Path, out: Path, use_llm: bool, selected: list[dict],
                       time_budget: float, trial_timeout: float, max_trials: int) -> dict:
    from agent.training_component import run_training
    agents = None if use_llm else OfflineAgents(selected)
    return asyncio.run(run_training(input_path, out, time_budget=time_budget, trial_timeout=trial_timeout,
                                    max_trials=max_trials, agents=agents))


# ============================================================================ 5. Метрики и проверки
def evaluate(task: dict, data_dir: Path, train_out: Path, report: dict, task_dir: Path) -> dict:
    from model_selection.metrics import score
    metric, tgt, ids = report["primary_metric"], task["target"], task["id_columns"]
    trials = [{"trial": t.get("trial_id"), "family": t.get("family"), "status": t.get("status"),
               "score": t.get("score"), "train_score": t.get("training_score"),
               "seconds": round(t.get("wall_duration_seconds") or t.get("duration_seconds") or 0, 1)}
              for t in report.get("trials", [])]
    # эталон на ТОМ ЖЕ внешнем split: константа (среднее по fit)
    train = pd.read_csv(data_dir / "train.csv")
    with np.load(train_out / "split.npz") as s:
        fit, valid = s["fit"], s["valid"]
    y = train[tgt].to_numpy(float)
    dummy = score(metric, y[valid], np.full(len(valid), y[fit].mean()))
    winner = dict(report.get("winner") or {})
    best = winner.get("score")
    winner["family"] = next((t["family"] for t in trials if t["trial"] == winner.get("trial_id")), None)
    # проверка submission
    sub_path = train_out / "submission.csv"
    sample = pd.read_csv(task_dir / "sample_submission.csv")
    checks = {"exists": sub_path.exists()}
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        checks.update(rows_match=len(sub) == len(sample), columns=list(sub.columns),
                      ids_order_match=bool((sub[ids].astype(str).values == sample[ids].astype(str).values).all()),
                      finite=bool(np.isfinite(sub[tgt]).all()),
                      pred_mean=round(float(sub[tgt].mean()), 3), train_target_mean=round(float(y.mean()), 3))
    rel = None if best is None else round(1 - best / dummy, 4) if metric != "r2" else None
    return {"metric": metric, "winner_family": winner.get("family"), "winner_validation_score": best,
            "dummy_validation_score": round(float(dummy), 4), "gain_vs_dummy": rel,
            "validation_rows": int(len(valid)), "fit_rows": int(len(fit)), "trials": trials,
            "final_refit_status": report.get("status"), "submission_checks": checks,
            "note": "Оценка на внешней отложенной выборке этапа обучения. Балл платформы "
                    "S=100·max(0,1−RMSE/B_RMSE) требует RMSE бейзлайна организаторов на их тесте."}


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--no-llm", action="store_true", help="без Qwen: правила + офлайн-планировщик")
    ap.add_argument("--total-budget", type=float, default=3 * 3600 - 600,
                    help="общий бюджет, с запасом 10 мин до лимита платформы 3 ч")
    ap.add_argument("--prep-budget", type=int, default=900)
    ap.add_argument("--probe-budget", type=int, default=300)
    ap.add_argument("--trial-timeout", type=float, default=600)
    ap.add_argument("--max-trials", type=int, default=12)
    a = ap.parse_args(argv)

    t0 = time.monotonic()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    import os
    use_llm = not a.no_llm
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Нет OPENAI_API_KEY: положите ключ в .env в корне репозитория или запустите с --no-llm")

    task_dir = a.task_dir.resolve()
    out = (a.output_dir or ROOT / "artifacts" / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    timings, warnings = {}, []
    log(f"Выход: {out} | Qwen: {'да' if use_llm else 'нет'}")

    ts = time.monotonic()
    task = read_task(task_dir)
    save(out / "00_task" / "task.json", task)
    log(f"0. Задача: target={task['target']} id={task['id_columns']} metric={task['metric']} ({task['metric_source']})")
    timings["0_task"] = round(time.monotonic() - ts, 1)

    ts = time.monotonic()
    prep = run_preparation(task_dir, out / "01_preparation", task, use_llm, a.prep_budget)
    data_info = assemble_tables(out / "01_preparation", out / "02_data", task, task_dir)
    save(out / "02_data" / "info.json", data_info)
    timings["1_preparation"] = round(time.monotonic() - ts, 1)
    log(f"1. Подготовка данных: {prep['status']}, признаков {data_info['n_features']}, "
        f"train {data_info['train_rows']}, test {data_info['test_rows']} ({timings['1_preparation']} с)")

    ts = time.monotonic()
    sel = run_selection(out / "02_data", out / "03_model_selection", task, out / "01_preparation", use_llm,
                        a.probe_budget)
    timings["2_model_selection"] = round(time.monotonic() - ts, 1)
    warnings += sel["warnings"]
    log(f"2. Выбор модели: source={sel['source']}, кандидаты: "
        + ", ".join(f"{c['model_id']}/{c['loss']}/{c['target_transform']}" for c in sel["candidates"])
        + f" ({timings['2_model_selection']} с)")
    if sel["validation_errors"]:
        log(f"   ошибки валидации ответа Qwen: {len(sel['validation_errors'])} (см. 03_model_selection/output.json)")

    ti, w = selection_to_training_input(sel, task, out / "02_data")
    warnings += w
    for f in ("train.csv", "test.csv", "sample_submission.csv"):
        ti["files"][f.split(".")[0] if f != "sample_submission.csv" else "sample_submission"] = \
            os.path.relpath(out / "02_data" / f, out / "04_training_input")
    save(out / "04_training_input" / "input.json", ti)

    remaining = a.total_budget - (time.monotonic() - t0) - 60
    log(f"3. Обучение: {len(ti['selected_models'])} моделей, бюджет {int(remaining)} с")
    ts = time.monotonic()
    report = run_training_stage(out / "04_training_input" / "input.json", out / "05_training", use_llm,
                                ti["selected_models"], remaining, a.trial_timeout, a.max_trials)
    timings["3_training"] = round(time.monotonic() - ts, 1)

    metrics = evaluate(task, out / "02_data", out / "05_training", report, task_dir)
    if (out / "05_training" / "submission.csv").exists():
        shutil.copy(out / "05_training" / "submission.csv", out / "submission.csv")
    timings["total"] = round(time.monotonic() - t0, 1)
    summary = {"task": task, "timings_s": timings, "metrics": metrics, "warnings": warnings,
               "llm": use_llm, "stop_reason": (report.get("coordinator") or {}).get("stop_reason")}
    save(out / "pipeline_report.json", summary)

    log("=" * 70)
    log(f"Метрика {metrics['metric']} на отложенной выборке ({metrics['validation_rows']} строк):")
    for t in sorted(metrics["trials"], key=lambda t: (t["score"] is None, t["score"] or 0)):
        log(f"   {t['family']:<24} {t['status']:<10} score={t['score']}  train={t['train_score']}  {t['seconds']} с")
    log(f"   {'константа (эталон)':<24} {'':<10} score={metrics['dummy_validation_score']}")
    log(f"Победитель: {metrics['winner_family']} = {metrics['winner_validation_score']}, "
        f"выигрыш к константе {metrics['gain_vs_dummy']}")
    log(f"Submission: {metrics['submission_checks']}")
    log(f"Время, с: {timings}")
    for w in warnings:
        log(f"   ⚠ {w}")
    return summary


if __name__ == "__main__":
    main()
