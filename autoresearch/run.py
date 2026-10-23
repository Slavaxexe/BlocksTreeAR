#!/usr/bin/env python3
"""
Сквозной AutoResearch-пайплайн регрессии.

    python -m autoresearch.run data/regression/task.md            # с Qwen (.env в корне репозитория)
    python -m autoresearch.run data/regression/task.md --no-llm   # без Qwen: эвристический исследователь

Этапы и что пишется в папку запуска (artifacts/autoresearch_<время>/):
  0 задача            → state.json/facts.task.*
  1 аудит данных      → facts.audit.*  (время, ключ объекта, сдвиг, хвост таргета)
  2 валидация         → folds.npz      (фиксирована для всех экспериментов)
  3 признаки          → features/*.parquet (блоки; история по таргету строго из прошлого)
  4 базовая линия + страховочный submission
  5 цикл AutoResearch → experiments/exp_NNN/ (spec, result, oof), journal.jsonl — каждый шаг и каждый ответ Qwen
  6 ансамбль          → ensemble.json
  7 финал             → submission.csv (и рядом с task.md), report.json, leaderboard.csv, dashboard.html
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_selection.diagnostics import diagnose_target  # noqa: E402
from model_selection.rules import loss_policy  # noqa: E402

from .audit import run_audit  # noqa: E402
from .config import Budget, Settings, load_env  # noqa: E402
from .ensemble import build_ensemble  # noqa: E402
from .features import FeatureStore  # noqa: E402
from .final import refit_members, write_submission  # noqa: E402
from .llm import LLM  # noqa: E402
from .memory import Memory  # noqa: E402
from .report import write_report  # noqa: E402
from .research import Lab, research_loop  # noqa: E402
from .task import read_task  # noqa: E402
from .validation import design_folds, save_folds  # noqa: E402


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(task_path: str, out_dir: str | None = None, no_llm: bool = False, total_budget: float = 3 * 3600,
        max_experiments: int = 60, llm_fn=None, write_next_to_task: bool = True, llm_calls: int | None = None) -> dict:
    load_env(ROOT)
    s = Settings(llm_enabled=not no_llm, max_experiments=max_experiments)
    if llm_calls:
        s.llm_max_calls = llm_calls
    budget = Budget(total_s=total_budget)
    run_dir = Path(out_dir).resolve() if out_dir else ROOT / "artifacts" / f"autoresearch_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mem = Memory(run_dir)
    llm = LLM(s, mem, fn=llm_fn)
    mem.fact("run", {"budget_s": total_budget, "n_jobs": s.n_jobs, "llm": llm.enabled, "model": s.model if llm.enabled else None})
    log(f"Запуск: {run_dir} | Qwen: {'да' if llm.enabled else 'нет'} | CPU: {s.n_jobs} | бюджет {int(total_budget)} с")

    # ------------------------------------------------------------------ 0. задача
    t = time.monotonic()
    task = read_task(task_path, mem, llm, budget.deadline)
    mem.stage("0_task", "ok", {"target": task.target, "id": task.id_cols, "metric": task.metric,
                               "metric_source": task.metric_source}, time.monotonic() - t)
    log(f"0. Задача: target={task.target}, id={task.id_cols}, метрика={task.metric} ({task.metric_source})")

    # ------------------------------------------------------------------ 1. аудит
    t = time.monotonic()
    train = pd.read_csv(task.files["train"])
    test = pd.read_csv(task.files["test"])
    audit = run_audit(train, test, task.target, task.id_cols, task.dictionary, mem, llm, budget.deadline, s.seed)
    tc = audit["time"]["col"] if audit["time"] else None
    ent = audit["entity"]["key"] if audit["entity"] else None
    mem.stage("1_audit", "ok", {"time_col": tc, "entity_key": ent, "adversarial_auc": (audit["adversarial"] or {}).get("auc"),
                                "target_top1pct_share": audit["target"]["top1pct_share_of_sq_dev"]}, time.monotonic() - t)
    log(f"1. Аудит: время={tc}, объект={ent} "
        f"(история у {audit['entity'] and audit['entity']['test_rows_with_history']} строк test), "
        f"сдвиг AUC={(audit['adversarial'] or {}).get('auc')}, хвост 1%={audit['target']['top1pct_share_of_sq_dev']}")

    # ------------------------------------------------------------------ 2. валидация
    t = time.monotonic()
    design = design_folds(train, tc, ent, s.seed)
    folds_path = run_dir / "folds.npz"
    folds_hash = save_folds(design, folds_path)
    mem.fact("validation", {"kind": design["kind"], "folds": design["labels"], "sizes": design["sizes"],
                            "weights": design["weights"], "sha256": folds_hash,
                            "why": "test — следующий период; фолды: учимся на прошлом, проверяем на следующем периоде"
                                   if tc else "нет колонки времени — k-fold (по группам объекта, если он найден)"})
    mem.stage("2_validation", "ok", {"kind": design["kind"], "folds": design["labels"]}, time.monotonic() - t)
    log(f"2. Валидация: {design['kind']}: {design['labels']}")

    # ------------------------------------------------------------------ 3. признаки
    t = time.monotonic()
    feat_cols = [c for c in test.columns if c not in task.id_cols]
    all_df = pd.concat([train, test], ignore_index=True)
    store = FeatureStore(run_dir / "features", all_df, len(train), task.target, task.id_cols, audit["cat_cols"],
                         audit["num_cols"], tc, ent, audit["group_cols"], s.seed)
    built = {}
    for b in store.available():
        built[b] = int(store.get(b).shape[1])
    mem.fact("feature_blocks", built)
    mem.stage("3_features", "ok", built, time.monotonic() - t)
    log(f"3. Блоки признаков: {built}")

    y_all = all_df[task.target].to_numpy(float)
    np.save(run_dir / "target.npy", y_all)
    ctx = {"cache_dir": str(store.dir), "folds_npz": str(folds_path), "target_npy": str(run_dir / "target.npy"),
           "n_train": len(train), "n_jobs": s.n_jobs, "seed": s.seed, "metric": task.metric,
           "fold_weights": design["weights"], "time_scale": max(0.5, 8 / max(1, s.n_jobs) * 0.5)}
    if tc:
        np.save(run_dir / "period.npy", all_df[tc].to_numpy(float))
        ctx["period_npy"] = str(run_dir / "period.npy")
    if audit["group_cols"]:
        np.save(run_dir / "segment.npy", all_df[audit["group_cols"][0]].astype(str).to_numpy(), allow_pickle=True)
        ctx["segment_col_npy"] = str(run_dir / "segment.npy")
    mem.fact("loss_policy", loss_policy(task.metric, diagnose_target(train[task.target])))

    # ------------------------------------------------------------------ 4–5. исследование
    lab = Lab(mem, s, budget, store, ctx, run_dir, all_df[feat_cols], task.target)
    t = time.monotonic()
    method = (Path(__file__).with_name("METHOD.md")).read_text(encoding="utf-8")
    test_ids = test[task.id_cols].copy()
    sub_paths = [run_dir / "submission.csv"] + ([Path(task.submission_path)] if write_next_to_task else [])

    def safety(rec):
        """Страховочный ответ сразу после базовой линии: даже если дальше всё упадёт, submission уже есть."""
        if rec.get("status") != "ok":
            return
        pred, info = refit_members({rec["id"]: 1.0}, mem, ctx, run_dir / "safety", budget.deadline, log)
        if pred is not None:
            chk = write_submission(pred, test_ids, task.id_cols, task.target, task.files.get("sample_submission"),
                                   y_all[: len(train)], sub_paths)
            mem.log("safety_submission", checks=chk)
            log(f"   страховочный submission записан: ok={chk['ok']}")

    log("4–5. Исследование:")
    stop = research_loop(lab, llm, method, log=log, on_baseline=safety)
    best = mem.best()
    mem.stage("5_research", "ok", {"stop_reason": stop, "experiments": len(mem.state["experiments"]),
                                   "best": best and {"id": best["id"], "cv": best["cv"], "summary": best["summary"]},
                                   "llm_calls": mem.state["llm"]["calls"]}, time.monotonic() - t)
    log(f"   стоп: {stop}; лучший {best and best['id']} cv={best and round(best['cv'], 4)}")

    # ------------------------------------------------------------------ 6. ансамбль
    t = time.monotonic()
    ens = build_ensemble(mem, run_dir / "experiments", y_all[: len(train)], task.metric, s.min_rel_improvement)
    (run_dir / "ensemble.json").write_text(json.dumps(ens, ensure_ascii=False, indent=1))
    mem.stage("6_ensemble", "ok", {k: ens.get(k) for k in ("used", "weights", "ensemble_score", "best_single_score", "rel_gain_pct")},
              time.monotonic() - t)
    log(f"6. Ансамбль: {ens.get('reason')}; веса {ens.get('weights')}; OOF {ens.get('ensemble_score')} vs лучшая {ens.get('best_single_score')}")

    # ------------------------------------------------------------------ 7. финал
    t = time.monotonic()
    pred, info = refit_members(ens["weights"], mem, ctx, run_dir, budget.deadline + budget.safety_s - 60, log)
    if pred is None and ens["weights"] != {best["id"]: 1.0}:
        pred, info2 = refit_members({best["id"]: 1.0}, mem, ctx, run_dir / "final_fallback", budget.deadline + budget.safety_s - 60, log)
        info.update(info2)
    checks = None
    if pred is not None:
        checks = write_submission(pred, test_ids, task.id_cols, task.target, task.files.get("sample_submission"),
                                  y_all[: len(train)], sub_paths)
    mem.stage("7_final", "ok" if checks and checks["ok"] else "fallback_safety", {"members": info, "checks": checks},
              time.monotonic() - t)
    log(f"7. Финал: {checks}")
    report = write_report(mem, run_dir, task, audit, design, ens, checks, budget)
    log(f"Готово за {int(budget.elapsed())} с. Отчёт: {run_dir / 'dashboard.html'}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", help="путь к task.md (или каталогу с ним)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--budget", type=float, default=3 * 3600, help="секунды на всю задачу (лимит платформы — 3 ч)")
    ap.add_argument("--max-experiments", type=int, default=60)
    ap.add_argument("--llm-calls", type=int, default=None, help="лимит вызовов Qwen за запуск (по умолчанию 45)")
    a = ap.parse_args(argv)
    try:
        run(a.task, a.out, a.no_llm, a.budget, a.max_experiments, llm_calls=a.llm_calls)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
