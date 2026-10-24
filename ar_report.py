"""Отчёт запуска: report.json, leaderboard.csv, SUMMARY.md и самодостаточный dashboard.html (открывается без сети)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_report(mem, run_dir: Path, task, audit: dict, design: dict, ens: dict, checks: dict | None, budget) -> dict:
    exps = mem.state["experiments"]
    best = mem.best()
    lb = pd.DataFrame([{"id": e["id"], "source": e.get("source"), "summary": e.get("summary"), "cv": e.get("cv"),
                        "cv_std": e.get("cv_std"), "train_cv": e.get("train_cv"), "folds": e.get("folds"),
                        "delta_best_pct": e.get("delta_best_pct"), "verdict": e.get("verdict"),
                        "seconds": e.get("seconds"), "hypothesis": e.get("hypothesis")} for e in exps])
    if len(lb):
        lb.sort_values("cv", ascending=mem.direction == "minimize").to_csv(run_dir / "leaderboard.csv", index=False)
    base = next((e for e in exps if e.get("verdict") == "baseline"), None)
    final_cv = ens.get("ensemble_score") if ens.get("used") else (best and best["cv"])
    report = {
        "task": {"target": task.target, "metric": task.metric, "metric_source": task.metric_source, "id": task.id_cols},
        "stages": mem.state["stages"],
        "facts": {k: v["value"] for k, v in mem.state["facts"].items()},
        "validation": {"kind": design["kind"], "folds": design["labels"], "sizes": design["sizes"]},
        "baseline": base and {"id": base["id"], "cv": base["cv"], "summary": base["summary"]},
        "best_single": best and {"id": best["id"], "cv": best["cv"], "summary": best["summary"], "segments": best.get("segments"),
                                 "top_features": best.get("top_features"), "folds": best.get("folds"), "train_cv": best.get("train_cv")},
        "ensemble": ens,
        "final_cv": final_cv,
        "improvement_vs_baseline_pct": round((base["cv"] - final_cv) / base["cv"] * 100, 2) if base and final_cv and mem.direction == "minimize" else None,
        "submission_checks": checks,
        "experiments": [{k: e.get(k) for k in ("id", "parent", "source", "hypothesis", "why", "summary", "status", "cv", "cv_std",
                                               "folds", "train_cv", "delta_best_pct", "delta_parent_pct", "verdict", "seconds",
                                               "analysis", "analysis_verdict", "error", "segments")} for e in exps],
        "insights": mem.state["insights"], "backlog": mem.state["backlog"],
        "rejected_proposals": mem.state["rejected_proposals"], "llm": mem.state["llm"],
        "elapsed_s": round(budget.elapsed(), 1),
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    mem.save()
    (run_dir / "SUMMARY.md").write_text(_summary_md(report), encoding="utf-8")
    (run_dir / "dashboard.html").write_text(render_dashboard(report, mem, task), encoding="utf-8")
    return report


def _summary_md(r: dict) -> str:
    L = [f"# AutoResearch: {r['task']['target']} ({r['task']['metric']})", ""]
    if r["baseline"]:
        L.append(f"- Базовая линия: **{r['baseline']['cv']:.4f}** ({r['baseline']['summary']})")
    if r["best_single"]:
        L.append(f"- Лучшая модель: **{r['best_single']['cv']:.4f}** ({r['best_single']['summary']})")
    L.append(f"- Финал (CV): **{r['final_cv']}**; улучшение к базовой линии: {r['improvement_vs_baseline_pct']}%")
    L.append(f"- Валидация: {r['validation']['kind']} {r['validation']['folds']}")
    L.append(f"- Экспериментов: {len(r['experiments'])}; вызовов Qwen: {r['llm']['calls']}; время {r['elapsed_s']} с")
    L += ["", "| id | источник | эксперимент | cv | Δ к лучшему % | вердикт |", "|---|---|---|---|---|---|"]
    for e in r["experiments"]:
        L.append(f"| {e['id']} | {e['source']} | {e['summary']} | {e['cv'] and round(e['cv'], 4)} | {e['delta_best_pct']} | {e['verdict']} |")
    L += ["", "## Выводы", *[f"- {i['text']} _({i['source']})_" for i in r["insights"]]]
    return "\n".join(L) + "\n"


TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def dashboard_data(report: dict, mem, task, extra: dict | None = None) -> dict:
    """Данные страницы: отчёт + название задачи, единицы таргета и память в том виде, как её видит Qwen."""
    d = dict(report)
    first = next((ln.lstrip("# ").strip() for ln in (task.description or "").splitlines() if ln.startswith("#")), "")
    d["title"] = first or f"Прогноз {task.target}"
    unit = next((str(r.get("unit", "")) for r in task.dictionary if str(r.get("feature")) == task.target), "")
    d["unit"] = unit
    ent = (report["facts"].get("audit.entity") or {})
    tc = report["facts"].get("audit.time") or {}
    d["lede"] = (f"Таргет {task.target}" + (f" ({unit})" if unit else "") + f", метрика {task.metric.upper()}. "
                 + (f"Test — период {tc.get('test_periods')}, train — {tc.get('train_periods')}. " if tc else "")
                 + (f"Объекты повторяются по периодам: у {round(ent.get('test_rows_with_history', 0) * 100)}% строк test есть своя история." if ent else ""))
    budget = {"seconds_left_for_research": 0, "experiments_done": len(report["experiments"])}
    d["memory_sample"] = mem.render(mem_chars(), budget)
    d.update(extra or {})
    return d


def mem_chars() -> int:
    from ar_config import Settings
    return Settings().memory_chars


def render_dashboard(report: dict, mem, task, extra: dict | None = None, standalone: bool = True) -> str:
    data = json.dumps(dashboard_data(report, mem, task, extra), ensure_ascii=False, default=str).replace("</", "<\\/")
    body = __import__("ar_assets").DASHBOARD.replace("__DATA__", data)
    if standalone:
        return ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + body + "</body></html>")
    return body
