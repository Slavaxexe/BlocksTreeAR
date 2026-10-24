"""
Этап 0. Понимание задачи.

Код находит файлы, таргет, id и метрику детерминированно (шаблон ответа + словарь данных + regex по task.md).
Qwen (если доступна) читает текст задачи и пишет в память предметную сводку и риски — это контекст для
всех следующих решений. Метрику Qwen может подтвердить, но только с цитатой из task.md (проверяется кодом).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from ar_llm import LLM
from ar_memory import Memory

METRIC_PATTERNS = [
    ("rmsle", r"\bRMSLE\b|root mean squared log"), ("msle", r"\bMSLE\b"),
    ("rmse", r"\bRMSE\b|root mean squared error"), ("mse", r"\bMSE\b|mean squared error"),
    ("mae", r"\bMAE\b|mean absolute error"), ("mape", r"\bMAPE\b|mean absolute percentage"),
    ("smape", r"\bSMAPE\b"), ("r2", r"\bR\^?2\b|R²|коэффициент детерминации"),
]
SUPPORTED_METRICS = {"rmse", "mse", "mae", "r2", "rmsle", "msle", "mape", "smape"}


class TaskNotSupported(RuntimeError):
    pass


@dataclass
class TaskSpec:
    task_md: str
    task_dir: str
    files: dict
    target: str
    id_cols: list
    metric: str
    metric_source: str
    task_type: str = "regression"
    submission_path: str = ""
    description: str = ""
    dictionary: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class TaskReading(BaseModel):
    domain_summary: str = Field(..., description="2–4 предложения: что за объекты, что предсказываем, единицы")
    metric: str = Field(..., description="метрика из текста задачи: rmse|mse|mae|r2|rmsle|msle|mape|smape")
    metric_quote: str = Field(..., description="точная цитата из task.md, где названа метрика")
    risks: list[str] = Field(default_factory=list, description="до 5 рисков: сдвиг, утечки, выбросы, повторы объектов")
    important_columns: list[str] = Field(default_factory=list, description="до 10 колонок, важных по смыслу")


def _find_files(task_md: Path, text: str) -> dict:
    root = task_md.parent
    found = {}
    for rel in re.findall(r"[\w./-]+\.csv", text):
        p = (root / rel).resolve()
        if p.exists():
            name = p.name
            for key in ("train", "test", "sample_submission", "data_dictionary"):
                if name == f"{key}.csv" and key not in found:
                    found[key] = str(p)
    for key in ("train", "test", "sample_submission", "data_dictionary"):
        if key not in found:
            hits = sorted(root.rglob(f"{key}.csv"), key=lambda p: len(p.parts))
            if hits:
                found[key] = str(hits[0])
    missing = [k for k in ("train", "test") if k not in found]
    if missing:
        raise FileNotFoundError(f"не найдены файлы {missing} рядом с {task_md}")
    return found


def read_task(path: str | Path, mem: Memory, llm: LLM, deadline: float) -> TaskSpec:
    p = Path(path).resolve()
    task_md = p / "task.md" if p.is_dir() else p
    text = task_md.read_text(encoding="utf-8") if task_md.exists() else ""
    files = _find_files(task_md, text)
    test_cols = list(pd.read_csv(files["test"], nrows=2).columns)
    train_cols = list(pd.read_csv(files["train"], nrows=2).columns)

    dictionary = []
    if "data_dictionary" in files:
        dd = pd.read_csv(files["data_dictionary"])
        dictionary = dd.fillna("").to_dict("records")
    roles = {str(r.get("feature")): str(r.get("role", "")).lower() for r in dictionary}

    if "sample_submission" in files:
        sample_cols = list(pd.read_csv(files["sample_submission"], nrows=2).columns)
        id_cols = [c for c in sample_cols if c in test_cols]
        targets = [c for c in sample_cols if c not in id_cols]
    else:
        id_cols = [c for c, r in roles.items() if r == "id"] or [c for c in test_cols if c.lower() == "id"]
        targets = [c for c in train_cols if c not in test_cols]
    if len(targets) != 1:
        raise TaskNotSupported(f"ожидался один таргет, найдено {targets}")
    target = targets[0]

    y = pd.read_csv(files["train"], usecols=[target])[target]
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
        raise TaskNotSupported(f"таргет {target} не похож на регрессию (dtype={y.dtype}, уникальных={y.nunique()})")

    metric, metric_source = None, "fallback"
    metric_lines = [ln for ln in text.splitlines() if re.search(r"метрик|metric|оценк", ln, re.I)]
    for scope in (metric_lines, [text]):
        for name, pat in METRIC_PATTERNS:
            if any(re.search(pat, s, re.I) for s in scope):
                metric, metric_source = name, "task_regex"
                break
        if metric:
            break

    spec = TaskSpec(task_md=str(task_md), task_dir=str(task_md.parent), files=files, target=target,
                    id_cols=id_cols, metric=metric or "rmse", metric_source=metric_source,
                    submission_path=str(task_md.parent / "submission.csv"),
                    description=text[:4000], dictionary=dictionary)

    # --- Qwen: предметная сводка, риски и независимое подтверждение метрики
    if llm.enabled and text:
        def validate(r: TaskReading) -> list[str]:
            errs = []
            if r.metric.lower() not in SUPPORTED_METRICS:
                errs.append(f"metric должна быть одной из {sorted(SUPPORTED_METRICS)}")
            if " ".join(r.metric_quote.split()).lower() not in " ".join(text.split()).lower():
                errs.append("metric_quote должна быть точной цитатой из task.md")
            return errs
        cols = [{"feature": d.get("feature"), "role": d.get("role"), "description": str(d.get("description_ru", d.get("description", "")))[:90]}
                for d in dictionary][:80]
        r = llm.ask("task_reader",
                    "Ты аналитик ML-задачи. Прочитай условие и словарь данных. Ответь одним JSON по схеме: "
                    + str(TaskReading.model_json_schema()["properties"]),
                    f"TASK.md:\n{text[:6000]}\n\nDATA_DICTIONARY:\n{cols}", TaskReading,
                    deadline=deadline, validator=validate)
        if r is not None:
            mem.fact("domain_summary", r.domain_summary, "qwen")
            mem.fact("domain_risks", r.risks[:5], "qwen")
            mem.fact("important_columns_by_meaning", r.important_columns[:10], "qwen")
            if metric is None:
                spec.metric, spec.metric_source = r.metric.lower(), "qwen_quote"
            elif r.metric.lower() != metric:
                mem.log("metric_disagreement", regex=metric, qwen=r.metric, quote=r.metric_quote)

    if spec.metric not in SUPPORTED_METRICS:
        raise TaskNotSupported(f"метрика {spec.metric} не поддержана")
    for k in ("target", "id_cols", "metric", "metric_source", "submission_path"):
        mem.fact(f"task.{k}", getattr(spec, k))
    mem.fact("direction", "maximize" if spec.metric == "r2" else "minimize")
    return spec
