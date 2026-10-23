"""
Память исследования. Qwen не хранит состояние между вызовами, поэтому вся история живёт здесь:

  journal.jsonl — каждый шаг (факты, решения, эксперименты, сырые ответы Qwen), только дописывается;
  state.json    — сжатое текущее состояние: факты, лидерборд, все гипотезы с вердиктами, выводы, бэклог идей.

Перед каждым вызовом Qwen получает render(): компактный JSON, где есть ВСЁ, что было сделано
(каждая гипотеза одной строкой), подробно — последние эксперименты и лучшие результаты.
Так модель «помнит каждый шаг», не упираясь в контекст.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


def _short(text: Any, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _norm(text: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", " ", str(text).lower()).strip()


class Memory:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.dir / "journal.jsonl"
        self.step = 0
        self.t0 = time.time()
        self.state: dict[str, Any] = {
            "facts": {}, "stages": {}, "experiments": [], "insights": [], "backlog": [],
            "rejected_proposals": [], "llm": {"calls": 0, "failures": 0, "prompt_tokens": 0,
                                              "completion_tokens": 0, "disabled_reason": None},
        }

    # ------------------------------------------------------------------ запись
    def log(self, kind: str, **data):
        self.step += 1
        rec = {"step": self.step, "t": round(time.time() - self.t0, 2), "kind": kind, **data}
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return self.step

    def fact(self, key: str, value: Any, source: str = "code"):
        self.state["facts"][key] = {"value": value, "source": source, "step": self.step}
        self.log("fact", key=key, value=value, source=source)

    def get(self, key: str, default=None):
        f = self.state["facts"].get(key)
        return default if f is None else f["value"]

    def stage(self, name: str, status: str, summary: dict | None = None, seconds: float | None = None):
        self.state["stages"][name] = {"status": status, "summary": summary or {},
                                      "seconds": None if seconds is None else round(seconds, 1)}
        self.log("stage", name=name, status=status, summary=summary, seconds=seconds)
        self.save()

    def add_experiment(self, rec: dict):
        self.state["experiments"].append(rec)
        self.log("experiment", **rec)
        self.save()

    def annotate_experiment(self, exp_id: str, **fields):
        for e in self.state["experiments"]:
            if e["id"] == exp_id:
                e.update(fields)
        self.log("experiment_annotation", id=exp_id, **fields)

    def add_insight(self, text: str, source: str, evidence: Any = None):
        if not text or not str(text).strip():
            return
        n = _norm(text)
        if any(_norm(i["text"]) == n for i in self.state["insights"]):
            return
        self.state["insights"].append({"text": _short(text, 300), "source": source, "step": self.step,
                                       "evidence": evidence})
        self.state["insights"] = self.state["insights"][-30:]
        self.log("insight", text=text, source=source, evidence=evidence)

    def add_backlog(self, ideas: list[str], source: str):
        for idea in ideas or []:
            if not idea or any(_norm(b["idea"]) == _norm(idea) for b in self.state["backlog"]):
                continue
            self.state["backlog"].append({"idea": _short(idea, 240), "source": source, "status": "open",
                                          "step": self.step})
        self.state["backlog"] = self.state["backlog"][-25:]

    def reject_proposal(self, summary: str, reasons: list[str], source: str):
        self.state["rejected_proposals"].append({"proposal": _short(summary, 200), "reasons": reasons[:6],
                                                 "source": source, "step": self.step})
        self.state["rejected_proposals"] = self.state["rejected_proposals"][-15:]
        self.log("proposal_rejected", proposal=summary, reasons=reasons, source=source)

    def save(self):
        (self.dir / "state.json").write_text(json.dumps(self.state, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")

    # ------------------------------------------------------------------ чтение
    @property
    def direction(self) -> str:
        return self.get("direction", "minimize")

    def ok_experiments(self) -> list[dict]:
        return [e for e in self.state["experiments"] if e.get("status") == "ok" and e.get("cv") is not None]

    def best(self) -> dict | None:
        ok = self.ok_experiments()
        if not ok:
            return None
        return (min if self.direction == "minimize" else max)(ok, key=lambda e: e["cv"])

    def spec_hashes(self) -> set[str]:
        return {e.get("spec_hash") for e in self.state["experiments"] if e.get("spec_hash")}

    # ------------------------------------------------------------------ для Qwen
    def render(self, max_chars: int, budget: dict) -> str:
        """Компактная память. Сначала полная версия, при превышении лимита — поэтапное сжатие."""
        exps = self.state["experiments"]
        best = self.best()
        sign = 1 if self.direction == "minimize" else -1
        ranked = sorted(self.ok_experiments(), key=lambda e: sign * e["cv"])

        def line(e, hyp_len):
            return {"id": e["id"], "parent": e.get("parent"), "src": e.get("source"),
                    "hypothesis": _short(e.get("hypothesis"), hyp_len), "status": e.get("status"),
                    "cv": None if e.get("cv") is None else round(e["cv"], 4),
                    "d_best_%": e.get("delta_best_pct"), "verdict": e.get("verdict"),
                    "qwen_verdict": e.get("analysis_verdict")}

        def detail(e):
            return {k: e.get(k) for k in ("id", "parent", "hypothesis", "spec", "status", "cv", "cv_std",
                                          "folds", "train_cv", "delta_best_pct", "delta_parent_pct",
                                          "verdict", "seconds", "best_iters", "top_features",
                                          "segments", "error", "analysis") if e.get(k) is not None}

        facts = {k: v["value"] for k, v in self.state["facts"].items()}

        def build(hyp_len, n_detail, n_top, keep_lines):
            lines = [line(e, hyp_len) for e in exps]
            if keep_lines is not None and len(lines) > keep_lines:
                lines = lines[:3] + [{"…": f"{len(lines) - keep_lines} строк скрыто"}] + lines[-(keep_lines - 3):]
            return json.dumps({
                "BUDGET": budget,
                "FACTS": facts,
                "BEST": detail(best) if best else None,
                "TOP": [line(e, hyp_len) for e in ranked[:n_top]],
                "ALL_EXPERIMENTS_IN_ORDER": lines,
                "LAST_EXPERIMENTS_DETAILED": [detail(e) for e in exps[-n_detail:]],
                "INSIGHTS": [i["text"] for i in self.state["insights"]],
                "BACKLOG_OPEN": [b["idea"] for b in self.state["backlog"] if b["status"] == "open"][-10:],
                "RECENTLY_REJECTED_PROPOSALS": self.state["rejected_proposals"][-4:],
            }, ensure_ascii=False, default=str)

        for args in ((140, 3, 8, None), (90, 2, 6, None), (70, 2, 5, 25), (50, 1, 5, 15), (40, 1, 3, 10)):
            text = build(*args)
            if len(text) <= max_chars:
                return text
        return text[:max_chars]
