"""Точка входа платформы: python solution.py path/to/task.md

Регрессия → AutoResearch-пайплайн (autoresearch/). Если задача им не поддержана (например, классификация) —
прежний агент команды (agent/), чтобы на приватных задачах всегда был ответ. Если AutoResearch упал, но уже
записал страховочный submission.csv в этом запуске, — оставляем его, а не начинаем заново.

Для локальной проверки (платформа эти флаги не передаёт):
    python solution.py task.md --budget 3600 --llm-calls 30 --no-llm
"""
import argparse
import sys
import time
import traceback
from pathlib import Path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("task", nargs="?", default="task.md")
    ap.add_argument("--budget", type=float, default=3 * 3600)
    ap.add_argument("--llm-calls", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true")
    known, rest = ap.parse_known_args(args)
    task = known.task
    started = time.time()
    p = Path(task)
    sub = (p if p.is_dir() else p.parent) / "submission.csv"
    try:
        from autoresearch.run import run
        from autoresearch.task import TaskNotSupported
        try:
            run(task, no_llm=known.no_llm, total_budget=known.budget, llm_calls=known.llm_calls,
                write_next_to_task=True)
            return 0
        except TaskNotSupported as e:
            print(f"AutoResearch: {e} → прежний агент", flush=True)
    except Exception:
        traceback.print_exc()
        if sub.exists() and sub.stat().st_mtime >= started:
            print("AutoResearch упал после страховочного ответа — оставляю его", flush=True)
            return 0
        print("AutoResearch упал → прежний агент", flush=True)
    from agent.__main__ import main as cli
    return cli(["search", "--task", task, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
