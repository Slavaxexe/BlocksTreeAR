"""Точка входа платформы: python solution.py path/to/task.md

Регрессия → AutoResearch-пайплайн (autoresearch/). Если задача им не поддержана (например, классификация) —
прежний агент команды (agent/), чтобы на приватных задачах всегда был ответ. Если AutoResearch упал, но уже
записал страховочный submission.csv в этом запуске, — оставляем его, а не начинаем заново.
"""
import sys
import time
import traceback
from pathlib import Path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    task = args[0] if args and not args[0].startswith("-") else "task.md"
    started = time.time()
    p = Path(task)
    sub = (p if p.is_dir() else p.parent) / "submission.csv"
    try:
        from autoresearch.run import run
        from autoresearch.task import TaskNotSupported
        try:
            run(task, write_next_to_task=True)
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
    rest = [a for a in args if a != task]
    return cli(["search", "--task", task, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
