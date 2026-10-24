"""Точка входа платформы: python solution.py path/to/task.md

Тип задачи определяется автоматически. Регрессия → прежний AutoResearch ar_* без изменений.
Классификация → отдельный AutoResearch ac_* с временными/групповыми фолдами и вероятностями классов.
Страховочный submission.csv сохраняется при позднем сбое. Резервный прежний агент применяется
только в исходном регрессионном сценарии.

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
    ap.add_argument("--task-type", choices=["auto", "regression", "classification"], default="auto")
    known, rest = ap.parse_known_args(args)
    task = known.task
    started = time.time()
    p = Path(task)
    sub = (p if p.is_dir() else p.parent) / "submission.csv"
    from ac_task import detect_task_type
    task_type = detect_task_type(task) if known.task_type == "auto" else known.task_type
    if task_type == "classification":
        from ac_run import run as run_classification
        # Classification owns its full scenario. It never goes through the old
        # regression extractor or silently substitutes a regression metric.
        classification_options = argparse.ArgumentParser(add_help=False)
        classification_options.add_argument("--max-experiments", type=int, default=60)
        classification_options.add_argument("--out", default=None)
        classification_options.add_argument("--cpu", type=int, default=None)
        classification_args, _ = classification_options.parse_known_args(rest)
        try:
            run_classification(task, out_dir=classification_args.out, no_llm=known.no_llm,
                total_budget=known.budget, llm_calls=known.llm_calls,
                max_experiments=classification_args.max_experiments, n_jobs=classification_args.cpu)
            return 0
        except Exception:
            traceback.print_exc()
            if sub.exists() and sub.stat().st_mtime >= started:
                print("Classification stopped after writing a full-data safety submission", flush=True)
                return 0
            return 2
    try:
        from ar_run import run
        from ar_task import TaskNotSupported
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
    from max_cli import main as cli
    return cli(["search", "--task", task, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
