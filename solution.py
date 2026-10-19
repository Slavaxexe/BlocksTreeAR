"""Hackathon-compatible entry point: python solution.py path/to/task.md."""

import argparse
import asyncio
from pathlib import Path
import sys

from alfa_agent.config import ConfigurationError, Settings
from alfa_agent.runner import Options, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="JSON-validated autonomous tabular ML prototype")
    parser.add_argument("task", nargs="?", default="task.md", type=Path)
    parser.add_argument("--offline", action="store_true", help="Exercise the loop without LLM calls")
    parser.add_argument("--max-experiments", type=int, default=4, help="Includes the dummy baseline")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-budget", type=float, default=300, help="Whole run, seconds")
    parser.add_argument("--experiment-timeout", type=float, default=90, help="Per worker, seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(argv)
    settings = None
    try:
        options = Options(max_experiments=args.max_experiments, workers=args.workers,
                          time_budget=args.time_budget, experiment_timeout=args.experiment_timeout,
                          offline=args.offline, output_root=args.output_dir)
        if not args.offline:
            settings = Settings.from_env(args.env_file)
        directory, report = asyncio.run(run(args.task, options, settings))
        print(f"Artifacts: {directory}\nSubmission: {args.task.resolve().parent / 'submission.csv'}")
        print(f"Best: {report.best.node_id}, {report.best.metric.name}={report.best.value:.6f}")
        print(f"Stopped: {report.stop_reason}")
        for warning in report.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        return 0
    except (ConfigurationError, ValueError, OSError) as error:
        message = str(error)
        if settings and settings.api_key:
            message = message.replace(settings.api_key, "[REDACTED]")
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
