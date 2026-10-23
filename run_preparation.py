#!/usr/bin/env python3
"""Standalone entry point for the data preparation module.

Usage:
    # With env vars:
    API_KEY=sk-... BASE_URL=https://... python run_preparation.py /path/to/task/dir

    # Without LLM (deterministic mode):
    python run_preparation.py /path/to/task/dir --no-llm

    # With .env file:
    pip install python-dotenv
    python run_preparation.py /path/to/task/dir
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Data preparation for tabular regression")
    parser.add_argument("task_dir", help="Directory with train.csv, test.csv, etc.")
    parser.add_argument("--target", default="site_eui", help="Target column name")
    parser.add_argument("--metric", default="rmse", help="Metric name")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: task_dir/output)")
    parser.add_argument("--time-budget", type=int, default=1800, help="Time budget in seconds")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM integration")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from data_preparation.config import PipelineConfig
    from data_preparation.pipeline import DataPreparationPipeline

    output_dir = args.output_dir or str(Path(args.task_dir) / "output")

    config = PipelineConfig(
        task_dir=args.task_dir,
        target_columns=[args.target],
        metric_name=args.metric,
        output_dir=output_dir,
        time_budget_seconds=args.time_budget,
        random_state=args.random_state,
    )

    if args.no_llm:
        config.llm.api_key = ""
        config.llm.base_url = ""

    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    pipeline = DataPreparationPipeline(config)
    result = pipeline.run()

    print(f"\nStatus: {result.status.value}")
    print(f"Features: {len(result.feature_names)}")
    if result.train_features is not None:
        print(f"Train: {result.train_features.shape}")
    if result.val_features is not None:
        print(f"Val:   {result.val_features.shape}")
    if result.test_features is not None:
        print(f"Test:  {result.test_features.shape}")
    print(f"Output: {output_dir}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings[:5]:
            print(f"  [{w.code}] {w.message}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors:
            print(f"  [{e.code}] {e.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
