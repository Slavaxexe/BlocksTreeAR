"""Offline oracle for generated fixtures; never import from the training component."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def evaluate(input_report: Path, submission: Path):
    report = json.loads(input_report.read_text(encoding="utf-8"))
    with (input_report.parent / "evaluation_labels.csv").open(encoding="utf-8", newline="") as stream:
        oracle = list(csv.DictReader(stream))
    with submission.open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    if len(predictions) != len(oracle):
        raise ValueError("Submission row count differs from the test row count")
    if [row["id"] for row in predictions] != [row["id"] for row in oracle]:
        raise ValueError("Submission must preserve exact string IDs and test order")
    prediction_column = report["submission"]["prediction_column"]
    actual = [float(row["y"]) for row in oracle]
    predicted = [float(row[prediction_column]) for row in predictions]
    if not all(math.isfinite(value) for value in predicted):
        raise ValueError("Submission contains non-finite predictions")
    squared = sum((y - p) ** 2 for y, p in zip(actual, predicted)) / len(actual)
    absolute = sum(abs(y - p) for y, p in zip(actual, predicted)) / len(actual)
    metric = report["primary_metric"]
    if metric == "rmse":
        score = math.sqrt(squared)
    elif metric == "mse":
        score = squared
    elif metric == "mae":
        score = absolute
    elif metric == "mape":
        score = sum(abs(y - p) / max(abs(y), 2.220446049250313e-16) for y, p in zip(actual, predicted)) / len(actual)
    elif metric == "r2":
        average = sum(actual) / len(actual)
        denominator = sum((y - average) ** 2 for y in actual)
        score = 1 - squared * len(actual) / denominator if denominator else float(squared == 0)
    else:
        raise ValueError(f"Unsupported oracle metric: {metric}")
    return {"metric": metric, "synthetic_test_score": score, "rows": len(actual),
            "purpose": "Post-run external evaluation only; do not feed this score into model selection."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_report", type=Path)
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input_report, args.submission), ensure_ascii=False, indent=2))
