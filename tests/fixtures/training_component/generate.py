"""Create reproducible regression component inputs; no network or ML packages needed."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def eda_facts(rows, features, **extra):
    return {
        "source": "Synthetic upstream EDA; component must consume supplied features as-is.",
        "train_rows": len(rows),
        "feature_columns": features,
        "missing_values": {key: sum(row[key] is None for row in rows) for key in features},
        "target_summary": {
            "minimum": min(row["y"] for row in rows),
            "maximum": max(row["y"] for row in rows),
            "mean": mean(row["y"] for row in rows),
        },
        **extra,
    }


def random_data(rng):
    rows = []
    for index in range(800):
        x1, x2, x3 = rng.uniform(-3, 3), rng.gauss(0, 1), rng.uniform(0, 2)
        segment = rng.choice(["retail", "corporate", "private"])
        effect = {"retail": 0, "corporate": 3, "private": -2}[segment]
        target = 25 + 4 * x1 ** 2 + 3 * math.sin(2 * x2) + x1 * x3 + effect + rng.gauss(0, .6)
        rows.append({"id": f"{index:06d}", "x1": x1, "x2": None if rng.random() < .08 else x2,
                     "x3": x3, "segment": segment, "y": target})
    rng.shuffle(rows)
    train, test = rows[:600], rows[600:]
    return train, test, eda_facts(train, ["x1", "x2", "x3", "segment"],
        categorical_columns=["segment"], categorical_cardinality={"segment": len({r["segment"] for r in train})},
        test_rows=len(test), relationship="Nonlinear target with interactions; random independent observations.")


def group_data(rng):
    rows = []
    for group in range(40):
        group_effect = rng.gauss(0, 2)
        for observation in range(20):
            x1, x2 = rng.uniform(-2, 2), rng.gauss(0, 1)
            rows.append({"id": f"{group * 20 + observation:06d}", "group_id": f"G{group:03d}",
                         "x1": x1, "x2": x2,
                         "y": 20 + 3 * x1 + x2 ** 2 + group_effect + rng.gauss(0, .3)})
    train, test = rows[:600], rows[600:]
    return train, test, eda_facts(train, ["group_id", "x1", "x2"],
        categorical_columns=["group_id"], train_groups=30, test_groups=10,
        test_rows=len(test), group_columns=["group_id"], train_test_groups_disjoint=True,
        relationship="Repeated observations within a group; validation must keep entire groups together.")


def time_data(rng):
    rows = []
    # Features were prepared upstream; no future target enters an input feature.
    previous, before_previous = 10.0, 10.0
    for timestamp in range(650):
        driver = math.sin(timestamp / 9)
        target = .72 * previous + .13 * before_previous + 2 * driver + .003 * timestamp + rng.gauss(0, .25)
        rows.append({"id": f"{timestamp:06d}", "time_index": timestamp,
                     "lag_y_1": previous, "lag_y_2": before_previous,
                     "driver": driver, "y": target})
        before_previous, previous = previous, target
    train, test = rows[:500], rows[500:]
    # Test is one-step-ahead regression with observed history supplied per row,
    # not a recursive multi-step forecast from the train/test cut-off.
    return train, test, eda_facts(train, ["time_index", "lag_y_1", "lag_y_2", "driver"],
        categorical_columns=[], test_rows=len(test), time_column="time_index",
        train_time_range=[0, 499], test_time_range=[500, 649],
        relationship="Ordered one-step-ahead regression; supplied lag features use observed past only.",
        forecasting_protocol="Each test row has its observed history supplied upstream; this is not a fixed-origin multi-step benchmark.")


def generate(destination: Path, seed: int):
    destination.mkdir(parents=True, exist_ok=True)
    for offset, (name, factory) in enumerate([("random", random_data), ("group", group_data), ("time", time_data)]):
        scenario = destination / name
        scenario.mkdir(parents=True, exist_ok=True)
        train, labelled_test, eda = factory(random.Random(seed + offset))
        test = [{key: value for key, value in row.items() if key != "y"} for row in labelled_test]
        sample = [{"id": row["id"], "prediction": 0.0} for row in test]
        oracle = [{"id": row["id"], "y": row["y"]} for row in labelled_test]
        write_csv(scenario / "train.csv", train)
        write_csv(scenario / "test.csv", test)
        write_csv(scenario / "sample_submission.csv", sample)
        write_csv(scenario / "evaluation_labels.csv", oracle)
        report = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
        report["eda"] = eda
        report["eda"]["generator_seed"] = seed + offset
        (scenario / "input.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(scenario / "input.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "artifacts" / "training_component_inputs")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    generate(args.output_dir.resolve(), args.seed)
