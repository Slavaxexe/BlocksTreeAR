"""One bounded CPU experiment, executed in a child process with no API credentials."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from .contracts import Finish, Profile, Start
from .data import dataset_hash, read_tables, validate_submission
from .storage import atomic_bytes, fingerprint, load, save


def build_pipeline(start: Start, profile: Profile, threads: int):
    spec = start.hypothesis.pipeline
    regression = start.task.problem.task_type == "regression"
    model = spec.model
    if model.family == "dummy":
        estimator = DummyRegressor(strategy="mean") if regression else DummyClassifier(strategy="most_frequent")
        # Dummy consumes no features and cannot fail on categorical/missing values.
        return estimator
    blocks = []
    if profile.numeric_columns:
        blocks.append(("numeric", SimpleImputer(strategy=spec.preprocessing.numeric_imputer,
                                               add_indicator=spec.preprocessing.missing_indicators,
                                               keep_empty_features=True), profile.numeric_columns))
    if profile.categorical_columns:
        blocks.append(("categorical", SKPipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                        encoded_missing_value=-1)),
        ]), profile.categorical_columns))
    if model.family == "extra_trees":
        kwargs = dict(n_estimators=model.n_estimators, max_depth=model.max_depth,
                      min_samples_leaf=model.min_samples_leaf, n_jobs=threads,
                      random_state=start.validation.seed)
        if regression:
            if model.class_weight != "none":
                raise ValueError("class_weight is only supported for classification")
            estimator = ExtraTreesRegressor(**kwargs)
        else:
            estimator = ExtraTreesClassifier(class_weight=None if model.class_weight == "none" else "balanced", **kwargs)
    else:
        constructor = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
        estimator = constructor(max_iter=model.max_iter, max_leaf_nodes=model.max_leaf_nodes,
                                learning_rate=model.learning_rate, l2_regularization=model.l2_regularization,
                                random_state=start.validation.seed, early_stopping=False)
    return SKPipeline([("preprocess", ColumnTransformer(blocks)), ("model", estimator)])


def evaluate(start_path: Path, task_directory: Path, run_directory: Path, threads: int) -> Finish:
    start = load(start_path, Start)
    profile = load(run_directory / "profile.json", Profile)
    started = time.monotonic()
    common = dict(node_id=start.node_id, metric=start.task.metric,
                  split_sha256=start.validation.split_sha256,
                  pipeline_sha256=fingerprint(start.hypothesis.pipeline))
    try:
        if dataset_hash(task_directory, start.task) != start.dataset_sha256:
            raise ValueError("Dataset changed since profiling")
        split = np.load(run_directory / "split.npz", allow_pickle=False)
        fit, valid = split["fit"], split["valid"]
        digest = hashlib.sha256(fit.tobytes() + valid.tobytes()).hexdigest()
        if digest != start.validation.split_sha256:
            raise ValueError("Validation split changed")
        train, test, sample = read_tables(task_directory, start.task)
        target, identifier = start.task.problem.target_column, start.task.data.id_column
        x = train.drop(columns=[target, identifier]).replace([np.inf, -np.inf], np.nan)
        x_test = test.drop(columns=identifier).replace([np.inf, -np.inf], np.nan)
        y = train[target]
        model = build_pipeline(start, profile, threads)
        # Preprocessing and model fit see ONLY the fitting fold at evaluation time.
        with threadpool_limits(limits=threads):
            model.fit(x.iloc[fit], y.iloc[fit])
            predicted = model.predict(x.iloc[valid])
            if start.task.metric.name == "rmse":
                value = float(root_mean_squared_error(y.iloc[valid], predicted))
            else:
                value = float(f1_score(y.iloc[valid], predicted, average="macro",
                                       labels=sorted(y.unique()), zero_division=0))
            # Refit only after the honest metric has been computed.
            model.fit(x, y)
            predictions = model.predict(x_test)
        submission = sample.copy()
        submission[target] = predictions
        validate_submission(submission, test, sample, train, start.task)
        relative = f"nodes/{start.node_id}/submission.csv"
        atomic_bytes(run_directory / relative, submission.to_csv(index=False).encode())
        result = Finish(status="completed", value=value, submission_file=relative,
                        validation_rows=len(valid), duration_seconds=time.monotonic() - started, **common)
    except Exception as error:
        # Never forward uncontrolled estimator/provider output into an agent prompt.
        result = Finish(status="failed", error=f"Experiment failed: {type(error).__name__}",
                        duration_seconds=time.monotonic() - started, **common)
    save(start_path.with_name("finish.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=Path)
    parser.add_argument("task_directory", type=Path)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("threads", type=int)
    args = parser.parse_args()
    result = evaluate(args.start, args.task_directory, args.run_directory, args.threads)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
