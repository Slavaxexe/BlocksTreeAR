"""Read a supported tabular task without hard-coded public target names."""

import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .contracts import DataFiles, Metric, Problem, Profile, Task, Validation


def confined_file(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError("Dataset paths must stay inside the task directory")
    if not path.is_file():
        raise ValueError(f"Missing task file: {relative}")
    return path


def local_task(task_path: Path) -> Task:
    """Conservative bootstrap. Ambiguous or unsupported tasks fail explicitly."""
    text = task_path.read_text(encoding="utf-8")
    paths = re.findall(r"`([^`\n]+\.csv)`", text)
    def find(name):
        matches = sorted({p for p in paths if Path(p).name == name})
        if len(matches) != 1:
            raise ValueError(f"Task must unambiguously name {name}")
        confined_file(task_path.parent, matches[0])
        return matches[0]
    train_file, test_file, sample_file = map(find, ("train.csv", "test.csv", "sample_submission.csv"))
    train_columns = list(pd.read_csv(task_path.parent / train_file, nrows=0))
    test_columns = list(pd.read_csv(task_path.parent / test_file, nrows=0))
    sample_columns = list(pd.read_csv(task_path.parent / sample_file, nrows=0))
    target = set(train_columns) - set(test_columns)
    if len(target) != 1 or len(sample_columns) != 2 or not target.issubset(sample_columns):
        raise ValueError("Expected one target absent from test and a two-column submission template")
    target = target.pop()
    identifier = next(c for c in sample_columns if c != target)
    # Recognise only supported primary metrics, never infer from target dtype.
    rmse = bool(re.search(r"\bRMSE\b", text, re.I))
    macro = bool(re.search(r"macro[\s_*-]*f1", text, re.I))
    if rmse == macro:
        raise ValueError("Expected one supported primary metric: RMSE or Macro F1")
    return Task(
        problem=Problem(task_type="regression" if rmse else "classification", target_column=target),
        data=DataFiles(train_file=train_file, test_file=test_file,
                       sample_submission_file=sample_file, id_column=identifier),
        metric=Metric(name="rmse" if rmse else "macro_f1",
                      direction="minimize" if rmse else "maximize"),
    )


def read_tables(directory: Path, task: Task):
    files = task.data
    # Keep IDs as literal strings (including leading zeros).
    dtype = {files.id_column: str}
    if task.problem.task_type == "classification":
        dtype[task.problem.target_column] = str
    train = pd.read_csv(confined_file(directory, files.train_file), dtype=dtype)
    test = pd.read_csv(confined_file(directory, files.test_file), dtype={files.id_column: str})
    sample = pd.read_csv(confined_file(directory, files.sample_submission_file), dtype={files.id_column: str})
    target, identifier = task.problem.target_column, files.id_column
    if target not in train or target in test or identifier not in test or identifier not in train:
        raise ValueError("Target or ID columns contradict the task")
    if list(train.drop(columns=target)) != list(test):
        raise ValueError("Train and test feature schemas differ")
    if set(sample.columns) != {identifier, target} or len(sample.columns) != 2:
        raise ValueError("Submission template must contain exactly the ID and target")
    for frame in (train, test, sample):
        if frame.empty or frame[identifier].isna().any() or not frame[identifier].is_unique:
            raise ValueError("Empty data, missing IDs or duplicate IDs")
    if not sample[identifier].equals(test[identifier]):
        raise ValueError("Template IDs differ from test IDs or their order")
    if train[target].isna().any():
        raise ValueError("Missing train targets")
    if task.problem.task_type == "regression" and not np.isfinite(train[target].to_numpy(dtype=float)).all():
        raise ValueError("Regression targets must be finite numbers")
    return train, test, sample


def dataset_hash(directory: Path, task: Task) -> str:
    digest = hashlib.sha256()
    for relative in (task.data.train_file, task.data.test_file, task.data.sample_submission_file):
        digest.update(relative.encode())
        with confined_file(directory, relative).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def profile_data(directory: Path, task: Task):
    train, test, _ = read_tables(directory, task)
    features = train.drop(columns=[task.problem.target_column, task.data.id_column])
    if features.shape[1] == 0:
        raise ValueError("No input features")
    numeric = list(features.select_dtypes(include="number"))
    categorical = [c for c in features if c not in numeric]
    classification = task.problem.task_type == "classification"
    counts = train[task.problem.target_column].value_counts() if classification else None
    if classification and (len(counts) < 2 or counts.min() < 2):
        raise ValueError("Stratified validation needs two classes and at least two rows per class")
    indices = np.arange(len(train), dtype=np.int64)
    fit_indices, valid_indices = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=train[task.problem.target_column] if classification else None,
    )
    split_hash = hashlib.sha256(fit_indices.tobytes() + valid_indices.tobytes()).hexdigest()
    profile = Profile(dataset_sha256=dataset_hash(directory, task), train_rows=len(train), test_rows=len(test),
                      numeric_columns=numeric, categorical_columns=categorical,
                      missing_train_cells=int(features.isna().sum().sum()),
                      class_counts={str(k): int(v) for k, v in counts.items()} if classification else {})
    validation = Validation(stratified=classification, split_sha256=split_hash)
    return profile, validation, fit_indices, valid_indices


def validate_submission(frame: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
                        train: pd.DataFrame, task: Task) -> None:
    identifier, target = task.data.id_column, task.problem.target_column
    if list(frame.columns) != list(sample.columns) or len(frame) != len(test):
        raise ValueError("Wrong submission schema or number of rows")
    if not frame[identifier].equals(test[identifier]) or frame.isna().any().any():
        raise ValueError("Submission IDs, order, or missing values are invalid")
    if task.problem.task_type == "regression":
        if not np.isfinite(frame[target].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite predictions")
    elif not frame[target].isin(train[target].unique()).all():
        raise ValueError("Unknown predicted class")
