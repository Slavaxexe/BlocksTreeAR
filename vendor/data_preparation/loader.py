"""Load and validate task files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_preparation.config import PipelineConfig
from data_preparation.schemas import (
    ArtifactType, ColumnDtype, DataProfilePayload, DatasetSummary,
    FileEntry, FileRole, InputManifestPayload, SchemaComparison,
    Status, make_envelope,
)

logger = logging.getLogger("data_preparation.loader")


class DataLoadResult:
    def __init__(self):
        self.train: pd.DataFrame | None = None
        self.test: pd.DataFrame | None = None
        self.task_description: str | None = None
        self.data_dictionary: pd.DataFrame | None = None
        self.sample_submission: pd.DataFrame | None = None
        self.files: list[FileEntry] = []
        self.warnings: list[dict] = []
        self.errors: list[dict] = []


def load_data(config: PipelineConfig) -> DataLoadResult:
    result = DataLoadResult()
    task_dir = Path(config.task_dir)

    train_path = _find_file(task_dir, "train.csv")
    test_path = _find_file(task_dir, "test.csv")

    if train_path is None:
        result.errors.append({
            "code": "MISSING_TRAIN", "message": "train.csv not found", "stage": "loading"
        })
        return result

    if test_path is None:
        result.errors.append({
            "code": "MISSING_TEST", "message": "test.csv not found", "stage": "loading"
        })
        return result

    logger.info("Loading train.csv from %s", train_path)
    result.train = pd.read_csv(train_path, encoding="utf-8")
    result.files.append(FileEntry(
        path=str(train_path), role=FileRole.TRAIN,
        rows=len(result.train), columns=len(result.train.columns),
        size_bytes=train_path.stat().st_size,
    ))

    logger.info("Loading test.csv from %s", test_path)
    result.test = pd.read_csv(test_path, encoding="utf-8")
    result.files.append(FileEntry(
        path=str(test_path), role=FileRole.TEST,
        rows=len(result.test), columns=len(result.test.columns),
        size_bytes=test_path.stat().st_size,
    ))

    target_col = config.target_columns[0]
    if target_col not in result.train.columns:
        result.errors.append({
            "code": "TARGET_NOT_FOUND",
            "message": f"Target column '{target_col}' not found in train. Available: {list(result.train.columns[:10])}",
            "stage": "loading",
        })
        return result

    if target_col in result.test.columns:
        result.warnings.append({
            "code": "TARGET_IN_TEST",
            "message": f"Target column '{target_col}' found in test.csv — potential leak",
            "stage": "loading",
        })

    for id_col in config.id_columns:
        if id_col not in result.train.columns:
            result.warnings.append({
                "code": "ID_NOT_FOUND",
                "message": f"ID column '{id_col}' not found in train",
                "stage": "loading",
            })

    dict_path = _find_file(task_dir, "data_dictionary.csv")
    if dict_path:
        try:
            result.data_dictionary = pd.read_csv(dict_path, encoding="utf-8")
            result.files.append(FileEntry(
                path=str(dict_path), role=FileRole.DATA_DICTIONARY,
                rows=len(result.data_dictionary), columns=len(result.data_dictionary.columns),
                size_bytes=dict_path.stat().st_size,
            ))
        except Exception as e:
            result.warnings.append({
                "code": "DICT_READ_ERROR", "message": str(e)[:200], "stage": "loading"
            })

    task_md = _find_file(task_dir, "task.md")
    if task_md:
        result.task_description = task_md.read_text(encoding="utf-8")[:5000]
        result.files.append(FileEntry(
            path=str(task_md), role=FileRole.TASK_DESCRIPTION,
            size_bytes=task_md.stat().st_size,
        ))

    submission_path = _find_file(task_dir, "sample_submission.csv")
    if submission_path:
        try:
            result.sample_submission = pd.read_csv(submission_path, encoding="utf-8")
            result.files.append(FileEntry(
                path=str(submission_path), role=FileRole.SAMPLE_SUBMISSION,
                rows=len(result.sample_submission), columns=len(result.sample_submission.columns),
                size_bytes=submission_path.stat().st_size,
            ))
        except Exception as e:
            result.warnings.append({
                "code": "SUBMISSION_READ_ERROR", "message": str(e)[:200], "stage": "loading"
            })

    _validate_schema_compatibility(result, config)

    return result


def build_input_manifest(config: PipelineConfig, load_result: DataLoadResult) -> dict:
    dd_text = None
    if load_result.data_dictionary is not None:
        dd_text = load_result.data_dictionary.to_csv(index=False)[:3000]

    payload = InputManifestPayload(
        task_type=config.task_type,
        metric_name=config.metric_name,
        target_columns=config.target_columns,
        id_columns=config.id_columns,
        files=load_result.files,
        time_column=config.time_column,
        group_column=config.group_column,
        random_state=config.random_state,
        time_budget_seconds=config.time_budget_seconds,
        task_description=(load_result.task_description or "")[:2000],
        data_dictionary_text=dd_text,
    )

    status = Status.SUCCESS if not load_result.errors else Status.FAILED
    return make_envelope(
        ArtifactType.INPUT_MANIFEST, "pending", status,
        payload.model_dump(mode="json"),
        warnings=[{"code": w["code"], "message": w["message"], "stage": w["stage"]}
                  for w in load_result.warnings],
        errors=[{"code": e["code"], "message": e["message"], "stage": e["stage"]}
                for e in load_result.errors],
    )


def _find_file(directory: Path, name: str) -> Path | None:
    direct = directory / name
    if direct.exists():
        return direct
    for p in directory.rglob(name):
        return p
    return None


def _validate_schema_compatibility(result: DataLoadResult, config: PipelineConfig):
    if result.train is None or result.test is None:
        return

    target_col = config.target_columns[0]
    train_features = set(result.train.columns) - {target_col}
    test_features = set(result.test.columns)

    only_train = train_features - test_features
    only_test = test_features - train_features

    for col in sorted(only_train):
        if col not in config.id_columns:
            result.warnings.append({
                "code": "COLUMN_ONLY_IN_TRAIN",
                "message": f"Column '{col}' in train but not test",
                "stage": "schema_check",
            })

    for col in sorted(only_test):
        result.warnings.append({
            "code": "COLUMN_ONLY_IN_TEST",
            "message": f"Column '{col}' in test but not train",
            "stage": "schema_check",
        })

    common = train_features & test_features
    for col in sorted(common):
        train_dtype = str(result.train[col].dtype)
        test_dtype = str(result.test[col].dtype)
        if train_dtype != test_dtype:
            result.warnings.append({
                "code": "DTYPE_MISMATCH",
                "message": f"Column '{col}': train={train_dtype}, test={test_dtype}",
                "stage": "schema_check",
            })
