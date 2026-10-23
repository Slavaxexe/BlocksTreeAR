"""Data quality checks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from data_preparation.schemas import ColumnProfile, ColumnDtype, DataProfilePayload

logger = logging.getLogger("data_preparation.quality")


class QualityReport:
    def __init__(self):
        self.issues: list[dict[str, Any]] = []

    def add(self, code: str, severity: str, message: str, column: str | None = None, **details: Any):
        self.issues.append({
            "code": code,
            "severity": severity,
            "message": message,
            "column": column,
            **details,
        })

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i["severity"] == "critical")


def run_quality_checks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    profile_payload: DataProfilePayload,
    target_col: str,
    id_cols: list[str],
) -> QualityReport:
    report = QualityReport()

    _check_missing(train, "train", report)
    _check_missing(test, "test", report)
    _check_inf_values(train, "train", report)
    _check_constant_columns(profile_payload.train_columns, report)
    _check_all_missing_columns(profile_payload.train_columns, report)
    _check_duplicates(train, "train", id_cols, report)
    _check_target_quality(train, target_col, report)
    _check_schema_issues(profile_payload, report)
    _check_potential_leaks(profile_payload, report)
    _check_outliers(train, profile_payload.train_columns, report)

    logger.info("Quality checks complete: %d issues found (%d critical)",
                len(report.issues), report.critical_count)
    return report


def _check_missing(df: pd.DataFrame, dataset: str, report: QualityReport):
    missing = df.isnull().sum()
    total = len(df)
    for col, count in missing.items():
        if count > 0:
            ratio = count / total
            severity = "warning" if ratio < 0.5 else "high"
            report.add(
                "MISSING_VALUES", severity,
                f"{dataset}.{col}: {count} missing ({ratio:.1%})",
                column=str(col), dataset=dataset, count=int(count), ratio=round(ratio, 4),
            )


def _check_inf_values(df: pd.DataFrame, dataset: str, report: QualityReport):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_inf = int(np.isinf(df[col]).sum())
        if n_inf > 0:
            report.add(
                "INF_VALUES", "high",
                f"{dataset}.{col}: {n_inf} infinite values",
                column=str(col), dataset=dataset, count=n_inf,
            )


def _check_constant_columns(profiles: list[ColumnProfile], report: QualityReport):
    for p in profiles:
        if p.is_constant:
            report.add(
                "CONSTANT_COLUMN", "warning",
                f"Column '{p.name}' is constant — carries no information",
                column=p.name,
            )


def _check_all_missing_columns(profiles: list[ColumnProfile], report: QualityReport):
    for p in profiles:
        if p.is_all_missing:
            report.add(
                "ALL_MISSING_COLUMN", "high",
                f"Column '{p.name}' is entirely missing",
                column=p.name,
            )


def _check_duplicates(df: pd.DataFrame, dataset: str, id_cols: list[str], report: QualityReport):
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        report.add(
            "DUPLICATE_ROWS", "info",
            f"{dataset}: {n_dup} fully duplicate rows",
            dataset=dataset, count=n_dup,
        )
    for col in id_cols:
        if col in df.columns:
            n_dup_id = int(df[col].duplicated().sum())
            if n_dup_id > 0:
                report.add(
                    "DUPLICATE_IDS", "warning",
                    f"{dataset}.{col}: {n_dup_id} duplicate IDs",
                    column=col, dataset=dataset, count=n_dup_id,
                )


def _check_target_quality(df: pd.DataFrame, target_col: str, report: QualityReport):
    if target_col not in df.columns:
        return
    target = df[target_col]
    n_missing = int(target.isna().sum())
    if n_missing > 0:
        report.add(
            "TARGET_MISSING", "critical",
            f"Target '{target_col}' has {n_missing} missing values. "
            "These rows cannot be used for training without explicit policy.",
            column=target_col, count=n_missing,
        )

    numeric_target = pd.to_numeric(target, errors="coerce")
    n_inf = int(np.isinf(numeric_target.dropna()).sum())
    if n_inf > 0:
        report.add(
            "TARGET_INF", "critical",
            f"Target '{target_col}' has {n_inf} infinite values",
            column=target_col, count=n_inf,
        )

    n_neg = int((numeric_target.dropna() < 0).sum())
    if n_neg > 0:
        report.add(
            "TARGET_NEGATIVE", "info",
            f"Target '{target_col}' has {n_neg} negative values",
            column=target_col, count=n_neg,
        )

    n_zero = int((numeric_target.dropna() == 0).sum())
    if n_zero > 0:
        report.add(
            "TARGET_ZEROS", "info",
            f"Target '{target_col}' has {n_zero} zero values",
            column=target_col, count=n_zero,
        )


def _check_schema_issues(profile: DataProfilePayload, report: QualityReport):
    if profile.schema_comparison.columns_only_in_train:
        for col in profile.schema_comparison.columns_only_in_train:
            report.add(
                "SCHEMA_TRAIN_ONLY", "warning",
                f"Column '{col}' present in train but not test",
                column=col,
            )
    if profile.schema_comparison.columns_only_in_test:
        for col in profile.schema_comparison.columns_only_in_test:
            report.add(
                "SCHEMA_TEST_ONLY", "warning",
                f"Column '{col}' present in test but not train",
                column=col,
            )
    for m in profile.schema_comparison.dtype_mismatches:
        report.add(
            "DTYPE_MISMATCH", "warning",
            f"Column '{m['column']}': dtype differs (train={m['train_dtype']}, test={m['test_dtype']})",
            column=m["column"],
        )


def _check_potential_leaks(profile: DataProfilePayload, report: QualityReport):
    for col in profile.potential_leaks:
        report.add(
            "POTENTIAL_LEAK", "high",
            f"Column '{col}' may leak target information",
            column=col,
        )


def _check_outliers(df: pd.DataFrame, profiles: list[ColumnProfile], report: QualityReport):
    for p in profiles:
        if p.detected_dtype != ColumnDtype.NUMERIC or p.numeric_stats is None:
            continue
        stats = p.numeric_stats
        if stats.q25 is None or stats.q75 is None:
            continue
        iqr = stats.q75 - stats.q25
        if iqr <= 0:
            continue
        lower = stats.q25 - 3.0 * iqr
        upper = stats.q75 + 3.0 * iqr
        if p.name not in df.columns:
            continue
        series = pd.to_numeric(df[p.name], errors="coerce").dropna()
        n_outliers = int(((series < lower) | (series > upper)).sum())
        if n_outliers > 0:
            ratio = n_outliers / max(len(series), 1)
            if ratio > 0.001:
                report.add(
                    "OUTLIERS", "info",
                    f"Column '{p.name}': {n_outliers} extreme outliers ({ratio:.1%}) by 3*IQR",
                    column=p.name, count=n_outliers, ratio=round(ratio, 4),
                )
