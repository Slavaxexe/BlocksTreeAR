"""Data profiling: statistics, type detection, quality checks."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from data_preparation.config import PipelineConfig
from data_preparation.schemas import (
    ArtifactType, CategoricalStats, ColumnDtype, ColumnProfile,
    DataProfilePayload, DatasetSummary, NumericStats, SchemaComparison,
    Status, TargetProfile, make_envelope,
)

logger = logging.getLogger("data_preparation.profiler")

_CATEGORY_THRESHOLD = 50
_TEXT_MIN_AVG_LEN = 30
_ID_UNIQUENESS_THRESHOLD = 0.95
_TOP_VALUES_LIMIT = 10


def profile_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: PipelineConfig,
    data_dictionary: pd.DataFrame | None = None,
) -> dict:
    target_col = config.target_columns[0]
    id_cols = set(config.id_columns)

    dd_lookup = _build_dd_lookup(data_dictionary) if data_dictionary is not None else {}

    train_summary = _dataset_summary(train)
    test_summary = _dataset_summary(test)

    target_profile = _profile_target(train, target_col)

    feature_cols = [c for c in train.columns if c != target_col]
    train_profiles = [_profile_column(train, col, dd_lookup, id_cols) for col in feature_cols]
    test_profiles = [_profile_column(test, col, dd_lookup, id_cols) for col in test.columns]

    schema_comp = _compare_schemas(train, test, target_col)

    potential_leaks = _detect_leaks(train_profiles, target_col)
    potential_groups = _detect_groups(train, feature_cols)
    potential_time = _detect_time_columns(train, feature_cols)

    train_summary.n_numeric = sum(1 for p in train_profiles if p.detected_dtype == ColumnDtype.NUMERIC)
    train_summary.n_categorical = sum(1 for p in train_profiles if p.detected_dtype == ColumnDtype.CATEGORICAL)
    train_summary.n_datetime = sum(1 for p in train_profiles if p.detected_dtype == ColumnDtype.DATETIME)
    train_summary.n_text = sum(1 for p in train_profiles if p.detected_dtype == ColumnDtype.TEXT)
    train_summary.n_constant_columns = sum(1 for p in train_profiles if p.is_constant)
    train_summary.n_all_missing_columns = sum(1 for p in train_profiles if p.is_all_missing)
    train_summary.n_duplicate_rows = int(train.duplicated().sum())

    payload = DataProfilePayload(
        train_summary=train_summary,
        test_summary=test_summary,
        target=target_profile,
        schema_comparison=schema_comp,
        train_columns=train_profiles,
        test_columns=test_profiles,
        potential_leaks=potential_leaks,
        potential_groups=potential_groups,
        potential_time_columns=potential_time,
    )

    return make_envelope(
        ArtifactType.DATA_PROFILE, "pending", Status.SUCCESS,
        payload.model_dump(mode="json"),
    )


def _dataset_summary(df: pd.DataFrame) -> DatasetSummary:
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    return DatasetSummary(
        n_rows=len(df),
        n_columns=len(df.columns),
        memory_mb=round(mem_mb, 2),
    )


def _profile_target(df: pd.DataFrame, col: str) -> TargetProfile:
    series = df[col]
    n_missing = int(series.isna().sum())
    numeric = pd.to_numeric(series, errors="coerce")
    n_inf = int(np.isinf(numeric.dropna()).sum())
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    stats = None
    if len(clean) > 0:
        stats = NumericStats(
            min=_safe_float(clean.min()),
            max=_safe_float(clean.max()),
            mean=_safe_float(clean.mean()),
            median=_safe_float(clean.median()),
            std=_safe_float(clean.std()),
            q25=_safe_float(clean.quantile(0.25)),
            q75=_safe_float(clean.quantile(0.75)),
            n_unique=int(clean.nunique()),
            n_missing=n_missing,
            missing_ratio=round(n_missing / len(df), 4) if len(df) > 0 else 0,
            n_zeros=int((clean == 0).sum()),
            n_inf=n_inf,
            n_negative=int((clean < 0).sum()),
        )

    return TargetProfile(name=col, n_missing=n_missing, n_inf=n_inf, stats=stats)


def _profile_column(
    df: pd.DataFrame,
    col: str,
    dd_lookup: dict[str, dict[str, str]],
    id_cols: set[str],
) -> ColumnProfile:
    series = df[col]
    n_total = len(series)
    n_missing = int(series.isna().sum())
    is_all_missing = n_missing == n_total

    detected_dtype = _detect_dtype(series)
    declared = dd_lookup.get(col, {}).get("dtype")
    role = dd_lookup.get(col, {}).get("role")
    description = dd_lookup.get(col, {}).get("description_ru")

    n_unique = int(series.nunique())
    is_constant = n_unique <= 1 and not is_all_missing
    is_potential_id = (
        col in id_cols
        or (n_unique / max(n_total, 1) > _ID_UNIQUENESS_THRESHOLD and n_total > 100)
    )

    numeric_stats = None
    categorical_stats = None

    if detected_dtype == ColumnDtype.NUMERIC:
        numeric_stats = _numeric_stats(series)
    elif detected_dtype == ColumnDtype.CATEGORICAL:
        categorical_stats = _categorical_stats(series)

    return ColumnProfile(
        name=col,
        detected_dtype=detected_dtype,
        declared_dtype=declared,
        role=role,
        is_constant=is_constant,
        is_all_missing=is_all_missing,
        is_potential_id=is_potential_id,
        is_potential_leak=False,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        description=description,
    )


def _detect_dtype(series: pd.Series) -> ColumnDtype:
    if pd.api.types.is_numeric_dtype(series):
        return ColumnDtype.NUMERIC
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnDtype.DATETIME

    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnDtype.UNKNOWN

    if non_null.dtype == object:
        sample = non_null.head(100)
        n_unique = non_null.nunique()
        avg_len = sample.astype(str).str.len().mean()

        if avg_len > _TEXT_MIN_AVG_LEN and n_unique / max(len(non_null), 1) > 0.5:
            return ColumnDtype.TEXT

        try:
            pd.to_datetime(sample, format="mixed", dayfirst=False)
            if avg_len > 6:
                return ColumnDtype.DATETIME
        except (ValueError, TypeError):
            pass

        return ColumnDtype.CATEGORICAL

    return ColumnDtype.UNKNOWN


def _numeric_stats(series: pd.Series) -> NumericStats:
    numeric = pd.to_numeric(series, errors="coerce")
    n_missing = int(numeric.isna().sum())
    n_total = len(numeric)
    n_inf = int(np.isinf(numeric.dropna()).sum()) if len(numeric.dropna()) > 0 else 0
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    if len(clean) == 0:
        return NumericStats(n_missing=n_missing, missing_ratio=round(n_missing / max(n_total, 1), 4))

    return NumericStats(
        min=_safe_float(clean.min()),
        max=_safe_float(clean.max()),
        mean=_safe_float(clean.mean()),
        median=_safe_float(clean.median()),
        std=_safe_float(clean.std()),
        q25=_safe_float(clean.quantile(0.25)),
        q75=_safe_float(clean.quantile(0.75)),
        n_unique=int(clean.nunique()),
        n_missing=n_missing,
        missing_ratio=round(n_missing / max(n_total, 1), 4),
        n_zeros=int((clean == 0).sum()),
        n_inf=n_inf,
        n_negative=int((clean < 0).sum()),
    )


def _categorical_stats(series: pd.Series) -> CategoricalStats:
    n_missing = int(series.isna().sum())
    n_total = len(series)
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    vc = non_null.value_counts().head(_TOP_VALUES_LIMIT)
    top_values = [
        {"value": str(val), "count": int(cnt), "ratio": round(cnt / max(n_total, 1), 4)}
        for val, cnt in vc.items()
    ]

    return CategoricalStats(
        n_unique=n_unique,
        n_missing=n_missing,
        missing_ratio=round(n_missing / max(n_total, 1), 4),
        top_values=top_values,
    )


def _compare_schemas(train: pd.DataFrame, test: pd.DataFrame, target: str) -> SchemaComparison:
    train_cols = set(train.columns) - {target}
    test_cols = set(test.columns)

    mismatches = []
    for col in sorted(train_cols & test_cols):
        t1 = str(train[col].dtype)
        t2 = str(test[col].dtype)
        if t1 != t2:
            mismatches.append({"column": col, "train_dtype": t1, "test_dtype": t2})

    return SchemaComparison(
        columns_only_in_train=sorted(train_cols - test_cols),
        columns_only_in_test=sorted(test_cols - train_cols),
        dtype_mismatches=mismatches,
    )


def _detect_leaks(profiles: list[ColumnProfile], target: str) -> list[str]:
    leaks = []
    for p in profiles:
        if p.name == target:
            continue
        if p.is_potential_id:
            continue
        if p.numeric_stats and p.numeric_stats.n_unique == 1:
            continue
        name_lower = p.name.lower()
        if any(kw in name_lower for kw in ["target", "label", "eui", "output", "prediction"]):
            if p.name != target:
                leaks.append(p.name)
    return leaks


def _detect_groups(df: pd.DataFrame, cols: list[str]) -> list[str]:
    groups = []
    for col in cols:
        if df[col].dtype == object:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 100:
                groups.append(col)
    return groups[:5]


def _detect_time_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    time_cols = []
    for col in cols:
        name_lower = col.lower()
        if any(kw in name_lower for kw in ["year", "date", "time", "month", "day"]):
            time_cols.append(col)
    return time_cols


def _build_dd_lookup(dd: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    name_col = None
    for candidate in ["feature", "column", "name", "field"]:
        if candidate in dd.columns:
            name_col = candidate
            break
    if name_col is None and len(dd.columns) > 0:
        name_col = dd.columns[0]
    if name_col is None:
        return lookup

    for _, row in dd.iterrows():
        key = str(row[name_col])
        entry: dict[str, str] = {}
        for c in dd.columns:
            if c != name_col:
                entry[c] = str(row[c]) if pd.notna(row[c]) else ""
        lookup[key] = entry
    return lookup


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None
