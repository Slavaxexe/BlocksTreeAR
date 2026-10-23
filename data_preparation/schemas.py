"""Fixed JSON schemas for all pipeline artifacts.

Every field name is snake_case and used consistently across all artifacts.
Schema version changes require explicit version bump.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


class ArtifactType(str, Enum):
    INPUT_MANIFEST = "input_manifest"
    DATA_PROFILE = "data_profile"
    FEATURE_PLAN = "feature_plan"
    PREPARATION_REPORT = "preparation_report"
    OUTPUT_MANIFEST = "output_manifest"


class Status(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class DiagnosticEntry(BaseModel):
    code: str
    message: str
    stage: str
    details: dict[str, Any] | None = None


class ArtifactEnvelope(BaseModel):
    schema_version: str = SCHEMA_VERSION
    artifact_type: ArtifactType
    run_id: str
    status: Status
    payload: dict[str, Any]
    warnings: list[DiagnosticEntry] = Field(default_factory=list)
    errors: list[DiagnosticEntry] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Invalid schema_version: {v}")
        return v


# ── input_manifest ──────────────────────────────────────────────

class FileRole(str, Enum):
    TRAIN = "train"
    TEST = "test"
    DATA_DICTIONARY = "data_dictionary"
    SAMPLE_SUBMISSION = "sample_submission"
    TASK_DESCRIPTION = "task_description"


class FileEntry(BaseModel):
    path: str
    role: FileRole
    rows: int | None = None
    columns: int | None = None
    size_bytes: int | None = None


class InputManifestPayload(BaseModel):
    task_type: str  # "regression"
    metric_name: str  # "rmse"
    target_columns: list[str] = Field(min_length=1, max_length=1)
    id_columns: list[str] = Field(default_factory=list)
    files: list[FileEntry]
    time_column: str | None = None
    group_column: str | None = None
    random_state: int = 42
    time_budget_seconds: int = 3600
    task_description: str | None = None
    data_dictionary_text: str | None = None


# ── data_profile ────────────────────────────────────────────────

class ColumnDtype(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


class NumericStats(BaseModel):
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q25: float | None = None
    q75: float | None = None
    n_unique: int = 0
    n_missing: int = 0
    missing_ratio: float = 0.0
    n_zeros: int = 0
    n_inf: int = 0
    n_negative: int = 0

    @field_validator("min", "max", "mean", "median", "std", "q25", "q75", mode="before")
    @classmethod
    def _reject_nan_inf(cls, v: Any) -> Any:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v


class CategoricalStats(BaseModel):
    n_unique: int = 0
    n_missing: int = 0
    missing_ratio: float = 0.0
    top_values: list[dict[str, Any]] = Field(default_factory=list)


class ColumnProfile(BaseModel):
    name: str
    detected_dtype: ColumnDtype
    declared_dtype: str | None = None
    role: str | None = None
    is_constant: bool = False
    is_all_missing: bool = False
    is_potential_id: bool = False
    is_potential_leak: bool = False
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    description: str | None = None


class TargetProfile(BaseModel):
    name: str
    n_missing: int = 0
    n_inf: int = 0
    stats: NumericStats | None = None


class DatasetSummary(BaseModel):
    n_rows: int
    n_columns: int
    memory_mb: float
    n_numeric: int = 0
    n_categorical: int = 0
    n_datetime: int = 0
    n_text: int = 0
    n_constant_columns: int = 0
    n_all_missing_columns: int = 0
    n_duplicate_rows: int = 0


class SchemaComparison(BaseModel):
    columns_only_in_train: list[str] = Field(default_factory=list)
    columns_only_in_test: list[str] = Field(default_factory=list)
    dtype_mismatches: list[dict[str, str]] = Field(default_factory=list)


class DataProfilePayload(BaseModel):
    train_summary: DatasetSummary
    test_summary: DatasetSummary
    target: TargetProfile
    schema_comparison: SchemaComparison
    train_columns: list[ColumnProfile]
    test_columns: list[ColumnProfile] = Field(default_factory=list)
    potential_leaks: list[str] = Field(default_factory=list)
    potential_groups: list[str] = Field(default_factory=list)
    potential_time_columns: list[str] = Field(default_factory=list)


# ── feature_plan ────────────────────────────────────────────────

class TransformOp(str, Enum):
    DROP = "drop"
    PASSTHROUGH = "passthrough"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MODE = "impute_mode"
    IMPUTE_CONSTANT = "impute_constant"
    ORDINAL_ENCODE = "ordinal_encode"
    TARGET_ENCODE = "target_encode"  # reserved, not used in v1
    ONE_HOT_ENCODE = "one_hot_encode"
    FREQUENCY_ENCODE = "frequency_encode"
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    STANDARD_SCALE = "standard_scale"
    CLIP_OUTLIERS = "clip_outliers"
    BINNING = "binning"
    INTERACTION = "interaction"
    RATIO = "ratio"
    POLYNOMIAL = "polynomial"
    DATE_COMPONENTS = "date_components"
    MISSING_INDICATOR = "missing_indicator"


class ColumnTransform(BaseModel):
    column: str
    operations: list[TransformOp]
    impute_value: Any | None = None
    reason: str | None = None


class NewFeatureDef(BaseModel):
    name: str
    source_columns: list[str]
    transform_type: TransformOp
    definition: str
    rationale: str
    missing_handling: str
    requires_fit: bool = False


class FeaturePlanPayload(BaseModel):
    columns_to_drop: list[str] = Field(default_factory=list)
    drop_reasons: dict[str, str] = Field(default_factory=dict)
    column_transforms: list[ColumnTransform] = Field(default_factory=list)
    new_features: list[NewFeatureDef] = Field(default_factory=list)
    scaling_mode: str | None = None
    llm_suggestions_used: bool = False
    llm_rationale: str | None = None


# ── preparation_report ──────────────────────────────────────────

class StepReport(BaseModel):
    step_name: str
    status: Status
    duration_seconds: float = 0.0
    rows_before: int | None = None
    rows_after: int | None = None
    columns_before: int | None = None
    columns_after: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CorrelationEntry(BaseModel):
    column_a: str
    column_b: str
    correlation: float
    method: str = "pearson"


class PreparationReportPayload(BaseModel):
    steps: list[StepReport] = Field(default_factory=list)
    new_features_created: list[str] = Field(default_factory=list)
    columns_dropped: list[str] = Field(default_factory=list)
    high_correlations: list[CorrelationEntry] = Field(default_factory=list)
    target_correlations: list[dict[str, Any]] = Field(default_factory=list)
    train_val_split: dict[str, Any] = Field(default_factory=dict)
    total_duration_seconds: float = 0.0
    llm_calls_count: int = 0
    llm_tokens_used: int = 0


# ── output_manifest ─────────────────────────────────────────────

class DataFileRef(BaseModel):
    path: str
    format: str = "parquet"
    n_rows: int = 0
    n_columns: int = 0
    columns: list[str] = Field(default_factory=list)


class OutputManifestPayload(BaseModel):
    train_features: DataFileRef
    val_features: DataFileRef | None = None
    test_features: DataFileRef
    train_target: DataFileRef
    val_target: DataFileRef | None = None
    train_ids: DataFileRef
    val_ids: DataFileRef | None = None
    test_ids: DataFileRef
    transformer_path: str
    recipe_path: str
    feature_names: list[str]
    target_columns: list[str]
    id_columns: list[str]
    split_indices: dict[str, list[int]] | None = None


# ── Helpers ─────────────────────────────────────────────────────

def make_envelope(
    artifact_type: ArtifactType,
    run_id: str,
    status: Status,
    payload: dict[str, Any],
    warnings: list[DiagnosticEntry] | None = None,
    errors: list[DiagnosticEntry] | None = None,
) -> dict[str, Any]:
    env = ArtifactEnvelope(
        artifact_type=artifact_type,
        run_id=run_id,
        status=status,
        payload=payload,
        warnings=warnings or [],
        errors=errors or [],
    )
    return env.model_dump(mode="json")


def validate_payload(artifact_type: ArtifactType, data: dict[str, Any]) -> BaseModel:
    mapping: dict[ArtifactType, type[BaseModel]] = {
        ArtifactType.INPUT_MANIFEST: InputManifestPayload,
        ArtifactType.DATA_PROFILE: DataProfilePayload,
        ArtifactType.FEATURE_PLAN: FeaturePlanPayload,
        ArtifactType.PREPARATION_REPORT: PreparationReportPayload,
        ArtifactType.OUTPUT_MANIFEST: OutputManifestPayload,
    }
    model_cls = mapping[artifact_type]
    return model_cls.model_validate(data)


# ── LLM response schemas ───────────────────────────────────────

class LLMFeatureSuggestion(BaseModel):
    name: str
    source_columns: list[str]
    transform_type: str
    definition: str
    rationale: str
    missing_handling: str
    requires_fit: bool = False

    @field_validator("transform_type")
    @classmethod
    def _valid_transform(cls, v: str) -> str:
        valid = {e.value for e in TransformOp}
        if v not in valid:
            raise ValueError(f"Unknown transform_type: {v}. Must be one of {valid}")
        return v


class LLMProfileInterpretation(BaseModel):
    summary: str
    key_observations: list[str] = Field(default_factory=list)
    suggested_features: list[LLMFeatureSuggestion] = Field(default_factory=list)
    suggested_drops: list[dict[str, str]] = Field(default_factory=list)
    suggested_transforms: list[dict[str, Any]] = Field(default_factory=list)


class LLMResponse(BaseModel):
    interpretation: LLMProfileInterpretation

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, data: Any) -> Any:
        if isinstance(data, dict) and "interpretation" not in data:
            return {"interpretation": data}
        return data
