"""The protocol is data, not a conversation history or executable LLM output."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
NodeID = Annotated[str, Field(pattern=r"^node_[0-9]{3}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


class Problem(Contract):
    task_type: Literal["regression", "classification"]
    target_column: str = Field(min_length=1)


class DataFiles(Contract):
    train_file: str = Field(min_length=1)
    test_file: str = Field(min_length=1)
    sample_submission_file: str = Field(min_length=1)
    id_column: str = Field(min_length=1)


class Metric(Contract):
    name: Literal["rmse", "macro_f1"]
    direction: Literal["minimize", "maximize"]

    @model_validator(mode="after")
    def correct_direction(self):
        expected = "minimize" if self.name == "rmse" else "maximize"
        if self.direction != expected:
            raise ValueError("Metric direction contradicts its definition")
        return self


class Task(Contract):
    schema_version: Literal[1] = 1
    problem: Problem
    data: DataFiles
    metric: Metric

    @model_validator(mode="after")
    def compatible_metric(self):
        if (self.problem.task_type == "regression") != (self.metric.name == "rmse"):
            raise ValueError("Unsupported task/metric combination")
        if self.problem.target_column == self.data.id_column:
            raise ValueError("Target cannot be the ID")
        return self


class Profile(Contract):
    dataset_sha256: SHA256
    train_rows: int = Field(gt=0)
    test_rows: int = Field(gt=0)
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_train_cells: int = Field(ge=0)
    class_counts: dict[str, int]


class Validation(Contract):
    strategy: Literal["holdout"] = "holdout"
    seed: int = Field(default=42, ge=0)
    fraction: float = Field(default=0.2, gt=0, lt=1)
    stratified: bool
    split_sha256: SHA256


class Preprocessing(Contract):
    numeric_imputer: Literal["median", "mean"] = "median"
    categorical_encoder: Literal["ordinal"] = "ordinal"
    missing_indicators: bool = False
    augmentation: Literal["none"] = "none"


class Dummy(Contract):
    family: Literal["dummy"] = "dummy"


class ExtraTrees(Contract):
    family: Literal["extra_trees"] = "extra_trees"
    n_estimators: int = Field(default=64, ge=16, le=160)
    max_depth: int | None = Field(default=16, ge=2, le=24)
    min_samples_leaf: int = Field(default=1, ge=1, le=64)
    class_weight: Literal["none", "balanced"] = "none"


class HistBoosting(Contract):
    family: Literal["hist_gradient_boosting"] = "hist_gradient_boosting"
    max_iter: int = Field(default=80, ge=20, le=200)
    max_leaf_nodes: int = Field(default=15, ge=3, le=63)
    learning_rate: float = Field(default=0.1, ge=0.01, le=0.3)
    l2_regularization: float = Field(default=1.0, ge=0.0, le=10.0)


class Pipeline(Contract):
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    model: Annotated[Dummy | ExtraTrees | HistBoosting, Field(discriminator="family")]


class Hypothesis(Contract):
    rationale: str = Field(min_length=1, max_length=1600)
    expected_effect: str = Field(min_length=1, max_length=800)
    pipeline: Pipeline


class Start(Contract):
    schema_version: Literal[1] = 1
    node_id: NodeID
    parent_id: NodeID | None
    source: Literal["baseline", "qwen", "offline"]
    task: Task
    dataset_sha256: SHA256
    validation: Validation
    hypothesis: Hypothesis


class Finish(Contract):
    schema_version: Literal[1] = 1
    node_id: NodeID
    status: Literal["completed", "failed", "timeout"]
    metric: Metric
    value: float | None = None
    duration_seconds: float = Field(ge=0)
    split_sha256: SHA256
    pipeline_sha256: SHA256
    submission_file: str | None = None
    validation_rows: int = Field(default=0, ge=0)
    error: str | None = Field(default=None, max_length=1200)

    @model_validator(mode="after")
    def result_consistency(self):
        if self.status == "completed":
            if self.value is None or not self.submission_file or self.error:
                raise ValueError("Completed runs require a metric and predictions, without an error")
            if self.value < 0 or (self.metric.name == "macro_f1" and self.value > 1):
                raise ValueError("Metric outside its mathematical range")
        elif self.value is not None or self.submission_file is not None or not self.error:
            raise ValueError("Failed runs require an error and cannot have a score or predictions")
        return self


class Score(Contract):
    node_id: NodeID
    value: float
    family: str


class Change(Contract):
    path: str
    before: str
    after: str


class Comparison(Contract):
    reference_id: NodeID
    candidate_id: NodeID
    metric_delta: float
    improvement: float
    changes: list[Change]


class Delta(Contract):
    schema_version: Literal[1] = 1
    latest_node_id: NodeID
    best_node_id: NodeID | None
    metric: Metric
    leaderboard: list[Score]
    comparisons: list[Comparison]
    failed_node_ids: list[NodeID]


class Best(Contract):
    node_id: NodeID
    metric: Metric
    value: float
    submission_file: str
    split_sha256: SHA256


class LLMCall(Contract):
    role: str
    attempt: int
    status: Literal["completed", "invalid_json", "api_error"]
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RunReport(Contract):
    schema_version: Literal[1] = 1
    mode: Literal["qwen", "offline"]
    stop_reason: str
    best: Best
    experiments: int
    elapsed_seconds: float
    llm_calls: list[LLMCall]
    warnings: list[str]


SCHEMAS = (Problem, DataFiles, Metric, Task, Profile, Validation, Hypothesis,
           Start, Finish, Delta, Best, LLMCall, RunReport)
