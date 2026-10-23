"""Boundary between upstream model selection/EDA and regression training."""
from typing import Any, Literal
from pathlib import Path

from pydantic import Field, model_validator

from .contracts import Contract


Family = Literal['ridge', 'elasticnet', 'sgd', 'random_forest', 'extra_trees',
                 'hist_gradient_boosting', 'lightgbm', 'catboost', 'xgboost', 'mlp', 'flaml', 'autogluon']
Metric = Literal['rmse', 'mae', 'mse', 'r2', 'mape']


class TrainingFiles(Contract):
    train: str
    test: str
    sample_submission: str | None = None


class SelectedModel(Contract):
    family: Family
    rationale: str = ''
    parameters: dict[str, Any] = Field(default_factory=dict)


class ValidationConstraints(Contract):
    strategy: Literal['random', 'group', 'time'] = 'random'
    group_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None

    @model_validator(mode='after')
    def columns_required(self):
        if self.strategy == 'group' and not self.group_columns:
            raise ValueError('group validation requires group_columns')
        if self.strategy == 'time' and not self.time_column:
            raise ValueError('time validation requires time_column')
        return self


class SubmissionSpec(Contract):
    filename: str = 'submission.csv'
    prediction_column: str | None = None

    @model_validator(mode='after')
    def root_csv(self):
        if Path(self.filename).name != self.filename or '/' in self.filename or '\\' in self.filename or not self.filename.endswith('.csv'):
            raise ValueError('submission filename must be a root CSV filename')
        return self


class TrainingInput(Contract):
    schema_version: Literal[1] = 1
    task_type: Literal['regression'] = 'regression'
    files: TrainingFiles
    target: str
    id_columns: list[str] = Field(default_factory=list)
    primary_metric: Metric
    selected_models: list[SelectedModel] = Field(min_length=1, max_length=12)
    eda: dict[str, Any] = Field(default_factory=dict)
    validation_constraints: ValidationConstraints = Field(default_factory=ValidationConstraints)
    submission: SubmissionSpec = Field(default_factory=SubmissionSpec)

    @model_validator(mode='after')
    def identifiers(self):
        if self.target in self.id_columns or len(set(self.id_columns)) != len(self.id_columns):
            raise ValueError('target and unique identifier columns must be distinct')
        if self.submission.prediction_column in self.id_columns:
            raise ValueError('prediction column cannot be an identifier')
        if self.files.train == self.files.test:
            raise ValueError('train and test files must differ')
        return self


class ValidationDecision(Contract):
    validation_fraction: float = Field(ge=.05, le=.4)
    inner_validation: Literal['holdout', 'cv'] = 'holdout'
    folds: int = Field(default=3, ge=2, le=5)
    seed: int = 42
    rationale: str


class TrainingCandidate(Contract):
    name: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,50}$')
    model_index: int = Field(ge=0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    optimizer: str = 'not_applicable'
    regularization: str = 'engine/model defaults'
    estimated_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False,
                                            description='Estimated wall time for this complete trial, including CV and worker startup')


class TrainingPlan(Contract):
    candidates: list[TrainingCandidate] = Field(min_length=1, max_length=12)
    skipped_models: dict[str, str] = Field(default_factory=dict)
    rationale: str


class TrainingAssessment(Contract):
    """Legacy combined response; the component now uses two separate role contracts."""
    findings: list[str] = Field(default_factory=list)
    next_candidates: list[TrainingCandidate] = Field(default_factory=list, max_length=6)
    stop: bool = False
    rationale: str


class TrainingAnalysis(Contract):
    observations: list[str] = Field(default_factory=list, max_length=12)
    hypotheses: list[str] = Field(default_factory=list, max_length=8)
    risks: list[str] = Field(default_factory=list, max_length=8)
    recommendation: Literal['continue', 'stop', 'repair']
    rationale: str


class TrainingNextPlan(Contract):
    next_candidates: list[TrainingCandidate] = Field(default_factory=list, max_length=6)
    stop: bool = False
    rationale: str

    @model_validator(mode='after')
    def stop_or_candidates(self):
        if self.stop and self.next_candidates:
            raise ValueError('A stop plan must not contain new candidates')
        return self
