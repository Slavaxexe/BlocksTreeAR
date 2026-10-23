"""JSON-контракты этапа (pydantic 2.10). Вход от Марго, решение Qwen, выход Максу."""
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# ----------------------------- ВХОД (от этапа Марго) -----------------------------
class TaskInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_type: Literal["regression"] = "regression"
    target: str
    id_col: Optional[str] = "id"
    target_metric: Optional[str] = None          # распарсено из описания задачи
    metric_params: dict[str, Any] = Field(default_factory=dict)  # напр. {"alpha": 0.9} для pinball


class Paths(BaseModel):
    model_config = ConfigDict(extra="allow")
    train: str
    test: Optional[str] = None


class Hints(BaseModel):
    """Подсказки из EDA. Всё опционально: этап работает и без них."""
    model_config = ConfigDict(extra="allow")
    categorical_cols: Optional[list[str]] = None
    time_col: Optional[str] = None
    group_col: Optional[str] = None
    drop_cols: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    probe_seconds: float = 300
    probe_max_rows: int = 30000
    probe_folds: int = 3


class StageInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    task: TaskInfo
    paths: Paths
    hints: Hints = Field(default_factory=Hints)
    budget: Budget = Field(default_factory=Budget)
    eda: dict[str, Any] = Field(default_factory=dict)  # пробрасывается дальше как есть


# ----------------------------- РЕШЕНИЕ QWEN -----------------------------
Role = Literal["primary", "ensemble_member"]


class ReasoningStep(BaseModel):
    claim: str = Field(..., description="Вывод одним предложением")
    evidence: dict[str, Any] = Field(..., description="Ключи из FACTS и их значения")


class LLMCandidate(BaseModel):
    model_id: str
    loss: str
    target_transform: Literal["none", "log1p"] = "none"
    priority: int = Field(..., ge=1, le=9)
    role: Role
    rationale: str
    evidence: dict[str, Any]


class LLMRejected(BaseModel):
    model_id: str
    reason: str


class LLMDecision(BaseModel):
    reasoning: list[ReasoningStep]
    candidates: list[LLMCandidate]
    rejected: list[LLMRejected] = Field(default_factory=list)


# ----------------------------- ВЫХОД (Максу) -----------------------------
class OutCandidate(BaseModel):
    model_id: str
    library: str
    estimator: str
    loss: str
    loss_native: dict[str, Any]
    target_transform: str
    metric_consistent: bool
    priority: int
    role: Role
    needs_scaling: bool
    native_categorical: bool
    optimizer_applicable: bool       # ветка Макса "градиентные методы" применима
    rationale: str
    evidence: dict[str, Any]


class AutoMLHint(BaseModel):
    flaml_metric: Optional[str]
    flaml_estimator_list: list[str]
    autogluon_eval_metric: Optional[str]
    autogluon_models: list[str]
    label_transform: str
    notes: list[str]


class StageOutput(BaseModel):
    stage: str = "model_family_and_loss_selection"
    version: str = "1.0"
    task: dict[str, Any]
    target_metric: str
    metric_source: Literal["task_description", "fallback"]
    metric_statistic: str
    loss_policy: dict[str, Any]
    diagnostics: dict[str, Any]
    probe: dict[str, Any]
    candidates: list[OutCandidate]
    rejected: list[dict[str, Any]]
    reasoning: list[dict[str, Any]]
    automl_hint: AutoMLHint
    source: Literal["qwen", "rules_fallback"]
    validation_errors: list[str]
    warnings: list[str]
    hints: dict[str, Any]
    eda: dict[str, Any]
    timings: dict[str, float]
