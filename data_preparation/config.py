"""Pipeline configuration. Reads env vars and provides defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = "qwen/qwen3.8-27b"
    timeout_seconds: int = 120
    max_retries: int = 3
    max_calls: int = 10
    max_tokens_per_call: int = 4096
    temperature: float = 0.3

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            api_key=os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url=os.environ.get("BASE_URL", os.environ.get("OPENAI_BASE_URL", "")),
            model=os.environ.get("OPENAI_MODEL", "qwen/qwen3.8-27b"),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)


@dataclass
class PipelineConfig:
    task_dir: str = "."
    task_type: str = "regression"
    metric_name: str = "rmse"
    target_columns: list[str] = field(default_factory=lambda: ["site_eui"])
    id_columns: list[str] = field(default_factory=lambda: ["id"])
    time_column: str | None = None
    group_column: str | None = None
    random_state: int = 42
    time_budget_seconds: int = 1800
    max_features_to_create: int = 20
    val_fraction: float = 0.2
    output_dir: str = "output"
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    train_indices: list[int] | None = None
    val_indices: list[int] | None = None
    cpu_limit: int = 8
    memory_limit_gb: int = 32
    scaling_mode: str | None = None

    def validate(self) -> list[str]:
        errors = []
        if self.task_type != "regression":
            errors.append(f"Unsupported task_type: {self.task_type}. Only 'regression' is supported.")
        if len(self.target_columns) != 1:
            errors.append(f"target_columns must contain exactly 1 element, got {len(self.target_columns)}.")
        if self.time_budget_seconds <= 0:
            errors.append("time_budget_seconds must be positive.")
        if not (0.0 < self.val_fraction < 1.0):
            errors.append(f"val_fraction must be in (0, 1), got {self.val_fraction}.")
        return errors
