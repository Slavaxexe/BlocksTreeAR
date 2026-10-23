"""Единая конфигурация: бюджеты времени, ресурсы, доступ к Qwen."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


def load_env(root: Path):
    """.env рядом с solution.py. Непустые значения из .env важнее окружения: локально это защищает от чужого
    OPENAI_API_KEY в shell (например, ключа OpenAI). На платформе .env нет — работают переменные окружения."""
    path = root / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ[k.strip()] = v


@dataclass
class Budget:
    """Все сроки — абсолютные моменты time.monotonic(). Лимит платформы — 3 часа на задачу."""
    total_s: float = 3 * 3600
    safety_s: float = 420          # запас на запись ответа и непредвиденное
    started: float = field(default_factory=time.monotonic)

    @property
    def deadline(self) -> float:
        return self.started + self.total_s - self.safety_s

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started


@dataclass
class Settings:
    n_jobs: int = field(default_factory=lambda: os.cpu_count() or 1)
    seed: int = 42
    # исследование
    max_experiments: int = 60
    experiment_timeout_s: float = 900
    patience: int = 10                  # экспериментов без значимого улучшения до остановки
    min_rel_improvement: float = 0.002  # 0.2% метрики — порог «значимого» улучшения
    max_custom_codes: int = 6
    # LLM
    llm_enabled: bool = True
    llm_max_calls: int = 45
    llm_timeout_s: float = 180
    llm_max_tokens: int = 12000
    memory_chars: int = 15000           # сколько памяти отдаём Qwen за один вызов

    @property
    def api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "").strip()

    @property
    def base_url(self) -> str:
        return os.getenv("OPENAI_BASE_URL", "https://changellenge.up.railway.app/v1").strip()

    @property
    def model(self) -> str:
        return os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b").strip()
