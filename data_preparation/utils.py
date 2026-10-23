"""Timing, memory, and logging utilities."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import psutil

logger = logging.getLogger("data_preparation")


class TimeBudget:
    def __init__(self, total_seconds: float):
        self._total = total_seconds
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self._total - self.elapsed)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def check(self, required_seconds: float = 0) -> bool:
        return self.remaining > required_seconds


@contextmanager
def timed_step(name: str, budget: TimeBudget | None = None) -> Generator[dict[str, float], None, None]:
    info: dict[str, float] = {}
    start = time.monotonic()
    logger.info("Step '%s' started", name)
    try:
        yield info
    finally:
        duration = time.monotonic() - start
        info["duration_seconds"] = round(duration, 3)
        remaining = budget.remaining if budget else -1
        logger.info("Step '%s' finished in %.1fs (budget remaining: %.0fs)", name, duration, remaining)


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def safe_json_dump(data: Any, path: Path, tmp_suffix: str = ".tmp") -> None:
    """Write JSON atomically: write to tmp file, then rename."""
    tmp_path = path.with_suffix(path.suffix + tmp_suffix)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf with None for JSON serialization."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj
