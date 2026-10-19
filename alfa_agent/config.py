"""Read baseline-compatible configuration without exposing credentials."""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """Configuration is incomplete or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_seconds: float = 90
    max_tokens: int = 2048

    @classmethod
    def from_env(
        cls,
        env_file: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        path = env_file if env_file is not None else Path(__file__).resolve().parents[1] / ".env"
        # Disable expansion so credentials containing '$' stay literal.
        values = dict(dotenv_values(path, encoding="utf-8", interpolate=False))
        values.update(os.environ if environ is None else environ)
        try:
            timeout = float(values.get("QWEN_TIMEOUT_SECONDS") or "90")
            max_tokens = int(values.get("QWEN_MAX_TOKENS") or "2048")
        except (ValueError, TypeError):
            raise ConfigurationError("QWEN_TIMEOUT_SECONDS и QWEN_MAX_TOKENS должны быть числами.") from None
        return cls(
            api_key=(values.get("OPENAI_API_KEY") or "").strip(),
            base_url=(values.get("OPENAI_BASE_URL") or "").strip(),
            model=(values.get("OPENAI_MODEL") or "").strip(),
            timeout_seconds=timeout,
            max_tokens=max_tokens,
        )

    def validate(self) -> None:
        missing = [name for name, value in (
            ("OPENAI_API_KEY", self.api_key),
            ("OPENAI_BASE_URL", self.base_url),
            ("OPENAI_MODEL", self.model),
        ) if not value]
        if missing:
            raise ConfigurationError("Заполните настройки: " + ", ".join(missing))
        try:
            url = urlsplit(self.base_url)
            valid_url = (
                bool(url.hostname)
                and not url.username and not url.password
                and not url.query and not url.fragment
                and (url.scheme == "https" or (
                    url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
                ))
            )
            url.port  # Also validate an explicitly supplied port.
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ConfigurationError("OPENAI_BASE_URL: нужен HTTPS base URL без логина, пароля и query-параметров.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError("QWEN_TIMEOUT_SECONDS должен быть конечным положительным числом.")
        if self.max_tokens <= 0:
            raise ConfigurationError("QWEN_MAX_TOKENS должен быть положительным целым числом.")

    def public_status(self) -> dict:
        return {
            "api_key_configured": bool(self.api_key),
            "base_url_configured": bool(self.base_url),
            "model_configured": bool(self.model),
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
        }
