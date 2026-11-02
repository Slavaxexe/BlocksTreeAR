"""Connection settings for an OpenAI-compatible model API."""

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
    timeout_seconds: float | None = None
    max_tokens: int | None = None
    api_profile: str = "standard"
    tools_enabled: bool = True

    @classmethod
    def from_env(
        cls,
        env_file: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        path = env_file if env_file is not None else Path(__file__).resolve().parents[1] / ".env"
        # Disable expansion so credentials containing '$' stay literal.
        values = {}
        for source in (dict(dotenv_values(path, encoding="utf-8", interpolate=False)),
                       dict(os.environ if environ is None else environ)):
            # Accept old variable names while preserving environment-over-file precedence.
            for name in ('TIMEOUT_SECONDS', 'MAX_TOKENS'):
                if 'LLM_' + name not in source and 'QWEN_' + name in source:
                    source['LLM_' + name] = source['QWEN_' + name]
            values.update(source)
        tools = str(values.get('LLM_TOOLS_ENABLED') or 'true').strip().lower()
        if tools not in {'true', 'false', '1', '0'}:
            raise ConfigurationError('LLM_TOOLS_ENABLED: используйте true или false.')
        try:
            timeout = float(values.get("LLM_TIMEOUT_SECONDS") or "0")
            max_tokens = int(values.get("LLM_MAX_TOKENS") or "0")
        except (ValueError, TypeError):
            raise ConfigurationError("LLM_TIMEOUT_SECONDS и LLM_MAX_TOKENS должны быть числами.") from None
        return cls(
            api_key=(values.get("OPENAI_API_KEY") or "").strip(),
            base_url=(values.get("OPENAI_BASE_URL") or "").strip(),
            model=(values.get("OPENAI_MODEL") or "").strip(),
            timeout_seconds=timeout if timeout else None,
            max_tokens=max_tokens if max_tokens else None,
            api_profile=(values.get("LLM_API_PROFILE") or "standard").strip().lower(),
            tools_enabled=tools in {"true", "1"},
        )

    def validate(self) -> None:
        if self.api_profile not in {"standard", "qwen"}:
            raise ConfigurationError("LLM_API_PROFILE: используйте standard или qwen.")
        missing = [name for name, value in (
            ("OPENAI_API_KEY", self.api_key),
            ("OPENAI_BASE_URL", self.base_url),
            ("OPENAI_MODEL", self.model),
        ) if not value]
        if missing:
            raise ConfigurationError("Заполните настройки: " + ", ".join(missing))
        try:
            url = urlsplit(self.base_url)
            valid_url = bool(url.hostname) and url.scheme in {'http', 'https'}
            url.port  # Validate an explicitly supplied port.
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ConfigurationError("OPENAI_BASE_URL: нужен абсолютный HTTP/HTTPS URL с корректным адресом и портом.")
        if self.timeout_seconds is not None and (not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ConfigurationError("LLM_TIMEOUT_SECONDS должен быть конечным положительным числом.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ConfigurationError("LLM_MAX_TOKENS должен быть положительным целым числом.")

    def reasoning_options(self, thinking=False):
        if self.api_profile == 'qwen':
            options = {'enable_thinking': thinking, 'reasoning_effort': 'medium' if thinking else 'low'}
            return {'temperature': 0, 'extra_body': {**options, 'chat_template_kwargs': options}}
        return {}

    def public_status(self) -> dict:
        return {
            "api_key_configured": bool(self.api_key),
            "base_url_configured": bool(self.base_url),
            "model_configured": bool(self.model),
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "api_profile": self.api_profile,
            "tools_enabled": self.tools_enabled,
        }
