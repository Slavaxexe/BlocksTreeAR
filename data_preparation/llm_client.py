"""LLM client for Qwen API (OpenAI-compatible)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from data_preparation.config import LLMConfig
from data_preparation.schemas import LLMResponse

logger = logging.getLogger("data_preparation.llm")

_REDACTED = "***"


class LLMBudgetExhausted(Exception):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig):
        self._config = config
        self._calls_made = 0
        self._tokens_used = 0
        self._client = None

    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def is_available(self) -> bool:
        return self._config.is_configured

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
                max_retries=0,
            )
        return self._client

    def health_check(self) -> tuple[bool, str]:
        if not self.is_available:
            return False, "API not configured (missing API_KEY or BASE_URL)"
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=10,
                temperature=0,
            )
            text = resp.choices[0].message.content or ""
            self._calls_made += 1
            if hasattr(resp, "usage") and resp.usage:
                self._tokens_used += (resp.usage.total_tokens or 0)
            return True, f"API OK: {text.strip()[:50]}"
        except Exception as e:
            err_msg = str(e)
            if self._config.api_key in err_msg:
                err_msg = err_msg.replace(self._config.api_key, _REDACTED)
            return False, f"API error: {err_msg[:200]}"

    def request_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries_parse: int = 2,
    ) -> LLMResponse | None:
        if not self.is_available:
            logger.warning("LLM not available, skipping request")
            return None
        if self._calls_made >= self._config.max_calls:
            logger.warning("LLM call budget exhausted (%d/%d)", self._calls_made, self._config.max_calls)
            return None

        for attempt in range(self._config.max_retries):
            try:
                return self._do_request(system_prompt, user_prompt, max_retries_parse)
            except Exception as e:
                err_msg = str(e)
                if self._config.api_key in err_msg:
                    err_msg = err_msg.replace(self._config.api_key, _REDACTED)
                logger.warning("LLM request attempt %d failed: %s", attempt + 1, err_msg[:200])

                is_auth_error = any(code in err_msg.lower() for code in ["401", "403", "unauthorized", "forbidden"])
                is_budget_error = "budget" in err_msg.lower() or "exceeded" in err_msg.lower()
                if is_auth_error or is_budget_error:
                    logger.error("Unrecoverable API error, not retrying")
                    return None

                if attempt < self._config.max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _do_request(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries_parse: int,
    ) -> LLMResponse | None:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._config.max_tokens_per_call,
            temperature=self._config.temperature,
        )
        self._calls_made += 1
        if hasattr(resp, "usage") and resp.usage:
            self._tokens_used += (resp.usage.total_tokens or 0)
        else:
            logger.info("API did not return usage info")

        raw = resp.choices[0].message.content or ""
        return self._parse_response(raw, max_retries_parse)

    def _parse_response(self, raw: str, max_retries: int) -> LLMResponse | None:
        for attempt in range(max_retries + 1):
            json_str = self._extract_json(raw)
            if json_str is None:
                logger.warning("No JSON found in LLM response (attempt %d)", attempt + 1)
                return None
            try:
                data = json.loads(json_str)
                return LLMResponse.model_validate(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("JSON parse/validate failed (attempt %d): %s", attempt + 1, str(e)[:200])
                if attempt < max_retries:
                    raw = json_str
        return None

    @staticmethod
    def _extract_json(text: str) -> str | None:
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            return text[brace_start:brace_end + 1]
        return None


class MockLLMClient(LLMClient):
    """For testing without real API."""

    def __init__(self, responses: list[dict[str, Any]] | None = None):
        super().__init__(LLMConfig())
        self._mock_responses = responses or []
        self._mock_idx = 0

    @property
    def is_available(self) -> bool:
        return True

    def health_check(self) -> tuple[bool, str]:
        return True, "Mock API OK"

    def request_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries_parse: int = 2,
    ) -> LLMResponse | None:
        self._calls_made += 1
        if self._mock_idx < len(self._mock_responses):
            data = self._mock_responses[self._mock_idx]
            self._mock_idx += 1
            try:
                return LLMResponse.model_validate(data)
            except Exception:
                return None
        return None
