"""Client for OpenAI-compatible chat completions."""

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import math
import re
import time

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import Settings


class LLMError(RuntimeError):
    """A printable error without raw provider responses or credentials."""


class LLMTemporaryError(LLMError):
    """A connection, server or rate-limit failure that may be retried within run limits."""


def api_error_details(error: APIStatusError) -> dict:
    """Classify provider failures without saving their raw message or response body."""
    body = error.body
    if isinstance(body, dict):
        body = body.get('error', body)
    fields = body if isinstance(body, dict) else {}
    message = fields.get('message', '') if fields else body
    description = ' '.join(str(value or '') for value in (message, fields.get('code'), fields.get('type'))).casefold()
    quota = any(marker in description for marker in (
        'insufficient_quota', 'quota_exceeded', 'exceeded your current quota',
        'billing_hard_limit', 'insufficient balance', 'insufficient credits',
        'credit balance is too low', 'out of credits',
    )) or ('budget' in description and any(word in description for word in ('exceeded', 'exhausted', 'insufficient')))
    kind = ('quota_exceeded' if quota else 'rate_limit' if error.status_code == 429
            else 'server_error' if error.status_code >= 500 else 'api_error')
    details = {'status_code': error.status_code, 'kind': kind}
    for name in ('code', 'type'):
        value = fields.get(name)
        if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value):
            details['provider_' + name] = value
    # Retry-After supports seconds or an HTTP date; some proxies send milliseconds.
    if kind == 'rate_limit':
        headers = error.response.headers
        delay = None
        try:
            if 'retry-after-ms' in headers:
                delay = float(headers['retry-after-ms']) / 1000
            elif 'retry-after' in headers:
                value = headers['retry-after']
                try:
                    delay = float(value)
                except ValueError:
                    delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            pass
        if delay is not None and math.isfinite(delay):
            details['retry_after_seconds'] = max(1., delay)
    return details


@dataclass(frozen=True)
class Answer:
    text: str
    model: str
    usage: dict | None


def ask_llm(
    settings: Settings,
    prompt: str,
    *,
    system_prompt: str = "Ты помощник по машинному обучению. Отвечай кратко и по существу.",
    http_client: httpx.Client | None = None,
) -> Answer:
    settings.validate()
    if not prompt.strip():
        raise ValueError("Промпт не должен быть пустым.")
    try:
        with OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
            http_client=http_client,
        ) as client:
            kwargs = settings.reasoning_options()
            if settings.max_tokens is not None:
                kwargs['max_tokens'] = settings.max_tokens
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            )
    except APITimeoutError:
        raise LLMError("LLM не ответил до истечения таймаута. Повторный запрос автоматически не отправлялся.") from None
    except APIConnectionError:
        raise LLMError("Не удалось подключиться к LLM. Проверьте адрес API и сеть.") from None
    except APIStatusError as error:
        hints = {
            401: "Проверьте ключ и срок его действия.",
            403: "Проверьте доступ к этому API.",
            404: "Проверьте base URL и API-имя модели.",
            429: "Достигнут лимит запросов или бюджет API.",
        }
        hint = hints.get(error.status_code, "Проверьте совместимость API и состояние сервиса.")
        raise LLMError(f"LLM API вернул HTTP {error.status_code}. {hint}") from None
    if not response.choices or not response.choices[0].message.content:
        raise LLMError("API не вернул текст ответа. Проверьте формат ответа и лимит генерации.")
    return Answer(
        text=response.choices[0].message.content,
        model=response.model,
        usage=response.usage.model_dump() if response.usage else None,
    )
