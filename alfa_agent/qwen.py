"""Small adapter for the OpenAI-compatible interface used by the baseline."""

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import math
import re
import time

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, DefaultHttpxClient, DefaultAsyncHttpxClient

from .config import Settings


class QwenError(RuntimeError):
    """A printable error without raw provider responses or credentials."""


def connection_options(settings: Settings, *, http_client=None, asynchronous=False) -> dict:
    """Keep proxy routing parameters out of the SDK's endpoint path concatenation."""
    url = httpx.URL(settings.base_url)
    options = {'base_url': settings.base_url}
    if url.query:
        options['base_url'] = url.copy_with(query=None, fragment=None)
        # HTTPX preserves repeated keys and empty values; the SDK's default_query
        # serializer changes arrays into key[] and drops empty values.
        if http_client is None:
            factory = DefaultAsyncHttpxClient if asynchronous else DefaultHttpxClient
            http_client = factory(params=url.params)
        else:
            http_client.params = http_client.params.merge(url.params)
    if http_client is not None:
        options['http_client'] = http_client
    return options


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


def reply_text(response) -> str:
    """Text of the first choice, tolerating a missing message and content sent as a list of parts."""
    choice = (getattr(response, 'choices', None) or [None])[0]
    content = getattr(getattr(choice, 'message', None), 'content', None)
    if isinstance(content, list):  # [{'type': 'text', 'text': ...}] or objects with .text
        parts = [part.get('text') if isinstance(part, dict) else getattr(part, 'text', None)
                 for part in content]
        # Reject the whole malformed answer: dropping a bad part could accept incomplete JSON.
        if any(part is not None and not isinstance(part, str) for part in parts):
            return ''
        content = ''.join(part or '' for part in parts)
    return content if isinstance(content, str) else '' if content is None else str(content)


@dataclass(frozen=True)
class Answer:
    text: str
    model: str
    usage: dict | None


def ask_qwen(
    settings: Settings,
    prompt: str,
    *,
    system_prompt: str = "Ты помощник команды ML-хакатона. Отвечай кратко и по существу.",
    http_client: httpx.Client | None = None,
) -> Answer:
    settings.validate()
    if not prompt.strip():
        raise ValueError("Промпт не должен быть пустым.")
    try:
        with OpenAI(
            api_key=settings.api_key,
            **connection_options(settings, http_client=http_client),
            timeout=settings.timeout_seconds,  # None waits for the answer
            max_retries=0,
        ) as client:
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                # The organizer's endpoint uses this compatible chat interface.
                **({'max_tokens': settings.max_tokens} if settings.max_tokens is not None else {}),
                reasoning_effort="low",
            )
    except APITimeoutError:
        raise QwenError("Qwen не ответил до истечения таймаута. Повторный запрос автоматически не отправлялся.") from None
    except APIConnectionError:
        raise QwenError("Не удалось подключиться к Qwen. Проверьте адрес API и сеть.") from None
    except APIStatusError as error:
        hints = {
            401: "Проверьте ключ и срок его действия.",
            403: "Проверьте доступ команды к этому API.",
            404: "Проверьте base URL и API-имя модели.",
            429: "Достигнут лимит запросов или бюджет команды.",
        }
        hint = hints.get(error.status_code, "Проверьте совместимость API и состояние сервиса.")
        raise QwenError(f"Qwen API вернул HTTP {error.status_code}. {hint}") from None
    text = reply_text(response)
    if not text:
        raise QwenError("API не вернул текст ответа. Проверьте формат ответа и лимит генерации.")
    return Answer(
        text=text,
        model=response.model,
        usage=response.usage.model_dump() if response.usage else None,
    )
