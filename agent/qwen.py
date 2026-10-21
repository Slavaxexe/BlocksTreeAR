"""Small adapter for the OpenAI-compatible interface used by the baseline."""

from dataclasses import dataclass

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import Settings


class QwenError(RuntimeError):
    """A printable error without raw provider responses or credentials."""


class QwenTemporaryError(QwenError):
    """A timeout, connection failure or server error that may be retried within run limits."""


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
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
            http_client=http_client,
        ) as client:
            kwargs = {'max_tokens': settings.max_tokens} if settings.max_tokens is not None else {}
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
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
    if not response.choices or not response.choices[0].message.content:
        raise QwenError("API не вернул текст ответа. Проверьте формат ответа и лимит генерации.")
    return Answer(
        text=response.choices[0].message.content,
        model=response.model,
        usage=response.usage.model_dump() if response.usage else None,
    )
