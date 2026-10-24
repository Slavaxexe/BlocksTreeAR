"""Small adapter for the OpenAI-compatible interface used by the baseline."""

from dataclasses import dataclass

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import Settings


class QwenError(RuntimeError):
    """A printable error without raw provider responses or credentials."""


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
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,  # None waits for the answer
            max_retries=0,
            http_client=http_client,
        ) as client:
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                # The organizer's endpoint uses this compatible chat interface.
                max_tokens=settings.max_tokens,
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
