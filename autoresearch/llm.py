"""
Клиент Qwen для всех ролей пайплайна.

- Ответ всегда JSON, проверяется pydantic-схемой и смысловым валидатором вызывающего кода.
  При ошибке — один повтор, в который отдаются ошибки; затем вызывающий код использует фолбэк.
- 429 с retry-after — ждём (в пределах дедлайна). Исчерпание квоты/бюджета или 3 сбоя подряд —
  LLM отключается до конца запуска, пайплайн продолжает на эвристиках (не падает).
- Каждый вызов (роль, размер промпта, сырой ответ, токены) пишется в журнал памяти.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from pydantic import BaseModel, ValidationError

from .config import Settings
from .memory import Memory

LLMFn = Callable[[list[dict]], tuple[str, dict]]


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("в ответе нет JSON-объекта")
    return json.loads(text[s:e + 1])


class LLM:
    def __init__(self, settings: Settings, memory: Memory, fn: LLMFn | None = None):
        self.s, self.mem, self.fn = settings, memory, fn
        self.enabled = settings.llm_enabled and (fn is not None or bool(settings.api_key))
        self.json_mode = True
        self.consecutive_failures = 0
        if not self.enabled:
            self._disable("нет OPENAI_API_KEY или запуск с --no-llm")

    # --------------------------------------------------------------- транспорт
    def _call_api(self, messages: list[dict], timeout: float) -> tuple[str, dict]:
        if self.fn is not None:
            return self.fn(messages)
        from openai import APIStatusError, OpenAI
        client = OpenAI(base_url=self.s.base_url, api_key=self.s.api_key, timeout=timeout, max_retries=0)
        kwargs = dict(model=self.s.model, messages=messages, temperature=0.3, max_tokens=self.s.llm_max_tokens)
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            r = client.chat.completions.create(**kwargs)
        except APIStatusError as e:
            # 400 от сервера: сначала отказываемся от JSON-режима, затем уменьшаем max_tokens — и пробуем снова
            if e.status_code == 400 and self.json_mode:
                self.json_mode = False
                self.mem.log("llm_adapt", change="json_mode off", error=str(e)[:200])
                return self._call_api(messages, timeout)
            if e.status_code == 400 and self.s.llm_max_tokens > 2048:
                self.s.llm_max_tokens = 2048
                self.mem.log("llm_adapt", change="max_tokens=2048", error=str(e)[:200])
                return self._call_api(messages, timeout)
            raise
        usage = r.usage.model_dump() if r.usage else {}
        choice = r.choices[0]
        text = choice.message.content or ""
        if not text.strip() and getattr(choice.message, "reasoning_content", None):
            text = choice.message.reasoning_content  # некоторые серверы кладут ответ рядом с рассуждением
        usage["finish_reason"] = choice.finish_reason
        return text, usage

    def ping(self) -> tuple[bool, str]:
        """Короткая проверка ключа и адреса: один крошечный запрос."""
        try:
            raw, usage = self._call_api([{"role": "user", "content": 'Ответь JSON: {"ok": true}'}], timeout=60)
            ok = '"ok"' in raw.replace(" ", "") or "ok" in raw.lower()
            return ok, f"модель {self.s.model} @ {self.s.base_url}: ответ «{raw.strip()[:80]}», токены {usage}"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:300]}"

    def _disable(self, reason: str):
        self.enabled = False
        self.mem.state["llm"]["disabled_reason"] = reason
        self.mem.log("llm_disabled", reason=reason)

    def _transport(self, messages: list[dict], deadline: float) -> tuple[str, dict] | None:
        for attempt in range(4):
            remaining = deadline - time.monotonic()
            if remaining < 20:
                self.mem.log("llm_skip", reason="нет времени до дедлайна")
                return None
            try:
                return self._call_api(messages, timeout=min(self.s.llm_timeout_s, remaining - 10))
            except Exception as e:  # сеть, 429, 5xx
                code = getattr(e, "status_code", None)
                text = str(e)[:300]
                self.mem.state["llm"]["failures"] += 1
                self.mem.log("llm_error", attempt=attempt + 1, status=code, error=text)
                if code in (401, 403) or "quota" in text.lower() or "budget" in text.lower() or "insufficient" in text.lower():
                    self._disable(f"доступ/квота: {code} {text[:120]}")
                    return None
                wait = 5 * (attempt + 1)
                headers = getattr(getattr(e, "response", None), "headers", {}) or {}
                if code == 429:
                    try:
                        wait = float(headers.get("retry-after-ms", 0)) / 1000 or float(headers.get("retry-after", wait))
                    except (TypeError, ValueError):
                        pass
                if time.monotonic() + wait > deadline - 30:
                    return None
                time.sleep(min(wait, 60))
        return None

    # --------------------------------------------------------------- основной вызов
    def ask(self, role: str, system: str, user: str, schema: type[BaseModel], *, deadline: float,
            validator: Callable[[BaseModel], list[str]] | None = None, max_attempts: int = 2) -> BaseModel | None:
        if not self.enabled:
            return None
        if self.mem.state["llm"]["calls"] >= self.s.llm_max_calls:
            self._disable(f"достигнут лимит вызовов {self.s.llm_max_calls}")
            return None
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(max_attempts):
            t = time.monotonic()
            out = self._transport(messages, deadline)
            if out is None:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3 and self.enabled:
                    self._disable("3 сбоя API подряд")
                return None
            raw, usage = out
            llm = self.mem.state["llm"]
            llm["calls"] += 1
            llm["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            llm["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            errors: list[str] = []
            obj = None
            try:
                obj = schema.model_validate(extract_json(raw))
                errors = validator(obj) if validator else []
            except (ValueError, ValidationError) as e:
                errors = [f"формат: {str(e)[:600]}"]
                if usage.get("finish_reason") == "length":
                    errors.append("ответ обрезан по длине: верни только JSON, коротко, без рассуждений вне него")
            self.mem.log("llm_call", role=role, attempt=attempt + 1, prompt_chars=len(system) + len(user),
                         seconds=round(time.monotonic() - t, 1), usage=usage, raw=raw[:20000], errors=errors)
            if not errors:
                self.consecutive_failures = 0
                return obj
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Код-валидатор отклонил ответ. Исправь ТОЛЬКО это и верни полный JSON заново:\n- "
                                            + "\n- ".join(errors[:12])}]
        self.mem.log("llm_rejected", role=role)
        return None
