"""
Шаг Qwen: она получает FACTS (числа из диагностик и probe), политику лосса, дефолт правил
и реестр, и возвращает решение по схеме LLMDecision. Код проверяет каждое утверждение.

Принцип: Qwen не считает и не придумывает варианты вне реестра. Её работа — связать
факты с методичкой и обосновать выбор. Любое число, на которое она ссылается, обязано
совпасть с FACTS — иначе ответ отклоняется (защита от галлюцинаций).
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from .registry import METRICS, MODELS, TRANSFORMS, compact_registry, is_metric_consistent, supports
from .rules import THRESHOLDS
from .schemas import LLMDecision

SKILL_PATH = Path(__file__).with_name("SKILL.md")

# Те же переменные, что в baseline.py организаторов и в этапах Марго и Макса.
# Читаются в момент вызова, чтобы .env, загруженный раннером, точно подхватился.
def llm_settings() -> dict:
    return {"base_url": os.getenv("OPENAI_BASE_URL", "https://inference.airi.net/v1"),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("OPENAI_MODEL", "Qwen/Qwen3.8-27B"),
            "max_tokens": int(os.getenv("MODEL_SELECTION_MAX_TOKENS", "4000"))}


# ----------------------------------------------------------------------------- facts
def build_facts(diag: dict, probe: dict, rule_evidence: dict) -> dict:
    """Плоская таблица фактов. Только на эти ключи Qwen может ссылаться."""
    f = {}
    for k, v in diag["target"].items():
        f[k] = v
    for k, v in diag["features"].items():
        if isinstance(v, (int, float, bool)) or v is None:
            f[k] = v
    f["n_high_card_cols"] = len(diag["features"]["high_card_cols"])
    for r in probe["results"]:
        f[f"probe.{r['id']}.metric_mean"] = r["metric_mean"]
        f[f"probe.{r['id']}.metric_std"] = r["metric_std"]
    for k, v in rule_evidence.items():
        if isinstance(v, dict):
            f[f"{k}.rel_gain"] = v["rel_gain"]
            f[f"{k}.significant"] = v["significant"]
        else:
            f[k] = v
    return f


# ----------------------------------------------------------------------------- prompt
def build_messages(task: dict, policy: dict, facts: dict, probe: dict, rule_default: dict) -> list[dict]:
    skill = SKILL_PATH.read_text(encoding="utf-8") if SKILL_PATH.exists() else ""
    allowed = {mid: v for mid, v in compact_registry().items() if mid != "dummy_mean"}
    user = {
        "TASK": task,
        "LOSS_POLICY": policy,
        "FACTS": facts,
        "PROBE_TABLE": [{k: r[k] for k in ("id", "metric_mean", "metric_std", "fit_seconds")}
                        for r in probe["results"]],
        "RULES_DEFAULT": rule_default["candidates"],
        "ALLOWED_MODELS": allowed,
        "LIMITS": {"max_candidates": THRESHOLDS["max_candidates"], "transforms": list(TRANSFORMS)},
        "OUTPUT_SCHEMA": {
            "reasoning": [{"claim": "str", "evidence": {"<FACTS key>": "<value as in FACTS>"}}],
            "candidates": [{"model_id": "str", "loss": "str", "target_transform": "none|log1p",
                            "priority": "1..9", "role": "primary|ensemble_member",
                            "rationale": "str", "evidence": {"<FACTS key>": "<value>"}}],
            "rejected": [{"model_id": "str", "reason": "str"}],
        },
    }
    system = (
        "Ты — ML-исследователь в агентной системе. Твоя задача на этом шаге: выбрать пары "
        "(модель, лосс, трансформ таргета) для дальнейшего обучения. Гиперпараметры НЕ выбирай.\n"
        "Строго следуй методичке ниже. Отвечай ТОЛЬКО одним JSON-объектом по OUTPUT_SCHEMA, без текста вокруг.\n"
        "Каждый вывод и каждый кандидат подкрепляй ключами из FACTS с ТОЧНЫМИ значениями оттуда.\n"
        "Начни с RULES_DEFAULT и меняй его только если FACTS дают для этого основание — назови его.\n\n"
        "=== МЕТОДИЧКА ===\n" + skill
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, default=str)}]


# ----------------------------------------------------------------------------- call
def default_llm_fn(messages: list[dict]) -> str:
    from openai import OpenAI
    s = llm_settings()
    if not s["api_key"]:
        raise RuntimeError("OPENAI_API_KEY не задан")
    client = OpenAI(base_url=s["base_url"], api_key=s["api_key"], timeout=180, max_retries=2)
    resp = client.chat.completions.create(model=s["model"], messages=messages, temperature=0.2,
                                          max_tokens=s["max_tokens"])
    return resp.choices[0].message.content or ""


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("в ответе нет JSON-объекта")
    return json.loads(text[s:e + 1])


# ----------------------------------------------------------------------------- validate
def _close(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    return abs(a - b) <= max(1e-6, 0.02 * abs(b))


def validate_decision(dec: LLMDecision, facts: dict, policy: dict, diag: dict, rule_evidence: dict) -> list[str]:
    errs = []
    metric = policy["metric"]
    y_min = diag["target"]["y_min"]
    n_rows = diag["features"]["n_rows"]

    def check_evidence(ev: dict, where: str):
        if not ev:
            errs.append(f"{where}: нет evidence")
        for k, v in ev.items():
            if k not in facts:
                errs.append(f"{where}: ключа '{k}' нет в FACTS")
            elif not _close(v, facts[k]):
                errs.append(f"{where}: '{k}'={v}, а в FACTS {facts[k]}")

    for i, st in enumerate(dec.reasoning):
        check_evidence(st.evidence, f"reasoning[{i}]")

    seen = set()
    if not 1 <= len(dec.candidates) <= THRESHOLDS["max_candidates"]:
        errs.append(f"кандидатов {len(dec.candidates)}, допустимо 1..{THRESHOLDS['max_candidates']}")
    for c in dec.candidates:
        w = f"candidate {c.model_id}/{c.loss}/{c.target_transform}"
        if c.model_id not in MODELS or c.model_id == "dummy_mean":
            errs.append(f"{w}: model_id вне реестра")
            continue
        if not supports(c.model_id, c.loss):
            errs.append(f"{w}: модель не поддерживает лосс; допустимо {list(MODELS[c.model_id]['losses'])}")
        if c.target_transform == "log1p" and (y_min is None or y_min <= -1):
            errs.append(f"{w}: log1p невозможен при y_min={y_min}")
        hint = MODELS[c.model_id]["max_rows_hint"]
        if hint and n_rows > hint:
            errs.append(f"{w}: n_rows={n_rows} > max_rows_hint={hint}")
        key = (c.model_id, c.loss, c.target_transform)
        if key in seen:
            errs.append(f"{w}: дубликат")
        seen.add(key)
        # primary обязан быть согласован с метрикой ЛИБО иметь значимый выигрыш в probe
        if c.role == "primary" and c.loss in [*MODELS[c.model_id]["losses"]]:
            consistent = (c.loss == policy["base"]["loss"] and c.target_transform == policy["base"]["transform"]) \
                or any(a["consistent"] and a["loss"] == c.loss and a["transform"] == c.target_transform
                       for a in policy["alternatives"])
            gain = rule_evidence.get(f"gain_lightgbm_{c.loss}_{c.target_transform}_vs_base")
            if not consistent and not (gain and gain["significant"]):
                errs.append(f"{w}: role=primary, но пара не согласована с метрикой {metric} "
                            f"и не выиграла значимо в probe")
        check_evidence(c.evidence, w)
    if not any(MODELS.get(c.model_id, {}).get("family") == "boosting" for c in dec.candidates):
        errs.append("нет ни одного бустинга: на табличных данных это почти всегда ошибка")
    return errs


# ----------------------------------------------------------------------------- run
def decide(task: dict, policy: dict, diag: dict, probe: dict, rule_default: dict,
           llm_fn: Callable[[list[dict]], str] | None = None, max_attempts: int = 2) -> dict:
    """Возвращает {"decision": LLMDecision|None, "errors": [...], "raw": [...]}.
    None -> вызывающий код берёт фолбэк правил."""
    llm_fn = llm_fn or default_llm_fn
    facts = build_facts(diag, probe, rule_default["evidence"])
    messages = build_messages(task, policy, facts, probe, rule_default)
    raws, errors = [], []
    for attempt in range(max_attempts):
        try:
            raw = llm_fn(messages)
        except Exception as e:
            errors.append(f"attempt {attempt + 1}: LLM call failed: {type(e).__name__}: {str(e)[:200]}")
            break
        raws.append(raw)
        try:
            dec = LLMDecision.model_validate(extract_json(raw))
            errs = validate_decision(dec, facts, policy, diag, rule_default["evidence"])
        except (ValueError, ValidationError) as e:
            errs = [f"парсинг/схема: {str(e)[:400]}"]
            dec = None
        if not errs:
            return {"decision": dec, "errors": errors, "raw": raws, "facts": facts}
        errors.extend(f"attempt {attempt + 1}: {e}" for e in errs)
        # ретрай: возвращаем Qwen её ответ и список ошибок
        messages = messages + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": "Валидатор отклонил ответ. Исправь ТОЛЬКО эти ошибки "
                                "и верни полный JSON заново:\n- " + "\n- ".join(errs)}]
    return {"decision": None, "errors": errors, "raw": raws, "facts": facts}
