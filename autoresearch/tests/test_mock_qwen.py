"""
Проверка ветки Qwen без API: мок отвечает как Qwen, включая заведомо плохие ответы.
Проверяем: память доходит до модели, валидатор ловит ошибки, повтор с ошибками работает,
код признаков Qwen исполняется в песочнице, запрещённый код отклоняется, ранний stop отклоняется,
выводы и уроки Qwen попадают в память и отчёт.

    python -m autoresearch.tests.test_mock_qwen
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autoresearch.run import run  # noqa: E402

CALLS = {"researcher": 0}
SEEN = {"memory_has_all_experiments": [], "retries_after_errors": 0}


def spec(blocks, **model):
    return {"blocks": blocks, "model": {"model_id": "lightgbm", "loss": "l2", "target_transform": "none", "params": {}, **model},
            "training": {"sample_weight": "none", "recent_decay": 0.7, "drop_patterns": [], "target_clip_quantile": None}}


def mock(messages):
    system, user = messages[0]["content"], messages[1]["content"]
    retry = len(messages) > 2
    if retry:
        SEEN["retries_after_errors"] += 1
    if "аналитик ML-задачи" in system:
        return json.dumps({"domain_summary": "Годовое удельное энергопотребление зданий (site_eui), наблюдения по годам.",
                           "metric": "rmse", "metric_quote": "RMSE", "risks": ["повторы зданий по годам", "сдвиг погоды"],
                           "important_columns": ["facility_type", "floor_area"]}), {}
    if "По фактам аудита" in system:
        alt = [["floor_area", "year_built", "facility_type", "State_Factor"]] + ([] if retry else [["no_such_col"]])
        return json.dumps({"key_findings": ["Хвост: 1% строк = 49% квадрата отклонений", "Test — следующий год, AUC сдвига 1.0"],
                           "strategy": ["валидация по времени", "история здания"], "entity_key_alternatives": alt}), {}
    # researcher
    mem = json.loads(user.split("MEMORY:\n", 1)[1])
    exps = mem["ALL_EXPERIMENTS_IN_ORDER"]
    SEEN["memory_has_all_experiments"].append(len(exps))
    last = exps[-1]["id"]
    best = mem["BEST"]
    analysis = {"experiment_id": last, "verdict": "confirmed" if exps[-1].get("verdict") in ("new_best", "baseline") else "rejected",
                "lesson": f"{last}: cv={exps[-1].get('cv')}"}
    if not retry:
        CALLS["researcher"] += 1
    n = CALLS["researcher"] - 1
    b = best["spec"]["blocks"]
    script = [
        # 0: невалидный параметр → повтор → валидный
        ({"decision": "experiment", "proposal": {"hypothesis": "история здания", "why": "test_rows_with_history=0.79", "parent": best["id"],
                                                  "spec": spec(b + ["entity_history"], params={"num_leaves": 5000})}},
         {"decision": "experiment", "proposal": {"hypothesis": "история здания", "why": "test_rows_with_history=0.79", "parent": best["id"],
                                                  "spec": spec(b + ["entity_history"])}}),
        # 1: код признаков Qwen
        ({"decision": "experiment", "proposal": {"hypothesis": "возраст здания и площадь на год", "why": "year_built и floor_area в топе",
                                                  "parent": best["id"], "spec": spec(b),
                                                  "custom_feature_code": "def add_features(df):\n    out = pd.DataFrame(index=df.index)\n"
                                                  "    out['age'] = df['Year_Factor'] + 2010 - df['year_built']\n"
                                                  "    out['log_area'] = np.log1p(df['floor_area'])\n    out['area_x_age'] = out['log_area'] * out['age']\n    return out\n",
                                                  "custom_feature_description": "возраст и площадь"}}, None),
        # 2: запрещённый код → повтор → безопасная гипотеза
        ({"decision": "experiment", "proposal": {"hypothesis": "читаю файл", "why": "-", "parent": best["id"], "spec": spec(b),
                                                  "custom_feature_code": "import os\ndef add_features(df):\n    return pd.DataFrame({'a': [1]})\n"}},
         {"decision": "experiment", "proposal": {"hypothesis": "группы в прошлом", "why": "строки без истории", "parent": best["id"],
                                                  "spec": spec(b + ["group_history"])}}),
        # 3: ранний stop → отклоняется → эксперимент с лоссом gamma
        ({"decision": "stop"},
         {"decision": "experiment", "proposal": {"hypothesis": "gamma согласован с RMSE", "why": "skew 4.8, y>0", "parent": best["id"],
                                                  "spec": spec(b, loss="gamma")}}),
        ({"decision": "experiment", "proposal": {"hypothesis": "сильнее регуляризация", "why": "разрыв train/val", "parent": best["id"],
                                                  "spec": spec(b, loss=best["spec"]["model"]["loss"], params={"num_leaves": 31, "min_child_samples": 100})}}, None),
    ]
    if n >= len(script):
        step = {"decision": "stop"}
    else:
        first, second = script[n]
        step = second if (retry and second is not None) else first
    step.update(analysis=analysis, new_insights=[f"мок-вывод после {last}"], backlog_add=["проверить catboost"])
    return json.dumps(step, ensure_ascii=False), {"prompt_tokens": len(system + user) // 4, "completion_tokens": 200}


if __name__ == "__main__":
    out = ROOT / "artifacts" / "mock_qwen"
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    rep = run(str(ROOT / "data/regression/task.md"), str(out), no_llm=False, total_budget=1200, max_experiments=12,
              llm_fn=mock, write_next_to_task=False)
    exps = rep["experiments"]
    j = [json.loads(l) for l in (out / "journal.jsonl").read_text().splitlines()]
    checks = {
        "qwen_experiments": sum(e["source"] == "qwen" for e in exps),
        "qwen_custom_block_used": any("custom_1" in (e["summary"] or "") for e in exps),
        "forbidden_code_rejected": any(x["kind"] == "llm_call" and any("import" in err for err in x.get("errors", [])) for x in j),
        "bad_param_rejected": any(x["kind"] == "llm_call" and any("num_leaves=5000" in err for err in x.get("errors", [])) for x in j),
        "early_stop_rejected": any(x["kind"] == "llm_call" and any("рано останавливаться" in err for err in x.get("errors", [])) for x in j),
        "bad_entity_key_rejected": any(x["kind"] == "llm_call" and any("no_such_col" in err for err in x.get("errors", [])) for x in j),
        "retries_after_errors": SEEN["retries_after_errors"],
        "memory_grows_each_call": SEEN["memory_has_all_experiments"],
        "qwen_lessons_in_report": sum(1 for e in exps if e.get("analysis")),
        "qwen_insights": sum(1 for i in rep["insights"] if i["source"].startswith("qwen")),
        "stop_reason": rep["stages"]["5_research"]["summary"]["stop_reason"],
        "final_ok": rep["submission_checks"]["ok"],
    }
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    assert checks["qwen_experiments"] >= 4 and checks["qwen_custom_block_used"] and checks["forbidden_code_rejected"]
    assert checks["bad_param_rejected"] and checks["early_stop_rejected"] and checks["final_ok"]
    print("OK")
