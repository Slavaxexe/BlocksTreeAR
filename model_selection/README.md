# Этап: выбор семейств моделей и функции потерь (регрессия)

Место в пайплайне: **Марго (понимание задачи + EDA) → этот этап → Макс (сплит, гиперпараметры, обучение, валидация)**.

Отвечает на один вопрос: **какие пары (модель, лосс, трансформ таргета) обучать и в каком приоритете**.
Гиперпараметры, оптимизаторы, регуляризация, схема валидации — НЕ здесь, это этап Макса.

## Как устроено

```
target_metric (из описания задачи) ─┐
train.csv ─► diagnostics ───────────┼─► loss_policy ─► probe_plan ─► probe (≤5 мин) ─► rules default
                                    │                                                     │
                                    └──────────── FACTS ◄──────────────────────────────────┤
                                                    │                                      │
                                         Qwen + SKILL.md ─► validator ─(ok)─► output ◄─────┘ (fallback)
```

| Слой | Файл | Кто решает |
|---|---|---|
| Реестр моделей/лоссов/метрик | `registry.py` | общий контракт с Максом |
| Сборка эстиматора по (model, loss, transform) | `builders.py` | код (Макс может импортировать) |
| Метрики на исходной шкале | `metrics.py` | код |
| Диагностики таргета и признаков | `diagnostics.py` | код |
| Быстрый пробный прогон | `probe.py` | код |
| Политика лосса, план probe, дефолт | `rules.py` | код (фолбэк) |
| Методичка для Qwen | `SKILL.md` | Qwen читает |
| Промпт, вызов, валидатор, ретрай | `llm_step.py` | Qwen выбирает, код проверяет |
| Точка входа | `stage.py` | — |

Принцип: **Qwen не считает и не придумывает варианты**. Она получает FACTS (числа),
политику лосса и дефолт правил, и выбирает внутри реестра с обоснованием. Валидатор
отклоняет ответ, если: модель вне реестра, лосс не поддерживается моделью, log1p при y≤-1,
основная пара не согласована с метрикой без значимого выигрыша в probe, ссылка на
несуществующий ключ FACTS или число не совпадает с FACTS, нет ни одного бустинга, дубликаты.
1 ретрай с ошибками → иначе дефолт правил (`source: "rules_fallback"`). Пайплайн не падает.

## Ключевая логика выбора лосса
Метрика вознаграждает статистику: RMSE → среднее, MAE → медиану, RMSLE → среднее в log-шкале,
pinball → квантиль. Основная пара должна оценивать ту же статистику. Для RMSE это `l2`, а также
`poisson/gamma/tweedie` (log-link, тоже оценивают среднее). `huber`, `l1`, `l2+log1p` при RMSE —
только если значимо выиграли в probe (на синтетике со скошенным y все трое проиграли l2 — как и
предсказывает теория).

## Запуск
```bash
python -m model_selection.stage model_selection/contracts/example_input.json output.json            # с Qwen
python -m model_selection.stage model_selection/contracts/example_input.json output.json --no-llm   # только правила
```
```python
from model_selection.stage import run_stage
out = run_stage(input_dict, out_path="stage_output.json")
```
Переменные окружения LLM: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (фолбэк на `OPENAI_*`) —
**сверить имена с baseline.py организаторов**. Клиент — пакет `openai` (в requirements.txt нет
`langchain-openai`).

## Контракты
- `contracts/stage_input.schema.json` — что ждём от Марго. Обязательно: `task.target`,
  `task.target_metric` (распарсенная из описания), `paths.train`. Остальное опционально.
- `contracts/stage_output.schema.json` — что отдаём Максу. Пример: `contracts/example_output.json`.

Запуск — из корня репозитория (папка `model_selection` — Python-пакет с относительными импортами,
не конфликтует с одноимёнными модулями команды).
- `contracts/llm_decision.schema.json` — формат ответа Qwen.

Что берёт Макс из выхода:
- `candidates[]` по `priority`: `model_id`, `estimator`, `loss_native` (готовые kwargs лосса),
  `target_transform` (если `log1p` — обучать на log1p(y), предсказывать expm1),
  `optimizer_applicable` (к модели применима его ветка SGD/momentum/Nesterov/Adam — сейчас это
  `sgd_linear` и `mlp`; для бустингов ветка не нужна), `needs_scaling`, `native_categorical`.
- `target_metric` — по ней оценивать (на исходной шкале), не по лоссу.
- `automl_hint` — готовые `estimator_list`/`eval_metric` для FLAML и AutoGluon.
- `hints.time_col/group_col` — чтобы его сплит совпал по типу со сплитом probe.

## Бюджет
Probe ≈ 25 с на 15k строк × 43 признака на 2 CPU (синтетика). На платформе (8 CPU, 30k строк) —
ожидаемо 30–90 с. Жёсткий лимит `budget.probe_seconds` (по умолчанию 300): при превышении
оставшиеся конфигурации пропускаются и записываются в `probe.skipped`.

## Тест
```bash
python -m model_selection.tests.run_synthetic   # фолбэк, валидный мок Qwen, мок с галлюцинациями, метрики MAE/RMSLE/неизвестная
```

## Проверено на версиях из requirements.txt
Все нативные имена лоссов прогнаны на sklearn 1.5.2, lightgbm 4.6.0, xgboost 2.1.4, catboost 1.2.8.
Имена FLAML 2.3.6 и AutoGluon 1.3.1 (модели и метрики) сверены с установленными пакетами.
LightAutoML не включён в реестр (совместимость с Python 3.12 не проверена).
