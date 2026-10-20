# Минимальные контракты агентов и structured output Qwen

Проверено 20 октября 2026. Исследование документации; реальный endpoint, модель и скриншот в этой подзадаче не проверялись. Основные источники: Pydantic, vLLM, Alibaba Cloud Model Studio, scikit-learn, Anthropic. Документация Qwen дополнительно использована для проверки расхождения версий API.

## Какие контракты и исполнение стоит реализовать сегодня?

### Takeaway
Рекомендация: передавать между этапами маленькие Pydantic-модели; каждый ответ LLM локально валидировать. LLM предлагает типизированную конфигурацию эксперимента, а заранее написанный Python-код строит и выполняет разрешённый pipeline.

### Cited Findings
- `model_validate_json()` проверяет JSON по модели; Pydantic поддерживает strict validation, а `extra='forbid'` запрещает неизвестные поля. По умолчанию лишние поля игнорируются, а некоторые значения приводятся к ожидаемому типу. `model_construct()` пропускает валидацию и для этой границы не подходит. — [Pydantic Models](https://pydantic.dev/docs/validation/latest/concepts/models/)
- `Pipeline` объединяет преобразования и финальную модель; его шаги можно совместно проверять cross-validation. Официальный пример обучает scaler и модель внутри pipeline, избегая попадания test set в обучение. — [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

### Inferences
Следующая схема — инженерная рекомендация для прототипа, а не готовый стандарт из источников:

- `TaskSpec`: `task_kind` из enum, `target_column`, запрещённые признаки, ограничения, `unresolved_fields`, ссылки на исходные свидетельства.
- `DataProfile`: вычисленные кодом размеры, типы, доли пропусков, уникальность, распределение target, кандидаты group/time/id. Не поручать LLM подсчёт статистик.
- `MetricSpec`: разрешённый `metric_id`, направление сравнения, параметры scorer, требуемый вид prediction, `split_kind`, число folds, seed. Не угадывать неизвестную метрику.
- `HypothesisSpec`: id, краткое объяснение, `evidence_ids`, `pipeline_id` из enum, типизированные и ограниченные параметры, оценка затрат. Эффект описывать качественно, без выдуманного будущего score.
- `ExperimentResult`: id гипотезы и split, статус `ok/failed/timeout`, реальные scores по folds, длительность, seed, ошибка. `RunSummary`: лучший результат, изменение к baseline, использованный бюджет, причины отклонений.

Общий envelope: `schema_version`, `run_id`, `status`, `warnings`. Ограничить длины строк/списков, допустимые enum и диапазоны чисел. Дополнительно проверять существование колонок, совместимость task/metric, согласованность split и доступность `evidence_ids`. Такие проверки должны исполняться локально: JSON-форма сама по себе не доказывает смысловую корректность. — вывод из [механизмов валидации Pydantic](https://pydantic.dev/docs/validation/latest/concepts/models/)

Исполняемый план должен ссылаться на статический registry разработчика: например, разрешённые preprocessing/model combinations с ограниченными параметрами. Запретить поля с Python-кодом, `eval`, `exec`, shell-командами, произвольными import/class/path. Сам `Pipeline` не является sandbox: ограничение создаёт registry. Использовать один зафиксированный split для baseline и кандидатов, обучать preprocessing внутри train folds. — предлагаемая граница исполнения, опирающаяся на [интерфейс sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

### Gaps
- Без исходной ML-задачи нельзя выбрать точную метрику, вид split или состав pipeline. Их следует оставить явно неразрешёнными до получения данных/условий.
- Поддержку конкретных версий зависимостей на Python 3.12 здесь не проверяли запуском; зафиксировать версии и выполнить минимальный интеграционный прогон.

## Что действительно гарантируют Qwen, JSON mode и structured outputs?

### Takeaway
`json_object` и schema-constrained output — разные возможности. Заявление на скриншоте «structured outputs» не подтверждает, что конкретный провайдер применяет переданную JSON Schema; нужны документация endpoint, модель/режим/версия сервера и проверка запросом.

### Cited Findings
- Alibaba Model Studio отличает JSON Object mode — корректный JSON без гарантии ключей/типов — от JSON Schema mode, ограничивающего структуру. Для первого используется `response_format={"type":"json_object"}`, для второго — `type="json_schema"` и схема; поддержка зависит от модели. Документация также предупреждает: некоторые модели в thinking mode принимают JSON mode без ошибки, но могут вернуть невалидный JSON. — [Alibaba Qwen structured output](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)
- vLLM показывает Chat Completions с `response_format.type="json_schema"`, схемой из `model_json_schema()` и альтернативными `structured_outputs` параметрами. Старые поля `guided_json` и аналогичные удалены в v0.12.0. Для Qwen3 Coder с reasoning есть отдельное предупреждение о возможном отключении structured outputs и требуемом server flag. — [vLLM Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/)
- Страница Qwen о vLLM ещё направляет к `guided_json`; на ней также отмечено, что `enable_thinking` и parsing reasoning не являются полностью OpenAI-compatible возможностями. Это реальное расхождение документации разных версий, а не универсальный формат запроса. — [Qwen vLLM deployment](https://qwen.readthedocs.io/en/stable/deployment/vllm.html)

### Inferences
- Реализовать маленький adapter возможностей: `json_schema`, `json_object` или `prompt_only`, выбранный по проверенной конфигурации endpoint. Не объявлять schema guarantee только потому, что сервер принимает OpenAI-compatible Chat Completions.
- При старте проверить запрос с обязательным полем, enum и запретом дополнительных полей; добавить противоречащую схеме инструкцию и проверить локальный результат. Проба выявляет игнорирование параметров, но конечное число успешных запросов не доказывает гарантию декодера. Устойчивую гарантию связывать с задокументированной и контролируемой серверной реализацией.
- При неподдерживаемой schema mode явно включать fallback с локальной валидацией и фиксировать downgrade. Даже при constrained decoding продолжать проверять значения и связи между полями. Не отправлять автоматически vLLM-specific параметры произвольному провайдеру.
- Для коротких извлечений и гипотез предпочесть non-thinking, если точная модель и endpoint это поддерживают; не смешивать reasoning-текст с JSON `content`.

Эти рекомендации следуют из различия гарантий [Alibaba](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output), версионных/API-ограничений [vLLM](https://docs.vllm.ai/en/stable/features/structured_outputs/) и [Qwen](https://qwen.readthedocs.io/en/stable/deployment/vllm.html).

### Gaps
- Неизвестны поставщик, точный model id, serving backend/version, thinking mode и поддерживаемый subset JSON Schema. Данные о Model Studio нельзя автоматически переносить на любой endpoint Qwen.
- У страницы Alibaba есть несогласованность: сравнительная строка говорит о selected qwen-plus, а подробный список JSON Schema включает другие серии. Не фиксировать перечень моделей по этой строке; проверять документацию конкретного deployment и реальный ответ.
- Фраза Alibaba об устранении дополнительной валидации относится к структуре; она не устанавливает истинность ML-гипотез, отсутствие leakage или корректность выбора target.

## Какая минимальная оркестрация и восстановление ошибок нужны?

### Takeaway
Рекомендация: фиксированный workflow с независимыми короткими извлечениями, программным join и ограниченным циклом экспериментов. Детерминированный код считает и ранжирует результаты; LLM объясняет и предлагает следующую гипотезу.

### Cited Findings
- Anthropic разделяет заранее определённые workflows и автономных agents, рекомендует начинать с простых компонуемых паттернов. Sectioning запускает независимые подзадачи параллельно и агрегирует выходы программно. Для agent loops подчёркнуты реальные результаты среды и ограничение числа итераций. — [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

### Inferences
1. Заморозить исходный brief и вычисленный `DataProfile`; запустить task/data-description/metric extractors асинхронно, с ограничением конкурентности, timeout и одинаковой неизменяемой входной информацией. Это функциональные вызовы, им не нужны чаты друг с другом.
2. После join проверить смысловую совместимость трёх выходов. Независимость вызовов не означает независимость решений: выбор метрики и split часто зависит от task. Конфликт вернуть как unresolved/error, не скрывать автоматическим «консенсусом».
3. `Summarizer` реализовать обычной функцией: она объединяет валидированные поля, вычисляет delta, учитывает направление метрики и выбирает лучший сопоставимый trial. Если нужен красивый текст, отдельный LLM получает уже рассчитанные цифры и ссылки на trial.
4. Генератор гипотез каждый раз получает только `TaskSpec`, компактный `DataProfile`, `MetricSpec`, baseline и последние измеренные результаты. Stateless-вызов возвращает одну или несколько ограниченных `HypothesisSpec`; полный растущий диалог не нужен.
5. На JSON/validation error разрешить не более одного repair-вызова: схема, исходный ответ и короткие ошибки валидации. Затем повторно валидировать; при провале зафиксировать ошибку этапа. Не исправлять JSON regex-подстановками, не проглатывать неизвестные поля, не запускать бесконечные repair loops. Отдельно ограничить transport retries и общий wall-clock/token budget.
6. Лимиты сегодняшнего MVP: фиксированное число гипотез, разрешённых pipelines и параллельных обучений; остановка по числу trials или времени. Это проектное решение для управляемости, не заявленный оптимум.

Основание для фиксированной декомпозиции, программной агрегации, feedback от реального исполнения и stop conditions: [Anthropic](https://www.anthropic.com/engineering/building-effective-agents). Значения лимитов и предложенные контракты являются рекомендациями автора исследования.

### Gaps
- Источники не устанавливают оптимальное число агентов, repair attempts или лимиты времени для этого датасета. Один repair и малый фиксированный workflow — осознанное упрощение для хакатона, которое затем проверяется на нескольких реальных кейсах.
