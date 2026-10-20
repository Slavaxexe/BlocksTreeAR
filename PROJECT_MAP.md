# Карта проекта

Код компактный и читаемый, без искусственных слоёв и дробления на мелкие модули.
Короткие выражения не растягиваем, длинные не минифицируем. Комментарии объясняют
причины и ограничения. При изменениях обновляем эту карту и архитектуру.

```text
AlfaHakaton/
├── solution.py              Точка входа платформы: task.md → первый baseline
├── viewer.py                Один скрипт: HTML-граф, JSON / evidence / исходники, localhost / экспорт
├── alfa_agent/
│   ├── __init__.py          Пакет
│   ├── __main__.py          CLI check-config / ping / ask / extract / baseline
│   ├── config.py            Env / .env, проверка настроек, скрытие ключа
│   ├── qwen.py              Текстовые диагностические вызовы API
│   ├── contracts.py         Pydantic-контракты фактов, подготовки, плана, кода, start / finish / report
│   ├── agents.py            Четыре extractor’а, preparer, baker, реализатор; bounded JSON-repair
│   ├── data.py              Инвентаризация, профиль таблиц, пути, хеши и интерфейс генерируемого кода
│   ├── runner.py            План до реализации, bounded repair, subprocess / таймер, публикация
│   ├── worker.py            Исполнение prepare.py, smoke-test, модель, общий scorer, запись finish
│   └── storage.py           Атомарная запись, сериализация и загрузка контрактов
├── tests/
│   ├── test_extraction.py   Источники / схемы, параллельность, свободные имена / метрики / форматы
│   ├── test_pipeline.py     Mock HTTP → настоящий код / subprocess → предсказания и кастомный score
│   ├── test_qwen.py         Настройки, HTTP, таймауты, отсутствие ключа в диагностике
│   └── test_viewer.py       Новая раскладка, незавершённые записи, исходники, безопасный HTML
├── schemas/                 JSON Schema публичных артефактов из contracts.py
├── artifacts/<run-id>/      Генерируемые файлы; исключены из git
│   ├── extraction/          problem.json, data.json, metric.json, output.json
│   ├── extraction.json      Четыре блока с цитатами
│   ├── report.json          Статус, ошибки, замечания, попытки API
│   ├── prepare.py           Общий загрузчик, splitter, scorer, writer от Qwen
│   ├── preparation.json     Профиль, конфигурация, готовность, хеши
│   ├── preparation.log      Диагностика подготовки
│   ├── split.npz            Индексы общей validation
│   ├── nodes/node_000/      start.json, experiment.py, finish.json, execution.log, предсказания
│   └── best.json            Указатель на оценённую ноду
├── data/raw/                Публичные условия, исходные данные и словари признаков
├── docs/
│   ├── ARCHITECTURE.md      Текущая архитектура, контракты, границы и последующие этапы
│   ├── TASK_BRIEF.md        Требования организаторов из исходного PDF
│   └── ...                 Источники, инвентаризация данных, настройка API, интерфейс vendor baseline
├── vendor/                  Неизменённый baseline организаторов и их requirements
├── history/                 Исторические research_notes / reports и первая проверка API
├── .env.example             Имена настроек без ключа
├── pyproject.toml, uv.lock  Зависимости и зафиксированное окружение
└── README.md                Запуск и поведение
```

Генерируемый Python хранится внутри запуска. Это не новые исходные модули проекта.
Файл ответа также публикуется рядом с Markdown согласно условию; генерируемые
submission исключены из git. Данные в каждую ноду не копируются.

`solution.py` → CLI baseline → `runner.run()` → extractor’ы → prepare.py через
worker → baker → start.json → реализатор → проверка → experiment.py через worker → finish / output.
Start неизменен; finish хранит хеши плана / кода, заметки и этап возможного отказа.
`viewer.py` независимо читает известные JSON и два известных исходника; CSV,
.env, API и выполнение кода ему не нужны. Конкретные загрузчик и scorer создаются
под условие; в контроллере не зашиты публичные target / имена файлов / метрики.

Старый цикл с фиксированными sklearn pipeline и summarizer удалён. Новый путь
останавливается после node_000; следующие гипотезы обсудим после проверки baseline.
