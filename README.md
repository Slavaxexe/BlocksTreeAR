# Альфа Хак: AI-агент для ML

Материалы задания и оба публичных датасета сохранены локально 19 октября 2026 года.
Агент должен самостоятельно читать описание новой задачи, исследовать данные,
проверять гипотезы, обучать модель и формировать `submission.csv`.

## Где что лежит

| Путь | Содержимое |
| --- | --- |
| [docs/TASK_BRIEF.md](docs/TASK_BRIEF.md) | Требования, ограничения, метрики и вопросы к организаторам |
| [docs/DATASETS.md](docs/DATASETS.md) | Состав данных и результаты локальной проверки |
| [docs/BASELINE_INTERFACE.md](docs/BASELINE_INTERFACE.md) | Интерфейс baseline и настройки API |
| [docs/source/alfa_automl_case.pdf](docs/source/alfa_automl_case.pdf) | Копия исходного задания |
| [docs/source/case_text.txt](docs/source/case_text.txt) | Текст PDF с номерами страниц; формулы смотреть в PDF |
| [data/raw/regression/](data/raw/regression/) | Энергопотребление зданий: CSV и описание задачи |
| [data/raw/classification/](data/raw/classification/) | Типы сетевых атак: CSV и описание задачи |
| [vendor/baseline.py](vendor/baseline.py) | Исходный baseline организаторов, без изменений |
| [vendor/requirements_pdf.txt](vendor/requirements_pdf.txt) | Зависимости по прямой ссылке из PDF |
| [vendor/requirements_data_folder.txt](vendor/requirements_data_folder.txt) | Другая версия зависимостей из папки данных |
| [docs/sources.json](docs/sources.json) | Источники, идентификаторы файлов, размеры и SHA-256 |
| [docs/data_inventory.json](docs/data_inventory.json) | Машиночитаемый отчёт проверки таблиц |

В каждой папке датасета лежат `train.csv`, `test.csv`, `data_dictionary.csv`,
`sample_submission.csv` и исходный `task.md`. Для нашей плоской структуры создан
`task.local.md`: в нём исправлены только относительные пути к CSV. Оригиналы сохранены.

## Доступ к Qwen

По заданию общий ключ выдаётся команде организаторами, дневной лимит составляет
600 рублей. В кабинете участника или чате хакатона нужно получить
`OPENAI_API_KEY`, `OPENAI_BASE_URL` и точное имя модели для `OPENAI_MODEL`.
PDF называет модель `Qwen3.8-27B`, но API-идентификатор может отличаться.
Самих реквизитов API в материалах нет.

## Статус подготовки

- Скачаны 8 CSV, 2 описания задач, baseline и обе версии requirements.
- Проверены размеры скачанных файлов, чтение CSV, размеры таблиц, ID, схема и словари.
- Сохранены исходный PDF, его текст и краткие требования.
- Оба датасета включены в Git: после клонирования все 8 CSV доступны локально. Логи исключены из репозитория.
- Решение `solution.py` ещё не разрабатывалось. Обучение, вызовы LLM и отправка на платформу не выполнялись.

Перед реализацией важно учесть расхождение requirements и отсутствие `mcp`
в обоих списках: исходный baseline импортирует этот пакет.
