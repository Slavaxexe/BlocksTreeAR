# Подключение Qwen

Python 3.12. Зависимости устанавливаются из корня проекта:

```powershell
uv sync --locked
```

Скопируйте `.env.example` в `.env` и заполните `OPENAI_API_KEY`.
Остальные параметры уже указаны в примере:

| Переменная | Значение |
| --- | --- |
| `OPENAI_BASE_URL` | `https://inference.airi.net/v1` |
| `OPENAI_MODEL` | `Qwen/Qwen3.8-27B` |
| `QWEN_TIMEOUT_SECONDS` | `60` |
| `QWEN_MAX_TOKENS` | `1024` |

`.env` исключён из Git. Переменные окружения имеют приоритет над файлом.

Проверка настроек без обращения к API:

```powershell
uv run --locked python -m agent check-config
```

Запрос к модели:

```powershell
uv run --locked python -m agent ask --prompt "Текст запроса"
```

Короткая проверка подключения: `uv run --locked python -m agent ping`.
Команды `ask` и `ping` расходуют бюджет API. Автоматические повторы отключены.

Клиент использует `/v1/chat/completions` и выводит ответ со статистикой токенов.
Обучение моделей, выполнение сгенерированного кода и цикл AutoResearch пока не реализованы.
