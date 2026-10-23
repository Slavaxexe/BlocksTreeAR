# data_preparation — модуль подготовки данных для табулярной регрессии

## Быстрый старт

```bash
# Установка зависимостей
pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 \
    pydantic==2.10.6 openai==1.68.2 cloudpickle==3.1.1 psutil==6.1.1

# Запуск без LLM (детерминированный режим)
python run_preparation.py /path/to/task_dir --target site_eui --no-llm

# Запуск с LLM (Qwen3.8-27B)
API_KEY=sk-... BASE_URL=https://changellenge.up.railway.app/v1 \
    python run_preparation.py /path/to/task_dir --target site_eui
```

## Настройка API

Переменные окружения (приоритет → fallback):
| Переменная | Fallback | Описание |
|---|---|---|
| `API_KEY` | `OPENAI_API_KEY` | Ключ доступа к LLM API |
| `BASE_URL` | `OPENAI_BASE_URL` | URL OpenAI-совместимого API |
| `OPENAI_MODEL` | — | Название модели (по умолчанию `qwen/qwen3.8-27b`) |

Можно задать через `.env` файл (скопируйте `.env.example`). Ключ **никогда** не попадает в логи, артефакты или код.

---

## Внутренняя логика (pipeline)

```
load → profile → quality_checks → split → feature_plan [+LLM] → transform → save → report
```

1. **load** — загрузка `train.csv`, `test.csv`, `data_dictionary.csv`, `task.md`. Валидация наличия таргета и совместимости схем train/test.
2. **profile** — статистика по каждому столбцу (числовой/категориальный/datetime/текст), детекция констант, полностью пустых, утечек и групп.
3. **quality_checks** — 37 типов проверок: пропуски, inf, дубликаты, выбросы (3×IQR), качество таргета, расхождения схем.
4. **split** — разбиение train на train/val (80/20 по умолчанию) **до** подбора параметров. Поддержка предзаданных индексов.
5. **feature_plan** — детерминированный план + запрос к LLM за дополнительными идеями. LLM возвращает JSON, который валидируется Pydantic.
6. **transform** — `fit()` на train, `transform()` на train/val/test идентично. Никакой информации из val/test в параметры.
7. **save** — parquet-файлы + 5 JSON-артефактов + `transformer.pkl` + `recipe.json`.
8. **report** — итоговый `preparation_report.json` со статистикой шагов, корреляциями, использованием LLM.

---

## Структура модуля

| Файл | Назначение |
|---|---|
| `schemas.py` | Все Pydantic-схемы и контракты (версия `1.0.0`) |
| `config.py` | `PipelineConfig`, `LLMConfig` — конфигурация через код или CLI |
| `loader.py` | Загрузка CSV/MD, валидация таргета и схемы |
| `profiler.py` | Статистика столбцов, детекция типов и утечек |
| `quality.py` | Проверки качества данных (37 типов) |
| `features.py` | Построение feature plan + merge LLM-предложений |
| `transformer.py` | `DataTransformer` — fit/transform, save/load через cloudpickle |
| `llm_client.py` | `LLMClient` + `MockLLMClient` для тестов |
| `llm_prompts.py` | Промпты для LLM (system + user) |
| `pipeline.py` | `DataPreparationPipeline` — оркестратор |
| `utils.py` | `TimeBudget`, atomic JSON write, sanitize NaN/Inf |
| `tests/` | 29 тестов на синтетических данных (без API) |

---

## JSON-артефакты (схемы)

Все артефакты обёрнуты в `ArtifactEnvelope`:

```json
{
  "schema_version": "1.0.0",
  "artifact_type": "<имя>",
  "run_id": "a07aaea5",
  "status": "success",
  "payload": { ... },
  "warnings": [],
  "errors": []
}
```

### input_manifest.json
```json
{
  "task_type": "regression",
  "metric_name": "rmse",
  "target_columns": ["site_eui"],
  "files": [
    {"name": "train.csv", "path": "...", "size_bytes": 43456789, "n_rows": 53308, "n_columns": 65}
  ]
}
```

### data_profile.json
```json
{
  "train_summary": {"n_rows": 53308, "n_columns": 65, "n_numeric": 60, "n_categorical": 4, ...},
  "test_summary": {...},
  "target": {"name": "site_eui", "stats": {"min": 0.1, "max": 999.9, "mean": 73.2, ...}},
  "train_columns": [
    {
      "name": "floor_area", "dtype": "float64", "detected_type": "numeric",
      "numeric_stats": {"min": 1.0, "max": 1500000.0, "mean": 32456.7, "n_missing": 0, ...},
      "is_constant": false, "is_all_missing": false
    }
  ],
  "schema_comparison": {"missing_in_test": [], "missing_in_train": [], "dtype_mismatches": []}
}
```

### feature_plan.json
```json
{
  "columns_to_drop": ["id", "const_col", "all_missing"],
  "column_transforms": {
    "floor_area": [{"op": "impute_median"}, {"op": "passthrough"}],
    "facility_type": [{"op": "impute_constant", "params": {"value": "__MISSING__"}}, {"op": "one_hot_encode"}]
  },
  "new_features": [
    {"name": "temp_range_annual", "source_columns": ["january_avg_temp_c", "july_avg_temp_c"],
     "transform_type": "interaction", "definition": "max(month_avgs) - min(month_avgs)", "rationale": "..."},
    {"name": "log_floor_area", "source_columns": ["floor_area"],
     "transform_type": "log_transform", "definition": "log1p(floor_area)", "rationale": "..."}
  ],
  "scaling_mode": "none",
  "llm_suggestions_used": false
}
```

### preparation_report.json
```json
{
  "steps": [{"name": "load", "duration_seconds": 0.1, "status": "ok"}, ...],
  "new_features_created": ["temp_range_annual", "log_floor_area", "hdd_cdd_ratio", "energy_star_missing"],
  "high_correlations": [{"col_a": "january_min_temp_c", "col_b": "january_avg_temp_c", "corr": 0.98}],
  "target_correlations": [{"column": "floor_area", "corr": 0.45}, ...],
  "llm_calls_made": 0, "llm_tokens_used": 0
}
```

### output_manifest.json
```json
{
  "train_features": {"path": "train_features.parquet", "format": "parquet", "n_rows": 42646, "n_columns": 66},
  "val_features": {"path": "val_features.parquet", ...},
  "test_features": {"path": "test_features.parquet", ...},
  "train_target": {"path": "train_target.parquet", ...},
  "val_target": {"path": "val_target.parquet", ...},
  "transformer_path": "transformer.pkl",
  "recipe_path": "recipe.json",
  "feature_names": ["Year_Factor", "State_Factor", "building_class", ...]
}
```

---

## Подключение к общему пайплайну

```python
from data_preparation.config import PipelineConfig
from data_preparation.pipeline import DataPreparationPipeline

config = PipelineConfig(
    task_dir="/path/to/regression/task",
    target_columns=["site_eui"],
    metric_name="rmse",
    output_dir="/path/to/output",
    time_budget_seconds=1800,
    random_state=42,
)

pipeline = DataPreparationPipeline(config)
result = pipeline.run()

# Готовые данные для модели
X_train = result.train_features   # pd.DataFrame
y_train = result.train_target     # pd.Series
X_val   = result.val_features
y_val   = result.val_target
X_test  = result.test_features

# Сохранённый трансформер для финальной подготовки
transformer = result.transformer
feature_names = result.feature_names

# Финальное обучение на полном train (train+val) с тем же планом
full_train = pd.concat([result.train_features, result.val_features])  # до transform
# Нужен исходный DataFrame с таргетом — загрузите заново или сохраните до split
from data_preparation.loader import load_data
lr = load_data(config)
test_final, final_transformer = pipeline.run_final(lr.train, lr.test, result.transformer.plan)
# test_final — готовые фичи для предсказания на полном train
```

### С LLM-обогащением

```python
from data_preparation.llm_client import LLMClient
from data_preparation.config import LLMConfig

llm = LLMClient(LLMConfig())  # читает API_KEY/BASE_URL из env
ok, msg = llm.health_check()
if ok:
    pipeline = DataPreparationPipeline(config, llm_client=llm)
else:
    print(f"LLM unavailable: {msg}")
    pipeline = DataPreparationPipeline(config)  # fallback без LLM

result = pipeline.run()
```

### С предзаданным split

```python
config = PipelineConfig(
    task_dir="/path/to/task",
    target_columns=["site_eui"],
    output_dir="/path/to/output",
    train_indices=list(range(0, 40000)),
    val_indices=list(range(40000, 53308)),
)
```

---

## Поддерживаемые трансформации

| Операция | Описание |
|---|---|
| `drop` | Удаление столбца |
| `passthrough` | Без изменений |
| `impute_median` / `impute_mean` / `impute_mode` | Заполнение пропусков |
| `impute_constant` | Заполнение константой (params: `value`) |
| `ordinal_encode` | Порядковое кодирование категорий |
| `one_hot_encode` | One-hot (для ≤15 уникальных значений) |
| `frequency_encode` | Частотное кодирование |
| `log_transform` / `sqrt_transform` | Логарифм / корень |
| `standard_scale` | Стандартизация (mean=0, std=1) |
| `clip_outliers` | Обрезка выбросов по IQR |
| `binning` | Квантильная бинаризация |
| `interaction` | Произведение двух столбцов |
| `ratio` | Отношение двух столбцов |
| `polynomial` | Полиномиальные признаки |
| `date_components` | Извлечение year/month/day/dayofweek |
| `missing_indicator` | Индикатор пропуска |

---

## Ограничения и непроверенные возможности

### Известные ограничения
- **Только регрессия** с одним числовым таргетом — классификация не поддерживается
- **target_encode** зарезервирован, но не реализован (опасность утечки)
- LLM-предложения проходят валидацию — невалидный JSON или несуществующие столбцы игнорируются
- One-hot encoding применяется только для категорий с ≤15 уникальными значениями
- Текстовые столбцы удаляются (NLP-фичи не генерируются)
- Максимум 5 повторов при ошибке LLM API

### Непроверенные в данном запуске
- Работа с очень большими данными (>1M строк) — может потребоваться больше памяти
- Реальный LLM-запрос к API (проверен только MockLLMClient)
- `run_final()` на реальных данных (логика протестирована на синтетике)
- Параллельная работа нескольких экземпляров на одной директории
- Datetime-колонки (в данных хакатона не встретились)

### Безопасность
- API-ключ читается **только** из переменных окружения
- Ключ **не** попадает в JSON-артефакты, логи или сообщения об ошибках
- LLM получает только профиль данных (статистики), **не** сырые строки
- Атомарная запись JSON (tmp + rename)

---

## Тесты

```bash
# Запуск всех 29 тестов (без API-токенов)
python -m pytest data_preparation/tests/ -v

# Только конкретный класс
python -m pytest data_preparation/tests/test_pipeline.py::TestTransformer -v
```

Тесты работают на синтетических данных (200 train / 80 test) и не требуют реального API.
