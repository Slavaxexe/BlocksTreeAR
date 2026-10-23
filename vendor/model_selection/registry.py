"""
Единый реестр моделей и лоссов для регрессии.

Это ОБЩИЙ контракт с этапом Макса: model_id отсюда = ключ его шаблонов.
Все нативные имена проверены на версиях из requirements.txt
(sklearn 1.5.2, lightgbm 4.6.0, xgboost 2.1.4, catboost 1.2.8, flaml 2.3.6, autogluon 1.3.1).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Канонические лоссы и какую статистику условного распределения они оценивают.
# Ключевая идея: метрика тоже "хочет" определённую статистику
# (RMSE -> среднее, MAE -> медиану, pinball -> квантиль, RMSLE -> среднее в log-шкале).
# Лосс согласован с метрикой, если оценивает ту же статистику.
# ---------------------------------------------------------------------------
LOSSES: dict[str, dict] = {
    "l2":           {"statistic": "mean",     "desc": "MSE. Оценивает условное среднее."},
    "poisson":      {"statistic": "mean",     "desc": "Пуассон, log-link. Оценивает среднее, требует y>=0. Хорош для счётных/скошенных неотрицательных y."},
    "gamma":        {"statistic": "mean",     "desc": "Гамма, log-link. Оценивает среднее, требует y>0. Для мультипликативного шума."},
    "tweedie":      {"statistic": "mean",     "desc": "Твиди, log-link. Оценивает среднее, y>=0, много нулей + длинный хвост."},
    "l1":           {"statistic": "median",   "desc": "MAE. Оценивает условную медиану."},
    "huber":        {"statistic": "robust",   "desc": "Хьюбер: L2 в центре, L1 на хвостах. Между средним и медианой."},
    "pseudo_huber": {"statistic": "robust",   "desc": "Гладкий Хьюбер (XGBoost)."},
    "fair":         {"statistic": "robust",   "desc": "Fair loss (LightGBM), робастный."},
    "logcosh":      {"statistic": "robust",   "desc": "log(cosh) — гладкий робастный, только CatBoost."},
    "quantile":     {"statistic": "quantile", "desc": "Pinball loss для квантиля alpha."},
    "mape":         {"statistic": "relative", "desc": "MAPE как objective (LightGBM, CatBoost)."},
    "squared_log":  {"statistic": "log_mean", "desc": "Квадрат ошибки в log1p-шкале (XGBoost), эквивалент RMSLE."},
}

# Трансформы таргета. Трансформ меняет оцениваемую статистику:
# l2 по log1p(y) оценивает E[log1p y], а не E[y] -> после обратного перевода прогноз смещён вниз.
TRANSFORMS: dict[str, dict] = {
    "none":  {"requires": None,     "statistic_shift": None},
    "log1p": {"requires": "y>-1",   "statistic_shift": "log_mean"},
}

# ---------------------------------------------------------------------------
# Модели. Поля:
#   family               — семейство (для правил и отчёта)
#   losses               — канонический лосс -> как задать в библиотеке
#   needs_scaling        — нужен StandardScaler/OHE (линейные, kNN, MLP)
#   handles_nan          — умеет NaN без импутации
#   native_categorical   — умеет категориальные без кодирования
#   max_rows_hint        — выше этого объёма модель непрактична по времени
#   cost                 — относительная стоимость обучения (1 = Ridge)
#   optimizer_applicable — к модели применима ветка Макса "градиентные методы
#                          (SGD, momentum, Nesterov, AdaGrad, Adam)"
#   flaml / autogluon    — имя в AutoML (None = нет)
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "dummy_mean": {
        "family": "reference", "library": "sklearn", "estimator": "sklearn.dummy.DummyRegressor",
        "losses": {"l2": {"strategy": "mean"}, "l1": {"strategy": "median"}},
        "needs_scaling": False, "handles_nan": True, "native_categorical": True,
        "max_rows_hint": None, "cost": 0.1, "optimizer_applicable": False,
        "flaml": None, "autogluon": "DUMMY",
        "note": "Нижняя граница качества. Не кандидат, а точка отсчёта.",
    },
    "ridge": {
        "family": "linear", "library": "sklearn", "estimator": "sklearn.linear_model.Ridge",
        "losses": {"l2": {}},
        "needs_scaling": True, "handles_nan": False, "native_categorical": False,
        "max_rows_hint": 2_000_000, "cost": 1, "optimizer_applicable": False,
        "flaml": None, "autogluon": "LR",
        "note": "Закрытое решение, L2-регуляризация. Базовая линейная модель.",
    },
    "elasticnet": {
        "family": "linear", "library": "sklearn", "estimator": "sklearn.linear_model.ElasticNet",
        "losses": {"l2": {}},
        "needs_scaling": True, "handles_nan": False, "native_categorical": False,
        "max_rows_hint": 2_000_000, "cost": 3, "optimizer_applicable": False,
        "flaml": "enet", "autogluon": None,
        "note": "L1+L2: отбор признаков + устойчивость к коллинеарности.",
    },
    "huber_linear": {
        "family": "linear", "library": "sklearn", "estimator": "sklearn.linear_model.HuberRegressor",
        "losses": {"huber": {"epsilon": 1.35}},
        "needs_scaling": True, "handles_nan": False, "native_categorical": False,
        "max_rows_hint": 300_000, "cost": 5, "optimizer_applicable": False,
        "flaml": None, "autogluon": None,
        "note": "Линейная модель с робастным лоссом.",
    },
    "sgd_linear": {
        "family": "linear_sgd", "library": "sklearn", "estimator": "sklearn.linear_model.SGDRegressor",
        "losses": {"l2": {"loss": "squared_error"}, "huber": {"loss": "huber"},
                   "l1": {"loss": "epsilon_insensitive", "epsilon": 0.0}},
        "needs_scaling": True, "handles_nan": False, "native_categorical": False,
        "max_rows_hint": None, "cost": 1, "optimizer_applicable": True,
        "flaml": "sgd", "autogluon": None,
        "note": "Оправдан при >1M строк или когда нужен робастный лосс на линейной модели. average=True = ASGD.",
    },
    "mlp": {
        "family": "neural", "library": "sklearn", "estimator": "sklearn.neural_network.MLPRegressor",
        "losses": {"l2": {}},
        "needs_scaling": True, "handles_nan": False, "native_categorical": False,
        "max_rows_hint": 500_000, "cost": 20, "optimizer_applicable": True,
        "flaml": None, "autogluon": "NN_TORCH",
        "note": "solver=sgd (momentum, nesterovs_momentum) или adam. Главный потребитель ветки оптимизаторов Макса. На таблицах обычно слабее GBM, ценен как разнообразие в ансамбле.",
    },
    "knn": {
        "family": "instance", "library": "sklearn", "estimator": "sklearn.neighbors.KNeighborsRegressor",
        "losses": {"l2": {}},
        "needs_scaling": True, "handles_nan": False, "native_categorical": False,
        "max_rows_hint": 200_000, "cost": 5, "optimizer_applicable": False,
        "flaml": "kneighbor", "autogluon": "KNN",
        "note": "Только при малом числе признаков (<~20). Разнообразие для ансамбля.",
    },
    "random_forest": {
        "family": "bagging_trees", "library": "sklearn", "estimator": "sklearn.ensemble.RandomForestRegressor",
        "losses": {"l2": {"criterion": "squared_error"}, "poisson": {"criterion": "poisson"},
                   "l1": {"criterion": "absolute_error"}},
        "needs_scaling": False, "handles_nan": True, "native_categorical": False,
        "max_rows_hint": 1_000_000, "cost": 15, "optimizer_applicable": False,
        "flaml": "rf", "autogluon": "RF",
        "note": "criterion=absolute_error очень медленный — только на малых данных.",
    },
    "extra_trees": {
        "family": "bagging_trees", "library": "sklearn", "estimator": "sklearn.ensemble.ExtraTreesRegressor",
        "losses": {"l2": {"criterion": "squared_error"}, "poisson": {"criterion": "poisson"}},
        "needs_scaling": False, "handles_nan": True, "native_categorical": False,
        "max_rows_hint": 1_000_000, "cost": 8, "optimizer_applicable": False,
        "flaml": "extra_tree", "autogluon": "XT",
        "note": "Быстрее RF, сильнее декоррелирован — хороший член ансамбля.",
    },
    "hist_gb": {
        "family": "boosting", "library": "sklearn", "estimator": "sklearn.ensemble.HistGradientBoostingRegressor",
        "losses": {"l2": {"loss": "squared_error"}, "l1": {"loss": "absolute_error"},
                   "poisson": {"loss": "poisson"}, "gamma": {"loss": "gamma"},
                   "quantile": {"loss": "quantile"}},
        "needs_scaling": False, "handles_nan": True, "native_categorical": True,
        "max_rows_hint": None, "cost": 4, "optimizer_applicable": False,
        "flaml": "histgb", "autogluon": None,
        "note": "Запасной бустинг без внешних зависимостей.",
    },
    "lightgbm": {
        "family": "boosting", "library": "lightgbm", "estimator": "lightgbm.LGBMRegressor",
        "losses": {"l2": {"objective": "regression"}, "l1": {"objective": "regression_l1"},
                   "huber": {"objective": "huber"}, "fair": {"objective": "fair"},
                   "poisson": {"objective": "poisson"}, "gamma": {"objective": "gamma"},
                   "tweedie": {"objective": "tweedie"}, "quantile": {"objective": "quantile"},
                   "mape": {"objective": "mape"}},
        "needs_scaling": False, "handles_nan": True, "native_categorical": True,
        "max_rows_hint": None, "cost": 3, "optimizer_applicable": False,
        "flaml": "lgbm", "autogluon": "GBM",
        "note": "Основной кандидат на табличных данных: быстрый, NaN и категории из коробки.",
    },
    "xgboost": {
        "family": "boosting", "library": "xgboost", "estimator": "xgboost.XGBRegressor",
        "losses": {"l2": {"objective": "reg:squarederror"}, "l1": {"objective": "reg:absoluteerror"},
                   "pseudo_huber": {"objective": "reg:pseudohubererror"},
                   "poisson": {"objective": "count:poisson"}, "gamma": {"objective": "reg:gamma"},
                   "tweedie": {"objective": "reg:tweedie"}, "quantile": {"objective": "reg:quantileerror"},
                   "squared_log": {"objective": "reg:squaredlogerror"}},
        "needs_scaling": False, "handles_nan": True, "native_categorical": True,
        "max_rows_hint": None, "cost": 5, "optimizer_applicable": False,
        "flaml": "xgboost", "autogluon": "XGB",
        "note": "Второй бустинг для разнообразия ансамбля. Категории: enable_categorical=True.",
    },
    "catboost": {
        "family": "boosting", "library": "catboost", "estimator": "catboost.CatBoostRegressor",
        "losses": {"l2": {"loss_function": "RMSE"}, "l1": {"loss_function": "MAE"},
                   "huber": {"loss_function": "Huber:delta={delta}"},
                   "logcosh": {"loss_function": "LogCosh"},
                   "poisson": {"loss_function": "Poisson"},
                   "tweedie": {"loss_function": "Tweedie:variance_power={p}"},
                   "quantile": {"loss_function": "Quantile:alpha={alpha}"},
                   "mape": {"loss_function": "MAPE"}},
        "needs_scaling": False, "handles_nan": True, "native_categorical": True,
        "max_rows_hint": None, "cost": 8, "optimizer_applicable": False,
        "flaml": "catboost", "autogluon": "CAT",
        "note": "Приоритет при категориальных признаках высокой кардинальности (ordered target statistics).",
    },
}

# ---------------------------------------------------------------------------
# Метрики: какую статистику они вознаграждают + имена в AutoML.
# ---------------------------------------------------------------------------
METRICS: dict[str, dict] = {
    "rmse":  {"statistic": "mean",     "greater_is_better": False, "flaml": "rmse", "autogluon": "root_mean_squared_error"},
    "mse":   {"statistic": "mean",     "greater_is_better": False, "flaml": "mse",  "autogluon": "mean_squared_error"},
    "r2":    {"statistic": "mean",     "greater_is_better": True,  "flaml": "r2",   "autogluon": "r2"},
    "mae":   {"statistic": "median",   "greater_is_better": False, "flaml": "mae",  "autogluon": "mean_absolute_error"},
    "medae": {"statistic": "median",   "greater_is_better": False, "flaml": None,   "autogluon": "median_absolute_error"},
    "rmsle": {"statistic": "log_mean", "greater_is_better": False, "flaml": None,   "autogluon": None},
    "msle":  {"statistic": "log_mean", "greater_is_better": False, "flaml": None,   "autogluon": None},
    "mape":  {"statistic": "relative", "greater_is_better": False, "flaml": "mape", "autogluon": "mean_absolute_percentage_error"},
    "smape": {"statistic": "relative", "greater_is_better": False, "flaml": None,   "autogluon": "symmetric_mean_absolute_percentage_error"},
    "pinball": {"statistic": "quantile", "greater_is_better": False, "flaml": None, "autogluon": None},
}
# Для rmsle/msle в AutoML: обучать на log1p(y) с метрикой rmse и делать expm1 — это точная эквивалентность.

METRIC_ALIASES = {
    "root mean squared error": "rmse", "root_mean_squared_error": "rmse",
    "mean squared error": "mse", "mean_squared_error": "mse",
    "mean absolute error": "mae", "mean_absolute_error": "mae",
    "median absolute error": "medae",
    "rmsle": "rmsle", "root mean squared log error": "rmsle", "msle": "msle",
    "mean absolute percentage error": "mape", "smape": "smape",
    "r^2": "r2", "r2_score": "r2", "pinball": "pinball", "quantile": "pinball",
}


def normalize_metric(name: str | None) -> str | None:
    if not name:
        return None
    k = name.strip().lower()
    if k in METRICS:
        return k
    return METRIC_ALIASES.get(k)


def loss_statistic(loss: str, transform: str = "none") -> str:
    """Какую статистику реально оценивает пара (лосс, трансформ)."""
    if transform != "none":
        shift = TRANSFORMS[transform]["statistic_shift"]
        if shift and LOSSES[loss]["statistic"] in ("mean", "robust"):
            return shift
    return LOSSES[loss]["statistic"]


def is_metric_consistent(loss: str, transform: str, metric: str) -> bool:
    """Пара согласована с метрикой, если оценивает ту же статистику."""
    return loss_statistic(loss, transform) == METRICS[metric]["statistic"]


def supports(model_id: str, loss: str) -> bool:
    return model_id in MODELS and loss in MODELS[model_id]["losses"]


def compact_registry() -> dict:
    """Сжатый вид реестра для промпта Qwen (экономим токены)."""
    return {
        mid: {"family": m["family"], "losses": list(m["losses"]), "cost": m["cost"],
              "native_categorical": m["native_categorical"], "max_rows_hint": m["max_rows_hint"],
              "optimizer_applicable": m["optimizer_applicable"]}
        for mid, m in MODELS.items()
    }
