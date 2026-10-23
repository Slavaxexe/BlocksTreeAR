"""Shared fixtures for tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preparation.config import PipelineConfig


@pytest.fixture
def synth_task_dir(tmp_path: Path) -> Path:
    rng = np.random.RandomState(42)
    n_train, n_test = 200, 80

    train = pd.DataFrame({
        "id": range(1, n_train + 1),
        "num_a": rng.normal(100, 20, n_train),
        "num_b": rng.exponential(50, n_train),
        "cat_small": rng.choice(["A", "B", "C"], n_train),
        "cat_large": rng.choice([f"cat_{i}" for i in range(30)], n_train),
        "const_col": 1.0,
        "all_missing": np.nan,
        "year_built": rng.choice([0, 1990, 2000, 2010, np.nan], n_train),
        "energy_star_rating": rng.choice([10, 50, 90, np.nan], n_train),
        "floor_area": rng.exponential(10000, n_train),
        "heating_degree_days_c": rng.uniform(500, 5000, n_train),
        "cooling_degree_days_c": rng.uniform(100, 3000, n_train),
        "january_avg_temp_c": rng.normal(0, 10, n_train),
        "july_avg_temp_c": rng.normal(25, 5, n_train),
    })
    train["site_eui"] = (
        50 + 0.001 * train["num_a"] + 0.5 * np.log1p(train["floor_area"])
        + rng.normal(0, 5, n_train)
    )
    train.loc[5, "num_a"] = np.nan
    train.loc[10, "num_b"] = np.inf

    test = pd.DataFrame({
        "id": range(n_train + 1, n_train + n_test + 1),
        "num_a": rng.normal(100, 20, n_test),
        "num_b": rng.exponential(50, n_test),
        "cat_small": rng.choice(["A", "B", "C", "D"], n_test),
        "cat_large": rng.choice([f"cat_{i}" for i in range(30)] + ["cat_new"], n_test),
        "const_col": 1.0,
        "all_missing": np.nan,
        "year_built": rng.choice([1990, 2005, 2015], n_test),
        "energy_star_rating": rng.choice([20, 60, np.nan], n_test),
        "floor_area": rng.exponential(10000, n_test),
        "heating_degree_days_c": rng.uniform(500, 5000, n_test),
        "cooling_degree_days_c": rng.uniform(100, 3000, n_test),
        "january_avg_temp_c": rng.normal(0, 10, n_test),
        "july_avg_temp_c": rng.normal(25, 5, n_test),
    })

    dd = pd.DataFrame({
        "feature": ["id", "num_a", "num_b", "cat_small", "cat_large",
                     "const_col", "all_missing", "year_built", "energy_star_rating",
                     "floor_area", "heating_degree_days_c", "cooling_degree_days_c",
                     "january_avg_temp_c", "july_avg_temp_c", "site_eui"],
        "role": ["id"] + ["feature"] * 13 + ["target"],
        "dtype": ["numeric", "numeric", "numeric", "category", "category",
                  "numeric", "numeric", "numeric", "numeric",
                  "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"],
        "unit": ["id", "", "", "", "", "", "", "year", "score",
                 "ft2", "degree-days", "degree-days", "C", "C", "kBtu/ft2"],
        "description_ru": ["ID", "Числовой A", "Числовой B", "Маленькая категория",
                           "Большая категория", "Константа", "Все пусто",
                           "Год постройки", "Рейтинг", "Площадь",
                           "Отопительные градусо-сутки", "Охлаждение градусо-сутки",
                           "Средняя темп. январь", "Средняя темп. июль", "Таргет"],
    })

    sample_sub = pd.DataFrame({
        "id": range(n_train + 1, n_train + n_test + 1),
        "site_eui": 0.0,
    })

    task_md = "# Test Task\n\nPredict site_eui using building features.\nMetric: RMSE\n"

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    dd.to_csv(tmp_path / "data_dictionary.csv", index=False)
    sample_sub.to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task.md").write_text(task_md, encoding="utf-8")

    return tmp_path


@pytest.fixture
def default_config(synth_task_dir: Path, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        task_dir=str(synth_task_dir),
        target_columns=["site_eui"],
        id_columns=["id"],
        random_state=42,
        time_budget_seconds=300,
        output_dir=str(tmp_path / "output"),
    )
