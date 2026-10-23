"""Data transformation pipeline: fit on train, apply to train/val/test."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preparation.schemas import (
    ColumnDtype, ColumnProfile, ColumnTransform, FeaturePlanPayload,
    NewFeatureDef, TransformOp,
)

logger = logging.getLogger("data_preparation.transformer")


class DataTransformer:
    """Fit on train split, apply identically to val/test.

    All learned parameters are stored in self._params and serialized with save/load.
    """

    def __init__(self, plan: FeaturePlanPayload, target_col: str, id_cols: list[str]):
        self._plan = plan
        self._target_col = target_col
        self._id_cols = id_cols
        self._params: dict[str, Any] = {}
        self._fitted = False
        self._feature_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def plan(self) -> FeaturePlanPayload:
        return self._plan

    def fit(self, train: pd.DataFrame) -> DataTransformer:
        logger.info("Fitting transformer on %d rows, %d columns", len(train), len(train.columns))
        self._params = {}

        for ct in self._plan.column_transforms:
            if ct.column not in train.columns:
                continue
            self._fit_column(train, ct)

        for nf in self._plan.new_features:
            self._fit_new_feature(train, nf)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        result = df.copy()

        for col in self._plan.columns_to_drop:
            if col in result.columns:
                result = result.drop(columns=[col])

        if self._target_col in result.columns and is_test:
            result = result.drop(columns=[self._target_col])

        for ct in self._plan.column_transforms:
            if ct.column not in result.columns:
                continue
            result = self._transform_column(result, ct)

        for nf in self._plan.new_features:
            result = self._create_new_feature(result, nf)

        for col in self._id_cols:
            if col in result.columns:
                result = result.drop(columns=[col])
        if self._target_col in result.columns:
            result = result.drop(columns=[self._target_col])

        self._feature_names = list(result.columns)
        return result

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.fit(train)
        return self.transform(train)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "plan": self._plan.model_dump(mode="json"),
            "target_col": self._target_col,
            "id_cols": self._id_cols,
            "params": self._params,
            "fitted": self._fitted,
            "feature_names": self._feature_names,
        }
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)
        logger.info("Transformer saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> DataTransformer:
        path = Path(path)
        with open(path, "rb") as f:
            data = cloudpickle.load(f)
        plan = FeaturePlanPayload.model_validate(data["plan"])
        t = cls(plan, data["target_col"], data["id_cols"])
        t._params = data["params"]
        t._fitted = data["fitted"]
        t._feature_names = data["feature_names"]
        logger.info("Transformer loaded from %s", path)
        return t

    def _fit_column(self, train: pd.DataFrame, ct: ColumnTransform):
        series = train[ct.column]

        for op in ct.operations:
            if op == TransformOp.IMPUTE_MEDIAN:
                numeric = pd.to_numeric(series, errors="coerce")
                self._params[f"{ct.column}__impute_median"] = float(numeric.median()) if not numeric.isna().all() else 0.0

            elif op == TransformOp.IMPUTE_MEAN:
                numeric = pd.to_numeric(series, errors="coerce")
                self._params[f"{ct.column}__impute_mean"] = float(numeric.mean()) if not numeric.isna().all() else 0.0

            elif op == TransformOp.IMPUTE_MODE:
                mode_val = series.mode()
                self._params[f"{ct.column}__impute_mode"] = mode_val.iloc[0] if len(mode_val) > 0 else ""

            elif op == TransformOp.IMPUTE_CONSTANT:
                self._params[f"{ct.column}__impute_constant"] = ct.impute_value or "__MISSING__"

            elif op == TransformOp.ORDINAL_ENCODE:
                cats = sorted(series.dropna().unique().tolist(), key=str)
                mapping = {str(c): i for i, c in enumerate(cats)}
                self._params[f"{ct.column}__ordinal_map"] = mapping
                self._params[f"{ct.column}__ordinal_unknown"] = len(mapping)

            elif op == TransformOp.ONE_HOT_ENCODE:
                cats = sorted(series.dropna().unique().tolist(), key=str)
                self._params[f"{ct.column}__ohe_cats"] = [str(c) for c in cats]

            elif op == TransformOp.FREQUENCY_ENCODE:
                vc = series.value_counts(normalize=True)
                self._params[f"{ct.column}__freq_map"] = {str(k): float(v) for k, v in vc.items()}

            elif op == TransformOp.STANDARD_SCALE:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                self._params[f"{ct.column}__scale_mean"] = float(numeric.mean()) if len(numeric) > 0 else 0.0
                self._params[f"{ct.column}__scale_std"] = float(numeric.std()) if len(numeric) > 0 else 1.0

            elif op == TransformOp.CLIP_OUTLIERS:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) > 0:
                    q1, q3 = numeric.quantile(0.01), numeric.quantile(0.99)
                    self._params[f"{ct.column}__clip_lower"] = float(q1)
                    self._params[f"{ct.column}__clip_upper"] = float(q3)

            elif op == TransformOp.BINNING:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) > 0:
                    _, edges = pd.qcut(numeric, q=10, retbins=True, duplicates="drop")
                    self._params[f"{ct.column}__bin_edges"] = edges.tolist()

    def _fit_new_feature(self, train: pd.DataFrame, nf: NewFeatureDef):
        pass

    def _transform_column(self, df: pd.DataFrame, ct: ColumnTransform) -> pd.DataFrame:
        for op in ct.operations:
            if op == TransformOp.PASSTHROUGH:
                continue

            elif op == TransformOp.IMPUTE_MEDIAN:
                val = self._params.get(f"{ct.column}__impute_median", 0)
                df[ct.column] = pd.to_numeric(df[ct.column], errors="coerce").fillna(val)

            elif op == TransformOp.IMPUTE_MEAN:
                val = self._params.get(f"{ct.column}__impute_mean", 0)
                df[ct.column] = pd.to_numeric(df[ct.column], errors="coerce").fillna(val)

            elif op == TransformOp.IMPUTE_MODE:
                val = self._params.get(f"{ct.column}__impute_mode", "")
                df[ct.column] = df[ct.column].fillna(val)

            elif op == TransformOp.IMPUTE_CONSTANT:
                val = self._params.get(f"{ct.column}__impute_constant", "__MISSING__")
                df[ct.column] = df[ct.column].fillna(val)

            elif op == TransformOp.ORDINAL_ENCODE:
                mapping = self._params.get(f"{ct.column}__ordinal_map", {})
                unknown = self._params.get(f"{ct.column}__ordinal_unknown", -1)
                df[ct.column] = df[ct.column].astype(str).map(
                    lambda x, m=mapping, u=unknown: m.get(x, u)
                )

            elif op == TransformOp.ONE_HOT_ENCODE:
                cats = self._params.get(f"{ct.column}__ohe_cats", [])
                col_str = df[ct.column].astype(str)
                for cat in cats:
                    df[f"{ct.column}__{cat}"] = (col_str == cat).astype(np.int8)
                df = df.drop(columns=[ct.column])

            elif op == TransformOp.FREQUENCY_ENCODE:
                freq_map = self._params.get(f"{ct.column}__freq_map", {})
                df[ct.column] = df[ct.column].astype(str).map(
                    lambda x, m=freq_map: m.get(x, 0.0)
                )

            elif op == TransformOp.STANDARD_SCALE:
                mean = self._params.get(f"{ct.column}__scale_mean", 0)
                std = self._params.get(f"{ct.column}__scale_std", 1)
                if std == 0:
                    std = 1.0
                df[ct.column] = (pd.to_numeric(df[ct.column], errors="coerce") - mean) / std

            elif op == TransformOp.CLIP_OUTLIERS:
                lower = self._params.get(f"{ct.column}__clip_lower")
                upper = self._params.get(f"{ct.column}__clip_upper")
                if lower is not None and upper is not None:
                    df[ct.column] = pd.to_numeric(df[ct.column], errors="coerce").clip(lower, upper)

            elif op == TransformOp.LOG_TRANSFORM:
                df[ct.column] = np.log1p(pd.to_numeric(df[ct.column], errors="coerce").clip(lower=0))

            elif op == TransformOp.SQRT_TRANSFORM:
                df[ct.column] = np.sqrt(pd.to_numeric(df[ct.column], errors="coerce").clip(lower=0))

            elif op == TransformOp.BINNING:
                edges = self._params.get(f"{ct.column}__bin_edges")
                if edges:
                    df[ct.column] = pd.cut(
                        pd.to_numeric(df[ct.column], errors="coerce"),
                        bins=edges, labels=False, include_lowest=True,
                    ).fillna(-1).astype(int)

            elif op == TransformOp.DATE_COMPONENTS:
                dt = pd.to_datetime(df[ct.column], errors="coerce")
                df[f"{ct.column}__year"] = dt.dt.year.fillna(-1).astype(int)
                df[f"{ct.column}__month"] = dt.dt.month.fillna(-1).astype(int)
                df[f"{ct.column}__dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
                df = df.drop(columns=[ct.column])

            elif op == TransformOp.MISSING_INDICATOR:
                df[f"{ct.column}__missing"] = df[ct.column].isna().astype(np.int8)

        return df

    def _create_new_feature(self, df: pd.DataFrame, nf: NewFeatureDef) -> pd.DataFrame:
        if not all(c in df.columns for c in nf.source_columns):
            logger.warning("Cannot create feature '%s': missing source columns %s", nf.name, nf.source_columns)
            return df

        try:
            if nf.transform_type == TransformOp.INTERACTION:
                if "range" in nf.definition.lower() and "temp" in nf.name.lower():
                    temp_vals = df[nf.source_columns].apply(pd.to_numeric, errors="coerce")
                    df[nf.name] = temp_vals.max(axis=1) - temp_vals.min(axis=1)
                elif "*" in nf.definition:
                    vals = [pd.to_numeric(df[c], errors="coerce") for c in nf.source_columns]
                    result = vals[0]
                    for v in vals[1:]:
                        result = result * v
                    df[nf.name] = result
                else:
                    vals = [pd.to_numeric(df[c], errors="coerce") for c in nf.source_columns]
                    result = vals[0]
                    for v in vals[1:]:
                        result = result * v
                    df[nf.name] = result

            elif nf.transform_type == TransformOp.LOG_TRANSFORM:
                df[nf.name] = np.log1p(pd.to_numeric(df[nf.source_columns[0]], errors="coerce").clip(lower=0))

            elif nf.transform_type == TransformOp.SQRT_TRANSFORM:
                df[nf.name] = np.sqrt(pd.to_numeric(df[nf.source_columns[0]], errors="coerce").clip(lower=0))

            elif nf.transform_type == TransformOp.RATIO:
                num = pd.to_numeric(df[nf.source_columns[0]], errors="coerce")
                denom = pd.to_numeric(df[nf.source_columns[1]], errors="coerce") + 1
                df[nf.name] = num / denom

            elif nf.transform_type == TransformOp.MISSING_INDICATOR:
                df[nf.name] = df[nf.source_columns[0]].isna().astype(np.int8)

            elif nf.transform_type == TransformOp.POLYNOMIAL:
                df[nf.name] = pd.to_numeric(df[nf.source_columns[0]], errors="coerce") ** 2

            else:
                logger.warning("Unknown new feature type '%s' for '%s'", nf.transform_type, nf.name)

        except Exception as e:
            logger.warning("Failed to create feature '%s': %s", nf.name, str(e)[:200])

        return df


def split_train_val(
    train: pd.DataFrame,
    target_col: str,
    val_fraction: float = 0.2,
    random_state: int = 42,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    if train_indices is not None and val_indices is not None:
        tr = train.iloc[train_indices].copy()
        val = train.iloc[val_indices].copy()
        return tr, val, train_indices, val_indices

    tr, val = train_test_split(
        train, test_size=val_fraction, random_state=random_state,
    )
    return tr, val, tr.index.tolist(), val.index.tolist()
