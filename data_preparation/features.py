"""Feature engineering: transformations, new features, correlations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from data_preparation.schemas import (
    ColumnDtype, ColumnProfile, ColumnTransform, CorrelationEntry,
    FeaturePlanPayload, NewFeatureDef, TransformOp,
)

logger = logging.getLogger("data_preparation.features")


def build_default_feature_plan(
    profiles: list[ColumnProfile],
    target_col: str,
    id_cols: list[str],
    train: pd.DataFrame,
) -> FeaturePlanPayload:
    columns_to_drop: list[str] = []
    drop_reasons: dict[str, str] = {}
    column_transforms: list[ColumnTransform] = []

    for p in profiles:
        if p.name == target_col:
            continue

        if p.name in id_cols:
            columns_to_drop.append(p.name)
            drop_reasons[p.name] = "ID column, not a feature"
            continue

        if p.is_all_missing:
            columns_to_drop.append(p.name)
            drop_reasons[p.name] = "Entirely missing"
            continue

        if p.is_constant:
            columns_to_drop.append(p.name)
            drop_reasons[p.name] = "Constant value, no information"
            continue

        if p.detected_dtype == ColumnDtype.NUMERIC:
            ops = []
            if p.numeric_stats and p.numeric_stats.n_missing > 0:
                ops.append(TransformOp.IMPUTE_MEDIAN)
            ops.append(TransformOp.PASSTHROUGH)
            column_transforms.append(ColumnTransform(
                column=p.name, operations=ops,
                reason=f"Numeric feature, {p.numeric_stats.n_missing if p.numeric_stats else 0} missing",
            ))

        elif p.detected_dtype == ColumnDtype.CATEGORICAL:
            ops = []
            if p.categorical_stats and p.categorical_stats.n_missing > 0:
                ops.append(TransformOp.IMPUTE_CONSTANT)

            n_unique = p.categorical_stats.n_unique if p.categorical_stats else 0
            if n_unique <= 15:
                ops.append(TransformOp.ONE_HOT_ENCODE)
            else:
                ops.append(TransformOp.ORDINAL_ENCODE)

            column_transforms.append(ColumnTransform(
                column=p.name, operations=ops,
                impute_value="__MISSING__" if TransformOp.IMPUTE_CONSTANT in ops else None,
                reason=f"Categorical, {n_unique} unique values",
            ))

        elif p.detected_dtype == ColumnDtype.TEXT:
            columns_to_drop.append(p.name)
            drop_reasons[p.name] = "Text column, not handled in v1"

        elif p.detected_dtype == ColumnDtype.DATETIME:
            ops = [TransformOp.DATE_COMPONENTS]
            column_transforms.append(ColumnTransform(
                column=p.name, operations=ops,
                reason="Datetime column, extracting components",
            ))

    new_features = _generate_default_features(profiles, target_col, id_cols, train)

    return FeaturePlanPayload(
        columns_to_drop=columns_to_drop,
        drop_reasons=drop_reasons,
        column_transforms=column_transforms,
        new_features=new_features,
        llm_suggestions_used=False,
    )


def _generate_default_features(
    profiles: list[ColumnProfile],
    target_col: str,
    id_cols: list[str],
    train: pd.DataFrame,
) -> list[NewFeatureDef]:
    features: list[NewFeatureDef] = []
    numeric_profiles = [
        p for p in profiles
        if p.detected_dtype == ColumnDtype.NUMERIC
        and p.name != target_col
        and p.name not in id_cols
        and not p.is_constant
        and not p.is_all_missing
    ]

    temp_cols = [p.name for p in numeric_profiles if "temp" in p.name.lower() and "avg" in p.name.lower()]
    if len(temp_cols) >= 2:
        features.append(NewFeatureDef(
            name="temp_range_annual",
            source_columns=temp_cols,
            transform_type=TransformOp.INTERACTION,
            definition="max(monthly_avg_temps) - min(monthly_avg_temps) across all months",
            rationale="Temperature range captures climate variability which affects heating/cooling energy demand",
            missing_handling="Result is NaN if any source is NaN; impute with median after creation",
            requires_fit=False,
        ))

    if "floor_area" in train.columns and "year_built" in train.columns:
        features.append(NewFeatureDef(
            name="log_floor_area",
            source_columns=["floor_area"],
            transform_type=TransformOp.LOG_TRANSFORM,
            definition="log1p(floor_area)",
            rationale="Floor area has right-skewed distribution; log transform may linearize relationship with EUI",
            missing_handling="Result is NaN if floor_area is NaN or <= 0; impute with median",
            requires_fit=False,
        ))

    if "heating_degree_days_c" in train.columns and "cooling_degree_days_c" in train.columns:
        features.append(NewFeatureDef(
            name="hdd_cdd_ratio",
            source_columns=["heating_degree_days_c", "cooling_degree_days_c"],
            transform_type=TransformOp.RATIO,
            definition="heating_degree_days_c / (cooling_degree_days_c + 1)",
            rationale="Ratio of heating to cooling demand characterizes heating-vs-cooling-dominated climates",
            missing_handling="Result is NaN if either source is NaN; denominator offset prevents division by zero",
            requires_fit=False,
        ))

    if "energy_star_rating" in train.columns:
        features.append(NewFeatureDef(
            name="energy_star_missing",
            source_columns=["energy_star_rating"],
            transform_type=TransformOp.MISSING_INDICATOR,
            definition="1 if energy_star_rating is NaN, else 0",
            rationale="Missingness of energy_star_rating may itself be informative (new/unrated buildings)",
            missing_handling="Never missing by definition",
            requires_fit=False,
        ))

    if len(features) < 2:
        if len(numeric_profiles) >= 2:
            c1, c2 = numeric_profiles[0].name, numeric_profiles[1].name
            features.append(NewFeatureDef(
                name=f"{c1}_x_{c2}",
                source_columns=[c1, c2],
                transform_type=TransformOp.INTERACTION,
                definition=f"{c1} * {c2}",
                rationale="Interaction between first two numeric features as a baseline candidate",
                missing_handling="NaN if either is NaN",
                requires_fit=False,
            ))

    return features[:10]


def compute_correlations(
    train: pd.DataFrame,
    target_col: str,
    id_cols: list[str],
    max_features: int = 50,
    threshold: float = 0.95,
) -> tuple[list[CorrelationEntry], list[dict[str, Any]]]:
    numeric_cols = [
        c for c in train.select_dtypes(include=[np.number]).columns
        if c not in id_cols
    ]

    feature_cols = [c for c in numeric_cols if c != target_col]
    if len(feature_cols) > max_features:
        feature_cols = feature_cols[:max_features]

    high_corrs: list[CorrelationEntry] = []
    target_corrs: list[dict[str, Any]] = []

    if len(feature_cols) < 2:
        return high_corrs, target_corrs

    corr_matrix = train[feature_cols].corr(method="pearson")

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            val = corr_matrix.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                high_corrs.append(CorrelationEntry(
                    column_a=feature_cols[i],
                    column_b=feature_cols[j],
                    correlation=round(float(val), 4),
                ))

    if target_col in train.columns:
        for col in feature_cols:
            corr_val = train[col].corr(train[target_col])
            if pd.notna(corr_val):
                target_corrs.append({
                    "column": col,
                    "correlation": round(float(corr_val), 4),
                    "abs_correlation": round(abs(float(corr_val)), 4),
                })
        target_corrs.sort(key=lambda x: x["abs_correlation"], reverse=True)

    return high_corrs, target_corrs


def merge_llm_suggestions(
    plan: FeaturePlanPayload,
    llm_features: list[dict[str, Any]],
    llm_drops: list[dict[str, str]],
    available_columns: list[str],
) -> FeaturePlanPayload:
    existing_names = {f.name for f in plan.new_features}
    col_set = set(available_columns)

    for feat_data in llm_features:
        try:
            feat = NewFeatureDef.model_validate(feat_data)
        except Exception as e:
            logger.warning("Invalid LLM feature suggestion: %s", str(e)[:200])
            continue

        if feat.name in existing_names:
            continue
        if not all(c in col_set for c in feat.source_columns):
            logger.warning("LLM feature '%s' references unknown columns: %s", feat.name, feat.source_columns)
            continue

        plan.new_features.append(feat)
        existing_names.add(feat.name)

    for drop in llm_drops:
        col = drop.get("column", "")
        reason = drop.get("reason", "LLM suggestion")
        if col in col_set and col not in plan.columns_to_drop:
            plan.columns_to_drop.append(col)
            plan.drop_reasons[col] = reason

    if llm_features or llm_drops:
        plan.llm_suggestions_used = True

    return plan
