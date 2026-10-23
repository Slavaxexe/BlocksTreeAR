"""Main orchestrator: ties together loading, profiling, features, transformation."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_preparation.config import PipelineConfig
from data_preparation.features import (
    build_default_feature_plan, compute_correlations, merge_llm_suggestions,
)
from data_preparation.llm_client import LLMClient, MockLLMClient
from data_preparation.llm_prompts import SYSTEM_PROMPT, build_profile_prompt
from data_preparation.loader import DataLoadResult, build_input_manifest, load_data
from data_preparation.profiler import profile_data
from data_preparation.quality import run_quality_checks
from data_preparation.schemas import (
    ArtifactType, DataFileRef, DataProfilePayload, DiagnosticEntry,
    FeaturePlanPayload, OutputManifestPayload, PreparationReportPayload,
    Status, StepReport, make_envelope, validate_payload,
)
from data_preparation.transformer import DataTransformer, split_train_val
from data_preparation.utils import TimeBudget, get_memory_mb, safe_json_dump, timed_step

logger = logging.getLogger("data_preparation")


class PipelineResult:
    def __init__(self):
        self.train_features: pd.DataFrame | None = None
        self.val_features: pd.DataFrame | None = None
        self.test_features: pd.DataFrame | None = None
        self.train_target: pd.Series | None = None
        self.val_target: pd.Series | None = None
        self.train_ids: pd.DataFrame | None = None
        self.val_ids: pd.DataFrame | None = None
        self.test_ids: pd.DataFrame | None = None
        self.transformer: DataTransformer | None = None
        self.feature_names: list[str] = []
        self.split_indices: dict[str, list[int]] | None = None
        self.artifacts: dict[str, dict] = {}
        self.status: Status = Status.SUCCESS
        self.warnings: list[DiagnosticEntry] = []
        self.errors: list[DiagnosticEntry] = []


class DataPreparationPipeline:
    def __init__(self, config: PipelineConfig, llm_client: LLMClient | None = None):
        self._config = config
        self._llm = llm_client or LLMClient(config.llm)
        self._run_id = str(uuid.uuid4())[:8]
        self._budget = TimeBudget(config.time_budget_seconds)
        self._report_steps: list[StepReport] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    def run(self) -> PipelineResult:
        result = PipelineResult()
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_errors = self._config.validate()
        if config_errors:
            for e in config_errors:
                result.errors.append(DiagnosticEntry(code="CONFIG_ERROR", message=e, stage="config"))
            result.status = Status.FAILED
            return result

        try:
            load_result = self._step_load()
            result.artifacts["input_manifest"] = build_input_manifest(self._config, load_result)
            safe_json_dump(result.artifacts["input_manifest"], output_dir / "input_manifest.json")

            if load_result.errors or load_result.train is None or load_result.test is None:
                result.status = Status.FAILED
                for e in load_result.errors:
                    result.errors.append(DiagnosticEntry(**e))
                return result

            train, test = load_result.train, load_result.test
            target_col = self._config.target_columns[0]

            profile_envelope = self._step_profile(train, test, load_result)
            result.artifacts["data_profile"] = profile_envelope
            safe_json_dump(profile_envelope, output_dir / "data_profile.json")

            profile_payload = DataProfilePayload.model_validate(profile_envelope["payload"])

            quality_report = self._step_quality(train, test, profile_payload)

            n_target_missing = sum(
                1 for i in quality_report.issues if i["code"] == "TARGET_MISSING"
            )
            if n_target_missing > 0:
                missing_count = train[target_col].isna().sum()
                logger.warning("Dropping %d rows with missing target", missing_count)
                train = train.dropna(subset=[target_col]).copy()
                result.warnings.append(DiagnosticEntry(
                    code="TARGET_ROWS_DROPPED",
                    message=f"Dropped {missing_count} rows with missing target",
                    stage="quality",
                ))

            n_target_inf = (np.isinf(pd.to_numeric(train[target_col], errors="coerce"))).sum()
            if n_target_inf > 0:
                train = train[~np.isinf(pd.to_numeric(train[target_col], errors="coerce"))].copy()
                result.warnings.append(DiagnosticEntry(
                    code="TARGET_INF_DROPPED",
                    message=f"Dropped {n_target_inf} rows with infinite target",
                    stage="quality",
                ))

            train_split, val_split, train_idx, val_idx = split_train_val(
                train, target_col,
                val_fraction=self._config.val_fraction,
                random_state=self._config.random_state,
                train_indices=self._config.train_indices,
                val_indices=self._config.val_indices,
            )
            result.split_indices = {"train": train_idx, "val": val_idx}

            feature_plan = self._step_feature_plan(train_split, profile_payload, load_result)
            result.artifacts["feature_plan"] = make_envelope(
                ArtifactType.FEATURE_PLAN, self._run_id, Status.SUCCESS,
                feature_plan.model_dump(mode="json"),
            )
            safe_json_dump(result.artifacts["feature_plan"], output_dir / "feature_plan.json")

            transformer, train_features, val_features, test_features = self._step_transform(
                train_split, val_split, test, feature_plan, target_col,
            )

            result.transformer = transformer
            result.train_features = train_features
            result.val_features = val_features
            result.test_features = test_features
            result.feature_names = transformer.feature_names
            result.train_target = train_split[target_col].reset_index(drop=True)
            result.val_target = val_split[target_col].reset_index(drop=True)

            for id_col in self._config.id_columns:
                if id_col in train_split.columns:
                    if result.train_ids is None:
                        result.train_ids = pd.DataFrame()
                        result.val_ids = pd.DataFrame()
                        result.test_ids = pd.DataFrame()
                    result.train_ids[id_col] = train_split[id_col].reset_index(drop=True)
                    result.val_ids[id_col] = val_split[id_col].reset_index(drop=True)
                    if id_col in test.columns:
                        result.test_ids[id_col] = test[id_col].reset_index(drop=True)

            high_corrs, target_corrs = [], []
            if self._budget.check(30):
                high_corrs, target_corrs = compute_correlations(
                    train_split, target_col, self._config.id_columns,
                )

            self._save_outputs(result, output_dir)

            prep_report = self._build_preparation_report(high_corrs, target_corrs, result)
            result.artifacts["preparation_report"] = make_envelope(
                ArtifactType.PREPARATION_REPORT, self._run_id, Status.SUCCESS,
                prep_report.model_dump(mode="json"),
            )
            safe_json_dump(result.artifacts["preparation_report"], output_dir / "preparation_report.json")

            output_manifest = self._build_output_manifest(result, output_dir)
            result.artifacts["output_manifest"] = make_envelope(
                ArtifactType.OUTPUT_MANIFEST, self._run_id, Status.SUCCESS,
                output_manifest.model_dump(mode="json"),
            )
            safe_json_dump(result.artifacts["output_manifest"], output_dir / "output_manifest.json")

            result.status = Status.SUCCESS

        except Exception as e:
            logger.exception("Pipeline failed")
            result.status = Status.FAILED
            result.errors.append(DiagnosticEntry(
                code="PIPELINE_ERROR", message=str(e)[:500], stage="pipeline",
            ))

        return result

    def run_final(self, full_train: pd.DataFrame, test: pd.DataFrame, plan: FeaturePlanPayload) -> tuple[pd.DataFrame, DataTransformer]:
        """Refit on full train with a fixed plan, transform test."""
        target_col = self._config.target_columns[0]
        transformer = DataTransformer(plan, target_col, self._config.id_columns)
        transformer.fit(full_train)
        test_features = transformer.transform(test, is_test=True)
        return test_features, transformer

    def _step_load(self) -> DataLoadResult:
        with timed_step("load", self._budget) as info:
            result = load_data(self._config)
        self._report_steps.append(StepReport(
            step_name="load", status=Status.SUCCESS if not result.errors else Status.FAILED,
            duration_seconds=info.get("duration_seconds", 0),
        ))
        return result

    def _step_profile(self, train: pd.DataFrame, test: pd.DataFrame, load_result: DataLoadResult) -> dict:
        with timed_step("profile", self._budget) as info:
            envelope = profile_data(train, test, self._config, load_result.data_dictionary)
        self._report_steps.append(StepReport(
            step_name="profile", status=Status.SUCCESS,
            duration_seconds=info.get("duration_seconds", 0),
            rows_before=len(train),
        ))
        return envelope

    def _step_quality(self, train, test, profile_payload):
        with timed_step("quality_checks", self._budget) as info:
            report = run_quality_checks(
                train, test, profile_payload,
                self._config.target_columns[0],
                self._config.id_columns,
            )
        self._report_steps.append(StepReport(
            step_name="quality_checks", status=Status.SUCCESS,
            duration_seconds=info.get("duration_seconds", 0),
            details={"issues_count": len(report.issues), "critical": report.critical_count},
        ))
        return report

    def _step_feature_plan(
        self, train_split: pd.DataFrame, profile: DataProfilePayload, load_result: DataLoadResult,
    ) -> FeaturePlanPayload:
        with timed_step("feature_plan", self._budget) as info:
            plan = build_default_feature_plan(
                profile.train_columns,
                self._config.target_columns[0],
                self._config.id_columns,
                train_split,
            )

            if self._budget.check(60) and self._llm.is_available:
                plan = self._enrich_plan_with_llm(plan, profile, load_result, train_split)

        n_new = len(plan.new_features)
        self._report_steps.append(StepReport(
            step_name="feature_plan", status=Status.SUCCESS,
            duration_seconds=info.get("duration_seconds", 0),
            details={"new_features": n_new, "drops": len(plan.columns_to_drop)},
        ))
        return plan

    def _enrich_plan_with_llm(
        self, plan: FeaturePlanPayload, profile: DataProfilePayload,
        load_result: DataLoadResult, train: pd.DataFrame,
    ) -> FeaturePlanPayload:
        col_summaries = []
        for cp in profile.train_columns[:40]:
            summary: dict[str, Any] = {"name": cp.name, "dtype": cp.detected_dtype.value}
            if cp.numeric_stats:
                s = cp.numeric_stats
                summary.update({
                    "min": s.min, "max": s.max, "mean": s.mean,
                    "n_missing": s.n_missing, "n_unique": s.n_unique,
                })
            if cp.categorical_stats:
                s = cp.categorical_stats
                summary.update({
                    "n_unique": s.n_unique, "n_missing": s.n_missing,
                    "top_values": s.top_values[:3],
                })
            col_summaries.append(summary)

        target_stats = {}
        if profile.target.stats:
            target_stats = profile.target.stats.model_dump(mode="json")

        user_prompt = build_profile_prompt(
            task_description=load_result.task_description,
            target_col=self._config.target_columns[0],
            metric=self._config.metric_name,
            column_summaries=col_summaries,
            target_stats=target_stats,
            n_rows_train=profile.train_summary.n_rows,
            n_rows_test=profile.test_summary.n_rows,
        )

        resp = self._llm.request_structured(SYSTEM_PROMPT, user_prompt)
        if resp and resp.interpretation:
            interp = resp.interpretation
            plan = merge_llm_suggestions(
                plan,
                [f.model_dump() for f in interp.suggested_features],
                interp.suggested_drops,
                list(train.columns),
            )
            plan.llm_rationale = interp.summary
            logger.info("LLM enrichment: %d features suggested, %d drops",
                        len(interp.suggested_features), len(interp.suggested_drops))
        else:
            logger.warning("LLM did not return valid response, using default plan")

        return plan

    def _step_transform(
        self,
        train_split: pd.DataFrame,
        val_split: pd.DataFrame,
        test: pd.DataFrame,
        plan: FeaturePlanPayload,
        target_col: str,
    ) -> tuple[DataTransformer, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with timed_step("transform", self._budget) as info:
            transformer = DataTransformer(plan, target_col, self._config.id_columns)
            train_features = transformer.fit_transform(train_split)
            val_features = transformer.transform(val_split)
            test_features = transformer.transform(test, is_test=True)

            common = set(train_features.columns) & set(val_features.columns) & set(test_features.columns)
            ordered = [c for c in train_features.columns if c in common]
            train_features = train_features[ordered]
            val_features = val_features[ordered]
            test_features = test_features[ordered]

        self._report_steps.append(StepReport(
            step_name="transform", status=Status.SUCCESS,
            duration_seconds=info.get("duration_seconds", 0),
            rows_before=len(train_split),
            rows_after=len(train_features),
            columns_before=len(train_split.columns),
            columns_after=len(train_features.columns),
        ))
        return transformer, train_features, val_features, test_features

    def _save_outputs(self, result: PipelineResult, output_dir: Path):
        if result.train_features is not None:
            result.train_features.to_parquet(output_dir / "train_features.parquet", index=False)
        if result.val_features is not None:
            result.val_features.to_parquet(output_dir / "val_features.parquet", index=False)
        if result.test_features is not None:
            result.test_features.to_parquet(output_dir / "test_features.parquet", index=False)
        if result.train_target is not None:
            pd.DataFrame({self._config.target_columns[0]: result.train_target}).to_parquet(
                output_dir / "train_target.parquet", index=False)
        if result.val_target is not None:
            pd.DataFrame({self._config.target_columns[0]: result.val_target}).to_parquet(
                output_dir / "val_target.parquet", index=False)
        if result.train_ids is not None:
            result.train_ids.to_parquet(output_dir / "train_ids.parquet", index=False)
        if result.val_ids is not None:
            result.val_ids.to_parquet(output_dir / "val_ids.parquet", index=False)
        if result.test_ids is not None:
            result.test_ids.to_parquet(output_dir / "test_ids.parquet", index=False)
        if result.transformer is not None:
            result.transformer.save(output_dir / "transformer.pkl")

        recipe = result.transformer.plan.model_dump(mode="json") if result.transformer else {}
        safe_json_dump(recipe, output_dir / "recipe.json")

    def _build_preparation_report(
        self, high_corrs, target_corrs, result: PipelineResult,
    ) -> PreparationReportPayload:
        new_feat_names = [nf.name for nf in result.transformer.plan.new_features] if result.transformer else []
        dropped = result.transformer.plan.columns_to_drop if result.transformer else []

        return PreparationReportPayload(
            steps=self._report_steps,
            new_features_created=new_feat_names,
            columns_dropped=dropped,
            high_correlations=high_corrs,
            target_correlations=target_corrs[:20],
            train_val_split={
                "method": "random" if self._config.train_indices is None else "predefined",
                "val_fraction": self._config.val_fraction,
                "random_state": self._config.random_state,
                "train_rows": len(result.train_features) if result.train_features is not None else 0,
                "val_rows": len(result.val_features) if result.val_features is not None else 0,
            },
            total_duration_seconds=round(self._budget.elapsed, 2),
            llm_calls_count=self._llm.calls_made,
            llm_tokens_used=self._llm.tokens_used,
        )

    def _build_output_manifest(self, result: PipelineResult, output_dir: Path) -> OutputManifestPayload:
        def _ref(name: str, df: pd.DataFrame | None) -> DataFileRef:
            if df is None:
                return DataFileRef(path="", n_rows=0, n_columns=0)
            return DataFileRef(
                path=str(output_dir / f"{name}.parquet"),
                n_rows=len(df), n_columns=len(df.columns),
                columns=list(df.columns),
            )

        return OutputManifestPayload(
            train_features=_ref("train_features", result.train_features),
            val_features=_ref("val_features", result.val_features),
            test_features=_ref("test_features", result.test_features),
            train_target=_ref("train_target",
                              pd.DataFrame({self._config.target_columns[0]: result.train_target})
                              if result.train_target is not None else None),
            val_target=_ref("val_target",
                            pd.DataFrame({self._config.target_columns[0]: result.val_target})
                            if result.val_target is not None else None),
            train_ids=_ref("train_ids", result.train_ids),
            val_ids=_ref("val_ids", result.val_ids),
            test_ids=_ref("test_ids", result.test_ids),
            transformer_path=str(output_dir / "transformer.pkl"),
            recipe_path=str(output_dir / "recipe.json"),
            feature_names=result.feature_names,
            target_columns=self._config.target_columns,
            id_columns=self._config.id_columns,
            split_indices=result.split_indices,
        )
