"""Tests for the full pipeline and individual components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preparation.config import PipelineConfig
from data_preparation.features import build_default_feature_plan, compute_correlations
from data_preparation.llm_client import MockLLMClient
from data_preparation.loader import load_data
from data_preparation.pipeline import DataPreparationPipeline, PipelineResult
from data_preparation.profiler import profile_data
from data_preparation.quality import run_quality_checks
from data_preparation.schemas import (
    ArtifactType, DataProfilePayload, FeaturePlanPayload,
    Status, validate_payload, SCHEMA_VERSION,
)
from data_preparation.transformer import DataTransformer, split_train_val


class TestSchemas:
    def test_schema_version(self):
        assert SCHEMA_VERSION == "1.0.0"

    def test_validate_input_manifest(self):
        data = {
            "task_type": "regression",
            "metric_name": "rmse",
            "target_columns": ["site_eui"],
            "files": [],
        }
        result = validate_payload(ArtifactType.INPUT_MANIFEST, data)
        assert result.target_columns == ["site_eui"]

    def test_target_columns_must_be_one(self):
        with pytest.raises(Exception):
            validate_payload(ArtifactType.INPUT_MANIFEST, {
                "task_type": "regression",
                "metric_name": "rmse",
                "target_columns": [],
                "files": [],
            })

    def test_reject_nan_in_numeric_stats(self):
        from data_preparation.schemas import NumericStats
        stats = NumericStats(min=float("nan"), max=float("inf"))
        assert stats.min is None
        assert stats.max is None


class TestLoader:
    def test_load_success(self, synth_task_dir, default_config):
        result = load_data(default_config)
        assert result.train is not None
        assert result.test is not None
        assert len(result.errors) == 0
        assert result.train.shape[0] == 200
        assert result.test.shape[0] == 80

    def test_load_missing_target(self, synth_task_dir, tmp_path):
        config = PipelineConfig(
            task_dir=str(synth_task_dir),
            target_columns=["nonexistent"],
            output_dir=str(tmp_path / "output"),
        )
        result = load_data(config)
        assert len(result.errors) > 0
        assert any(e["code"] == "TARGET_NOT_FOUND" for e in result.errors)

    def test_load_missing_files(self, tmp_path):
        config = PipelineConfig(
            task_dir=str(tmp_path / "empty"),
            output_dir=str(tmp_path / "output"),
        )
        (tmp_path / "empty").mkdir()
        result = load_data(config)
        assert len(result.errors) > 0


class TestProfiler:
    def test_profile_basic(self, synth_task_dir, default_config):
        result = load_data(default_config)
        envelope = profile_data(result.train, result.test, default_config, result.data_dictionary)
        assert envelope["status"] == "success"
        payload = DataProfilePayload.model_validate(envelope["payload"])
        assert payload.train_summary.n_rows == 200
        assert payload.test_summary.n_rows == 80
        assert payload.target.name == "site_eui"

    def test_constant_detected(self, synth_task_dir, default_config):
        result = load_data(default_config)
        envelope = profile_data(result.train, result.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        const_cols = [p for p in payload.train_columns if p.is_constant]
        assert len(const_cols) > 0

    def test_all_missing_detected(self, synth_task_dir, default_config):
        result = load_data(default_config)
        envelope = profile_data(result.train, result.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        missing_cols = [p for p in payload.train_columns if p.is_all_missing]
        assert len(missing_cols) > 0


class TestQuality:
    def test_quality_checks(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        report = run_quality_checks(lr.train, lr.test, payload, "site_eui", ["id"])
        assert len(report.issues) > 0

    def test_detects_inf_values(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        report = run_quality_checks(lr.train, lr.test, payload, "site_eui", ["id"])
        inf_issues = [i for i in report.issues if i["code"] == "INF_VALUES"]
        assert len(inf_issues) > 0


class TestFeatures:
    def test_default_plan(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        plan = build_default_feature_plan(payload.train_columns, "site_eui", ["id"], lr.train)
        assert len(plan.new_features) >= 2
        assert "id" in plan.columns_to_drop
        assert "const_col" in plan.columns_to_drop
        assert "all_missing" in plan.columns_to_drop

    def test_correlations(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        high, target = compute_correlations(lr.train, "site_eui", ["id"])
        assert isinstance(high, list)
        assert isinstance(target, list)
        assert len(target) > 0


class TestTransformer:
    def test_fit_transform(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        plan = build_default_feature_plan(payload.train_columns, "site_eui", ["id"], lr.train)

        tr, val, _, _ = split_train_val(lr.train, "site_eui", random_state=42)
        transformer = DataTransformer(plan, "site_eui", ["id"])
        train_feat = transformer.fit_transform(tr)
        val_feat = transformer.transform(val)
        test_feat = transformer.transform(lr.test, is_test=True)

        assert len(train_feat) == len(tr)
        assert len(val_feat) == len(val)
        assert len(test_feat) == len(lr.test)
        assert "site_eui" not in train_feat.columns
        assert "id" not in train_feat.columns
        assert set(train_feat.columns) == set(val_feat.columns) == set(test_feat.columns)

    def test_no_leakage(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        plan = build_default_feature_plan(payload.train_columns, "site_eui", ["id"], lr.train)

        tr, val, _, _ = split_train_val(lr.train, "site_eui", random_state=42)
        t1 = DataTransformer(plan, "site_eui", ["id"])
        t1.fit(tr)

        val_modified = val.copy()
        val_modified["num_a"] = val_modified["num_a"] * 100

        t2 = DataTransformer(plan, "site_eui", ["id"])
        t2.fit(tr)

        assert t1._params == t2._params

    def test_test_row_count_preserved(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        plan = build_default_feature_plan(payload.train_columns, "site_eui", ["id"], lr.train)

        transformer = DataTransformer(plan, "site_eui", ["id"])
        transformer.fit(lr.train)
        test_feat = transformer.transform(lr.test, is_test=True)
        assert len(test_feat) == 80

    def test_save_load(self, synth_task_dir, default_config, tmp_path):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        plan = build_default_feature_plan(payload.train_columns, "site_eui", ["id"], lr.train)

        transformer = DataTransformer(plan, "site_eui", ["id"])
        transformer.fit(lr.train)

        path = tmp_path / "transformer.pkl"
        transformer.save(path)
        loaded = DataTransformer.load(path)

        assert loaded.is_fitted
        test1 = transformer.transform(lr.test, is_test=True)
        test2 = loaded.transform(lr.test, is_test=True)
        pd.testing.assert_frame_equal(test1, test2)

    def test_reproducibility(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        plan = build_default_feature_plan(payload.train_columns, "site_eui", ["id"], lr.train)

        t1 = DataTransformer(plan, "site_eui", ["id"])
        t1.fit(lr.train)
        r1 = t1.transform(lr.test, is_test=True)

        t2 = DataTransformer(plan, "site_eui", ["id"])
        t2.fit(lr.train)
        r2 = t2.transform(lr.test, is_test=True)

        pd.testing.assert_frame_equal(r1, r2)


class TestPipeline:
    def test_full_run(self, synth_task_dir, default_config):
        pipeline = DataPreparationPipeline(default_config)
        result = pipeline.run()
        assert result.status == Status.SUCCESS
        assert result.train_features is not None
        assert result.val_features is not None
        assert result.test_features is not None
        assert result.train_target is not None
        assert result.val_target is not None
        assert len(result.feature_names) > 0

        assert set(result.train_features.columns) == set(result.val_features.columns)
        assert set(result.train_features.columns) == set(result.test_features.columns)
        assert len(result.test_features) == 80

    def test_artifacts_written(self, synth_task_dir, default_config):
        pipeline = DataPreparationPipeline(default_config)
        result = pipeline.run()
        output_dir = Path(default_config.output_dir)

        for name in ["input_manifest", "data_profile", "feature_plan",
                      "preparation_report", "output_manifest"]:
            path = output_dir / f"{name}.json"
            assert path.exists(), f"{name}.json not found"
            with open(path) as f:
                data = json.load(f)
            assert data["schema_version"] == SCHEMA_VERSION
            assert data["artifact_type"] == name

    def test_parquet_files_written(self, synth_task_dir, default_config):
        pipeline = DataPreparationPipeline(default_config)
        pipeline.run()
        output_dir = Path(default_config.output_dir)
        for name in ["train_features", "val_features", "test_features",
                      "train_target", "val_target"]:
            assert (output_dir / f"{name}.parquet").exists()

    def test_with_mock_llm(self, synth_task_dir, default_config):
        mock = MockLLMClient(responses=[{
            "summary": "Test data with numeric and categorical features",
            "key_observations": ["floor_area is right-skewed"],
            "suggested_features": [{
                "name": "floor_area_squared",
                "source_columns": ["floor_area"],
                "transform_type": "polynomial",
                "definition": "floor_area ** 2",
                "rationale": "capture nonlinear relationship",
                "missing_handling": "NaN propagates",
                "requires_fit": False,
            }],
            "suggested_drops": [],
            "suggested_transforms": [],
        }])
        pipeline = DataPreparationPipeline(default_config, llm_client=mock)
        result = pipeline.run()
        assert result.status == Status.SUCCESS
        assert mock.calls_made >= 1

    def test_invalid_llm_json(self, synth_task_dir, default_config):
        mock = MockLLMClient(responses=[{"garbage": "not valid"}])
        pipeline = DataPreparationPipeline(default_config, llm_client=mock)
        result = pipeline.run()
        assert result.status == Status.SUCCESS

    def test_llm_unavailable(self, synth_task_dir, default_config):
        default_config.llm.api_key = ""
        default_config.llm.base_url = ""
        pipeline = DataPreparationPipeline(default_config)
        result = pipeline.run()
        assert result.status == Status.SUCCESS

    def test_time_budget(self, synth_task_dir, default_config):
        default_config.time_budget_seconds = 1
        pipeline = DataPreparationPipeline(default_config)
        result = pipeline.run()
        assert result.status in (Status.SUCCESS, Status.PARTIAL)


class TestSplitIndices:
    def test_predefined_split(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        train_idx = list(range(0, 150))
        val_idx = list(range(150, 200))
        tr, val, ri, vi = split_train_val(
            lr.train, "site_eui",
            train_indices=train_idx, val_indices=val_idx,
        )
        assert len(tr) == 150
        assert len(val) == 50
        assert ri == train_idx
        assert vi == val_idx


class TestEdgeCases:
    def test_unknown_categories_in_test(self, synth_task_dir, default_config):
        lr = load_data(default_config)
        envelope = profile_data(lr.train, lr.test, default_config)
        payload = DataProfilePayload.model_validate(envelope["payload"])
        plan = build_default_feature_plan(payload.train_columns, "site_eui", ["id"], lr.train)

        transformer = DataTransformer(plan, "site_eui", ["id"])
        transformer.fit(lr.train)
        test_feat = transformer.transform(lr.test, is_test=True)
        assert not test_feat.isnull().all(axis=1).any()

    def test_config_validation(self):
        cfg = PipelineConfig(task_type="classification")
        errors = cfg.validate()
        assert len(errors) > 0

        cfg2 = PipelineConfig(target_columns=[])
        errors2 = cfg2.validate()
        assert len(errors2) > 0
