"""Protocol, honest evaluation, concurrency and recovery checks without paid API calls."""

import asyncio
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from openai import AsyncOpenAI
import pandas as pd
from pydantic import ValidationError

from alfa_agent.agents import JSONAgents
from alfa_agent.config import Settings
from alfa_agent.contracts import (DataFiles, Dummy, ExtraTrees, Finish, Hypothesis, Metric,
                                  Pipeline, Problem, Profile, Start, Task, Validation)
from alfa_agent.data import local_task, profile_data, read_tables, validate_submission
from alfa_agent.qwen import QwenError
from alfa_agent.runner import Options, execute, offline_hypothesis, run
from alfa_agent.storage import atomic_bytes, fingerprint, load, save
from alfa_agent.summarizer import summarize
from alfa_agent.worker import build_pipeline


def task_contract(classification=False):
    return Task(problem=Problem(task_type="classification" if classification else "regression",
                                target_column="answer"),
                data=DataFiles(train_file="train.csv", test_file="test.csv",
                               sample_submission_file="sample_submission.csv", id_column="row_id"),
                metric=Metric(name="macro_f1" if classification else "rmse",
                              direction="maximize" if classification else "minimize"))


def start_contract(node="node_000", classification=False, model=None):
    return Start(node_id=node, parent_id=None if node == "node_000" else "node_000",
                 source="baseline" if node == "node_000" else "offline", task=task_contract(classification),
                 dataset_sha256="a" * 64, validation=Validation(stratified=classification, split_sha256="b" * 64),
                 hypothesis=Hypothesis(rationale="test", expected_effect="test",
                                       pipeline=Pipeline(model=model or Dummy())))


def measured(start, value):
    return Finish(node_id=start.node_id, status="completed", metric=start.task.metric, value=value,
                  duration_seconds=1.0, split_sha256=start.validation.split_sha256,
                  pipeline_sha256=fingerprint(start.hypothesis.pipeline), submission_file="submission.csv")


def write_fixture(directory, classification=False):
    rng = np.random.default_rng(13)
    x = np.arange(100, dtype=float)
    target = np.where(x % 2 == 0, "yes", "no") if classification else x * 3 + rng.normal(size=100)
    train = pd.DataFrame({"row_id": [f"{i:05d}" for i in range(100)], "signal": x,
                          "category": np.where(x % 2 == 0, "even", "odd"), "answer": target})
    test = pd.DataFrame({"row_id": [f"{i:05d}" for i in range(110, 120)],
                         "signal": np.arange(110., 120.), "category": ["new"] * 10})
    sample = pd.DataFrame({"row_id": test.row_id, "answer": ["no" if classification else 0] * 10})
    for name, frame in (("train", train), ("test", test), ("sample_submission", sample)):
        frame.to_csv(directory / f"{name}.csv", index=False)
    path = directory / "task.md"
    path.write_text("Metric: " + ("Macro F1" if classification else "RMSE")
                    + ". Files: `train.csv`, `test.csv`, `sample_submission.csv`.", encoding="utf-8")
    return path


class ProtocolTests(unittest.TestCase):
    def test_unknown_fields_types_nan_and_parameter_limits_are_rejected(self):
        for payload in ({"family": "extra_trees", "n_estimators": "64"},
                        {"family": "extra_trees", "n_estimators": 100000},
                        {"family": "extra_trees", "shell": "arbitrary command"}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                ExtraTrees.model_validate_json(json.dumps(payload))
        with self.assertRaises(ValidationError):
            measured(start_contract(), float("nan"))
        with self.assertRaises(ValidationError):
            Metric(name="rmse", direction="maximize")
        with self.assertRaises(ValidationError):
            measured(start_contract(classification=True), 1.1)

    def test_minimize_and_maximize_choose_correct_winner_and_exact_delta(self):
        for classification, values in ((False, [10., 8.]), (True, [.2, .7])):
            a, b = start_contract(classification=classification), start_contract("node_001", classification, ExtraTrees())
            starts = {s.node_id: s for s in (a, b)}
            finishes = {s.node_id: measured(s, v) for s, v in zip((a, b), values)}
            delta = summarize(starts, finishes, b.node_id, a.node_id)
            self.assertEqual(delta.best_node_id, b.node_id)
            self.assertAlmostEqual(delta.comparisons[0].metric_delta, values[1] - values[0])
            self.assertGreater(delta.comparisons[0].improvement, 0)
            self.assertTrue(delta.comparisons[0].changes)

    def test_failed_and_incomparable_runs_never_win(self):
        a, b = start_contract(), start_contract("node_001", model=ExtraTrees())
        failure = Finish(node_id=b.node_id, status="failed", metric=b.task.metric, duration_seconds=1.0,
                         split_sha256=b.validation.split_sha256, pipeline_sha256=fingerprint(b.hypothesis.pipeline),
                         error="failed")
        starts = {a.node_id: a, b.node_id: b}
        finishes = {a.node_id: measured(a, 3.), b.node_id: failure}
        self.assertEqual(summarize(starts, finishes, b.node_id, a.node_id).best_node_id, a.node_id)
        finishes[b.node_id] = measured(b, 1.).model_copy(update={"split_sha256": "different"})
        with self.assertRaisesRegex(ValueError, "Cannot compare"):
            summarize(starts, finishes, b.node_id, a.node_id)

    def test_preprocessing_fits_only_on_supplied_training_rows_and_handles_unknown_category(self):
        start = start_contract("node_001", model=ExtraTrees(n_estimators=16))
        profile = Profile(dataset_sha256="a" * 64, train_rows=3, test_rows=1,
                          numeric_columns=["x"], categorical_columns=["cat"], missing_train_cells=0,
                          class_counts={})
        pipeline = build_pipeline(start, profile, threads=1)
        x = pd.DataFrame({"x": [-100., 0., 1e9], "cat": ["a", "b", "unseen"]})
        pipeline.fit(x.iloc[:2], [0., 1.])
        imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        self.assertEqual(imputer.statistics_[0], -50.)
        self.assertTrue(np.isfinite(pipeline.predict(x.iloc[2:])).all())

    def test_generic_target_leading_zero_ids_shared_split_and_submission_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = write_fixture(directory)
            task = local_task(path)
            self.assertEqual(task.problem.target_column, "answer")
            train, test, sample = read_tables(directory, task)
            self.assertEqual(train.row_id[0], "00000")
            _, first, fit, valid = profile_data(directory, task)
            _, second, fit2, valid2 = profile_data(directory, task)
            self.assertEqual(first.split_sha256, second.split_sha256)
            self.assertFalse(set(fit) & set(valid))
            np.testing.assert_equal(fit, fit2)
            np.testing.assert_equal(valid, valid2)
            with self.assertRaisesRegex(ValueError, "IDs"):
                validate_submission(sample.iloc[::-1].reset_index(drop=True), test, sample, train, task)
            path.write_text("RMSE and Macro F1. `train.csv`, `test.csv`, `sample_submission.csv`.", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "primary metric"):
                local_task(path)


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_extractors_run_concurrently_and_cross_check_contract(self):
        agents = JSONAgents(Settings("fake", "https://qwen.example/v1", "model"), time.monotonic() + 10)
        task = task_contract()
        entered = []
        barrier = asyncio.Event()
        async def ask(schema, prompt, role):
            entered.append(role)
            if len(entered) == 3:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), .5)
            return {"problem": task.problem, "data": task.data, "metric": task.metric}[role]
        with patch.object(agents, "ask", side_effect=ask):
            result = await agents.inspect("task data", task)
        self.assertEqual(len(result), 3)

    async def test_json_repair_is_bounded_and_errors_never_echo_provider_secret(self):
        settings = Settings("fake-secret", "https://qwen.example/v1", "model", 5, 256)
        requests = []
        def factory(handler):
            return AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url, max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "model", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"task_type":"regression","target_column":42}'}}]})
        agents = JSONAgents(settings, time.monotonic() + 10)
        with patch("alfa_agent.agents.AsyncOpenAI", side_effect=lambda **kwargs: factory(handler)):
            with self.assertRaisesRegex(QwenError, "дважды"):
                await agents.ask(Problem, "Extract", "test")
        self.assertEqual(len(requests), 2)
        self.assertEqual([c.status for c in agents.calls], ["invalid_json", "invalid_json"])
        requests.clear()
        def unauthorized(request):
            requests.append(request)
            return httpx.Response(401, json={"error": {"message": "fake-secret"}})
        agents = JSONAgents(settings, time.monotonic() + 10)
        with patch("alfa_agent.agents.AsyncOpenAI", side_effect=lambda **kwargs: factory(unauthorized)):
            with self.assertRaises(QwenError) as caught:
                await agents.ask(Problem, "Extract", "test")
        self.assertNotIn(settings.api_key, str(caught.exception))
        self.assertEqual(len(requests), 1)

    async def test_unsupported_json_mode_has_one_validated_text_fallback(self):
        requests = []
        settings = Settings("fake-secret", "https://qwen.example/v1", "model", 5, 256)
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if "response_format" in body:
                return httpx.Response(400, json={"error": {"message": "unsupported response format"}})
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "model", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"task_type":"regression","target_column":"answer"}'}}]})
        def factory(**kwargs):
            return AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url, max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        agents = JSONAgents(settings, time.monotonic() + 10)
        with patch("alfa_agent.agents.AsyncOpenAI", side_effect=factory):
            result = await agents.ask(Problem, "Extract", "test")
        self.assertEqual(result.target_column, "answer")
        self.assertEqual(len(requests), 2)
        self.assertNotIn("response_format", requests[1])

    async def test_conflicting_extractors_stop_search_instead_of_guessing(self):
        agents = JSONAgents(Settings("fake", "https://qwen.example/v1", "model"), time.monotonic() + 10)
        task = task_contract()
        async def wrong(schema, prompt, role):
            return {"problem": Problem(task_type="regression", target_column="wrong"),
                    "data": task.data, "metric": task.metric}[role]
        with patch.object(agents, "ask", side_effect=wrong), self.assertRaisesRegex(QwenError, "противоречит"):
            await agents.inspect("task data", task)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_baker_does_not_delay_publishing_completed_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            task_path = write_fixture(directory)
            published_during_bake = []
            class SlowAgents:
                calls = []
                def __init__(self, *args, **kwargs):
                    pass
                async def inspect(self, text, task):
                    return task.problem, task.data, task.metric
                async def bake(self, context, slot):
                    if slot == 2:
                        for _ in range(100):
                            if list((directory / 'outputs').glob('*/*/nodes/node_002/delta.json')):
                                published_during_bake.append(True)
                                break
                            await asyncio.sleep(.005)
                    return offline_hypothesis(slot + 1, False)
            async def fake_execute(start, task_dir, run_dir, deadline, timeout, threads):
                save(run_dir / 'nodes' / start.node_id / 'start.json', start)
                if start.node_id != 'node_000':
                    await asyncio.sleep(.01 if start.node_id == 'node_001' else .04)
                relative = f'nodes/{start.node_id}/submission.csv'
                atomic_bytes(run_dir / relative, (directory / 'sample_submission.csv').read_bytes())
                result = measured(start, 100. - int(start.node_id[-3:]))
                result = result.model_copy(update={'submission_file': relative})
                save(run_dir / 'nodes' / start.node_id / 'finish.json', result)
                return result
            with patch('alfa_agent.runner.JSONAgents', SlowAgents), \
                    patch('alfa_agent.runner.execute', side_effect=fake_execute), \
                    redirect_stdout(io.StringIO()):
                _, report = await run(task_path, Options(max_experiments=4, time_budget=10,
                    output_root=directory / 'outputs'), Settings('fake', 'https://qwen.example/v1', 'model'))
            self.assertEqual(report.experiments, 4)
            self.assertEqual(published_during_bake, [True])

    async def test_regression_and_classification_complete_loop_and_export_best(self):
        for classification in (False, True):
            with self.subTest(classification=classification), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                task_path = write_fixture(directory, classification)
                with redirect_stdout(io.StringIO()):
                    run_dir, report = await run(task_path, Options(offline=True, max_experiments=3,
                        time_budget=60, experiment_timeout=30, output_root=directory / "outputs"))
                self.assertEqual(report.experiments, 3)
                self.assertEqual(report.llm_calls, [])
                starts = [load(p, Start) for p in sorted(run_dir.glob("nodes/*/start.json"))]
                finishes = [load(p, Finish) for p in sorted(run_dir.glob("nodes/*/finish.json"))]
                self.assertEqual(len({s.validation.split_sha256 for s in starts}), 1)
                self.assertTrue(all(f.status == "completed" for f in finishes))
                baseline = finishes[0].value
                if classification:
                    self.assertGreaterEqual(report.best.value, baseline)
                else:
                    self.assertLessEqual(report.best.value, baseline)
                self.assertEqual((directory / "submission.csv").read_bytes(), (run_dir / report.best.submission_file).read_bytes())
                task = local_task(task_path)
                train, test, sample = read_tables(directory, task)
                frame = pd.read_csv(directory / "submission.csv", dtype={"row_id": str})
                validate_submission(frame, test, sample, train, task)

    async def test_timed_out_candidate_preserves_baseline_and_worker_has_no_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            task_path = write_fixture(directory)
            original_spawn = asyncio.create_subprocess_exec
            observed_environments = []
            async def spawn(*args, **kwargs):
                observed_environments.append(kwargs["env"])
                return await original_spawn(*args, **kwargs)
            async def forced_timeout(start, task_dir, run_dir, deadline, timeout, threads):
                return await execute(start, task_dir, run_dir, deadline,
                                     30 if start.node_id == "node_000" else .001, threads)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-secret"}), \
                    patch("alfa_agent.runner.execute", side_effect=forced_timeout), \
                    patch("alfa_agent.runner.asyncio.create_subprocess_exec", side_effect=spawn), \
                    redirect_stdout(io.StringIO()):
                run_dir, report = await run(task_path, Options(offline=True, max_experiments=2,
                    time_budget=60, output_root=directory / "outputs"))
            self.assertEqual(report.best.node_id, "node_000")
            self.assertEqual(load(run_dir / "nodes/node_001/finish.json", Finish).status, "timeout")
            self.assertEqual((directory / "submission.csv").read_bytes(), (run_dir / "nodes/node_000/submission.csv").read_bytes())
            self.assertTrue(observed_environments)
            self.assertTrue(all("OPENAI_API_KEY" not in env for env in observed_environments))


if __name__ == "__main__":
    unittest.main()
