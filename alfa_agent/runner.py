"""Bounded event-driven experiment loop, with a usable prediction file from node zero."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from .agents import JSONAgents
from .config import Settings
from .contracts import (Best, Dummy, ExtraTrees, Finish, HistBoosting, Hypothesis,
                        Pipeline, RunReport, Start)
from .data import confined_file, local_task, profile_data, read_tables, validate_submission
from .qwen import QwenError
from .storage import atomic_bytes, fingerprint, load, save
from .summarizer import summarize


@dataclass(frozen=True)
class Options:
    max_experiments: int = 4  # Includes the baseline.
    workers: int = 2
    time_budget: float = 300
    experiment_timeout: float = 90
    offline: bool = False
    output_root: Path = Path("outputs")

    def validate(self):
        if not 1 <= self.max_experiments <= 16 or not 1 <= self.workers <= 2:
            raise ValueError("Use 1–16 experiments (including baseline) and 1–2 workers")
        if not np.isfinite(self.time_budget) or not 5 <= self.time_budget <= 10800:
            raise ValueError("Time budget must be 5–10800 seconds")
        if not np.isfinite(self.experiment_timeout) or self.experiment_timeout <= 0:
            raise ValueError("Experiment timeout must be finite and positive")


def offline_hypothesis(index: int, classification: bool) -> Hypothesis:
    """An explicit offline policy exercises the exact same executor and contracts."""
    if index % 2:
        model = ExtraTrees(n_estimators=64, max_depth=min(24, 12 + index * 2),
                           min_samples_leaf=1 if index < 3 else 2,
                           class_weight="balanced" if classification else "none")
    else:
        model = HistBoosting(max_iter=min(200, 60 + index * 10), max_leaf_nodes=15,
                             learning_rate=0.1, l2_regularization=1.0)
    return Hypothesis(rationale="Deterministic offline candidate; not an LLM hypothesis.",
                      expected_effect="Test a nonlinear tree model against the incumbent.",
                      pipeline=Pipeline(model=model))


async def execute(start: Start, task_directory: Path, run_directory: Path,
                  deadline: float, timeout: float, threads: int) -> Finish:
    node_directory = run_directory / "nodes" / start.node_id
    save(node_directory / "start.json", start)
    started = time.monotonic()
    common = dict(node_id=start.node_id, metric=start.task.metric,
                  split_sha256=start.validation.split_sha256,
                  pipeline_sha256=fingerprint(start.hypothesis.pipeline))
    remaining = min(timeout, deadline - time.monotonic())
    if remaining <= 0:
        result = Finish(status="timeout", duration_seconds=0.0, error="Run deadline reached", **common)
    else:
        # No inherited environment; worker code never loads API credentials.
        env = {"PATH": os.defpath, "PYTHONIOENCODING": "utf-8",
               "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
               "MKL_NUM_THREADS": str(threads), "VECLIB_MAXIMUM_THREADS": str(threads)}
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "alfa_agent.worker", str(node_directory / "start.json"),
            str(task_directory), str(run_directory), str(threads),
            cwd=Path(__file__).resolve().parents[1], env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=remaining)
        except (TimeoutError, asyncio.CancelledError) as error:
            if process.returncode is None:
                process.kill()
            await process.wait()
            if isinstance(error, asyncio.CancelledError):
                raise
            result = Finish(status="timeout", duration_seconds=time.monotonic() - started,
                            error="Experiment exceeded its wall-clock budget", **common)
        else:
            try:
                result = load(node_directory / "finish.json", Finish)
                if any(getattr(result, key) != value for key, value in common.items()):
                    raise ValueError("Worker result does not match its start contract")
                if result.status == "completed" and process.returncode != 0:
                    raise ValueError("Worker failed after producing a result")
            except (ValueError, OSError):
                result = Finish(status="failed", duration_seconds=time.monotonic() - started,
                                error="Worker exited without a valid matching finish contract", **common)
    save(node_directory / "finish.json", result)
    return result


async def run(task_path: Path, options: Options, settings: Settings | None = None):
    options.validate()
    started = time.monotonic()
    deadline = started + options.time_budget
    task_path = task_path.resolve()
    task = local_task(task_path)
    profile, validation, fit, valid = profile_data(task_path.parent, task)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    run_directory = (options.output_root / task_path.parent.name / stamp).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    save(run_directory / "task.json", task)
    save(run_directory / "profile.json", profile)
    save(run_directory / "validation.json", validation)
    buffer = io.BytesIO()
    np.savez(buffer, fit=fit, valid=valid)
    atomic_bytes(run_directory / "split.npz", buffer.getvalue())
    starts, finishes = {}, {}
    best = None
    delta = None
    warnings = []
    agents = None
    threads = max(1, min(os.cpu_count() or 1, 8) // options.workers)
    train, test, sample = read_tables(task_path.parent, task)

    def make_start(hypothesis, source, parent):
        node_id = f"node_{len(starts):03d}"
        start = Start(node_id=node_id, parent_id=parent, source=source, task=task,
                      dataset_sha256=profile.dataset_sha256, validation=validation, hypothesis=hypothesis)
        starts[node_id] = start
        return start

    def accept(result):
        nonlocal best, delta
        previous_best = best.node_id if best else None
        if result.status == "completed":
            expected = f"nodes/{result.node_id}/submission.csv"
            try:
                if result.submission_file != expected:
                    raise ValueError("Unexpected submission artifact path")
                path = confined_file(run_directory, expected)
                frame = pd.read_csv(path, dtype={task.data.id_column: str,
                    **({task.problem.target_column: str} if task.problem.task_type == "classification" else {})})
                validate_submission(frame, test, sample, train, task)
            except (ValueError, OSError):
                result = result.model_copy(update={"status": "failed", "value": None,
                    "submission_file": None, "error": "Parent rejected the prediction artifact"})
                save(run_directory / "nodes" / result.node_id / "finish.json", result)
        finishes[result.node_id] = result
        delta = summarize(starts, finishes, result.node_id, previous_best)
        save(run_directory / "nodes" / result.node_id / "delta.json", delta)
        save(run_directory / "delta.json", delta)
        if delta.best_node_id and delta.best_node_id != previous_best:
            winner = finishes[delta.best_node_id]
            predictions = (run_directory / winner.submission_file).read_bytes()
            # Publish predictions before the manifest; a failed candidate never replaces the best.
            atomic_bytes(run_directory / "submission.csv", predictions)
            atomic_bytes(task_path.parent / "submission.csv", predictions)
            best = Best(node_id=winner.node_id, metric=winner.metric, value=winner.value,
                        submission_file=winner.submission_file, split_sha256=winner.split_sha256)
            save(run_directory / "best.json", best)
        score = f" {task.metric.name}={result.value:.6f}" if result.value is not None else ""
        print(f"{result.node_id}: {result.status}{score}", flush=True)

    baseline = make_start(Hypothesis(
        rationale="Establish a runnable floor and save valid predictions before any LLM work.",
        expected_effect="Mean target for regression; most frequent class for classification.",
        pipeline=Pipeline(model=Dummy())), "baseline", None)
    accept(await execute(baseline, task_path.parent, run_directory, deadline,
                         options.experiment_timeout, threads))
    if best is None:
        raise ValueError(f"Baseline failed; inspect {run_directory / 'nodes/node_000/finish.json'}")

    stop_reason = "max_experiments"
    if not options.offline and options.max_experiments > 1:
        try:
            if settings is None:
                raise QwenError("Нет настроек Qwen; baseline сохранён. Используйте --offline для проверки без API.")
            agents = JSONAgents(settings, deadline, max_calls=6 + 2 * (options.max_experiments - 1))
            extracted = await agents.inspect(task_path.read_text(encoding="utf-8"), task)
            for name, result in zip(("problem", "data", "metric"), extracted):
                save(run_directory / "extractors" / f"{name}.json", result)
        except (QwenError, ValueError) as error:
            warnings.append(str(error))
            stop_reason = "llm_unavailable_or_contract_conflict"

    workers = {}
    bakers = {}
    attempts = 0
    duplicate_streak = 0
    search_allowed = stop_reason == "max_experiments"

    async def propose(slot, parent_id):
        if options.offline:
            return offline_hypothesis(slot + 1, task.problem.task_type == "classification")
        context = {
            "task": task.model_dump(mode="json"), "profile": profile.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"),
            "leaderboard": [row.model_dump() for row in delta.leaderboard[:3]],
            "latest_comparisons": [row.model_dump() for row in delta.comparisons],
            "incumbent_pipeline": starts[parent_id].hypothesis.pipeline.model_dump(mode="json"),
            "tried_pipelines": [s.hypothesis.pipeline.model_dump(mode="json") for s in starts.values()],
            "failures": [{"node_id": f.node_id, "status": f.status, "error": f.error}
                         for f in list(finishes.values())[-5:] if f.status != "completed"],
        }
        return await agents.bake(context, slot)

    try:
        while True:
            if time.monotonic() >= deadline:
                search_allowed = False
                stop_reason = "time_budget"
            # Reserve one worker slot per in-flight hypothesis. Slow LLM calls never
            # block accepting measurements from an already completed worker.
            capacity = min(options.workers - len(workers) - len(bakers),
                           options.max_experiments - len(starts) - len(bakers))
            for _ in range(capacity if search_allowed else 0):
                if attempts >= 2 * options.max_experiments:
                    search_allowed = False
                    stop_reason = "proposal_limit"
                    break
                parent_id = best.node_id
                future = asyncio.create_task(propose(attempts, parent_id))
                bakers[future] = (attempts, parent_id)
                attempts += 1
            pending = set(workers) | set(bakers)
            if not pending:
                break
            completed, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for future in sorted(completed & workers.keys(), key=lambda f: workers[f]):
                workers.pop(future)
                accept(future.result())
            for future in sorted(completed & bakers.keys(), key=lambda f: bakers[f][0]):
                _, parent_id = bakers.pop(future)
                try:
                    hypothesis = future.result()
                except (QwenError, ValueError) as error:
                    warnings.append(str(error))
                    search_allowed = False
                    stop_reason = "llm_unavailable_or_invalid_hypothesis"
                    continue
                pipeline = hypothesis.pipeline
                if pipeline.model.family == "dummy" or any(
                    fingerprint(pipeline) == fingerprint(s.hypothesis.pipeline) for s in starts.values()
                ):
                    duplicate_streak += 1
                    warnings.append("Baker returned a dummy or duplicate pipeline; proposal rejected.")
                elif task.problem.task_type == "regression" and pipeline.model.family == "extra_trees" \
                        and pipeline.model.class_weight != "none":
                    warnings.append("Baker proposed classification class_weight for regression; rejected.")
                    duplicate_streak += 1
                else:
                    duplicate_streak = 0
                    start = make_start(hypothesis, "offline" if options.offline else "qwen", parent_id)
                    future = asyncio.create_task(execute(start, task_path.parent, run_directory, deadline,
                                                        options.experiment_timeout, threads))
                    workers[future] = start.node_id
                if duplicate_streak >= 2:
                    search_allowed = False
                    stop_reason = "proposal_limit"
    finally:
        pending = set(workers) | set(bakers)
        for future in pending:
            future.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # Model/provider errors can never be persisted with the organizer credential.
    if settings and settings.api_key:
        warnings = [message.replace(settings.api_key, "[REDACTED]") for message in warnings]
    report = RunReport(mode="offline" if options.offline else "qwen", stop_reason=stop_reason,
                       best=best, experiments=len(finishes), elapsed_seconds=time.monotonic() - started,
                       llm_calls=agents.calls if agents else [], warnings=warnings)
    save(run_directory / "report.json", report)
    return run_directory, report
