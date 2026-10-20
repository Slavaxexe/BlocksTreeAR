"""Bounded experiment tree: extraction → preparation → start / finish / delta → next plan."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from .agents import JSONAgents, extract_task
from .config import Settings
from .contracts import Change, Comparison, Delta, Extraction, Finish, NodeDelta, Preparation, Start
from .qwen import QwenError
from .data import check_code, file_hash, inventory, within
from .storage import atomic_bytes, load, save

ENTRY = "\nif __name__ == '__main__':\n    from alfa_agent.worker import main\n    main(fit_predict)\n"


@dataclass(frozen=True)
class Options:
    time_budget: float = 600
    experiment_timeout: float = 120
    output_dir: Path | None = None
    max_nodes: int = 4


async def process(arguments: list[str], cwd: Path, log: Path, deadline: float, timeout: float):
    remaining = min(timeout, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError('Run time budget exhausted')
    # No inherited credentials. This is a bounded process, not an OS sandbox.
    env = {'PATH': os.defpath, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
           'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    with log.open('wb') as stream:
        child = await asyncio.create_subprocess_exec(sys.executable, *arguments, cwd=cwd, env=env,
                                                     stdout=stream, stderr=asyncio.subprocess.STDOUT)
        try:
            code = await asyncio.wait_for(child.wait(), remaining)
        except (TimeoutError, asyncio.CancelledError):
            if child.returncode is None:
                child.kill()
            await child.wait()
            raise
    if code:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 8000))
            tail = stream.read().decode('utf-8', errors='replace')[-5000:]
        raise ValueError(tail or f'Child exited with code {code}')


def pipeline_changes(before, after, path='pipeline') -> list[Change]:
    if isinstance(before, dict) and isinstance(after, dict):
        return [change for key in sorted(before.keys() | after.keys())
                for change in pipeline_changes(before.get(key), after.get(key), path + '.' + key)]
    return [] if before == after else [Change(path=path, before=before, after=after)]


def summarize(delta: Delta, start: Start, finish: Finish) -> tuple[NodeDelta, Delta]:
    comparisons = []
    for previous in delta.nodes:
        reference = previous.result
        reason = None
        if finish.preparation_sha256 != reference.preparation_sha256 or finish.metric != reference.metric:
            reason = 'Different preparation or metric'
        elif finish.value is None or reference.value is None:
            reason = 'Score unavailable'
        elif finish.metric.direction is None:
            reason = 'Optimization direction unknown'
        difference = finish.value - reference.value if reason is None else None
        if difference is not None and not math.isfinite(difference):
            reason, difference = 'Metric difference is not finite', None
        comparisons.append(Comparison(reference_id=previous.node_id, comparable=reason is None, reason=reason,
            value_delta=difference, improvement=(-difference if finish.metric.direction == 'minimize' else difference) if difference is not None else None,
            changes=pipeline_changes(previous.hypothesis.pipeline.model_dump(), start.hypothesis.pipeline.model_dump())))
    row = NodeDelta(node_id=start.node_id, parent_id=start.parent_id, hypothesis=start.hypothesis,
                    result=finish, comparisons=comparisons)
    nodes = delta.nodes + [row]
    eligible = [n for n in nodes if n.result.value is not None and n.result.status == 'completed'
                and n.result.preparation_sha256 == delta.preparation_sha256 and n.result.metric == delta.metric]
    best = None
    if eligible and delta.metric.direction is not None:
        choose = min if delta.metric.direction == 'minimize' else max
        best = choose(eligible, key=lambda n: n.result.value).node_id
    row = row.model_copy(update={'best_node_id': best})
    return row, delta.model_copy(update={'nodes': delta.nodes + [row], 'best_node_id': best})


async def experiment(node: Path, start: Start, shared: dict, agents: JSONAgents, deadline: float, checkpoint) -> Finish:
    node.mkdir(parents=True)
    save(node / 'start.json', start)
    start_hash, started = file_hash(node / 'start.json'), time.monotonic()
    checkpoint()
    code_hash, notes, stage = None, [], 'implementation'
    try:
        preparation_path = node.parent.parent / 'preparation.json'
        context = {**shared, 'start': load(node / 'start.json', Start).model_dump(mode='json'),
                   'preparation_code': (node.parent.parent / 'prepare.py').read_text()}
        for attempt in range(2):
            stage = 'implementation'
            program = await agents.implement(context)
            notes = list(dict.fromkeys(notes + program.notes))
            if program.blocked_reason:
                raise ValueError('Implementation blocked: ' + program.blocked_reason)
            stage = 'check'
            try:
                check_code(program.code, ('fit_predict',))
                atomic_bytes(node / start.implementation, (program.code + ENTRY).encode())
                code_hash = file_hash(node / start.implementation)
                await process(['-m', 'alfa_agent.worker', '--check', str(node), '--preparation', str(preparation_path)],
                              node, node / 'execution.log', deadline, min(45, start.timeout_seconds))
                break
            except (ValueError, OSError, TimeoutError) as error:
                if attempt:
                    raise
                context.update(previous_code=program.code, execution_error=str(error)[-5000:])
                checkpoint(f'{start.node_id}: one code repair; saved start is unchanged')
        if file_hash(node / 'start.json') != start_hash or file_hash(preparation_path) != start.preparation_sha256:
            raise ValueError('Saved plan or preparation changed during implementation')
        stage = 'execution'
        await process([str(node / start.implementation), '--start', str(node / 'start.json'),
                       '--start-sha256', start_hash, '--code-sha256', code_hash], node,
                      node / 'execution.log', deadline, start.timeout_seconds)
        finish = load(node / 'finish.json', Finish)
        if (finish.node_id, finish.code_sha256, finish.preparation_sha256, finish.start_sha256) != (
                start.node_id, code_hash, start.preparation_sha256, start_hash):
            raise ValueError('Finish does not match its recorded node, plan, code and preparation')
        if finish.status != 'completed':
            raise ValueError(finish.error or 'Experiment failed')
    except Exception as error:
        message = str(error).replace(agents.settings.api_key, '[REDACTED]')[-5000:]
        finish = Finish(node_id=start.node_id, status='timeout' if isinstance(error, TimeoutError) else 'failed',
            stage=stage, metric=shared['preparation']['metric'], duration_seconds=0.,
            preparation_sha256=start.preparation_sha256, start_sha256=start_hash, code_sha256=code_hash, error=message)
    finish = finish.model_copy(update={'notes': notes, 'duration_seconds': time.monotonic() - started})
    save(node / 'finish.json', finish)
    checkpoint()
    return finish


async def run(task_path: Path, options: Options, settings: Settings):
    if not 0 < options.time_budget <= 10800 or not 0 < options.experiment_timeout <= 10800:
        raise ValueError('Time budgets must be finite, positive and at most three hours')
    if type(options.max_nodes) is not int or not 1 <= options.max_nodes <= 1000:
        raise ValueError('max_nodes must be an integer between 1 and 1000')
    task_path = task_path.resolve()
    deadline = time.monotonic() + options.time_budget
    agents = JSONAgents(settings, deadline, max_calls=12 + 6 * options.max_nodes)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = (options.output_dir or Path('artifacts') / f'{task_path.parent.name}-{stamp}-{uuid4().hex[:6]}').resolve()
    report = await extract_task(task_path, directory, settings, agents=agents)
    if report.status == 'failed':
        return directory, report

    def update(**changes):
        nonlocal report
        report = report.model_copy(update={**changes, 'llm_calls': list(agents.calls)})
        save(directory / 'report.json', report)

    delta, published, stop_reason = None, None, 'node_limit'
    try:
        extraction = load(directory / 'extraction.json', Extraction)
        update(status='running', stage='preparation')
        context = {'task_text': task_path.read_text(encoding='utf-8'),
                   'extraction': extraction.model_dump(mode='json'), 'inventory': inventory(task_path.parent)}
        for attempt in range(2):
            draft = await agents.prepare(context)
            try:
                check_code(draft.code, ('load_data', 'make_split', 'score', 'write_submission'))
                for relative in draft.data.files.values():
                    within(task_path.parent, relative)
                within(directory / 'nodes' / 'node_000', draft.output.filename)
                atomic_bytes(directory / 'prepare.py', draft.code.encode())
                preparation = Preparation(task_directory=str(task_path.parent), rationale=draft.rationale,
                    data=draft.data, validation=draft.validation, metric=draft.metric, output=draft.output,
                    code_sha256=file_hash(directory / 'prepare.py'))
                save(directory / 'preparation.json', preparation)
                await process(['-m', 'alfa_agent.worker', '--prepare', str(directory)], directory,
                              directory / 'preparation.log', deadline, options.experiment_timeout)
                preparation = load(directory / 'preparation.json', Preparation)
                break
            except (ValueError, OSError, TimeoutError) as error:
                if (directory / 'preparation.json').exists():
                    failed = load(directory / 'preparation.json', Preparation).model_copy(
                        update={'status': 'blocked', 'issues': [str(error)[-5000:]]})
                    save(directory / 'preparation.json', failed)
                if attempt:
                    raise
                context.update(previous_code=draft.code, execution_error=str(error)[-5000:])
                update(warnings=report.warnings + ['Preparation required one code repair'])
        update(stage='baseline', checks=report.checks + ['executed_preparation', 'measured_profile', 'fixed_split'])
        shared = {'task_text': task_path.read_text(encoding='utf-8'),
                  'extraction': extraction.model_dump(mode='json'), 'preparation': preparation.model_dump(mode='json'),
                  'installed_libraries': {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')}}
        delta = Delta(preparation_sha256=file_hash(directory / 'preparation.json'), metric=preparation.metric)
        save(directory / 'delta.json', delta)

        def checkpoint(warning=None):
            update(warnings=report.warnings + ([warning] if warning else []))

        for index in range(options.max_nodes):
            if time.monotonic() >= deadline:
                stop_reason = 'time_budget'
                break
            update(stage='baseline' if index == 0 else 'search')
            parent, context = None, dict(shared)
            if delta.nodes:
                parent = next((n for n in delta.nodes if n.node_id == delta.best_node_id), None)
                parent = parent or next((n for n in reversed(delta.nodes) if n.result.status == 'completed'), delta.nodes[-1])
                parent_path = directory / 'nodes' / parent.node_id
                context.update(delta=delta.model_dump(mode='json'), remaining_nodes=options.max_nodes - index,
                    remaining_seconds=max(0., deadline - time.monotonic()),
                    parent={'start': load(parent_path / 'start.json', Start).model_dump(mode='json'),
                            'finish': parent.result.model_dump(mode='json'),
                            'code': (parent_path / 'experiment.py').read_text() if (parent_path / 'experiment.py').exists() else None})
            hypothesis = await (agents.plan_baseline(context) if parent is None else agents.plan_next(context))
            node_id = f'node_{index:03d}'
            start = Start(node_id=node_id, parent_id=parent.node_id if parent else None,
                hypothesis=hypothesis, preparation_sha256=delta.preparation_sha256,
                context={'task': str(task_path), 'task_sha256': report.task_sha256,
                         'extraction': '../../extraction.json', 'extraction_sha256': file_hash(directory / 'extraction.json'),
                         'preparation_code': '../../prepare.py', 'installed_libraries': shared['installed_libraries']},
                outputs={'validation': 'validation_predictions.csv', 'submission': preparation.output.filename},
                timeout_seconds=float(options.experiment_timeout))
            finish = await experiment(directory / 'nodes' / node_id, start, context, agents, deadline, checkpoint)
            row, delta = summarize(delta, start, finish)
            save(directory / 'nodes' / node_id / 'delta.json', row)
            save(directory / 'delta.json', delta)
            selected = delta.best_node_id or next((n.node_id for n in delta.nodes if n.result.status == 'completed'), None)
            if selected is not None and selected != published:
                result = next(n.result for n in delta.nodes if n.node_id == selected)
                output_file = within(directory / 'nodes' / selected, result.submission_file)
                atomic_bytes(within(task_path.parent, preparation.output.filename), output_file.read_bytes())
                if delta.best_node_id:
                    atomic_bytes(directory / 'best.json', json.dumps({'node_id': selected, 'value': result.value,
                                 'submission_file': output_file.relative_to(directory).as_posix()}).encode())
                published = selected
            completed = sum(n.result.status == 'completed' for n in delta.nodes)
            update(completed_nodes=completed, failed_nodes=len(delta.nodes) - completed,
                   warnings=report.warnings + ([f'{node_id}: {finish.error}'] if finish.error else []))
        update(status='ready' if delta.best_node_id and stop_reason == 'node_limit' else 'partial' if published else 'failed',
               stage='complete', stop_reason=stop_reason,
               checks=report.checks + ['saved_starts', 'recorded_finishes', 'computed_deltas'] + (['submission_written'] if published else []))
    except Exception as error:
        message = str(error).replace(settings.api_key, '[REDACTED]')[-5000:]
        reason = ('preparation_failed' if delta is None else 'time_budget' if time.monotonic() >= deadline
                  else 'llm_error' if isinstance(error, QwenError) else 'error')
        update(status='partial' if published else 'failed', stop_reason=reason, errors=report.errors + [message])
    return directory, report
