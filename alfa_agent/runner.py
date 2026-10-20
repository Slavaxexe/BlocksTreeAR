"""One complete baseline: extraction → generated preparation → generated node_000."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from .agents import JSONAgents, extract_task
from .config import Settings
from .contracts import Extraction, Finish, Implementation, Preparation, Start
from .data import check_code, file_hash, inventory, within
from .storage import atomic_bytes, load, save

ENTRY = "\nif __name__ == '__main__':\n    from alfa_agent.worker import main\n    main(fit_predict)\n"


@dataclass(frozen=True)
class Options:
    time_budget: float = 600
    experiment_timeout: float = 120
    output_dir: Path | None = None


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


async def run(task_path: Path, options: Options, settings: Settings):
    if not 0 < options.time_budget <= 10800 or not 0 < options.experiment_timeout <= 10800:
        raise ValueError('Time budgets must be finite, positive and at most three hours')
    task_path = task_path.resolve()
    deadline = time.monotonic() + options.time_budget
    agents = JSONAgents(settings, deadline)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = (options.output_dir or Path('artifacts') / f'{task_path.parent.name}-{stamp}-{uuid4().hex[:6]}').resolve()
    report = await extract_task(task_path, directory, settings, agents=agents)
    if report.status == 'failed':
        return directory, report

    def update(**changes):
        nonlocal report
        report = report.model_copy(update={**changes, 'llm_calls': list(agents.calls)})
        save(directory / 'report.json', report)

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
        shared = {'extraction': extraction.model_dump(mode='json'), 'preparation': preparation.model_dump(mode='json'),
                  'installed_libraries': {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')}}
        hypothesis = await agents.plan_baseline(shared)
        node = directory / 'nodes' / 'node_000'
        node.mkdir(parents=True)
        context = {**shared, 'hypothesis': hypothesis.model_dump(mode='json'),
                   'preparation_code': (directory / 'prepare.py').read_text()}
        for attempt in range(2):
            program = await agents.implement(context)
            try:
                check_code(program.code, ('fit_predict',))
                atomic_bytes(node / 'experiment.py', (program.code + ENTRY).encode())
                await process(['-m', 'alfa_agent.worker', '--check', str(node),
                               '--preparation', str(directory / 'preparation.json')], node,
                              node / 'execution.log', deadline, min(45, options.experiment_timeout))
                break
            except (ValueError, OSError, TimeoutError) as error:
                if attempt:
                    raise
                context.update(previous_code=program.code, execution_error=str(error)[-5000:])
                update(warnings=report.warnings + ['Baseline required one code repair before start'])
        start = Start(hypothesis=program.hypothesis, preparation_sha256=file_hash(directory / 'preparation.json'),
            implementation=Implementation(sha256=file_hash(node / 'experiment.py')),
            outputs={'validation': 'validation_predictions.csv', 'submission': preparation.output.filename},
            timeout_seconds=float(options.experiment_timeout))
        save(node / 'start.json', start)
        started = time.monotonic()
        try:
            await process([str(node / 'experiment.py'), '--start', str(node / 'start.json')], node,
                          node / 'execution.log', deadline, options.experiment_timeout)
            finish = load(node / 'finish.json', Finish)
            if finish.code_sha256 != start.implementation.sha256 or finish.preparation_sha256 != start.preparation_sha256:
                raise ValueError('Finish does not match its recorded code and preparation')
            if finish.status != 'completed':
                raise ValueError(finish.error or 'Experiment failed')
        except (ValueError, OSError, TimeoutError) as error:
            finish = Finish(status='timeout' if isinstance(error, TimeoutError) else 'failed',
                metric=preparation.metric, duration_seconds=time.monotonic() - started,
                preparation_sha256=start.preparation_sha256, code_sha256=start.implementation.sha256,
                error=str(error)[-5000:])
            save(node / 'finish.json', finish)
            raise
        output_file = within(node, finish.submission_file)
        atomic_bytes(within(task_path.parent, preparation.output.filename), output_file.read_bytes())
        if finish.value is not None and preparation.metric.direction is not None:
            best = {'node_id': 'node_000', 'value': finish.value,
                    'submission_file': output_file.relative_to(directory).as_posix()}
            atomic_bytes(directory / 'best.json', json.dumps(best).encode())
        if finish.error:
            update(warnings=report.warnings + [finish.error])
        update(status='ready' if finish.evaluation_status == 'completed' else 'partial', stage='complete',
               checks=report.checks + ['code_smoke_test', 'executed_baseline', 'submission_written'])
    except Exception as error:
        update(status='failed', errors=report.errors + [str(error).replace(settings.api_key, '[REDACTED]')[-5000:]])
    return directory, report
