"""Module experiments: saved plans, bounded repairs, analysis and a rolling search queue."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import signal
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

from ml_reference_tools import MAX_REFERENCE_ROUNDS

from .agents import JSONAgents, extract_task
from .assembly import BlockError, assemble, block_catalog, initialize, sources
from .config import Settings
from .context import (execution_contract, history_view, model_feedback, preparation_view,
                      requirements, research_profile_view)
from .contracts import Assembly, Block, Delta, Extraction, Finish, Hypothesis, NodeDelta, Pipeline, Preparation, Proposal, Start
from .data import check_code, file_hash, installed_libraries, inventory, within
from .qwen import QwenError
from .search import Search
from .research import Research
from .storage import atomic_bytes, load, save


@dataclass(frozen=True)
class Options:
    time_budget: float = 600
    experiment_timeout: float = 240
    output_dir: Path | None = None
    max_experiments: int | None = 1
    parallel: int = 6
    parallel_agents: int = 8
    max_llm_calls: int | None = 80
    max_code_attempts: int = 5
    automl: bool = True
    research_parallel: int = 2
    max_research_nodes: int | None = 12
    research_timeout: float = 60
    final_refit_seconds: float = 300
    eda_fraction: float = .15
    eda_seconds: float | None = None
    research_call_fraction: float = .3
    research_unlimited: bool = False

    @property
    def experiment_limit(self):
        return math.inf if self.max_experiments is None else self.max_experiments

    @property
    def research_limit(self):
        return math.inf if self.max_research_nodes is None else self.max_research_nodes


async def process(arguments: list[str], cwd: Path, log: Path, deadline: float, timeout: float):
    remaining = min(timeout, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError('Run time budget exhausted')
    env = {'PATH': os.defpath, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
           'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    # Some ML libraries need the user/temp directories even without API credentials.
    for name in ('HOME', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'TEMP', 'TMP', 'APPDATA', 'LOCALAPPDATA'):
        if name in os.environ:
            env[name] = os.environ[name]
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    # Windows needs SystemRoot for system libraries used by asyncio and sklearn.
    if os.name == 'nt':
        env['SystemRoot'] = os.environ['SystemRoot']
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('wb') as stream:
        child = await asyncio.create_subprocess_exec(sys.executable, *arguments, cwd=cwd, env=env,
                                                     stdout=stream, stderr=asyncio.subprocess.STDOUT, start_new_session=os.name != 'nt')
        try:
            code = await asyncio.wait_for(child.wait(), remaining)
        except (TimeoutError, asyncio.CancelledError) as error:
            if child.returncode is None:
                # AutoML can start its own workers; cancel the whole experiment tree.
                if os.name == 'nt':
                    import psutil
                    try:
                        descendants = psutil.Process(child.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        descendants = []
                    for descendant in reversed(descendants):
                        try:
                            descendant.kill()
                        except psutil.NoSuchProcess:
                            pass
                    try:
                        child.kill()
                    except ProcessLookupError:
                        pass
                else:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            await child.wait()
            if isinstance(error, TimeoutError):
                raise TimeoutError(f'Process exceeded its {remaining:.1f}s time limit') from None
            raise
    if code:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 8000))
            tail = stream.read().decode('utf-8', errors='replace')[-5000:]
        raise ValueError(tail or f'Child exited with code {code}')


def changes(before, after, path='pipeline') -> list[dict]:
    if isinstance(before, dict) and isinstance(after, dict):
        return [change for key in sorted(before.keys() | after.keys())
                for change in changes(before.get(key), after.get(key), path + '.' + key)]
    return [] if before == after else [{'path': path, 'before': before, 'after': after}]


def summarize(delta: Delta, start: Start, finish: Finish) -> Delta:
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
            reason, difference = 'Non-finite metric difference', None
        comparisons.append({'reference_id': previous.node_id, 'comparable': reason is None, 'reason': reason,
            'improvement': (-difference if finish.metric.direction == 'minimize' else difference) if difference is not None else None,
            'changes': changes(previous.hypothesis.pipeline.model_dump(), start.hypothesis.pipeline.model_dump())
                       + changes(previous.assembly.model_dump(), start.assembly.model_dump(), 'assembly')})
    row = NodeDelta(node_id=start.node_id, parent_id=start.parent_id, hypothesis=start.hypothesis,
                    assembly=start.assembly, result=finish, comparisons=comparisons)
    nodes = delta.nodes + [row]
    eligible = [n for n in nodes if n.result.status == 'completed' and n.result.value is not None
                and n.result.preparation_sha256 == delta.preparation_sha256 and n.result.metric == delta.metric]
    best = None
    if eligible and delta.metric.direction:
        choose = min if delta.metric.direction == 'minimize' else max
        best = choose(eligible, key=lambda n: n.result.value).node_id
    return delta.model_copy(update={'nodes': nodes, 'best_node_id': best})


async def experiment(node: Path, start: Start, shared: dict, agents: JSONAgents,
                     deadline: float, checkpoint) -> Finish:
    started, start_hash = time.monotonic(), file_hash(node / 'start.json')
    target, stage, code_hash, notes = None, 'assembly', None, []
    attempts, clients, writers = {}, {}, []
    execution_state = None
    max_code_attempts = start.context.get('max_code_attempts', 5)
    block_states = {b.name: {'stage': 'queued' if b.name in start.assembly.write_blocks else 'reused',
                             'attempt': 0} for b in start.assembly.blocks}
    context = {**shared, 'node_id': start.node_id, 'hypothesis': start.hypothesis.model_dump(),
               'assembly': start.assembly.model_dump(), 'purpose': start.purpose,
               'experiment_timeout_seconds': start.timeout_seconds}

    def block_update(name, **values):
        block_states[name].update(values)
        checkpoint(blocks={key: dict(value) for key, value in block_states.items()})

    async def write_block_source(name, error=None, failed_stage=None):
        from .operations import CATALOG
        from .plans import ToolCall
        if error is not None:
            agents.observe_error('configuration_repair', error)

        spec = next(block for block in start.assembly.blocks if block.name == name)
        operation = spec.operation
        path = node / 'modules' / f'{name}.py'
        previous = node / 'modules' / f'{name}.json'
        if previous.exists():
            operation = ToolCall.model_validate(json.loads(previous.read_text(encoding='utf-8')))
        defaults = {key: shared['requirements'].get('problem' if key != 'prediction_type' else 'output', {}).get(key)
                    for key in ('task_type', 'class_labels', 'prediction_type')}
        while attempts.get(name, 0) < max_code_attempts:
            attempt = attempts[name] = attempts.get(name, 0) + 1
            attempt_dir = node / 'attempts' / name / f'{attempt:02d}'
            if error is not None:
                if name not in clients:
                    clients[name] = JSONAgents(agents.settings, deadline, max_code_attempts * (3 + MAX_REFERENCE_ROUNDS),
                                              shared=agents, node_id=start.node_id, block=name)
                operation = await clients[name].repair_operation({**context, 'target_block': spec.model_dump(),
                    'previous_operation': operation.model_dump() if operation else None, 'execution_error': error,
                    'failed_stage': failed_stage, 'operations': CATALOG})
            if operation is None:
                raise BlockError(name, 'New blocks must select an operation; Python generation is disabled')
            block_update(name, stage='implementation', attempt=attempt)
            save(attempt_dir / 'configuration.json', operation.model_dump())
            save(previous, operation.model_dump())
            code = ('from agent.operations import run_operation\n\n\n'
                    'def run(state):\n'
                    f'    return run_operation(state, {operation.model_dump()!r}, {defaults!r})\n')
            atomic_bytes(path, code.encode('utf-8'))
            check_code(code, ('run',))
            block_update(name, stage='ready', attempt=attempt)
            agents.log('block_configuration_saved', node_id=start.node_id, block=name, attempt=attempt,
                       operation=operation.kind)
            return
        raise BlockError(name, f'Block exhausted its {max_code_attempts} configuration versions')


    async def write_block(name, error=None, failed_stage=None):
        try:
            return await write_block_source(name, error, failed_stage)
        except (Exception, asyncio.CancelledError) as problem:
            detail = {'node_id': start.node_id, 'block': name, 'stage': 'implementation',
                      'error_type': type(problem).__name__, 'error': str(problem),
                      'traceback': traceback.format_exc()}
            agents.log('block_failed', **detail)
            block_update(name, stage='failed')
            if isinstance(problem, (BlockError, asyncio.CancelledError)):
                raise
            agents.record(Path('nodes') / start.node_id / 'implementation_error.json', detail)
            raise BlockError(name, str(problem)) from problem

    def execution_update(status):
        execution_state[stage] = status
        if status == 'running':
            stage_started[stage] = time.monotonic()
        else:
            execution_state['duration_seconds'][stage] = time.monotonic() - stage_started[stage]
        save(log_dir / 'status.json', execution_state)
        agents.log('execution_stage', node_id=start.node_id, execution=execution, stage=stage, status=status)

    try:
        initialize(node, start)
        stage = 'implementation'
        checkpoint(stage=stage, blocks=block_states)
        writers = [asyncio.create_task(write_block(name)) for name in start.assembly.write_blocks]
        results = await asyncio.gather(*writers, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        execution = 0
        while True:
            execution += 1
            log_dir = node / 'executions' / f'{execution:02d}'
            stage_started = {}
            execution_state = {'number': execution, 'assembly': 'not_started',
                               'check': 'not_started', 'execution': 'not_started', 'duration_seconds': {}}
            try:
                stage = 'assembly'
                checkpoint(stage=stage, block=None)
                execution_update('running')
                code_hash = assemble(node, start)
                if (file_hash(node / 'start.json') != start_hash or
                        file_hash(node.parent.parent / 'preparation.json') != start.preparation_sha256):
                    raise ValueError('Saved plan or preparation changed')
                execution_update('completed')
                (node / 'block_error.json').unlink(missing_ok=True)
                (node / 'progress.json').unlink(missing_ok=True)
                stage = 'check'
                checkpoint(stage=stage)
                execution_update('running')
                await process(['-m', 'agent.worker', '--check', str(node),
                               '--preparation', str(node.parent.parent / 'preparation.json'),
                               '--time-budget', str(min(45, start.timeout_seconds, deadline - time.monotonic()))],
                              node, log_dir / 'smoke.log', deadline, min(45, start.timeout_seconds))
                execution_update('completed')
                stage = 'execution'
                checkpoint(stage=stage)
                execution_update('running')
                await process([str(node / 'experiment.py'), '--start', str(node / 'start.json'),
                               '--start-sha256', start_hash, '--code-sha256', code_hash,
                               '--time-budget', str(min(start.timeout_seconds, deadline - time.monotonic()))],
                              node, log_dir / 'execution.log', deadline, start.timeout_seconds)
                finish = load(node / 'finish.json', Finish)
                if (finish.node_id, finish.code_sha256, finish.preparation_sha256, finish.start_sha256) != (
                        start.node_id, code_hash, start.preparation_sha256, start_hash):
                    raise ValueError('Finish does not match the saved plan and assembled code')
                if finish.status != 'completed':
                    raise ValueError(finish.error or 'Experiment failed')
                execution_update('completed')
                break
            except (ValueError, OSError, TimeoutError) as error:
                agents.log('execution_failed', node_id=start.node_id, stage=stage,
                           error_type=type(error).__name__, traceback=traceback.format_exc())
                if isinstance(error, OSError) and not isinstance(error, TimeoutError):
                    raise
                execution_update('failed')
                failed_block = getattr(error, 'block', None)
                error_text = str(error)
                if (node / 'block_error.json').exists():
                    block_error = json.loads((node / 'block_error.json').read_text(encoding='utf-8'))
                    failed_block = block_error['block']
                    # Library warnings can bury the exception in the combined stdout/stderr log.
                    error_text = block_error['error']
                if isinstance(error, TimeoutError) and (node / 'progress.json').exists():
                    failed_block = json.loads((node / 'progress.json').read_text(encoding='utf-8'))['block']
                target = failed_block
                detail = {'stage': stage, 'block': target, 'error': error_text[-5000:],
                          'traceback': traceback.format_exc()}
                save(log_dir / 'error.json', detail)
                if target is None:
                    raise
                if attempts.get(target, 0):
                    save(node / 'attempts' / target / f'{attempts[target]:02d}' / 'error.json', detail)
                block_update(target, stage='failed')
                if attempts.get(target, 0) >= max_code_attempts or time.monotonic() >= deadline:
                    raise
                failed_stage, stage, code_hash = stage, 'implementation', None
                checkpoint(stage=stage, block=target)
                await write_block(target, error_text[-5000:], failed_stage)
    except (Exception, asyncio.CancelledError) as error:
        target = getattr(error, 'block', None) or target
        agents.log('experiment_failed', node_id=start.node_id, block=target, stage=stage,
                   error_type=type(error).__name__, traceback=traceback.format_exc())
        if execution_state is not None and execution_state.get(stage) == 'running':
            execution_update('interrupted')
        message = str(error) or ('Run time budget exhausted' if time.monotonic() >= deadline
                                 else 'Experiment cancelled')
        message = message.replace(agents.settings.api_key, '[REDACTED]')[-5000:]
        finish = Finish(node_id=start.node_id,
            status='timeout' if isinstance(error, TimeoutError) or time.monotonic() >= deadline else 'failed',
            stage=stage, failed_block=target, metric=shared['preparation']['metric'], duration_seconds=0.,
            preparation_sha256=start.preparation_sha256, start_sha256=start_hash, code_sha256=code_hash, error=message)
    finally:
        for writer in writers:
            if not writer.done():
                writer.cancel()
        if writers:
            await asyncio.gather(*writers, return_exceptions=True)
    finish = finish.model_copy(update={'notes': list(dict.fromkeys(notes)), 'duration_seconds': time.monotonic() - started})
    save(node / 'finish.json', finish)
    agents.log('experiment_finished', node_id=start.node_id, status=finish.status, stage=finish.stage,
               value=finish.value, evaluation_status=finish.evaluation_status)
    checkpoint(stage=finish.status, block=target)
    return finish


async def run(task_path: Path, options: Options, settings: Settings):
    if not 0 < options.time_budget <= 10800 or not 0 < options.experiment_timeout <= 10800:
        raise ValueError('Time budgets must be finite, positive and at most three hours')
    if not 0 < options.research_timeout <= 10800 or options.research_limit < 0 or options.research_parallel < 0:
        raise ValueError('Research needs a positive timeout and nonnegative node/parallel limits')
    if (options.experiment_limit < 1 or options.parallel < 1
            or options.parallel_agents < 1 or (options.max_llm_calls is not None and options.max_llm_calls < 1)
            or options.max_code_attempts < 1):
        raise ValueError('Use positive experiment/parallel/call/attempt limits, or None for unlimited nodes/calls')
    task_path = task_path.resolve()
    if not math.isfinite(options.final_refit_seconds) or options.final_refit_seconds < 0:
        raise ValueError('Final refit reserve must be finite and nonnegative')
    if not 0 <= options.eda_fraction < 1:
        raise ValueError('EDA fraction must be between 0 (disabled) and 1 (exclusive)')
    if options.eda_seconds is not None and (not math.isfinite(options.eda_seconds) or options.eda_seconds < 0):
        raise ValueError('EDA seconds must be finite and nonnegative')
    if not 0 <= options.research_call_fraction <= 1:
        raise ValueError('Research call fraction must be between 0 and 1')
    overall_deadline = time.monotonic() + options.time_budget
    final_reserve = min(options.final_refit_seconds, options.time_budget * .2) if options.time_budget >= 60 else 0.
    deadline = overall_deadline - final_reserve
    agents = JSONAgents(settings, deadline, max_calls=options.max_llm_calls, parallel_agents=options.parallel_agents)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = (options.output_dir or Path('artifacts') / f'{task_path.parent.name}-{stamp}-{uuid4().hex[:6]}').resolve()
    report = await extract_task(task_path, directory, settings, agents=agents)
    if report.status == 'failed':
        return directory, report
    active, analyses, starts = {}, {}, {}
    revision, published, search, research = 0, None, None, None
    shared = {}
    delta, stop_reason = None, 'max_experiments'

    def update(**values):
        nonlocal report
        report = report.model_copy(update={**values, 'llm_calls': list(agents.calls),
            'running_nodes': [s.node_id for s in active.values()], 'queued_nodes': [],
            'analyzing_nodes': list(analyses.values()), **(search.report() if search else {}),
            **(research.report() if research else {}),
            'max_parallel_nodes': options.parallel, 'max_parallel_agents': options.parallel_agents,
            'max_parallel_research': options.research_parallel,
            'active_agents': list(agents.active.values()), 'api_budget': agents.budget_view(),
            'experiments': len(starts), 'best_node': delta.best_node_id if delta else None})
        try:
            save(directory / 'report.json', report)
        except OSError as error:
            # Live status is secondary; losing one refresh must not abort model generation.
            agents.log('report_write_failed', error_type=type(error).__name__, traceback=traceback.format_exc())
            warning = 'report.json could not be refreshed; details are in events.jsonl'
            if warning not in report.warnings:
                report = report.model_copy(update={'warnings': report.warnings + [warning]})
                print(warning, file=sys.stderr)

    def notify(message, *, fatal=False):
        key = 'errors' if fatal else 'warnings'
        message = message.replace(settings.api_key, '[REDACTED]')[-5000:]
        update(**{key: getattr(report, key) + [message]})

    def checkpoint(node_id, **values):
        states = {**report.node_states, node_id: {**report.node_states.get(node_id, {}), **values}}
        update(node_states=states)

    def save_delta():
        save(directory / 'delta.json', delta)
        save(directory / 'journal.json', {'nodes': [n.model_dump(mode='json') for n in delta.nodes]})
        if research:
            feedback_ids = {delta.best_node_id, *(row.node_id for row in delta.nodes[-2:])}
            research.model_feedback = [model_feedback(row) for row in delta.nodes if row.node_id in feedback_ids]
        for row in delta.nodes:
            save(directory / 'nodes' / row.node_id / 'delta.json', row)

    def record(start, finish):
        nonlocal delta, published
        delta = summarize(delta, start, finish)
        save_delta()
        selected = delta.best_node_id or next((n.node_id for n in delta.nodes if n.result.status == 'completed'), None)
        if selected is not None and selected != published:
            result = next(n.result for n in delta.nodes if n.node_id == selected)
            source = within(directory / 'nodes' / selected, result.submission_file)
            atomic_bytes(within(task_path.parent, preparation.output.filename), source.read_bytes())
            if delta.best_node_id:
                save(directory / 'best.json', {'node_id': selected, 'value': result.value,
                     'submission_file': source.relative_to(directory).as_posix()})
            published = selected

    def result_or_failure(task, start):
        # A task cancelled before its coroutine starts has no Finish of its own.
        try:
            return task.result()
        except (Exception, asyncio.CancelledError) as error:
            node = directory / 'nodes' / start.node_id
            finish = Finish(node_id=start.node_id, status='timeout' if time.monotonic() >= deadline else 'failed',
                stage='implementation', metric=preparation.metric, duration_seconds=0.,
                preparation_sha256=start.preparation_sha256, start_sha256=file_hash(node / 'start.json'),
                error=(str(error) or 'Experiment cancelled').replace(settings.api_key, '[REDACTED]')[-5000:])
            save(node / 'finish.json', finish)
            agents.log('node_task_failed', node_id=start.node_id, error_type=type(error).__name__)
            return finish

    def launch(node_id, proposal):
        catalog = block_catalog(directory, delta.nodes)
        hashes = {block.name: next(item['sha256'] for item in catalog[block.source_node] if item['name'] == block.name)
                  for block in proposal.assembly.blocks if block.source_node is not None}
        start = Start(node_id=node_id, parent_id=proposal.parent_id, purpose=proposal.purpose, source='builtin',
            hypothesis=proposal.hypothesis, assembly=proposal.assembly, preparation_sha256=delta.preparation_sha256,
            context={'task': str(task_path), 'task_sha256': report.task_sha256, 'source_hashes': hashes,
                     'analysis_revision': revision, 'installed_libraries': shared['installed_libraries'],
                     'research_revision': research.revision if research else 0,
                     'max_code_attempts': options.max_code_attempts},
            outputs={'validation': 'validation_predictions.csv', 'submission': preparation.output.filename},
            timeout_seconds=float(options.experiment_timeout))
        node = directory / 'nodes' / node_id
        node.mkdir(parents=True)
        save(node / 'start.json', start)
        starts[node_id] = start
        task = asyncio.create_task(experiment(node, start, shared, agents, deadline,
            lambda **values: checkpoint(node_id, **values)))
        active[task] = start
        checkpoint(node_id, stage='queued', blocks={
            block.name: {'stage': 'queued' if block.name in start.assembly.write_blocks else 'reused', 'attempt': 0}
            for block in start.assembly.blocks})
        return task

    async def analyze(row):
        references = {'node_000', row.parent_id, delta.best_node_id}
        comparisons = [c.model_dump() for c in row.comparisons if c.reference_id in references]
        node = directory / 'nodes' / row.node_id
        modules = sources(node)
        context = {**shared, 'experiment': history_view(row), 'comparisons': comparisons, 'modules': modules}
        manifest = node / 'assembly.json'
        context['executed_assembly'] = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None
        context['automl_diagnostics'] = [json.loads(path.read_text(encoding='utf-8'))
                                        for path in sorted(node.glob('*_full_diagnostics.json'))]
        from .model_evidence import comparison_facts
        context['comparison_facts'] = comparison_facts(directory, row.parent_id)
        context['tool_configurations'] = {p.stem: json.loads(p.read_text(encoding='utf-8'))
                                          for p in sorted((node / 'modules').glob('*.json'))}
        context['repair_attempts'] = []
        for path in sorted((node / 'attempts').rglob('error.json')):
            error = json.loads(path.read_text(encoding='utf-8'))
            # Keep the exception message and phase, not a full traceback from an old version.
            message = error.get('error', '').strip().splitlines()
            context['repair_attempts'].append({'block': error.get('block'), 'stage': error.get('stage'),
                'attempt': int(path.parent.name), 'error': message[-1][:700] if message else ''})
        context['execution_stages'] = [json.loads(path.read_text(encoding='utf-8'))
                                       for path in sorted((node / 'executions').glob('*/status.json'))]
        context['input_facts'] = {}
        for phase in ('smoke', 'execution'):
            path = node / f'{phase}_inputs.json'
            if path.exists():
                facts = json.loads(path.read_text(encoding='utf-8'))
                if facts['code_sha256'] == row.result.code_sha256:
                    context['input_facts'][phase] = {k: v for k, v in facts.items() if k != 'code_sha256'}
        local = JSONAgents(settings, deadline, 8 + MAX_REFERENCE_ROUNDS, shared=agents, node_id=row.node_id)
        return await local.analyze(context)

    agents.on_activity = update
    try:
        extraction = load(directory / 'extraction.json', Extraction)
        update(status='running', stage='preparation')
        context = {'task_text': task_path.read_text(encoding='utf-8'), 'task_directory': str(task_path.parent),
                   'extraction': requirements(extraction), 'inventory': inventory(task_path.parent),
                   'installed_libraries': installed_libraries()}
        async def prepare_inputs(inspect_only=False):
            for attempt in range(3):
                draft = await agents.prepare(context)
                try:
                    check_code(draft.code, ('load_data', 'score', 'write_submission'))
                    for relative in draft.data.files.values():
                        within(task_path.parent, relative)
                    within(directory / 'nodes' / 'node_000', draft.output.filename)
                    atomic_bytes(directory / 'prepare.py', draft.code.encode('utf-8'))
                    prepared = Preparation(task_directory=str(task_path.parent), rationale=draft.rationale,
                        data=draft.data, validation=draft.validation, metric=draft.metric, output=draft.output,
                        code_sha256=file_hash(directory / 'prepare.py'))
                    save(directory / 'preparation.json', prepared)
                    arguments = (['-m', 'agent.research_worker', '--root', str(directory)] if inspect_only
                                 else ['-m', 'agent.worker', '--prepare', str(directory)])
                    await process(arguments, directory, directory / 'preparation.log', deadline, options.experiment_timeout)
                    context.pop('execution_error', None)
                    context['previous_code'] = draft.code
                    return load(directory / 'preparation.json', Preparation)
                except (ValueError, OSError, TimeoutError) as error:
                    if attempt == 2:
                        raise
                    context.update(previous_code=draft.code, execution_error=str(error)[-5000:])
                    update(warnings=report.warnings + [f'Preparation required code repair {attempt + 1}/2'])

        shared.update(requirements=requirements(extraction, prepared=True), installed_libraries=context['installed_libraries'])
        preparation = await prepare_inputs(inspect_only=True)
        update(stage='baseline')
        await process(['-m', 'agent.worker', '--baseline', str(directory)], directory,
                      directory / 'baseline' / 'baseline.log', deadline, options.experiment_timeout)
        published = 'safety_baseline'
        update(checks=report.checks + ['published_cheap_baseline'])
        eda_budget = options.eda_seconds
        if eda_budget is None and not options.research_unlimited:
            eda_budget = options.time_budget * options.eda_fraction
        eda_deadline = deadline if eda_budget is None else min(deadline, time.monotonic() + eda_budget)
        if (options.research_parallel and options.experiment_limit > 1 and
                eda_budget != 0 and (options.research_unlimited or options.research_limit)):
            shared['preparation'] = preparation_view(preparation)
            save(directory / 'research' / 'structure_preparation.json', preparation)
            atomic_bytes(directory / 'research' / 'structure_prepare.py', (directory / 'prepare.py').read_bytes())
            research = Research(agents, directory, options, shared, process, notify, update)
            update(stage='data_research')
            agents.log('initial_eda_started', budget_seconds=eda_budget,
                       research_unlimited=options.research_unlimited)
            structure_deadline = eda_deadline if options.research_unlimited else min(eda_deadline, time.monotonic() + eda_budget * .4)
            await research.window('structure', structure_deadline,
                                  limit=options.research_parallel)
            context.update(data_research=research.view(), preparation_phase='finalize_validation',
                           data_profile=research_profile_view(json.loads((directory / 'research' / 'profile.json').read_text(encoding='utf-8'))))
        if research and research.revision:
            preparation = await prepare_inputs()
        else:
            try:
                await process(['-m', 'agent.worker', '--prepare', str(directory)], directory,
                              directory / 'preparation.log', deadline, options.experiment_timeout)
                preparation = load(directory / 'preparation.json', Preparation)
            except (ValueError, OSError, TimeoutError) as error:
                context['execution_error'] = str(error)[-5000:]
                preparation = await prepare_inputs()
        shared.update({'requirements': requirements(extraction, prepared=True), 'preparation': preparation_view(preparation),
                  'execution_contract': execution_contract(preparation),
                  'installed_libraries': context['installed_libraries']})
        delta = Delta(preparation_sha256=file_hash(directory / 'preparation.json'), metric=preparation.metric)
        save_delta()
        update(checks=report.checks + ['executed_preparation', 'measured_profile', 'fixed_split'])

        from .plans import ToolCall
        baseline_plan = Proposal(purpose='baseline', hypothesis=Hypothesis(
            rationale='Cheap constant predictor before model search', expected_effect='A working reference submission',
            pipeline=Pipeline(data={}, preprocessing={}, model={'name': 'Constant predictor'}, training={'fit_only': True})),
            assembly=Assembly(blocks=[Block(name='constant', operation=ToolCall(kind='constant'),
                description='Fit-only mean/mode/prior prediction', requires=['fit', 'validation', 'test', 'preparation'],
                provides=['validation_predictions', 'test_predictions'])], write_blocks=['constant']))
        update(stage='baseline')
        baseline_task = launch('node_000', baseline_plan)
        await asyncio.gather(baseline_task, return_exceptions=True)
        baseline_finish = result_or_failure(baseline_task, starts['node_000'])
        baseline_start = active.pop(baseline_task)
        record(baseline_start, baseline_finish)
        delta = delta.model_copy(update={'nodes': [row.model_copy(update={'analysis_status': 'skipped'}) for row in delta.nodes]})
        save_delta()
        if research:
            update(stage='data_research')
            await research.window('fit', eda_deadline)
            shared['initial_eda'] = True
            agents.log('initial_eda_finished', research_revision=research.revision, nodes=len(research.nodes))
        search = Search(agents, directory, options, shared, notify)
        search.next_node = 1
        while True:
            expired = time.monotonic() >= deadline
            if expired:
                stop_reason = 'time_budget'
                for task in [*active, *analyses, *search.tasks, *(research.tasks if research else [])]:
                    task.cancel()
            else:
                search.fill(delta, list(active.values()), revision)
                if research and (active or analyses or search.tasks):
                    research.fill()
            update(stage='search' if delta.nodes else 'baseline')
            model_tasks = [*active, *analyses, *search.tasks]
            tasks = [*model_tasks, *(research.tasks if research else [])]
            if not model_tasks:
                if not expired:
                    stop_reason = search.stop_reason
                    if not search.parents and stop_reason == 'max_experiments':
                        stop_reason = 'baseline_failed'
                break
            done, _ = await asyncio.wait(tasks, timeout=None if expired else max(0., deadline - time.monotonic()),
                                         return_when=asyncio.FIRST_COMPLETED)
            if research:
                research.collect(done)
            for task in [task for task in done if task in active]:
                start = active.pop(task)
                search.complete(start)
                record(start, result_or_failure(task, start))
                revision += 1  # New measured models are usable before their prose analysis finishes.
                row = next(row for row in delta.nodes if row.node_id == start.node_id)
                if not expired and agents.remaining_calls:
                    row = row.model_copy(update={'analysis_status': 'running'})
                    delta = delta.model_copy(update={'nodes': [row if n.node_id == row.node_id else n for n in delta.nodes]})
                    analyses[asyncio.create_task(analyze(row))] = row.node_id
                else:
                    row = row.model_copy(update={'analysis_status': 'skipped'})
                    delta = delta.model_copy(update={'nodes': [row if n.node_id == row.node_id else n for n in delta.nodes]})
                    revision += 1
                save_delta()
            for task in [task for task in done if task in analyses]:
                node_id = analyses.pop(task)
                try:
                    analysis = task.result()
                    save(directory / 'nodes' / node_id / 'analysis.json', analysis)
                    values = {'analysis': analysis, 'analysis_status': 'completed'}
                except (Exception, asyncio.CancelledError) as error:
                    agents.log('analysis_failed', node_id=node_id, error_type=type(error).__name__,
                               traceback=traceback.format_exc())
                    values = {'analysis_status': 'skipped' if expired else 'failed'}
                    save(directory / 'nodes' / node_id / 'analysis_error.json', {'error': str(error) or 'Analysis cancelled'})
                    update(warnings=report.warnings + [f'{node_id} analysis: {str(error)[:1000]}'])
                delta = delta.model_copy(update={'nodes': [n.model_copy(update=values) if n.node_id == node_id else n for n in delta.nodes]})
                revision += 1
                save_delta()
            for node_id, proposal in search.collect(done):
                if time.monotonic() >= deadline:
                    continue
                launch(node_id, proposal)
        completed_checks = (['saved_starts'] if starts else [])
        completed_checks += ['assembled_modules'] if any(n.result.code_sha256 for n in delta.nodes) else []
        completed_checks += ['recorded_finishes'] if delta.nodes else []
        completed_checks += ['analyzed_nodes'] if any(n.analysis_status == 'completed' for n in delta.nodes) else []
        update(status='ready' if delta.best_node_id else 'partial' if published else 'failed',
               stage='complete', stop_reason=stop_reason,
               checks=report.checks + completed_checks)
        agents.log('search_finished', status=report.status, stop_reason=stop_reason,
                   requests=agents.budget['requests'], experiments=len(starts), best_node=delta.best_node_id)
    except (Exception, asyncio.CancelledError) as error:
        agents.log('controller_failed', error_type=type(error).__name__, traceback=traceback.format_exc())
        update(status='ready' if delta and delta.best_node_id else 'partial' if published else 'failed',
               stage='complete', stop_reason='error', errors=report.errors + [str(error).replace(settings.api_key, '[REDACTED]')[-5000:]])
    finally:
        tasks = [*active, *analyses, *(search.tasks if search else [])]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for task, start in list(active.items()):
            active.pop(task)
            try:
                record(start, result_or_failure(task, start))
            except Exception as error:
                notify(f'{start.node_id} cleanup: {error}')
        if analyses:
            cancelled = set(analyses.values())
            delta = delta.model_copy(update={'nodes': [row.model_copy(update={'analysis_status': 'skipped'})
                if row.node_id in cancelled else row for row in delta.nodes]})
            analyses.clear()
            save_delta()
        if search:
            try:
                search.collect(set(search.tasks))
            except (Exception, asyncio.CancelledError) as error:
                notify(f'Planning cleanup failed: {error}')
        if research:
            try:
                await research.close(stop_reason)
            except (Exception, asyncio.CancelledError) as error:
                notify(f'Research cleanup failed: {type(error).__name__}: {error}')
        for node_id in list(agents.root.call_reservations):
            agents.release_calls(node_id)
        update()
    # Re-publish the last completed winner even when planning or cleanup failed.
    if published and published != 'safety_baseline':
        selected = delta.best_node_id or published
        result = next(row.result for row in delta.nodes if row.node_id == selected)
        source = within(directory / 'nodes' / selected, result.submission_file)
        atomic_bytes(within(task_path.parent, preparation.output.filename), source.read_bytes())
    if final_reserve and delta and delta.best_node_id and time.monotonic() < overall_deadline:
        selected = delta.best_node_id
        remaining = min(options.final_refit_seconds, overall_deadline - time.monotonic())
        update(status='running', stage='final_refit')
        try:
            await process(['-m', 'agent.finalize', '--run-directory', str(directory),
                           '--node-id', selected, '--time-budget', str(max(1., remaining - 1))],
                          directory, directory / 'final' / 'refit.log', overall_deadline, remaining)
            best = json.loads((directory / 'best.json').read_text(encoding='utf-8'))
            save(directory / 'best.json', {**best, 'submission_file': 'final/submission.csv', 'refitted': True})
            update(status='ready', stage='complete', checks=report.checks + ['final_refit'])
        except (ValueError, OSError, TimeoutError) as error:
            save(directory / 'final' / 'refit_error.json', {'error': str(error)[-3000:]})
            update(status='ready', stage='complete', warnings=report.warnings +
                   ['Final refit failed; the previously published experimental submission is preserved'])
    agents.log('run_finished', status=report.status, best_node=report.best_node, requests=agents.budget['requests'])
    return directory, report
