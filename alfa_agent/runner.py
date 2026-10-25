"""Bounded experiment tree: extraction → preparation → start / finish / delta → next plan."""

import asyncio
import ast
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
import math
import os
import re
from pathlib import Path
import signal
import sys
import time
from uuid import uuid4

from .agents import BAKER_ATTEMPTS, JSONAgents, extract_task, runtime_contract
from .autopilot import automl, template
from .config import Settings
from .contracts import (Change, Comparison, Delta, Extraction, Finalization, Finish, Hypothesis, NodeDelta, Preparation,
                        Program, ReviewRecord, Start)
from .qwen import QwenError
from .research import Research
from .data import check_code, file_hash, inventory, library_context, within
from .storage import atomic_bytes, canonical, load, save

ENTRY = "\nif __name__ == '__main__':\n    from alfa_agent.worker import main\n    main(fit_model)\n"
PREPARATION_ATTEMPTS = 10
IMPLEMENTATION_CALLS = 12  # more repairs rarely help: the hybrid repair and refusal limit settle sooner
AUTOML_CODE = re.compile(r'autogluon|lightautoml|flaml|\bautoml\(')
CHILD = {'threads': 2, 'shift_check': False}  # set per run from Options; children inherit nothing else


class RepairProgress:
    """Three consecutive unchanged executable code/failure pairs stop only this node."""
    def __init__(self):
        self.key, self.repeats = None, 0

    def observe(self, stage, code, error):
        try:
            code = ast.dump(ast.parse(code), include_attributes=False)
        except SyntaxError:
            code = code.strip()
        key = (stage, code, (str(error).strip().splitlines() or [type(error).__name__])[-1])
        self.repeats = self.repeats + 1 if key == self.key else 1
        self.key = key
        if self.repeats >= 3:
            kind = TimeoutError if isinstance(error, TimeoutError) else ValueError
            raise kind(f'{error}\nRepair stalled: unchanged code and failure repeated 3 times; return this node to planning.')
        return {'unchanged_failures': self.repeats, 'stop_after': 3, 'require_changed_code': True}


def apply_edits(code: str, edits) -> str:
    """Replace numbered line ranges of the original code; first_line guards against misnumbering."""
    lines, taken = code.split('\n'), set()
    for edit in edits:  # validate everything against the original before changing anything
        if not edit.from_line <= edit.to_line <= len(lines):
            raise ValueError(f'Edit lines {edit.from_line}-{edit.to_line} are outside 1-{len(lines)}')
        span = set(range(edit.from_line, edit.to_line + 1))
        if taken & span:
            raise ValueError(f'Edit lines {edit.from_line}-{edit.to_line} overlap another edit')
        actual = lines[edit.from_line - 1]
        if edit.first_line.strip() and ' '.join(edit.first_line.split()) != ' '.join(actual.split()):
            raise ValueError(f'first_line does not match line {edit.from_line}; that line is actually: {actual!r}')
        taken |= span
    for edit in sorted(edits, key=lambda e: -e.from_line):  # bottom-up keeps earlier numbers valid
        lines[edit.from_line - 1:edit.to_line] = edit.new.split('\n') if edit.new else []
    return '\n'.join(lines)


class DuplicateCode(Exception):
    """Not a ValueError: an exact repeat of measured code ends the node instead of entering repair."""


def code_key(code: str) -> str:
    try:
        return hashlib.sha256(ast.dump(ast.parse(code), include_attributes=False).encode()).hexdigest()
    except SyntaxError:
        return hashlib.sha256(code.strip().encode()).hexdigest()


class NeedsReplan(ValueError):
    pass


def plan_contains(value, quote):
    if isinstance(value, str):
        return quote in value
    values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return any(plan_contains(item, quote) for item in values)


@dataclass(frozen=True)
class Options:
    time_budget: float = 600
    experiment_timeout: float = 240
    output_dir: Path | None = None
    max_nodes: int = 10
    platform_output: bool = False
    final_refit: bool = True
    # Autopilot: measured shift-aware split, controller GBDT baseline, advisory code review, 4 threads per executor.
    autopilot: bool = False
    executors: int = 2  # concurrent experiment processes
    threads: int | None = None  # library threads per process; default 4 with autopilot, else 2
    llm_concurrency: int = 0  # simultaneous LLM requests per run; 0 = unbounded
    research_parallel: int = 0  # optional advisory workers, enabled by the platform entry


async def process(arguments: list[str], cwd: Path, log: Path, deadline: float, timeout: float):
    remaining = min(timeout, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError('Run time budget exhausted')
    # No inherited credentials. This is a bounded process, not an OS sandbox.
    env = {'PATH': os.defpath, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
           **{name: os.environ[name] for name in ('HOME', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH',
               'TMPDIR', 'TEMP', 'TMP', 'APPDATA', 'LOCALAPPDATA') if name in os.environ},
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
           'ALFA_THREADS': str(CHILD['threads']), 'ALFA_SHIFT_CHECK': '1' if CHILD['shift_check'] else '0',
           **{name: str(CHILD['threads']) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    with log.open('wb') as stream:
        child = await asyncio.create_subprocess_exec(sys.executable, *arguments, cwd=cwd, env=env,
                                                     stdout=stream, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        try:
            code = await asyncio.wait_for(child.wait(), remaining)
        except (TimeoutError, asyncio.CancelledError) as error:
            try:  # AutoML/joblib may spawn workers; the timeout applies to their whole process group.
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


def review_contract(preparation: dict) -> dict:
    override = preparation.get('validation', {}).get('shift', {}).get('override')
    return {**{k: preparation[k] for k in ('data', 'validation', 'metric', 'output')},
        **({'split_note': f'The controller measured train/test shift and replaced the make_split result ({override}); '
                          'split.npz is authoritative, so make_split itself is not a finding.'} if override else {}),
        'columns': {name: stats['dtype'] for name, stats in preparation.get('profile', {}).get('train', {}).get('columns', {}).items()},
        'runtime': runtime_contract(preparation)}


async def review_source(agents, path: Path, source: Path, context: dict, *, request_limit=None):
    """Review only this source/contract; reuse a verdict only for identical bytes and inputs."""
    code_hash = file_hash(source)
    context_hash = hashlib.sha256(canonical(context).encode()).hexdigest()
    previous = load(path, ReviewRecord) if path.exists() else None
    if previous and (previous.code_sha256, previous.context_sha256) == (code_hash, context_hash) and previous.result:
        return previous.result
    payload = {**context, 'previous_findings': previous.result.model_dump()['findings'] if previous and previous.result else []}
    try:
        result = await agents.review(payload, request_limit=request_limit)
        if file_hash(source) != code_hash:
            raise ValueError('Source changed during review')
    except (Exception, asyncio.CancelledError) as error:
        save(path, ReviewRecord(code_sha256=code_hash, context_sha256=context_hash,
             error=(str(error) or 'Review cancelled').replace(agents.settings.api_key, '[REDACTED]')[-2000:]))
        raise
    save(path, ReviewRecord(code_sha256=code_hash, context_sha256=context_hash, result=result))
    return result


async def experiment(node: Path, start: Start, shared: dict, agents: JSONAgents, deadline: float, checkpoint,
                     *, code: str | None = None, repairable: bool = True,
                     check_seconds: float = 45, measured: dict | None = None) -> Finish:
    """code: controller-written first source; the implementer repairs it only when repairable."""
    start_hash, started = file_hash(node / 'start.json'), time.monotonic()
    checkpoint()
    code_hash, notes, stage = None, [], 'implementation'
    progress, replan, refusals = RepairProgress(), {}, 0
    request_limit = min(agents.max_calls, agents.requests + IMPLEMENTATION_CALLS)
    try:
        preparation_path = node.parent.parent / 'preparation.json'
        context = {**shared, 'start': load(node / 'start.json', Start).model_dump(mode='json'),
                   'preparation_code': (node.parent.parent / 'prepare.py').read_text()}
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Run time budget exhausted')
            stage = 'implementation'
            context['requests_left'] = request_limit - agents.requests
            program = Program(code=code, notes=['Controller-written source']) if code else \
                await agents.implement(context, request_limit=request_limit)
            code = None
            notes = list(dict.fromkeys(notes + program.notes))
            if program.edits and not program.code.strip():
                try:
                    program = program.model_copy(update={'code': apply_edits(context.get('previous_code', ''), program.edits)})
                except ValueError as error:  # unusable edits: ask for the complete source next
                    context.update(execution_error='Edits not applied: ' + str(error), failed_stage='implementation', repair_mode='full')
                    if agents.requests >= request_limit or time.monotonic() >= deadline:
                        raise
                    continue
            if program.plan_conflict and not program.code.strip():
                conflict = program.plan_conflict
                if shared['preparation']['data']['target_columns'] and plan_contains(start.hypothesis.model_dump(), conflict.plan_quote):
                    replan = {**conflict.model_dump(), 'source': 'model_request', 'verified': False,
                              'contract_fact': 'Outer validation targets are withheld; labeled rows are available in fit.',
                              'action': 'Baker must propose a new plan; consider an inner split of fit.'}
                    raise NeedsReplan('Plan conflict request: ' + canonical(replan))
                context['last_refusal'] = {'source': 'model_claim', 'verified': False,
                                          'message': 'plan_conflict requires an exact saved-plan quote and actual withheld target columns'}
                continue
            if program.blocked_reason and not program.code.strip() and not program.edits:
                message = 'Implementation blocked: ' + program.blocked_reason
                context['last_refusal'] = {'source': 'model_claim', 'verified': False, 'message': program.blocked_reason}
                refusals += 1
                if refusals >= 2 or agents.requests >= request_limit or time.monotonic() >= deadline:  # a repeated refusal returns the plan to Baker
                    raise ValueError(message)
                checkpoint(f'{start.node_id}: reconsidering an unverified implementation refusal; saved start is unchanged')
                continue
            stage = 'check'
            try:
                atomic_bytes(node / start.implementation, (program.code + ENTRY).encode())
                code_hash = file_hash(node / start.implementation)
                check_code(program.code, ('fit_model',))
                twin = (measured or {}).get(code_key(program.code))
                if twin and twin != start.node_id:
                    raise DuplicateCode(f'Identical code was already measured as {twin}; a new node must change the experiment.')
                await process(['-m', 'alfa_agent.worker', '--check', str(node), '--preparation', str(preparation_path)],
                              node, node / 'execution.log', deadline,
                              # AutoML imports and searches take longer than a plain model on smoke rows
                              min(150 if AUTOML_CODE.search(program.code) else check_seconds, start.timeout_seconds))
                if file_hash(node / 'start.json') != start_hash or file_hash(preparation_path) != start.preparation_sha256:
                    raise ValueError('Saved plan or preparation changed during implementation')
                stage = 'review'
                review = await review_source(agents, node / 'review.json', node / start.implementation,
                    {'kind': 'experiment', 'plan': start.hypothesis.pipeline.model_dump(), 'code': program.code,
                     'contract': review_contract(shared['preparation']),
                     'smoke': json.loads((node / 'smoke.json').read_text()) if (node / 'smoke.json').exists()
                              else {'status': 'passed', 'checks': ['worker_exit_0']}}, request_limit=request_limit)
                if review is not None:
                    context['review_findings'] = [f.model_dump() for f in review.findings]
                    if review.status == 'revise':
                        raise ValueError('Reviewer requested correction: ' + canonical(context['review_findings']))
                    notes.extend('Reviewer warning: ' + f.explanation for f in review.findings)
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
                break
            except (ValueError, OSError, TimeoutError) as error:
                atomic_bytes(node / 'execution.log', str(error).encode())
                context['repair_progress'] = progress.observe(stage, program.code, error)
                if not repairable or agents.requests >= request_limit or time.monotonic() >= deadline:
                    raise
                context.update(previous_code=program.code, execution_error=str(error)[-5000:], failed_stage=stage,
                               # Line edits are cheap and keep working code intact; alternate with a full rewrite.
                               repair_mode='full' if context.get('repair_mode') == 'edits' else 'edits')
                try:  # real signatures of the calls in this code, read in a worker process
                    atomic_bytes(node / 'last_error.txt', str(error)[-5000:].encode())
                    await process(['-m', 'alfa_agent.worker', '--api', str(node)], node, node / 'api.log', deadline, 90)
                    context['api_reference'] = json.loads((node / 'api_reference.json').read_text())
                except (ValueError, OSError, TimeoutError):
                    context.pop('api_reference', None)
                if 'toolkit.py' in str(error):  # the frame is in project code the node cannot edit
                    context['error_location'] = ('Raised inside the project toolkit called from your code: change the '
                                                 'arguments or data you pass, or stop using that toolkit call.')
                else:
                    context.pop('error_location', None)
                checkpoint(f'{start.node_id}: code repair after {stage}; saved start is unchanged')
    except (Exception, asyncio.CancelledError) as error:
        message = (str(error) or 'Experiment cancelled').replace(agents.settings.api_key, '[REDACTED]')[-5000:]
        finish = Finish(node_id=start.node_id, status='needs_replan' if isinstance(error, NeedsReplan) else
            'timeout' if isinstance(error, TimeoutError) or time.monotonic() >= deadline else 'failed',
            stage=stage, metric=shared['preparation']['metric'], duration_seconds=0.,
            preparation_sha256=start.preparation_sha256, start_sha256=start_hash, code_sha256=code_hash, error=message, replan=replan)
    finish = finish.model_copy(update={'notes': notes, 'duration_seconds': time.monotonic() - started})
    save(node / 'finish.json', finish)
    checkpoint()
    return finish


async def run(task_path: Path, options: Options, settings: Settings):
    if not 0 < options.time_budget <= 10800 or not 0 < options.experiment_timeout <= 10800:
        raise ValueError('Time budgets must be finite, positive and at most three hours')
    if type(options.max_nodes) is not int or not 1 <= options.max_nodes <= 1000:
        raise ValueError('max_nodes must be an integer between 1 and 1000')
    if type(options.research_parallel) is not int or not 0 <= options.research_parallel <= 2:
        raise ValueError("research_parallel must be 0, 1 or 2")
    task_path = task_path.resolve()
    CHILD.update(threads=options.threads or (4 if options.autopilot else 2), shift_check=options.autopilot)
    final_deadline = time.monotonic() + options.time_budget
    # The blend adds selection and parallel member refits around the same reserve for the single-winner refit.
    reserve = min(options.time_budget * .15, options.experiment_timeout + (300 if options.autopilot else 0),
                  1500 if options.autopilot else 1200) if options.final_refit else 0
    deadline = final_deadline - reserve
    agents = JSONAgents(settings, deadline, max_calls=12 + (IMPLEMENTATION_CALLS + BAKER_ATTEMPTS) * options.max_nodes,
                        concurrency=options.llm_concurrency)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = (options.output_dir or Path('artifacts') / f'{task_path.parent.name}-{stamp}-{uuid4().hex[:6]}').resolve()
    report = await extract_task(task_path, directory, settings, agents=agents)
    if report.status == 'failed':
        return directory, report

    active, waiting, starts, planner, planning = {}, deque(), {}, None, None
    sources, baseline_id, references = {}, None, set()  # controller code by node; search gate; AutoML nodes
    research = None
    measured = {}  # AST hash of every completed node's code -> node id

    def update(**changes):
        nonlocal report
        report = report.model_copy(update={**changes, 'llm_calls': list(agents.calls),
            'running_nodes': [s.node_id for s in active.values()], 'queued_nodes': [s.node_id for s in waiting],
            'planning': planner is not None, 'research': research.report() if research else {}})
        try:
            save(directory / 'report.json', report)
        except OSError as error:
            # A diagnostic file must not prevent publishing/finalizing an available prediction.
            warning = 'report.json could not be refreshed; details were printed to stderr'
            if warning not in report.warnings:
                report = report.model_copy(update={'warnings': report.warnings + [warning]})
                print(warning + ': ' + error_message(error), file=sys.stderr)

    def error_message(error):
        return (type(error).__name__ + ': ' + (str(error) or 'operation cancelled or interrupted')).replace(settings.api_key, '[REDACTED]')[-5000:]

    delta, published, stop_reason, baseline_ready = None, None, 'node_limit', False
    try:
        extraction = load(directory / 'extraction.json', Extraction)
        update(status='running', stage='preparation')
        context = {'task_text': task_path.read_text(encoding='utf-8'),
                   'extraction': extraction.model_dump(mode='json'), 'inventory': inventory(task_path.parent, exclude=(directory,)),
                   'runtime_limits': {'executors': options.executors, 'threads_per_executor': CHILD['threads']},
                   **library_context()}
        if options.platform_output:
            context['platform_output'] = {'filename': 'submission.csv', 'format': 'CSV',
                'requirement': 'Mandatory platform contract: comma-separated CSV, explicit ordered columns, one row per test object.'}
        for attempt in range(PREPARATION_ATTEMPTS):
            draft = await agents.prepare(context)
            try:
                if options.platform_output and (draft.output.filename != 'submission.csv'
                        or (draft.output.format or '').upper() != 'CSV' or not draft.output.columns):
                    raise ValueError('Platform requires output.filename=submission.csv, output.format=CSV and explicit output.columns')
                if extraction.output.columns and draft.output.columns != extraction.output.columns:
                    raise ValueError('Output columns must match the ordered columns extracted from task.md')
                check_code(draft.code, ('load_data', 'make_split', 'score', 'write_submission'))
                for relative in draft.data.files.values():
                    within(task_path.parent, relative)
                within(directory / 'nodes' / 'node_000', draft.output.filename)
                atomic_bytes(directory / 'prepare.py', draft.code.encode())
                preparation = Preparation(task_directory=str(task_path.parent), rationale=draft.rationale,
                    data=draft.data, validation=draft.validation, metric=draft.metric, output=draft.output,
                    task_type=extraction.problem.task_type, prediction_scope=draft.prediction_scope,
                    code_sha256=file_hash(directory / 'prepare.py'))
                save(directory / 'preparation.json', preparation)
                await process(['-m', 'alfa_agent.worker', '--prepare', str(directory)], directory,
                              directory / 'preparation.log', deadline, options.experiment_timeout)
                preparation = load(directory / 'preparation.json', Preparation)
                try:
                    # Review is optional when the service is unavailable; keep time for a first prediction.
                    async with asyncio.timeout(min(60., max(0., deadline - time.monotonic()) * .1)):
                        review = await review_source(agents, directory / 'preparation_review.json', directory / 'prepare.py',
                            {'kind': 'preparation', 'code': draft.code, 'contract': review_contract(preparation.model_dump(mode='json')),
                             'requirements': {k: {field: value for field, value in block.items() if field != 'evidence'}
                                              for k, block in extraction.model_dump(mode='json').items() if isinstance(block, dict)},
                             'execution': {'status': 'passed', 'checks': ['loaded_tables', 'measured_profile', 'disjoint_split']}})
                except (QwenError, TimeoutError) as error:
                    message = (str(error) or 'Review time limit reached').replace(settings.api_key, '[REDACTED]')
                    update(warnings=report.warnings + ['Preparation review unavailable; runtime checks passed: ' + message])
                    break
                context['review_findings'] = [f.model_dump() for f in review.findings]
                if review.status == 'revise':
                    raise ValueError('Preparation reviewer requested correction: ' + canonical(context['review_findings']))
                if review.findings:
                    update(warnings=report.warnings + ['Preparation reviewer: ' + f.explanation for f in review.findings])
                break
            except (ValueError, OSError, TimeoutError, QwenError) as error:
                if (directory / 'preparation.json').exists():
                    failed = load(directory / 'preparation.json', Preparation).model_copy(
                        update={'status': 'blocked', 'issues': [str(error)[-5000:]]})
                    save(directory / 'preparation.json', failed)
                if attempt == PREPARATION_ATTEMPTS - 1 or isinstance(error, QwenError):
                    raise
                context.update(previous_code=draft.code, execution_error=str(error)[-5000:])
                update(warnings=report.warnings + [f'Preparation requires code repair {attempt + 1}/{PREPARATION_ATTEMPTS - 1}'])
        update(stage='baseline', checks=report.checks + ['executed_preparation', 'measured_profile', 'fixed_split'])
        shared = {'task_text': task_path.read_text(encoding='utf-8'),
                  'runtime_limits': context['runtime_limits'],
                  'extraction': extraction.model_dump(mode='json'), 'preparation': preparation.model_dump(mode='json'),
                  **library_context()}
        delta = Delta(preparation_sha256=file_hash(directory / 'preparation.json'), metric=preparation.metric)
        save(directory / 'delta.json', delta)

        def create(parent_id, purpose, hypothesis, source='qwen', seen=0, timeout=None):
            start = Start(node_id=f'node_{len(starts):03d}', parent_id=parent_id, hypothesis=hypothesis, source=source,
                preparation_sha256=delta.preparation_sha256,
                context={'task': str(task_path), 'task_sha256': report.task_sha256, 'purpose': purpose,
                    'delta_nodes': seen, 'extraction': '../../extraction.json',
                    'extraction_sha256': file_hash(directory / 'extraction.json'),
                    'preparation_code': '../../prepare.py', 'installed_libraries': shared['installed_libraries'],
                    'runtime_interface': 'fit_model_v1',
                    'allowed_libraries': shared['allowed_libraries']},
                outputs={'validation': 'validation_predictions.csv', 'submission': preparation.output.filename},
                timeout_seconds=float(timeout or options.experiment_timeout))
            node = directory / 'nodes' / start.node_id
            node.mkdir(parents=True)
            save(node / 'start.json', start)
            starts[start.node_id] = start
            waiting.append(start)
            return start

        if options.autopilot and preparation.metric.status == 'ready':
            prepared = template(preparation.model_dump(mode='json'), CHILD['threads'])
            if prepared:
                start = create(None, 'baseline', Hypothesis.model_validate(prepared[1]), 'controller')
                sources[start.node_id], baseline_id = prepared[0], start.node_id
                # AutoML references run beside the LLM search; 75% of the process limit leaves room for prediction checks.
                # Small debug runs keep their node budget for the LLM search; AutoML needs at least 6 nodes.
                # AutoGluon gets a double process limit (60% of it to search): its stacking keeps improving with time.
                for code, plan in automl(preparation.model_dump(mode='json'), CHILD['threads'], .75 * options.experiment_timeout,
                                         long_limit=1.2 * options.experiment_timeout)[:max(0, min(3, options.max_nodes - 3))]:
                    long = 'autogluon' in plan['pipeline']['model']['family']
                    start = create(None, 'reference', Hypothesis.model_validate(plan), 'controller',
                                   timeout=2 * options.experiment_timeout if long else None)
                    sources[start.node_id] = code
                    references.add(start.node_id)

        def checkpoint(warning=None):
            update(warnings=report.warnings + ([warning] if warning else []))

        if options.research_parallel and options.max_nodes > 1:
            research = Research(agents, directory, shared, process, checkpoint, options.research_parallel)
            research.start()

        def parent_view(node_id):
            node = directory / 'nodes' / node_id
            return {'start': load(node / 'start.json', Start).model_dump(mode='json'),
                    'finish': load(node / 'finish.json', Finish).model_dump(mode='json'),
                    'code': (node / 'experiment.py').read_text().removesuffix(ENTRY) if (node / 'experiment.py').exists() else None}

        def record(start, finish):
            nonlocal delta, published
            parent = next((n.result for n in delta.nodes if n.node_id == start.parent_id), None)
            classes = finish.diagnostics.get('classification', {})
            reference = parent.diagnostics.get('classification', {}) if parent else {}
            differences = {label: {key: values[key] - reference[label][key] for key in ('precision', 'recall', 'f1')}
                           for label, values in classes.items() if label in reference}
            if differences:
                finish = finish.model_copy(update={'diagnostics': {**finish.diagnostics, 'parent_delta': differences}})
                save(directory / 'nodes' / start.node_id / 'finish.json', finish)
            source = directory / 'nodes' / start.node_id / 'experiment.py'
            if finish.status == 'completed' and source.exists():
                measured.setdefault(code_key(source.read_text().removesuffix(ENTRY)), start.node_id)
            row, delta = summarize(delta, start, finish)
            save(directory / 'nodes' / start.node_id / 'delta.json', row)
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
            if research is not None:
                research.feedback = [{'node_id': n.node_id, 'pipeline': n.hypothesis.pipeline.model_dump(),
                    'status': n.result.status, 'score': n.result.value, 'diagnostics': n.result.diagnostics}
                    for n in delta.nodes[-4:]]
            completed = sum(n.result.status == 'completed' for n in delta.nodes)
            update(completed_nodes=completed, failed_nodes=len(delta.nodes) - completed,
                   warnings=report.warnings + ([f'{start.node_id}: {finish.error}'] if finish.error else []))

        def failed_finish(start, error):
            node = directory / 'nodes' / start.node_id
            finish = Finish(node_id=start.node_id,
                status='timeout' if time.monotonic() >= deadline else 'failed',
                stage='implementation', metric=preparation.metric, duration_seconds=0.,
                preparation_sha256=start.preparation_sha256, start_sha256=file_hash(node / 'start.json'),
                error=error_message(error))
            save(node / 'finish.json', finish)
            return finish

        def result_or_failure(task, start):
            # Cancellation before coroutine entry bypasses experiment's own exception handler.
            try:
                return task.result()
            except (Exception, asyncio.CancelledError) as error:
                return failed_finish(start, error)

        def planning_context():
            nonlocal snapshot
            finished = {n.node_id for n in delta.nodes}
            snapshot = {**shared, 'remaining_nodes': options.max_nodes - len(starts),
                'experiment_timeout_seconds': options.experiment_timeout,
                'remaining_seconds': max(0., deadline - time.monotonic()),
                'delta': {**delta.model_dump(mode='json'),
                          'nodes': [n.model_dump(mode='json', exclude={'comparisons'}) for n in delta.nodes]},
                'pending': [s.model_dump(mode='json') for key, s in starts.items() if key not in finished],
                'available_parents': [n.node_id for n in delta.nodes if n.result.status == 'completed']}
            leader = next((n for n in delta.nodes if n.node_id == delta.best_node_id), None)
            leader = leader or next((n for n in reversed(delta.nodes) if n.result.status == 'completed'), None)
            if leader is not None:
                snapshot.update(parent=parent_view(leader.node_id),
                    available_parents=[n.node_id for n in delta.nodes if n.result.status == 'completed'])
                scored = [n for n in delta.nodes if n.node_id != leader.node_id and n.result.value is not None]
                if scored and delta.metric.direction:
                    second = (min if delta.metric.direction == 'minimize' else max)(scored, key=lambda n: n.result.value)
                    snapshot['codes'] = {second.node_id: parent_view(second.node_id)['code']}
            return snapshot

        planning_failed, snapshot, rejections, blocked_version, batch_version = False, {}, {}, -1, -1
        while True:
            expired = time.monotonic() >= deadline
            if expired:
                stop_reason = 'time_budget'
                for task in [*active, *([planner] if planner else [])]:
                    if not task.done():
                        task.cancel()
            while not expired and waiting and len(active) < options.executors:
                start = waiting.popleft()
                context = {**shared, **({'parent': parent_view(start.parent_id)} if start.parent_id else {})}
                local = JSONAgents(settings, deadline, IMPLEMENTATION_CALLS, shared=agents, node_id=start.node_id)
                task = asyncio.create_task(experiment(directory / 'nodes' / start.node_id, start, context, local, deadline, checkpoint,
                                                      code=sources.pop(start.node_id, None),
                                                      # blends of several boosters need imports and fits beyond 45 s
                                                      check_seconds=150 if start.node_id in references else 120 if options.autopilot else 45,
                                                      measured=measured))
                active[task] = start
            if (not expired and not planning_failed and planner is None and (options.autopilot or len(active) < options.executors)
                    and all(s.node_id in references for s in waiting)  # autopilot: one batch may queue behind running nodes
                    and len(starts) < options.max_nodes and len(delta.nodes) > blocked_version and (baseline_id is None or baseline_ready)):
                batch_version, stop_reason = len(delta.nodes), 'node_limit'
                if baseline_ready:
                    planning = agents.plan_next(planning_context, min(2, options.max_nodes - len(starts)), rejections=rejections)
                    planner = asyncio.create_task(anext(planning))
                else:
                    planner = asyncio.create_task(agents.plan_baseline(planning_context()))
            update(stage='search' if delta.nodes else 'baseline')
            tasks = [*active, *([planner] if planner else [])]
            if not tasks:
                break
            done, _ = await asyncio.wait(tasks, timeout=None if expired else max(0., deadline - time.monotonic()),
                                         return_when=asyncio.FIRST_COMPLETED)
            # Only the controller merges results and publishes a winner; workers never write shared state.
            for task in sorted((t for t in done if t in active), key=lambda t: active[t].node_id):
                start = active.pop(task)
                finish = result_or_failure(task, start)
                record(start, finish)
                if start.node_id == baseline_id:
                    baseline_ready = finish.status == 'completed' and finish.evaluation_status != 'failed'
                    if baseline_ready:
                        update(checks=report.checks + ['working_baseline'])
                    elif start.source == 'controller':  # fall back to the LLM baseline
                        baseline_id = None
                        update(warnings=report.warnings + ['Controller baseline failed; requesting an LLM baseline'])
                    else:
                        planning_failed = True
                        stop_reason = 'time_budget' if time.monotonic() >= deadline else 'baseline_failed'
                        update(errors=report.errors + ['Baseline did not complete successfully; search was not started'])
            if planner in done:
                try:
                    result = planner.result()
                    proposals = [(None, 'baseline', result)] if baseline_id is None else [(p.parent_id, p.purpose, p.hypothesis) for p in result]
                    if time.monotonic() < deadline:
                        for parent_id, purpose, hypothesis in proposals:
                            start = create(parent_id, purpose, hypothesis, seen=len(snapshot['delta']['nodes']))
                            baseline_id = baseline_id or start.node_id
                except StopAsyncIteration:
                    planning = None
                except (Exception, asyncio.CancelledError) as error:
                    planning = None
                    planning_failed = not isinstance(error, QwenError)
                    blocked_version = batch_version
                    stop_reason = 'time_budget' if time.monotonic() >= deadline else 'llm_error' if isinstance(error, QwenError) else 'error'
                    if not isinstance(error, asyncio.CancelledError):
                        message = error_message(error)
                        update(warnings=report.warnings + [message])
                        if planning_failed or (not active and not waiting and len(delta.nodes) <= blocked_version):
                            update(errors=report.errors + [message])
                planner = asyncio.create_task(anext(planning)) if planning is not None and time.monotonic() < deadline else None
    except (Exception, asyncio.CancelledError) as error:
        stop_reason = ('preparation_failed' if delta is None else 'time_budget' if time.monotonic() >= deadline
                       else 'llm_error' if isinstance(error, QwenError) else 'error')
        update(status='partial' if published else 'failed', stop_reason=stop_reason,
               errors=report.errors + [error_message(error)])
    finally:
        if research is not None:
            try:
                await research.close()
            except (Exception, asyncio.CancelledError) as error:
                update(warnings=report.warnings + ["Research cleanup: " + error_message(error)])
        tasks = [*active, *([planner] if planner else [])]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # A worker can finish while cancellation/cleanup is in progress. Keep its result.
        for task, start in list(active.items()):
            active.pop(task)
            try:
                record(start, result_or_failure(task, start))
            except (Exception, asyncio.CancelledError) as error:
                update(warnings=report.warnings + [f'{start.node_id} cleanup: ' + error_message(error)])
        while waiting:
            start = waiting.popleft()
            try:
                record(start, failed_finish(start, asyncio.CancelledError('Queued experiment was not started')))
            except (Exception, asyncio.CancelledError) as error:
                update(warnings=report.warnings + [f'{start.node_id} cleanup: ' + error_message(error)])
        if planning is not None:
            try:
                await planning.aclose()
            except (Exception, asyncio.CancelledError) as error:
                update(warnings=report.warnings + ['Planning cleanup: ' + error_message(error)])
        planner = planning = None

    # Finalization must run after both normal search and recovered controller failures.
    try:
        if published:
            selected = next(n.result for n in delta.nodes if n.node_id == published)
            output = within(directory / 'nodes' / published, selected.submission_file)
            from .worker import validate_submission
            validate_submission(output, preparation.output, preparation.profile['test']['rows'],
                                id_columns=preparation.data.id_columns)
            atomic_bytes(within(task_path.parent, preparation.output.filename), output.read_bytes())
        blended = False
        if (options.autopilot and published and options.final_refit and time.monotonic() < final_deadline
                and sum(n.result.value is not None for n in delta.nodes) > 1):
            # A validation-chosen blend replaces the single winner only when it measurably improves the shared split.
            update(stage='finalization')
            worker = ['-m', 'alfa_agent.worker']
            try:
                await process(worker + ['--blend', str(directory), '--members', str(options.executors)], directory,
                              directory / 'blend.log', final_deadline, 300)
                plan = json.loads((directory / 'blend.json').read_text())
                if plan['selected']:
                    failures = await asyncio.gather(*(process(worker + ['--refit-member', str(directory / 'nodes' / node / 'start.json')],
                        directory, directory / 'nodes' / node / 'refit.log', final_deadline, min(1200, options.experiment_timeout))
                        for node in plan['weights']), return_exceptions=True)
                    await process(worker + ['--blend-final', str(directory)], directory, directory / 'blend_final.log', final_deadline, 300)
                    plan = json.loads((directory / 'blend.json').read_text())
                    output = within(directory, plan['submission_file'])
                    from .worker import validate_submission
                    validate_submission(output, preparation.output, preparation.profile['test']['rows'],
                                        id_columns=preparation.data.id_columns)
                    atomic_bytes(within(task_path.parent, preparation.output.filename), output.read_bytes())
                    blended = True
                    update(checks=report.checks + ['validated_blend', 'final_submission_validated'],
                           warnings=report.warnings + [f'Blend member refit failed, fit-only predictions used: {str(e)[-300:]}'
                                                       for e in failures if isinstance(e, Exception)])
            except (Exception, asyncio.CancelledError) as error:
                update(warnings=report.warnings + ['Blend skipped; refitting the single winner: ' + error_message(error)[-1000:]])
        if not blended and published and options.final_refit and time.monotonic() < final_deadline:
            selected = next(n.result for n in delta.nodes if n.node_id == published)
            final = Finalization(status='running', node_id=published, preparation_sha256=selected.preparation_sha256,
                                 start_sha256=selected.start_sha256, code_sha256=selected.code_sha256)
            save(directory / 'finalization.json', final)
            update(stage='finalization')
            started = time.monotonic()
            try:
                await process(['-m', 'alfa_agent.worker', '--refit', str(directory / 'nodes' / published / 'start.json')],
                              directory, directory / 'finalization.log', final_deadline, min(1200, options.experiment_timeout))
                final = load(directory / 'finalization.json', Finalization)
                if final.status != 'completed' or (final.node_id, final.code_sha256, final.preparation_sha256, final.start_sha256) != (
                        published, selected.code_sha256, selected.preparation_sha256, selected.start_sha256):
                    raise ValueError('Final refit does not match the selected source and preparation')
                output = within(directory, final.submission_file)
                from .worker import validate_submission
                validate_submission(output, preparation.output, preparation.profile['test']['rows'],
                                        id_columns=preparation.data.id_columns)
                atomic_bytes(within(task_path.parent, preparation.output.filename), output.read_bytes())
                if (directory / 'best.json').exists():
                    best = json.loads((directory / 'best.json').read_text())
                    best.update(validation_submission_file=best['submission_file'], submission_file=final.submission_file,
                                refit_on_full_train=True, refit_scored=False)
                    atomic_bytes(directory / 'best.json', canonical(best).encode())
                update(checks=report.checks + ['final_refit', 'final_submission_validated'])
            except (Exception, asyncio.CancelledError) as error:
                final = final.model_copy(update={'status': 'timeout' if isinstance(error, TimeoutError) else 'failed',
                    'error': error_message(error)[-2000:], 'duration_seconds': time.monotonic() - started})
                save(directory / 'finalization.json', final)
                update(warnings=report.warnings + ['Final refit failed; preserved validated winner: ' + error_message(error)[-1000:]])
    except (Exception, asyncio.CancelledError) as error:
        update(warnings=report.warnings + ['Finalization interrupted: ' + error_message(error)])
    valid_output = False
    if published:
        try:
            from .worker import validate_submission
            validate_submission(within(task_path.parent, preparation.output.filename), preparation.output,
                                preparation.profile['test']['rows'], id_columns=preparation.data.id_columns)
            valid_output = True
        except (Exception, asyncio.CancelledError) as error:
            update(errors=report.errors + ['Final output invalid: ' + error_message(error)])
    update(status='ready' if valid_output and delta.best_node_id and stop_reason == 'node_limit'
           else 'partial' if valid_output else 'failed', stage='complete', stop_reason=stop_reason,
           checks=report.checks + (['saved_starts', 'recorded_finishes', 'computed_deltas'] if delta is not None else [])
                  + (['submission_written', 'final_submission_validated'] if valid_output else []))
    return directory, report
