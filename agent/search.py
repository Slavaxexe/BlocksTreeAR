"""A short idea queue and independent node planners; ready plans never wait for their batch."""

import asyncio
import json
import math
from collections import deque
import time
import traceback

from .agents import JSONAgents, plan_key, REGRESSION_MODELS
from .assembly import block_catalog
from .context import catalog_view, idea_history_view
from .contracts import Assembly, Block, Hypothesis, Idea, Pipeline, Proposal, TrainingSelection
from .qwen import QwenTemporaryError
from .storage import fingerprint, save


class Search:
    def __init__(self, agents, directory, options, shared, notify):
        self.agents, self.directory, self.options = agents, directory, options
        self.shared, self.notify = shared, notify
        self.queue, self.records, self.planners = deque(), [], {}
        self.baker, self.round, self.next_node = None, 0, 0
        self.failures, self.misses, self.revision = 0, 0, 0
        self.failed, self.stop_reason = False, 'max_experiments'
        self.used_ideas, self.used_plans = {}, {}
        self.parents, self.leader, self.catalog = [], None, {}
        self.rows = []
        self.experiment_limit = options.max_experiments or math.inf
        explore = int(options.parallel > 1)
        self.caps = {'improve': options.parallel - explore, 'explore': explore}
        self.seed_portfolio()

    def remaining_seconds(self):
        return None if math.isinf(self.agents.deadline) else max(0., self.agents.deadline - time.monotonic())

    async def select_regression_portfolio(self):
        available = [record['builtin_engine'] for record in self.queue]
        selection = await self.agents.ask(TrainingSelection,
            REGRESSION_MODELS + 'Choose a SMALL ordered subset of available_engines for this regression task from the measured '
            'profile, missingness, categorical columns, row count, metric, validation policy and remaining time. '
            'Do not mechanically run every library. Explain each choice and omitted redundant approaches. '
            'When available_engines contains FLAML or AutoGluon, include at least one of them to obtain '
            'an AutoML result for subsequent Qwen analysis and improvements. '
            'Return engines=[] if custom Qwen-generated training is more appropriate. '
            'Choose only from available_engines. No business-model stage or invented measurements.\n' +
            json.dumps({'available_engines': available, **self.shared,
                                     'remaining_seconds': self.remaining_seconds()}, ensure_ascii=False),
            'training_selector')
        if len(set(selection.engines)) != len(selection.engines) or set(selection.engines) - set(available):
            raise ValueError('Qwen selected duplicate or unavailable training engines')
        if set(available) & {'flaml', 'autogluon'} and not set(selection.engines) & {'flaml', 'autogluon'}:
            raise ValueError('Qwen must include an available AutoML baseline')
        by_engine = {record['builtin_engine']: record for record in self.queue}
        self.queue.clear()
        self.records.clear()
        self.used_ideas.clear()
        for index, engine in enumerate(selection.engines[:self.options.max_experiments or len(selection.engines)]):
            record = by_engine[engine]
            record['id'] = f'idea_{index:03d}'
            record['idea']['purpose'] = 'baseline' if index == 0 else 'explore'
            record['idea']['change'] += '; Qwen selection: ' + selection.rationale
            self.records.append(record)
            self.queue.append(record)
            self.used_ideas[fingerprint(Idea.model_validate(record['idea']).model_dump(exclude={'expected_effect'}))] = record['id']
        self.shared['training_selection'] = selection.model_dump()
        save(self.directory / 'training_selection.json', selection)

    def seed_portfolio(self):
        """Seed budgeted model candidates without asking Qwen to write their code."""
        preparation = self.shared['preparation']
        requirements = self.shared['requirements']
        problem = requirements.get('problem', {})
        task = str(problem.get('task_type') or '').lower()
        metric = str(preparation['metric'].get('name') or '').lower().replace('-', '_').replace(' ', '_')
        validation = preparation.get('validation', {})
        split_description = ' '.join(str(validation.get(key) or '') for key in
                                     ('strategy', 'method', 'scheme', 'split_type')).lower()
        if (len(preparation['data']['target_columns']) != 1
                or not any(kind in task for kind in ('regression', 'classification', 'binary', 'multiclass'))
                or any(kind in task for kind in ('multioutput', 'multilabel', 'forecast', 'rank'))
                or validation.get('group_columns')
                or validation.get('time_column') or validation.get('time_columns')
                or any(kind in split_description for kind in ('temporal', 'time_series', 'chronological', 'time_based', 'group', 'time'))
                or metric not in {'rmse', 'root_mean_squared_error', 'mae', 'mean_absolute_error',
                                  'mse', 'mean_squared_error', 'r2', 'r2_score', 'accuracy', 'accuracy_score',
                                  'roc_auc', 'auc', 'log_loss', 'logloss', 'f1', 'f1_macro', 'macro_f1'}):
            return
        libraries = self.shared['installed_libraries']
        engines = [engine for engine, package in (('flaml', 'flaml'), ('autogluon', 'autogluon.tabular'))
                   if self.options.automl and package in libraries]
        if self.options.tuning and 'optuna' in libraries:
            engines.extend(engine for engine, package in (('optuna_lightgbm', 'lightgbm'), ('optuna_catboost', 'catboost'))
                           if package in libraries)
        if 'regress' in task and validation.get('inner_validation', {}).get('method') == 'cross_validation':
            # The other native adapters use holdout; never claim they performed CV.
            engines = [engine for engine in engines if engine == 'flaml']
        limit = len(engines) if 'regress' in task else self.options.max_experiments or len(engines)
        for engine in engines[:limit]:
            idea = Idea(purpose='baseline' if not self.records else 'explore',
                        change=f'Built-in {engine} fit-only model search',
                        change_blocks=[f'builtin_{engine}'],
                        expected_effect='Compare an automatically selected tabular model on the fixed outer split')
            record = {'id': f'idea_{len(self.records):03d}', 'idea': idea.model_dump(),
                      'status': 'queued', 'builtin_engine': engine}
            self.records.append(record)
            self.queue.append(record)
            self.used_ideas[fingerprint(idea.model_dump(exclude={'expected_effect'}))] = record['id']

    async def builtin_plan(self, idea, engine):
        name = f'builtin_{engine}'
        problem = self.shared['requirements']['problem']
        output = self.shared['requirements']['output']
        method = 'Optuna with held-out postprocessing' if engine.startswith('optuna_') else 'AutoML'
        return Proposal(purpose=idea.purpose, parent_id=idea.parent_id,
            hypothesis=Hypothesis(rationale=idea.change, expected_effect=idea.expected_effect,
                pipeline=Pipeline(data={'source': 'prepared fit only'},
                    preprocessing={'owner': engine, 'fit_only': True},
                    model={'engine': engine, 'task_type': problem.get('task_type'),
                           'prediction_type': output.get('prediction_type'),
                           'class_labels': problem.get('class_labels', [])},
                    training={'builtin_adapter': True, 'internal_validation': 'fit only', 'cpu_limit': 2,
                              'postprocessing': 'held-out within fit; frozen before outer predictions'})),
            assembly=Assembly(blocks=[Block(name=name,
                description=f'Built-in {engine}: fit-only {method} and prediction DataFrames in fixed row order',
                requires=['fit', 'validation', 'test', 'preparation', 'auxiliary'],
                provides=['validation_predictions', 'test_predictions'])], write_blocks=[name]))

    @property
    def tasks(self):
        return [*self.planners, *([self.baker] if self.baker else [])]

    def report(self):
        return {'planning': self.baker is not None,
                'planning_nodes': [record['node_id'] for record in self.planners.values()],
                'queued_ideas': [{'id': record['id'], **record['idea']} for record in self.queue]}

    def save(self):
        save(self.directory / 'ideas.json', {'ideas': self.records})

    def fill(self, delta, running, revision):
        self.rows = delta.nodes
        if revision != self.revision:
            self.misses, self.revision = 0, revision
            if self.stop_reason == 'planning_stalled':
                self.stop_reason = 'max_experiments'
        rows = [row for row in delta.nodes if row.result.status == 'completed'
                and row.analysis_status in {'completed', 'failed', 'skipped'}]
        self.parents = [row.node_id for row in rows]
        eligible = [row for row in rows if row.result.value is not None]
        if eligible and delta.metric.direction:
            choose = min if delta.metric.direction == 'minimize' else max
            self.leader = choose(eligible, key=lambda row: row.result.value).node_id
        else:
            self.leader = self.parents[-1] if self.parents else None
        self.catalog = block_catalog(self.directory, rows)

        # A new leader invalidates queued improvements, but never interrupts work already started.
        for record in list(self.queue):
            if record['idea']['purpose'] == 'improve' and record['idea']['parent_id'] != self.leader:
                record['status'] = 'superseded'
                self.queue.remove(record)
                self.agents.log('idea_superseded', idea_id=record['id'], leader_id=self.leader)
                self.save()
        self.dispatch(running)
        self.request_ideas(delta)

    def dispatch(self, running):
        occupied = [start.purpose for start in running] + [record['idea']['purpose'] for record in self.planners.values()]
        while (self.queue and len(occupied) < self.options.parallel
               and self.next_node < self.experiment_limit):
            record = next((record for record in self.queue
                if (record.get('builtin_engine') or self.agents.remaining_calls >= 2)
                and (record['idea']['purpose'] == 'baseline'
                     or (record.get('builtin_engine') and self.options.parallel == 1)
                     or occupied.count(record['idea']['purpose']) < self.caps[record['idea']['purpose']])), None)
            if record is None:
                break
            self.queue.remove(record)
            idea = Idea.model_validate(record['idea'])
            node_id = f'node_{self.next_node:03d}'
            self.next_node += 1
            record.update(status='planning', node_id=node_id)
            parent = next((row for row in self.rows if row.node_id == idea.parent_id), None)
            context = {**self.shared, 'idea': record['idea'],
                'parent': idea_history_view(parent) if parent else None,
                'module_catalog': catalog_view(self.catalog),
                'pending': self.pending(), 'remaining_seconds': self.remaining_seconds(),
                'experiment_timeout_seconds': self.options.experiment_timeout or None}
            save(self.directory / 'planning' / node_id / 'idea.json', record)
            if record.get('builtin_engine'):
                task = asyncio.create_task(self.builtin_plan(idea, record['builtin_engine']))
            else:
                client = JSONAgents(self.agents.settings, self.agents.deadline, 3, shared=self.agents, node_id=node_id)
                task = asyncio.create_task(client.plan_node(context, idea, self.catalog))
            self.planners[task] = record
            occupied.append(idea.purpose)
            self.agents.log('node_planning_started', node_id=node_id, idea_id=record['id'])
            self.save()

    def pending(self):
        return [{'id': record['id'], 'status': record['status'], 'node_id': record.get('node_id'),
                 'error': record.get('error'), 'duplicate_of': record.get('duplicate_of'), **record['idea']}
                for record in self.records]

    def request_ideas(self, delta):
        if self.baker or self.failed or self.misses >= 3:
            return
        # Let the fixed portfolio produce results before asking for follow-up hypotheses.
        if not self.parents and (self.records or self.planners):
            if not self.queue and not self.planners and self.rows:
                # A failed native adapter must not suppress the normal generated baseline.
                if self.next_node >= self.experiment_limit:
                    return
            else:
                return
        if self.agents.remaining_calls < 2:
            self.stop_reason = 'llm_limit'
            return
        room = self.experiment_limit - self.next_node - len(self.queue)
        if room <= 0:
            return
        if not self.parents and not self.queue:
            purposes = ['baseline']
        elif self.parents:
            size = max(3, min(5, self.options.parallel + 2))
            targets = {'improve': size - int(self.caps['explore'] > 0),
                       'explore': int(self.caps['explore'] > 0)}
            needed = {purpose: max(0, target - sum(r['idea']['purpose'] == purpose for r in self.queue))
                      for purpose, target in targets.items()}
            purposes = []
            while any(needed.values()):
                for purpose, cap in self.caps.items():
                    count = min(needed[purpose], cap)
                    purposes.extend([purpose] * count)
                    needed[purpose] -= count
        else:
            return
        purposes = purposes[:min(len(purposes), room, self.agents.remaining_calls - 1)]
        if not purposes:
            return
        context = {**self.shared, 'purposes': purposes, 'leader_id': self.leader,
            'available_parents': self.parents, 'experiments': [idea_history_view(row) for row in delta.nodes],
            'module_catalog': {node: [{'name': block['name'], 'provides': block['provides']} for block in blocks]
                               for node, blocks in catalog_view(self.catalog).items()},
            'previous_ideas': self.pending(), 'remaining_seconds': self.remaining_seconds(),
            'experiment_timeout_seconds': self.options.experiment_timeout or None}
        self.round += 1
        save(self.directory / 'planning' / f'batch_{self.round:03d}' / 'request.json', context)
        self.baker = asyncio.create_task(self.ask_ideas(context))

    async def ask_ideas(self, context):
        if self.failures:
            self.agents.log('planning_retry_wait', consecutive_failures=self.failures, delay_seconds=5)
            await asyncio.sleep(min(5., self.remaining_seconds() or 5.))
        return await self.agents.propose_ideas({**context,
            'remaining_seconds': self.remaining_seconds()})

    def collect(self, done):
        ready = []
        if self.baker in done:
            task, self.baker = self.baker, None
            try:
                ideas = task.result()
                self.failures = 0
                accepted = 0
                for idea in ideas:
                    key = fingerprint(idea.model_dump(exclude={'expected_effect'}))
                    if key in self.used_ideas:
                        self.agents.log('duplicate_idea', duplicate_of=self.used_ideas[key])
                        continue
                    record = {'id': f'idea_{len(self.records):03d}', 'idea': idea.model_dump(), 'status': 'queued'}
                    self.records.append(record)
                    self.queue.append(record)
                    self.used_ideas[key] = record['id']
                    accepted += 1
                self.misses = 0 if accepted else self.misses + 1
                if accepted:
                    self.stop_reason = 'max_experiments'
                elif self.misses >= 3:
                    self.stop_reason = 'planning_stalled'
                self.save()
            except asyncio.CancelledError:
                pass
            except OSError:
                raise
            except Exception as error:
                self.failures += 1
                self.agents.log('idea_generation_failed', traceback=traceback.format_exc())
                temporary = isinstance(error, QwenTemporaryError) and self.failures < 3
                self.failed = not temporary
                self.stop_reason = 'llm_error'
                self.notify(f'Baker: {error}', fatal=not temporary)

        for task in [task for task in done if task in self.planners]:
            record = self.planners.pop(task)
            node_id = record['node_id']
            try:
                proposal = task.result()
                key = plan_key(proposal)
                if key in self.used_plans:
                    record.update(status='rejected', duplicate_of=self.used_plans[key])
                    self.notify(f'{node_id}: duplicate plan matching {self.used_plans[key]}')
                    continue
                self.used_plans[key] = node_id
                record['status'] = 'planned'
                save(self.directory / 'planning' / node_id / 'proposal.json', proposal)
                ready.append((node_id, proposal))
            except OSError:
                raise
            except (Exception, asyncio.CancelledError) as error:
                message = (str(error) or 'Planning cancelled').replace(self.agents.settings.api_key, '[REDACTED]')
                record.update(status='failed', error=message[:2000])
                self.agents.log('node_planning_failed', node_id=node_id, traceback=traceback.format_exc())
                save(self.directory / 'planning' / node_id / 'error.json', {'error': message})
                if not isinstance(error, asyncio.CancelledError):
                    self.notify(f'{node_id} planner: {message}')
            finally:
                self.save()
        return ready
