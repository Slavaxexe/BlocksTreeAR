"""A short idea queue and independent node planners; ready plans never wait for their batch."""

import asyncio
from collections import Counter, deque
import json
import math
import time
import traceback

from .agents import JSONAgents, plan_key
from .assembly import block_catalog
from .context import (catalog_view, compact, experiment_index, idea_history_view,
                      operation_view, select_experiments)
from .contracts import Assembly, Block, Hypothesis, Idea, Ideas, Pipeline, Proposal
from .plans import AutoMLPlan, ToolCall
from .llm import LLMError
from .storage import fingerprint, save


class Search:
    def __init__(self, agents, directory, options, shared, notify):
        self.agents, self.directory, self.options = agents, directory, options
        self.shared, self.notify = shared, notify
        self.queue, self.records, self.planners = deque(), [], {}
        self.baker, self.round, self.next_node = None, 0, 0
        self.failures, self.misses, self.revision = 0, 0, (0, 0)
        self.failed, self.stop_reason = False, 'max_experiments'
        self.used_ideas, self.used_plans = {}, {}
        self.parents, self.leader, self.catalog = [], None, {}
        self.rows = []
        self.baker_research_revision = 0
        self.baker_feedback = None
        self.occupied = {}
        self.parallel_limit = min(3, options.parallel)
        self.budget_wait = None
        self.automl_seeded = False
        self.prediction_cache = {}
        self.feature_decisions = {}
        self.last_planning_error = None
        self.baker_candidates = []
        self.node_builds = {}
        explore = int(options.parallel > 1)
        self.caps = {'improve': options.parallel - explore, 'explore': explore}

    def seed_automl(self):
        """Reserve one configured experiment per installed adapter after initial EDA."""
        if self.automl_seeded:
            return
        self.automl_seeded = True
        if not self.options.automl:
            return
        preparation = self.shared['preparation']
        requirements = self.shared['requirements']
        problem = requirements.get('problem', {})
        task = str(problem.get('task_type') or '').lower()
        metric = str(preparation['metric'].get('name') or '').lower().replace('-', '_').replace(' ', '_')
        from .task_types import family
        kind = family(task, preparation['data']['target_columns'])
        standard_metric = metric in {'rmse', 'root_mean_squared_error', 'mae', 'mean_absolute_error',
                                  'mse', 'mean_squared_error', 'r2', 'r2_score', 'accuracy', 'accuracy_score',
                                  'roc_auc', 'auc', 'log_loss', 'logloss', 'f1', 'f1_macro', 'macro_f1',
                                  'f1_micro', 'micro_f1', 'hamming_loss', 'subset_accuracy'}
        if (not preparation['data']['target_columns'] or kind == 'clustering'
                or not (standard_metric or kind == 'ranking' and metric.startswith(('ndcg', 'map')))):
            return
        libraries = self.shared['installed_libraries']
        engines = [engine for engine, package in (('flaml', 'flaml'), ('autogluon', 'autogluon.tabular'))
                   if package in libraries and (engine == 'flaml' or kind not in {'ranking', 'forecasting'})]
        remaining = self.options.experiment_limit - self.next_node
        for engine in engines[:min(len(engines), remaining)]:
            idea = Idea(purpose='improve', parent_id=self.leader,
                        change=f'Dedicated {engine} search using EDA and selected feature recipes',
                        change_blocks=[f'builtin_{engine}'], priority=100,
                        expected_effect='Compare an automatically selected task-compatible model on the fixed outer split')
            record = {'id': f'idea_{len(self.records):03d}', 'idea': idea.model_dump(),
                      'status': 'queued', 'builtin_engine': engine, 'dedicated_automl': True,
                      'research_revision': self.shared.get('data_research', {}).get('revision', 0)}
            self.records.append(record)
            self.queue.append(record)
            self.used_ideas[fingerprint(idea.model_dump(exclude={'expected_effect'}))] = record['id']

        # Cheap independent reference families complement the configurable AutoML trials.
        # LLM still chooses their features/settings; unsupported task families skip these probes.
        if kind in {'regression', 'classification'} and self.agents.deadline - time.monotonic() > 90:
            for name in ('ExtraTrees', 'regularized linear'):
                if len(self.queue) >= remaining:
                    break
                idea = Idea(purpose='improve', parent_id=self.leader,
                    change=f'Test a cheap {name} reference model with suitable preprocessing and promising EDA features; '
                           'keep the outer split fixed. Use no large hyperparameter search.',
                    change_blocks=['features', 'preprocess', 'fit', 'predict'], priority=90,
                    expected_effect='Measure a complementary family; keep it only if its validation evidence is useful')
                record = {'id': f'idea_{len(self.records):03d}', 'idea': idea.model_dump(),
                          'status': 'queued', 'reference_family': name,
                          'research_revision': self.shared.get('data_research', {}).get('revision', 0)}
                self.records.append(record)
                self.queue.append(record)

        self.save()

    async def dedicated_automl_plan(self, idea, engine, context, node_id):
        client = JSONAgents(self.agents.settings, self.agents.deadline, 6,
                            shared=self.agents, node_id=node_id)
        try:
            plan = await client.configure_automl(context, engine)
        except LLMError as error:
            # A broken planning reply must not silently remove the reserved adapter trial.
            self.notify(f'{node_id}: AutoML configuration failed; using {engine} defaults: {error}')
            self.agents.log('automl_configuration_fallback', node_id=node_id, engine=engine, error=str(error))
            plan = AutoMLPlan(engine=engine, ensemble=False)
        configured = idea.model_copy(update={'automl': plan})
        return await self.builtin_plan(configured, engine)

    async def builtin_plan(self, idea, engine, candidates=None):
        name = f'builtin_{engine}'
        plan = idea.automl or AutoMLPlan(engine=engine)
        from .feature_trials import selected_candidates
        from .plans import FeatureSpec
        selected = selected_candidates(candidates if candidates is not None else self.shared.get('data_research', {}).get('feature_candidates', []), idea.feature_candidate_ids)
        recipes = {feature.name: feature for feature in plan.features}
        for candidate in selected:
            recipes.update({feature['name']: FeatureSpec.model_validate(feature) for feature in candidate['features']})
        plan = plan.model_copy(update={'features': list(recipes.values())})
        problem = self.shared['requirements']['problem']
        output = self.shared['requirements']['output']
        proposal = Proposal(purpose=idea.purpose, parent_id=idea.parent_id,
            hypothesis=Hypothesis(rationale=idea.change, expected_effect=idea.expected_effect, research_ids=idea.research_ids,
                pipeline=Pipeline(data={'source': 'prepared fit only'},
                    preprocessing={'owner': engine, 'fit_only': True},
                    model={'name': 'FLAML AutoML' if engine == 'flaml' else 'AutoGluon Tabular',
                           'engine': engine, 'task_type': problem.get('task_type'),
                           'prediction_type': output.get('prediction_type'),
                           'class_labels': problem.get('class_labels', [])},
                    training={'builtin_adapter': True, 'automl': plan.model_dump(), 'internal_validation': 'fit only', 'cpu_limit': 1})),
            assembly=Assembly(blocks=[Block(name=name, operation=ToolCall(kind='automl', parameters=plan.model_dump()),
                description=f'Built-in {engine}: fit-only AutoML and prediction DataFrames in fixed row order',
                requires=['fit', 'validation', 'test', 'preparation', 'auxiliary'],
                provides=['validation_predictions', 'test_predictions'])], write_blocks=[name]))
        from .feature_trials import check_recipes
        check_recipes(proposal, selected)
        return proposal

    async def ensemble_plan(self, idea):
        plan = idea.ensemble
        return Proposal(purpose=idea.purpose, parent_id=idea.parent_id,
            hypothesis=Hypothesis(rationale=idea.change, expected_effect=idea.expected_effect,
                research_ids=idea.research_ids,
                pipeline=Pipeline(data={'prediction_sources': plan.source_nodes},
                    preprocessing={'owner': 'saved_ensemble'},
                    model={'name': plan.method + ' ensemble', 'sources': plan.source_nodes},
                    training={'ensemble': plan.model_dump(), 'fit_scope': 'OOF inside fit for learned methods'})),
            assembly=Assembly(blocks=[Block(name='combine', description='Combine completed sources: ' + ', '.join(plan.source_nodes),
                operation=ToolCall(kind='saved_ensemble', parameters=plan.model_dump()),
                requires=['fit', 'validation', 'test', 'preparation', 'auxiliary'],
                provides=['validation_predictions', 'test_predictions'])], write_blocks=['combine']))

    async def ready_plan(self, idea):
        from .builds import compile_build, load_build
        return compile_build(load_build(self.directory, idea.build_id), idea.changes, idea)

    @property
    def tasks(self):
        return [*self.planners, *([self.baker] if self.baker else [])]

    def report(self):
        return {'planning': self.baker is not None,
                'planning_nodes': [record['node_id'] for record in self.planners.values()],
                'queued_ideas': self.queued_view(), 'search_parallel_limit': self.parallel_limit}

    def estimate_calls(self, blocks, *, planning=True):
        per_block = 1  # Reserve a configuration repair per block; initial source is compiled locally.
        # Include a repair margin; measured calls include tool continuations and JSON retries.
        return (3 if planning else 0) + per_block * blocks + 2

    def complete(self, start):
        self.agents.release_calls(start.node_id)

    def wait_for_budget(self, calls):
        state = (calls, self.agents.free_calls)
        if state != self.budget_wait:
            self.agents.log('node_budget_wait', needed_calls=calls, free_calls=self.agents.free_calls,
                            reservations=dict(self.agents.root.call_reservations))
            self.budget_wait = state
        self.stop_reason = 'llm_limit'

    def slot_limits(self):
        if self.parallel_limit == self.options.parallel:
            return self.caps
        explore = min(1, self.caps['explore'])
        return {'improve': self.parallel_limit - explore, 'explore': explore}

    def queued_view(self):
        return [{'id': record['id'], **record['idea']} for record in
                sorted(self.queue, key=lambda row: -row['idea'].get('priority', 50))]

    def save(self):
        save(self.directory / 'ideas.json', {'ideas': self.records})

    def fill(self, delta, running, revision):
        self.rows = delta.nodes
        revision = (revision, self.shared.get('data_research', {}).get('revision', 0))
        if revision != self.revision:
            self.misses, self.revision = 0, revision
            if self.stop_reason == 'planning_stalled':
                self.stop_reason = 'max_experiments'
        rows = [row for row in delta.nodes if row.result.status == 'completed']
        self.parents = [row.node_id for row in rows]
        eligible = [row for row in rows if row.result.value is not None]
        if eligible and delta.metric.direction:
            choose = min if delta.metric.direction == 'minimize' else max
            self.leader = choose(eligible, key=lambda row: row.result.value).node_id
        else:
            self.leader = self.parents[-1] if self.parents else None
        self.catalog = block_catalog(self.directory, rows)
        from .builds import register_node
        for row in rows:
            manifest = self.directory / 'nodes' / row.node_id / 'assembly.json'
            if row.node_id != 'node_000' and row.node_id not in self.node_builds and manifest.exists():
                try:
                    self.node_builds[row.node_id] = register_node(self.directory, row)
                except (ValueError, OSError, KeyError) as error:
                    self.node_builds[row.node_id] = None
                    self.agents.log('build_registration_failed', node_id=row.node_id, error=str(error))
        baseline = next((row for row in eligible if row.node_id == 'node_000'), None)
        leader = next((row for row in eligible if row.node_id == self.leader), None)
        if baseline and leader and delta.metric.direction:
            better = (leader.result.value < baseline.result.value if delta.metric.direction == 'minimize'
                      else leader.result.value > baseline.result.value)
            if better and self.parallel_limit < self.options.parallel:
                self.parallel_limit = self.options.parallel
                self.agents.log('search_parallel_expanded', limit=self.parallel_limit, leader=self.leader)

        if self.parents and not self.automl_seeded:
            self.seed_automl()

        # A new leader invalidates queued improvements, but never interrupts work already started.
        for record in list(self.queue):
            stale_parent = (not record.get('builtin_engine') and not record.get('reference_family') and not record['idea'].get('build_id')
                            and record['idea']['purpose'] == 'improve'
                            and record['idea']['parent_id'] != self.leader)
            if stale_parent:
                record['status'] = 'superseded'
                self.queue.remove(record)
                self.agents.log('idea_superseded', idea_id=record['id'], leader_id=self.leader)
                self.save()
        self.dispatch(running)
        self.request_ideas()

    def dispatch(self, running):
        occupied = [start.purpose for start in running] + [record['idea']['purpose'] for record in self.planners.values()]
        caps = self.slot_limits()
        while (self.queue and len(occupied) < self.parallel_limit
               and self.next_node < self.options.experiment_limit and self.agents.remaining_calls >= 2):
            record = max((record for record in self.queue
                if record['idea']['purpose'] == 'baseline'
                or occupied.count(record['idea']['purpose']) < caps[record['idea']['purpose']]),
                key=lambda row: (row.get('dedicated_automl', False), row['idea'].get('priority', 50)), default=None)
            if record is None:
                break
            if record.get('dedicated_automl') or record.get('reference_family'):
                record['idea']['parent_id'] = self.leader
            idea = Idea.model_validate(record['idea'])
            blocks = max(1, len(idea.change_blocks))
            calls = (8 if record.get('dedicated_automl') else 2
                     if record.get('builtin_engine') or record.get('builtin_ensemble') or record['idea'].get('build_id') else self.estimate_calls(blocks))
            node_id = f'node_{self.next_node:03d}'
            if not self.agents.reserve_calls(node_id, calls):
                self.wait_for_budget(calls)
                break
            self.budget_wait = None
            self.queue.remove(record)
            self.next_node += 1
            record.update(status='planning', node_id=node_id)
            parent = next((row for row in self.rows if row.node_id == idea.parent_id), None)
            parent_context = idea_history_view(parent) if parent else None
            if parent_context:
                # Baker chose the change already; old rationales/next ideas are not planner choices.
                parent_context['hypothesis'] = {'pipeline': parent.hypothesis.pipeline.model_dump()}
                parent_context.pop('analysis')
            context = {**self.shared,
                'parent': parent_context,
                'parent_operations': [operation_view(block.get('operation')) for block in self.catalog.get(idea.parent_id, [])],
                'module_catalog': catalog_view(self.catalog),
                'remaining_seconds': max(0., self.agents.deadline - time.monotonic()),
                'experiment_timeout_seconds': self.options.experiment_timeout,
                'idea': record['idea']}
            if not record.get('reference_family'):
                context['parent_build'] = self.node_builds.get(idea.parent_id)
            if record.get('feature_candidates'):
                context['data_research'] = {**context.get('data_research', {}),
                    'feature_candidates': record['feature_candidates']}
            save(self.directory / 'planning' / node_id / 'idea.json', record)
            if idea.build_id:
                task = asyncio.create_task(self.ready_plan(idea))
            elif record.get('dedicated_automl'):
                task = asyncio.create_task(self.dedicated_automl_plan(idea, record['builtin_engine'], context, node_id))
            elif record.get('builtin_ensemble'):
                task = asyncio.create_task(self.ensemble_plan(idea))
            elif record.get('builtin_engine'):
                task = asyncio.create_task(self.builtin_plan(idea, record['builtin_engine'], record.get('feature_candidates')))
            else:
                client = JSONAgents(self.agents.settings, self.agents.deadline, 8, shared=self.agents, node_id=node_id)
                task = asyncio.create_task(client.plan_node(context, idea, self.catalog))
            self.planners[task] = record
            occupied.append(idea.purpose)
            self.agents.log('node_planning_started', node_id=node_id, idea_id=record['id'])
            self.save()
        self.occupied = dict(Counter(occupied))

    def idea_history(self, *, active_only=False):
        result = []
        experiments = {row.node_id for row in self.rows}
        records = self.records
        if not active_only:
            active = [row for row in records if row['status'] in {'queued', 'planning', 'planned'}
                      and row.get('node_id') not in experiments]
            recent = [row for row in records if row['status'] in {'failed', 'rejected', 'superseded', 'removed'}][-8:]
            records = active + recent
        for record in records:
            if active_only and record['status'] not in {'queued', 'planning'}:
                continue
            item = {'id': record['id'], 'status': record['status'], **record['idea']}
            # Past forecasts are not observations; retain the concrete change to prevent repeats.
            item.pop('expected_effect')
            if record['status'] == 'queued' or record.get('node_id') in experiments:
                # Queue details and completed nodes already have their own views.
                item = {'id': record['id'], 'status': record['status']}
            for key in ('node_id', 'error', 'duplicate_of', 'decision'):
                if record.get(key) is not None:
                    item[key] = record[key]
            result.append(compact(item))
        return result

    def request_ideas(self):
        if self.baker or self.failed or self.misses >= 3:
            return
        if self.agents.free_calls < self.estimate_calls(1) + 1:
            self.stop_reason = 'llm_limit'
            return
        remaining = self.options.experiment_limit - self.next_node
        if remaining <= 0:
            return
        if not self.parents and (self.next_node or self.queue):
            return
        free_slots = max(0, self.parallel_limit - sum(self.occupied.values()))
        needs_ideas = len(self.queue) < min(free_slots, remaining)
        new_analysis = self.baker_feedback is None or self.revision[0] != self.baker_feedback[0]
        if not needs_ideas and not (self.queue and new_analysis):
            return
        self.round += 1
        self.baker = asyncio.create_task(self.ask_ideas('new_analysis' if new_analysis else 'queue_low'))

    def experiment_history(self):
        cards = [{'node_id': row.node_id, 'parent_id': getattr(row, 'parent_id', None),
                  **idea_history_view(row)} for row in self.rows]
        direction = self.shared.get('preparation', {}).get('metric', {}).get('direction')
        selected = select_experiments(cards, self.leader, direction, limit=6)
        baseline = next((row.get('result', {}).get('value') for row in cards if row['node_id'] == 'node_000'), None)
        result = []
        for item in selected:
            node = self.directory / 'nodes' / item['node_id']
            facts = [json.loads(path.read_text(encoding='utf-8')) for path in node.glob('*_full_diagnostics.json')]
            manifest = node / 'assembly.json'
            blocks = json.loads(manifest.read_text(encoding='utf-8'))['blocks'] if manifest.exists() else []
            recipes = []
            for block in blocks:
                call = block.get('operation') or {}
                settings = call.get('parameters', {})
                recipes.extend(settings.get('recipes', settings.get('features', [])))
            score = item['result'].get('value')
            model = next((f.get('native_estimator') or f.get('best_estimator') or f.get('best_model')
                          for f in facts if f.get('native_estimator') or f.get('best_estimator') or f.get('best_model')),
                         item['hypothesis']['pipeline']['model'].get('name'))
            card = {'node_id': item['node_id'], 'parent_id': item['parent_id'], 'model': model,
                    'status': item['result']['status'], 'val_score': score,
                    'delta_from_baseline': (baseline - score if direction == 'minimize' else score - baseline)
                        if score is not None and baseline is not None else None,
                    'key_features': [recipe['name'] for recipe in recipes],
                    'input_feature_counts': sorted({f['input_feature_count'] for f in facts if 'input_feature_count' in f}),
                    'feature_warnings': [warning for f in facts for warning in f.get('feature_warnings', [])],
                    'seconds': item['result'].get('duration_seconds'), 'error': item['result'].get('error')}
            if item.get('analysis'):
                card['analysis'] = {'summary': item['analysis'].get('summary')}
            if item['node_id'] == self.leader:
                card['feature_recipes'] = recipes
                card['model_facts'] = [{key: f[key] for key in ('native_estimator', 'native_parameters',
                    'best_estimator', 'best_config', 'inner_best_loss', 'inner_validation', 'selected_model_count',
                    'selected_columns', 'input_feature_count')
                    if key in f} for f in facts if f.get('best_estimator') or f.get('native_estimator') or f.get('selected_columns')]
            result.append(compact(card))
        # The full journal remains local for duplicate detection. Prompt size is bounded.
        index = select_experiments(cards, self.leader, direction, limit=16)
        return result, experiment_index(index)

    def baker_context(self):
        from .predictions import prediction_catalog
        remaining = self.options.experiment_limit - self.next_node
        problem = self.shared.get('requirements', {}).get('problem', {})
        prediction_sources = prediction_catalog(self.directory, self.rows, self.leader, self.prediction_cache,
            {**problem, **self.shared.get('requirements', {}).get('output', {})},
            self.shared.get('preparation', {}).get('metric', {}).get('name'))
        experiments, index = self.experiment_history()
        selected = {item['node_id'] for item in experiments}
        from .feature_trials import candidate_view
        research = dict(self.shared.get('data_research', {}))
        if research.get('feature_candidates'):
            research['feature_candidates'] = candidate_view(research['feature_candidates'],
                self.directory, self.rows, self.records, self.feature_decisions)
        from .builds import card, load_build
        build_lookup = {node: value for node, value in self.node_builds.items() if node in selected and value}
        for item in research.get('feature_candidates', []):
            if item.get('build_id'):
                build_lookup[item['build_id']] = load_build(self.directory, item['build_id'])
        ready_builds = [card(value) for value in build_lookup.values()]
        return {**self.shared, 'data_research': research, 'purposes': ['baseline'] if self.next_node == 0 else
                [purpose for purpose, cap in self.caps.items() if cap],
            'max_new_ideas': min(1 if self.next_node == 0 or self.failures else 2, remaining,
                                 max(1 if self.next_node == 0 else 2, self.parallel_limit - len(self.queue))),
            'queue_capacity': min(self.parallel_limit, remaining),
            'remaining_experiments': None if math.isinf(remaining) else remaining,
            'slot_limits': self.slot_limits(), 'occupied_slots': self.occupied,
            'api_budget': self.agents.budget_view(),
            'automl_portfolio': [{'engine': record['builtin_engine'], 'status': record['status'],
                                 'node_id': record.get('node_id'), 'dedicated': record.get('dedicated_automl', False)}
                                for record in self.records if record.get('builtin_engine')],
            'prediction_sources': prediction_sources,
            'queued_ideas': self.queued_view(), 'leader_id': self.leader,
            'available_parents': self.parents, 'experiments': experiments, 'experiment_index': index,
            'ready_builds': ready_builds,
            '_build_lookup': build_lookup,
            'module_catalog': {node: [{key: block[key] for key in ('name', 'requires', 'provides')} for block in blocks]
                               for node, blocks in catalog_view({node: blocks for node, blocks in self.catalog.items()
                                                               if node in selected}).items()},
            'previous_ideas': self.idea_history(), 'idea_coverage': dict(Counter(row['status'] for row in self.records)),
            'remaining_seconds': max(0., self.agents.deadline - time.monotonic()),
            'feedback_revision': {'experiments': self.revision[0],
                                  'data_research': self.shared.get('data_research', {}).get('revision', 0)},
            'experiment_timeout_seconds': self.options.experiment_timeout}

    async def ask_ideas(self, trigger):
        if self.failures:
            delay = min(60., 5. * 2 ** min(self.failures - 1, 4))
            self.agents.log('planning_retry_wait', consecutive_failures=self.failures, delay_seconds=delay)
            await asyncio.sleep(min(delay, max(0., self.agents.deadline - time.monotonic())))
        context = {**self.baker_context(), 'trigger': trigger}
        self.baker_research_revision = context.get('data_research', {}).get('revision', 0)
        self.baker_candidates = context.get('data_research', {}).get('feature_candidates', [])
        self.baker_feedback = (context['feedback_revision']['experiments'], self.baker_research_revision)
        save(self.directory / 'planning' / f'batch_{self.round:03d}' / 'request.json', context)
        if context['remaining_experiments'] is not None and context['remaining_experiments'] <= 0:
            return Ideas()
        if self.failures:
            context['planning_error'] = self.last_planning_error
        return await self.agents.propose_ideas(context)

    def apply_updates(self, updates):
        queued = {row['id']: row for row in self.queue}
        for update in updates:
            record = queued.get(update.idea_id)
            if record is None:
                self.agents.log('idea_update_skipped', idea_id=update.idea_id, reason='Already dispatched or removed')
                continue
            record['decision'] = update.model_dump()
            if record.get('dedicated_automl') and update.action == 'remove':
                self.agents.log('idea_update_skipped', idea_id=record['id'], reason='Reserved AutoML experiment')
                continue
            if update.action == 'remove':
                record['status'] = 'removed'
                self.queue.remove(record)
            else:
                record['idea']['priority'] = update.priority
            self.agents.log('idea_updated', **update.model_dump())

    def collect(self, done):
        ready = []
        if self.baker in done:
            task, self.baker = self.baker, None
            try:
                response = task.result()
                self.failures = 0
                save(self.directory / 'planning' / f'batch_{self.round:03d}' / 'response.json', response)
                self.apply_updates(response.updates)
                for decision in response.feature_decisions:
                    self.feature_decisions[decision.candidate_id] = decision.model_dump()
                save(self.directory / 'feature_decisions.json', self.feature_decisions)
                accepted = 0
                for idea in response.ideas:
                    if len(self.queue) >= min(self.parallel_limit, self.options.experiment_limit - self.next_node):
                        self.agents.log('idea_capacity_reached')
                        break
                    identity = idea.model_dump(exclude={'expected_effect', 'priority', 'research_ids'})
                    identity['change'] = ' '.join(identity['change'].split())
                    identity['change_blocks'] = sorted(identity['change_blocks'])
                    key = fingerprint(identity)
                    if key in self.used_ideas:
                        self.agents.log('duplicate_idea', duplicate_of=self.used_ideas[key])
                        continue
                    record = {'id': f'idea_{len(self.records):03d}', 'idea': idea.model_dump(), 'status': 'queued',
                              'research_revision': self.baker_research_revision}
                    from .feature_trials import selected_candidates
                    record['feature_candidates'] = selected_candidates(self.baker_candidates, idea.feature_candidate_ids)
                    if idea.automl is not None:
                        record['builtin_engine'] = idea.automl.engine
                    if idea.ensemble is not None:
                        record['builtin_ensemble'] = True
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
                self.last_planning_error = str(error)
                self.agents.log('idea_generation_failed', traceback=traceback.format_exc())
                temporary = not self.agents.root.api_error and time.monotonic() < self.agents.deadline
                self.failed = not temporary
                self.stop_reason = 'llm_error'
                self.notify(f'Baker: {error}', fatal=not temporary)

        for task in [task for task in done if task in self.planners]:
            record = self.planners.pop(task)
            node_id = record['node_id']
            accepted = False
            try:
                proposal = task.result()
                key = (fingerprint({'automl': proposal.hypothesis.pipeline.training['automl']})
                       if record.get('builtin_engine') else plan_key(proposal))
                if key in self.used_plans:
                    record.update(status='rejected', duplicate_of=self.used_plans[key])
                    self.notify(f'{node_id}: duplicate plan matching {self.used_plans[key]}')
                    continue
                calls = 2 if record.get('builtin_engine') or record.get('builtin_ensemble') or record['idea'].get('build_id') else self.estimate_calls(len(proposal.assembly.write_blocks), planning=False)
                if not self.agents.reserve_calls(node_id, calls):
                    record.update(status='rejected', error='Insufficient API budget for the planned blocks')
                    self.wait_for_budget(calls)
                    continue
                self.used_plans[key] = node_id
                accepted = True
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
                if not accepted:
                    self.agents.release_calls(node_id)
                self.save()
        return ready
