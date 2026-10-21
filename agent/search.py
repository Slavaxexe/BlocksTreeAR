"""A short idea queue and independent node planners; ready plans never wait for their batch."""

import asyncio
from collections import Counter, deque
import time
import traceback

from .agents import JSONAgents, plan_key
from .assembly import block_catalog
from .context import catalog_view, idea_history_view
from .contracts import Idea, Ideas
from .qwen import QwenTemporaryError
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
        explore = int(options.parallel > 1)
        self.caps = {'improve': options.parallel - explore, 'explore': explore}

    @property
    def tasks(self):
        return [*self.planners, *([self.baker] if self.baker else [])]

    def report(self):
        return {'planning': self.baker is not None,
                'planning_nodes': [record['node_id'] for record in self.planners.values()],
                'queued_ideas': self.queued_view()}

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
            stale_parent = record['idea']['purpose'] == 'improve' and record['idea']['parent_id'] != self.leader
            if stale_parent:
                record['status'] = 'superseded'
                self.queue.remove(record)
                self.agents.log('idea_superseded', idea_id=record['id'], leader_id=self.leader)
                self.save()
        self.request_ideas()
        self.dispatch(running)

    def dispatch(self, running):
        occupied = [start.purpose for start in running] + [record['idea']['purpose'] for record in self.planners.values()]
        while (self.queue and len(occupied) < self.options.parallel
               and self.next_node < self.options.max_experiments and self.agents.remaining_calls >= 2):
            record = max((record for record in self.queue
                if record['idea']['purpose'] == 'baseline'
                or occupied.count(record['idea']['purpose']) < self.caps[record['idea']['purpose']]),
                key=lambda row: row['idea'].get('priority', 50), default=None)
            if record is None:
                break
            self.queue.remove(record)
            idea = Idea.model_validate(record['idea'])
            node_id = f'node_{self.next_node:03d}'
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
                'module_catalog': catalog_view(self.catalog),
                'remaining_seconds': max(0., self.agents.deadline - time.monotonic()),
                'experiment_timeout_seconds': self.options.experiment_timeout,
                'idea': record['idea']}
            save(self.directory / 'planning' / node_id / 'idea.json', record)
            client = JSONAgents(self.agents.settings, self.agents.deadline, 3, shared=self.agents, node_id=node_id)
            task = asyncio.create_task(client.plan_node(context, idea, self.catalog))
            self.planners[task] = record
            occupied.append(idea.purpose)
            self.agents.log('node_planning_started', node_id=node_id, idea_id=record['id'])
            self.save()
        self.occupied = dict(Counter(occupied))

    def idea_history(self, *, active_only=False):
        result = []
        experiments = {row.node_id for row in self.rows}
        for record in self.records:
            if active_only and record['status'] not in {'queued', 'planning'}:
                continue
            item = {'id': record['id'], 'status': record['status'], **record['idea']}
            # Past forecasts are not observations; retain the concrete change to prevent repeats.
            item.pop('expected_effect')
            if record.get('node_id') in experiments:
                # The complete change and configuration are already in experiments.
                item = {'id': record['id'], 'status': record['status']}
            for key in ('node_id', 'error', 'duplicate_of', 'decision'):
                if record.get(key) is not None:
                    item[key] = record[key]
            result.append(item)
        return result

    def request_ideas(self):
        if self.baker or self.failed or self.misses >= 3:
            return
        if self.agents.remaining_calls < 2:
            self.stop_reason = 'llm_limit'
            return
        remaining = self.options.max_experiments - self.next_node
        if remaining <= 0:
            return
        if not self.parents and (self.next_node or self.queue):
            return
        needs_ideas = len(self.queue) < min(self.options.parallel, remaining)
        new_feedback = self.revision != self.baker_feedback
        if not needs_ideas and not new_feedback and not self.failures:
            return
        self.round += 1
        self.baker = asyncio.create_task(self.ask_ideas('new_analysis' if new_feedback else 'queue_low'))

    def baker_context(self):
        remaining = self.options.max_experiments - self.next_node
        return {**self.shared, 'purposes': ['baseline'] if self.next_node == 0 else
                [purpose for purpose, cap in self.caps.items() if cap],
            'max_new_ideas': min(1 if self.next_node == 0 else 3, remaining),
            'queue_capacity': min(self.options.parallel, remaining), 'remaining_experiments': remaining,
            'slot_limits': self.caps, 'occupied_slots': self.occupied,
            'queued_ideas': self.queued_view(), 'leader_id': self.leader,
            'available_parents': self.parents, 'experiments': [idea_history_view(row) for row in self.rows],
            'module_catalog': {node: [{key: block[key] for key in ('name', 'requires', 'provides')} for block in blocks]
                               for node, blocks in catalog_view(self.catalog).items()},
            'previous_ideas': self.idea_history(), 'remaining_seconds': max(0., self.agents.deadline - time.monotonic()),
            'feedback_revision': {'experiments': self.revision[0],
                                  'data_research': self.shared.get('data_research', {}).get('revision', 0)},
            'experiment_timeout_seconds': self.options.experiment_timeout}

    async def ask_ideas(self, trigger):
        if self.failures:
            self.agents.log('planning_retry_wait', consecutive_failures=self.failures, delay_seconds=5)
            await asyncio.sleep(min(5., max(0., self.agents.deadline - time.monotonic())))
        context = {**self.baker_context(), 'trigger': trigger}
        self.baker_research_revision = context.get('data_research', {}).get('revision', 0)
        self.baker_feedback = (context['feedback_revision']['experiments'], self.baker_research_revision)
        save(self.directory / 'planning' / f'batch_{self.round:03d}' / 'request.json', context)
        if context['remaining_experiments'] <= 0:
            return Ideas()
        return await self.agents.propose_ideas(context)

    def apply_updates(self, updates):
        queued = {row['id']: row for row in self.queue}
        for update in updates:
            record = queued.get(update.idea_id)
            if record is None:
                self.agents.log('idea_update_skipped', idea_id=update.idea_id, reason='Already dispatched or removed')
                continue
            record['decision'] = update.model_dump()
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
                accepted = 0
                for idea in response.ideas:
                    if len(self.queue) >= min(self.options.parallel, self.options.max_experiments - self.next_node):
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
