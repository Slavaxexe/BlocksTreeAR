"""An independent research tree and evidence-backed context shared with Baker."""

import asyncio
import json
import time
import traceback

from .agents import JSONAgents
from .contracts import ResearchPlan, ResearchReview
from .context import (compact, research_history, research_profile_view, select_research,
                      table, task_context)
from .data import file_hash
from .storage import fingerprint, save
from .eda import CATALOG


RESEARCH_RULES = (
    'Investigate the supplied dataset to inform validation, features and model choices. '
    'Use measured evidence, not dataset-name guesses. Do not search for hidden labels. '
    'Structure phase supplies fit WITHOUT row-level targets: defer label-dependent questions to fit phase. '
    'Marginal profiles are available; investigate schema, meaning, '
    'quality, duplicates, groups and train/test shift. Fit phase supplies labelled fit only; '
    'validation/test never have targets. A date or group-like column alone does not determine the split. '
    'Do not change data, split, scorer, preparation or submissions. '
)


class Research:
    def __init__(self, agents, directory, options, shared, process, notify, refresh):
        self.agents, self.directory, self.options = agents, directory, options
        self.shared, self.process, self.notify, self.refresh = shared, process, notify, refresh
        self.active, self.clients, self.nodes, self.insights = {}, {}, [], []
        self.questions = []
        self.model_feedback = []
        self.revision, self.failures = 0, 0
        self.stop_reason = None
        # Research shares the global API budget and reserves most calls for model search.
        self.call_limit = min(24, max(0, (options.max_llm_calls - 8) // 3))
        self.persist()

    @property
    def tasks(self):
        return list(self.active)

    def view(self):
        return {'revision': self.revision,
                'insights': [{k: row[k] for k in ('id', 'phase', 'statement', 'kind')}
                             for row in self.insights if row['active']],
                'open_questions': self.questions,
                'validation_policy': 'Research proposes changes; the fixed split is never modified during model search'}

    def report(self):
        return {'max_parallel_research': self.options.research_parallel,
                'researching_nodes': [row['node_id'] for row in self.active.values()],
                'research_nodes': len(self.nodes), 'research_revision': self.revision,
                'research_insights': sum(row['active'] for row in self.insights),
                'research_stop_reason': self.stop_reason}

    def persist(self):
        save(self.directory / 'research' / 'context.json', {
            **self.view(), 'insight_history': self.insights, 'stop_reason': self.stop_reason})
        save(self.directory / 'research' / 'tree.json', {'nodes': self.nodes})
        self.shared['data_research'] = self.view()

    def fill(self, phase='fit', limit=None):
        cap = min(self.options.max_research_nodes, limit or self.options.max_research_nodes)
        while (not self.stop_reason and len(self.active) < self.options.research_parallel
               and len(self.nodes) < cap and self.failures < 3):
            spent = sum(client.requests for client in self.clients.values())
            reserved = sum(self.clients[row['node_id']].max_calls - self.clients[row['node_id']].requests
                           for row in self.active.values())
            allowance = min(6, self.call_limit - spent - reserved)
            # A structured analysis plan and its review each need an API call. Preserve model-search headroom.
            if allowance < 2 or self.agents.remaining_calls < reserved + allowance + 4:
                if not self.active:
                    self.stop_reason = 'llm_limit'
                break
            node_id = f'research_{len(self.nodes):03d}'
            row = {'node_id': node_id, 'phase': phase, 'stage': 'planning', 'parent_id': None,
                   'context_revision': self.revision,
                   'focus': 'quality, semantics and grouping' if len(self.nodes) % 2 == 0
                            else 'distributions and useful feature relationships'}
            client = JSONAgents(self.agents.settings, self.agents.deadline, allowance,
                                shared=self.agents, node_id=node_id)
            self.clients[node_id] = client
            self.nodes.append(row)
            task = asyncio.create_task(self.investigate(row, client))
            self.active[task] = row
        if len(self.nodes) >= self.options.max_research_nodes and not self.active:
            self.stop_reason = 'node_limit'
        elif self.failures >= 3:
            self.stop_reason = 'repeated_failures'
        self.persist()
        self.refresh()

    async def bootstrap(self):
        self.fill('structure', limit=self.options.research_parallel)
        while self.active:
            done, _ = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
            self.collect(done)

    async def investigate(self, row, client):
        started = time.monotonic()
        node = self.directory / 'research' / 'nodes' / row['node_id']
        node.mkdir(parents=True, exist_ok=True)
        context = {**task_context(self.shared, profile=False), 'phase': row['phase'], 'node_id': row['node_id'],
                   'model_feedback': self.model_feedback,
                   'data_research': select_research(self.view(), [row['focus'], self.model_feedback]),
                   'profile': research_profile_view(json.loads((self.directory / 'research' / 'profile.json').read_text(encoding='utf-8'))),
                   'analysis_catalog': CATALOG,
                   'history': research_history([item for item in self.nodes if item is not row]),
                   'target_access': {'fit': row['phase'] == 'fit', 'validation': False, 'test': False},
                   'focus': row['focus'], 'timeout_seconds': self.options.research_timeout}
        context.pop('installed_libraries', None)  # AnalysisRequest already enumerates its ready estimators.
        context = compact(context)
        try:
            save(node / 'request.json', context)
            plan = await client.ask(ResearchPlan, RESEARCH_RULES +
                'Choose a specific decision-relevant question and invoke the supplied analysis catalog through analyses. '
                'Return structured requests, NEVER Python code. Each request selects kind, input columns and parameters. '
                'dataset selects fit/validation/test for descriptive checks. compare selects the second table ONLY for shift; '
                'a name containing test does not select test. Label-dependent checks always use fit. '
                'Use a few related checks instead of an exhaustive Cartesian search. Prefer existing measurements '
                'over repeating them; a changed question does not make identical measurements new. '
                'Use feature recipes to test proposed features, ablation to test removal, interactions for joint '
                'signal missed by univariate tests, permutation for held-out importance, and stability for groups. '
                'Use validation=auto for proxy models to respect the task time/group policy. '
                'sample_rows/top_k control the computation and result size; set null when the full result is needed. '
                'Proxy scores are internal fit-only evidence, never an experiment outer score or test score. '
                'parent_id may reference a completed research node. Keep hypothesis, method and decision concise.\n' +
                json.dumps(context, ensure_ascii=False, separators=(',', ':')), 'data_researcher')
            parents = {item['node_id'] for item in self.nodes if item['stage'] == 'completed'}
            if plan.parent_id is not None and plan.parent_id not in parents:
                raise ValueError('Research parent must be a completed node')
            start = {'node_id': row['node_id'], 'phase': row['phase'], 'context_revision': self.revision,
                     **plan.model_dump(), 'task_type': self.shared.get('requirements', {}).get('problem', {}).get('task_type'),
                     'preparation_sha256': file_hash(self.directory / 'preparation.json')}
            start['analysis_sha256'] = fingerprint(start['analyses'])
            save(node / 'start.json', start)
            row.update({key: start[key] for key in ('parent_id', 'question', 'hypothesis', 'method', 'decision', 'analyses')})
            self.stage(row, 'execution')
            try:
                await self.process(['-m', 'agent.research_worker', '--root', str(self.directory), '--node', str(node)],
                                   node, node / 'execution.log', self.agents.deadline, self.options.research_timeout)
            except TimeoutError:
                if not (node / 'measurements.json').exists():
                    raise
            measured = json.loads((node / 'measurements.json').read_text(encoding='utf-8'))
            if (measured['analysis_sha256'], measured['preparation_sha256']) != (start['analysis_sha256'], start['preparation_sha256']):
                raise ValueError('Measurements do not match the saved requests/preparation')
            self.stage(row, 'review')
            review_context = {'plan': {key: value for key, value in start.items()
                                      if key not in {'analysis_sha256', 'preparation_sha256', 'context_revision'}},
                              'measurements': table(measured),
                              'data_research': select_research(self.view(), plan.model_dump(), limit=8)}
            review_context['plan']['analyses'] = [request.model_dump(exclude_defaults=True, exclude_none=True)
                                                  for request in plan.analyses]
            review = await client.ask(ResearchReview, RESEARCH_RULES +
                'Interpret ONLY completed recorded measurements from the standard analysis catalog. '
                'Failed/unavailable requests are not evidence. Respect rows, sampling, scope and metric direction. '
                'A difference on one proxy holdout does not prove noise, causality, universal feature harm or safety of removal. '
                'Do not infer a mechanism from a score change. An unmeasured ablation cannot be replaced by a correlation argument. '
                'Proxy model improvements describe only the stated inner holdout and estimator; propose an '
                'outer experiment before claiming better solution quality. Internal fit/validation must not '
                'be confused with the experiment outer validation. Correlation, MI and shift do not prove causality. '
                'A weak single feature may be useful jointly; marginal distributions do not establish class-conditional shift. '
                'Return 1-2 concise summary sentences, up to 3 genuinely new insights with exact top-level '
                'evidence_keys from measurements.values, and up to 3 unresolved questions. Zero insights is valid. '
                'Mark interpretations explicitly. Do not rephrase cached/repeated findings as new discoveries. '
                'Use supersedes/resolves only for existing IDs/questions.\n' +
                json.dumps(review_context, ensure_ascii=False, separators=(',', ':')), 'data_reviewer',
                validation={'measurements': measured['values'], 'insight_ids': [item['id'] for item in self.insights],
                            'open_questions': self.questions})
            return {'status': 'completed', 'review': review.model_dump(), 'duration_seconds': time.monotonic() - started}
        except (Exception, asyncio.CancelledError) as problem:
            save(node / 'error.json', {'error': str(problem), 'traceback': traceback.format_exc()})
            return {'status': 'cancelled' if isinstance(problem, asyncio.CancelledError) else 'failed',
                    'error': (str(problem) or 'Research cancelled').replace(self.agents.settings.api_key, '[REDACTED]'),
                    'duration_seconds': time.monotonic() - started}


    def stage(self, row, stage):
        row['stage'] = stage
        self.persist()
        self.refresh()
        self.agents.log('research_stage', research_node=row['node_id'], stage=stage)

    def collect(self, done):
        for task in [task for task in done if task in self.active]:
            row = self.active.pop(task)
            result = task.result() if not task.cancelled() else {'status': 'cancelled', 'error': 'Research cancelled'}
            row.update(stage=result['status'], **{k: v for k, v in result.items() if k != 'status'})
            if result['status'] == 'completed':
                self.failures = 0
                review = result['review']
                self.questions = list(dict.fromkeys([q for q in self.questions if q not in review['resolves']]
                                                   + review['open_questions']))
                for insight in self.insights:
                    if insight['id'] in review['supersedes']:
                        insight.update(active=False, superseded_by=row['node_id'])
                known = {(r['phase'], r['kind'], ' '.join(r['statement'].split()))
                         for r in self.insights if r['active']}
                for index, insight in enumerate(review['insights']):
                    identity = (row['phase'], insight['kind'], ' '.join(insight['statement'].split()))
                    if identity in known:
                        continue
                    known.add(identity)
                    self.insights.append({**insight, 'id': f"{row['node_id']}:{index}",
                        'node_id': row['node_id'], 'phase': row['phase'], 'active': True,
                        'evidence_file': f"research/nodes/{row['node_id']}/measurements.json"})
                self.revision += 1
            elif result['status'] == 'failed':
                self.failures += 1
                self.notify(f"{row['node_id']}: {result['error']}")
            save(self.directory / 'research' / 'nodes' / row['node_id'] / 'finish.json', result)
            self.agents.log('research_finished', research_node=row['node_id'], status=result['status'])
        self.persist()
        self.refresh()

    async def close(self, reason):
        self.stop_reason = self.stop_reason or reason
        tasks = self.tasks
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.collect(set(tasks))
        self.persist()
        self.refresh()
