"""An independent research tree and evidence-backed context shared with Baker."""

import asyncio
import json
import math
import time
import traceback

from .agents import JSONAgents
from .contracts import ResearchPlan, ResearchReview
from .context import (compact, research_history, research_memory_view, research_profile_view, select_research, research_columns,
                      table, task_context)
from .data import file_hash
from .storage import fingerprint, save
from .eda import CATALOG
from .research_policy import (DIRECTIONS, choose_direction, coverage, directions, evidence_summary, importance_score,
                              measurement_key, select_candidates, select_questions)


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
        self.question_metadata = {}
        self.measurement_sources = {}
        self.model_feedback = []
        self.feature_candidates = []
        self.revision, self.failures = 0, 0
        self.stop_reason = None
        self.pause_reason = None
        self.phase_deadline = None
        # Research shares the global API budget and reserves most calls for model search.
        self.call_limit = (None if options.max_llm_calls is None else
                           int(options.max_llm_calls * options.research_call_fraction))
        self.persist()

    @property
    def tasks(self):
        return list(self.active)

    def view(self):
        memory = {'revision': self.revision,
                'insights': [{k: row[k] for k in ('id', 'node_id', 'phase', 'statement', 'kind',
                    'implication', 'importance', 'confidence', 'topic', 'direction', 'evidence') if k in row}
                             for row in self.insights if row['active']],
                'open_questions': select_questions(self.questions, self.question_metadata),
                'feature_candidates': select_candidates(self.feature_candidates),
                'coverage': coverage(self.nodes),
                'api_budget': {'limit': self.call_limit, 'spent': sum(client.requests for client in self.clients.values()),
                               'paused': self.pause_reason},
                'validation_policy': 'Research proposes changes; the fixed split is never modified during model search'}
        return select_research(memory, '')

    def report(self):
        return {'max_parallel_research': self.options.research_parallel,
                'researching_nodes': [row['node_id'] for row in self.active.values()],
                'research_nodes': len(self.nodes), 'research_revision': self.revision,
                'research_insights': sum(row['active'] for row in self.insights),
                'research_stop_reason': self.stop_reason}

    def persist(self):
        save(self.directory / 'research' / 'context.json', {
            **self.view(), 'insight_history': self.insights, 'question_history': self.question_metadata,
            'feature_candidate_history': self.feature_candidates, 'stop_reason': self.stop_reason})
        save(self.directory / 'research' / 'tree.json', {'nodes': self.nodes})
        self.shared['data_research'] = self.view()

    def fill(self, phase='fit', limit=None):
        unlimited = self.options.research_unlimited
        cap = limit if limit is not None else (math.inf if unlimited else self.options.research_limit)
        if not unlimited:
            cap = min(cap, self.options.research_limit)
        while (not self.stop_reason and len(self.active) < self.options.research_parallel
               and len(self.nodes) < cap and (unlimited or self.failures < 3)):
            spent = sum(client.requests for client in self.clients.values())
            reserved = sum(self.agents.root.call_reservations.get(row['node_id'], 0) for row in self.active.values())
            remaining = math.inf if self.call_limit is None else self.call_limit - spent - reserved
            allowance = min(6, remaining, self.agents.free_calls - 8)
            # A structured analysis plan and its review each need an API call. Preserve model-search headroom.
            if allowance < 2:
                if not self.active:
                    if remaining < 2:
                        self.stop_reason = 'research_call_limit'
                    elif self.agents.free_calls < 10:
                        self.stop_reason = 'llm_limit'
                    else:
                        self.pause_reason = 'model_reservations'
                break
            self.pause_reason = None
            node_id = f'research_{len(self.nodes):03d}'
            if not self.agents.reserve_calls(node_id, allowance):
                break
            direction, mode = choose_direction(self.nodes, phase, self.insights)
            row = {'node_id': node_id, 'phase': phase, 'stage': 'planning', 'parent_id': None,
                   'context_revision': self.revision,
                   'direction': direction, 'mode': mode, 'focus': DIRECTIONS[direction]}
            client = JSONAgents(self.agents.settings, min(self.agents.deadline, self.phase_deadline or self.agents.deadline), allowance,
                                shared=self.agents, node_id=node_id)
            self.clients[node_id] = client
            self.nodes.append(row)
            task = asyncio.create_task(self.investigate(row, client))
            self.active[task] = row
        if not unlimited and len(self.nodes) >= self.options.research_limit and not self.active:
            self.stop_reason = 'node_limit'
        elif not unlimited and self.failures >= 3:
            self.stop_reason = 'repeated_failures'
        self.persist()
        self.refresh()

    async def bootstrap(self):
        self.fill('structure', limit=self.options.research_parallel)
        while self.active:
            done, _ = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
            self.collect(done)

    async def window(self, phase, deadline, limit=None):
        """Bound initial EDA by wall time; completed evidence survives cancellation."""
        self.phase_deadline = min(deadline, self.agents.deadline)
        try:
            while time.monotonic() < self.phase_deadline:
                self.fill(phase, limit=limit)
                if not self.active:
                    break
                done, _ = await asyncio.wait(self.tasks, timeout=max(0., self.phase_deadline - time.monotonic()),
                                             return_when=asyncio.FIRST_COMPLETED)
                self.collect(done)
        finally:
            tasks = self.tasks
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.collect(set(tasks))
            self.phase_deadline = None

    async def investigate(self, row, client):
        started = time.monotonic()
        node = self.directory / 'research' / 'nodes' / row['node_id']
        node.mkdir(parents=True, exist_ok=True)
        profile = json.loads((self.directory / 'research' / 'profile.json').read_text(encoding='utf-8'))
        memory = research_memory_view(select_research(self.view(), [row['focus'], self.model_feedback], limit=5))
        columns = research_columns(profile, row['direction'], [row['focus'], memory.get('open_questions', [])],
                                   offset=len(self.nodes) * 5)
        context = {**task_context(self.shared, profile=False), 'phase': row['phase'], 'node_id': row['node_id'],
                   'model_feedback': self.model_feedback,
                   'data_research': memory,
                   'profile': research_profile_view(profile, detailed=False, focus_columns=columns),
                   'analysis_catalog': {key: value for key, value in CATALOG.items()
                       if row['phase'] == 'fit' or key in {'quality', 'correlation', 'shift', 'groups', 'redundancy', 'entities', 'queries', 'series', 'clusters'}},
                   'history': research_history([item for item in self.nodes if item is not row]),
                   'target_access': {'fit': row['phase'] == 'fit', 'validation': False, 'test': False},
                   'search': {'direction': row['direction'], 'mode': row['mode'],
                              'coverage': coverage(self.nodes), 'available_directions': DIRECTIONS},
                   'focus': row['focus'], 'timeout_seconds': max(0., client.deadline - time.monotonic())
                   if self.options.research_unlimited else self.options.research_timeout}
        context.pop('installed_libraries', None)  # AnalysisRequest already enumerates its ready estimators.
        context = compact(context)
        try:
            save(node / 'request.json', context)
            plan = await client.ask(ResearchPlan, RESEARCH_RULES +
                'Choose a specific decision-relevant question and invoke the supplied analysis catalog through analyses. '
                'Return structured requests, NEVER Python code. Each request selects kind, input columns and parameters. '
                'dataset selects fit/validation/test for descriptive checks. compare selects the second table ONLY for shift; '
                'a name containing test does not select test. Label-dependent checks always use fit. '
                'For relationships to labels use association with target specified separately; columns contains inputs, not targets. '
                'Use a few related checks instead of an exhaustive Cartesian search. Prefer existing measurements '
                'over repeating them; a changed question does not make identical measurements new. '
                'For model checks features are actually applied; baseline_features holds existing recipes. '
                'To test a new feature alone, keep raw columns and baseline_features identical and put only '
                'the new/changed recipe in features. For ablation baseline_columns is the original set and '
                'columns contains ONLY removed inputs; preserve shared recipes and leave a nonempty model. '
                'Check selected populations exist. Hidden validation/test target distributions cannot be queried. '
                'search allocates exploration versus refinement. Explore should test the assigned under-explored '
                'direction, not another small parameter variant of the current favorite. Refine should verify '
                'a decision-changing finding on comparable data or explain its boundary. Do not call a tool just '
                'to fill coverage. If the data cannot support the assigned direction, give alternative_reason '
                'with the concrete missing prerequisite and test a genuinely new alternative. A negative result '
                'for one key, feature, population or model does not rule out untested alternatives. '
                'Use feature recipes to test proposed features, ablation to test removal, interactions for joint '
                'signal missed by univariate tests, permutation for held-out importance, and stability for groups. '
                'Use validation=auto for proxy models to respect the task time/group policy. '
                'sample_rows/top_k control the computation and result size; set null when the full result is needed. '
                'Proxy scores are internal fit-only evidence, never an experiment outer score or test score. '
                'parent_id may reference a completed research node. Keep hypothesis, method and decision concise.\n' +
                json.dumps(context, ensure_ascii=False, separators=(',', ':')), 'data_researcher',
                validation={'assigned_direction': row['direction']})
            if row['direction'] not in directions(plan.analyses) and not plan.alternative_reason:
                row['direction_mismatch'] = True
                self.agents.log('research_direction_mismatch', research_node=row['node_id'],
                                assigned=row['direction'], measured=sorted(directions(plan.analyses)))
            parents = {item['node_id'] for item in self.nodes if item['stage'] == 'completed'}
            if plan.parent_id is not None and plan.parent_id not in parents:
                self.agents.log('research_parent_ignored', research_node=row['node_id'], parent_id=plan.parent_id)
                plan = plan.model_copy(update={'parent_id': None})
            requests = []
            for request in plan.analyses:
                keys = request.key_candidates or [request.columns]
                if request.kind == 'entities' and len(keys) > 1:
                    requests.extend(request.model_copy(update={'name': f'{request.name[:56]}_key_{index}',
                        'columns': key, 'key_candidates': [key]}) for index, key in enumerate(keys))
                else:
                    requests.append(request)
            unique = {}
            for request in requests:
                unique.setdefault(measurement_key(request), request)
            plan = plan.model_copy(update={'analyses': list(unique.values())})
            keys = [(row['phase'], measurement_key(request)) for request in plan.analyses]
            prior = {self.measurement_sources[key] for key in keys if key in self.measurement_sources}
            row.update(question=plan.question, alternative_reason=plan.alternative_reason, reuses=sorted(prior))
            if prior and all(key in self.measurement_sources for key in keys):
                return {'status': 'duplicate', 'reuses': sorted(prior), 'duration_seconds': time.monotonic() - started}
            # Requests are independent: execute/review only new measurements in a mixed plan.
            plan = plan.model_copy(update={'analyses': [request for request, key in zip(plan.analyses, keys)
                                                       if key not in self.measurement_sources]})
            for key in keys:
                self.measurement_sources.setdefault(key, row['node_id'])
            start = {'node_id': row['node_id'], 'phase': row['phase'], 'context_revision': self.revision,
                     **plan.model_dump(), 'reuses': sorted(prior),
                     'task_type': self.shared.get('requirements', {}).get('problem', {}).get('task_type'),
                     'preparation_sha256': file_hash(self.directory / 'preparation.json')}
            start['analysis_sha256'] = fingerprint(start['analyses'])
            save(node / 'start.json', start)
            row.update({key: start[key] for key in ('parent_id', 'question', 'hypothesis', 'method', 'decision', 'analyses', 'alternative_reason')})
            self.stage(row, 'execution')
            try:
                await self.process(['-m', 'agent.research_worker', '--root', str(self.directory), '--node', str(node)],
                                   node, node / 'execution.log', min(self.agents.deadline, self.phase_deadline or self.agents.deadline),
                                   max(0., client.deadline - time.monotonic()) if self.options.research_unlimited else self.options.research_timeout)
            except TimeoutError:
                if not (node / 'measurements.json').exists():
                    raise
            measured = json.loads((node / 'measurements.json').read_text(encoding='utf-8'))
            for value in measured['values'].values():
                if isinstance(value, dict) and value.get('status') == 'failed':
                    self.agents.observe_error('data_researcher', value.get('error', ''))
            if (measured['analysis_sha256'], measured['preparation_sha256']) != (start['analysis_sha256'], start['preparation_sha256']):
                raise ValueError('Measurements do not match the saved requests/preparation')
            self.stage(row, 'review')
            review_context = {'plan': {key: value for key, value in start.items()
                                      if key not in {'analysis_sha256', 'preparation_sha256', 'context_revision'}},
                              'measurements': table(measured),
                              'data_research': research_memory_view(select_research(self.view(), plan.model_dump(), limit=4), review=True)}
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
                'Assess importance by the modeling decision affected, and confidence by the actual evidence. '
                'critical is reserved for invalid task/validation assumptions; high changes a major choice; '
                'medium is a useful refinement; low is incidental. Do not rate every result high. A large '
                'gain on a tiny or filtered proxy subset is not automatically important for the full task. '
                'Give a stable topic per claim. State one measured conclusion and one concrete implication '
                'per insight, with population/estimator limitations. Do not generalize a failed single-key '
                'test to all possible entity keys, or a proxy result to test performance. '
                'feature_warnings reports constant/all-missing generated inputs on its stated scope. Do not attribute '
                'signal to an all-missing feature or isolate its effect when other recipes were added simultaneously. '
                'Temporal target_encoding uses strictly earlier fit periods and ignores folds. '
                'Read actual feature_columns, recipe_names and inner_split_sha256: describe the executed '
                'comparison, not its requested name. Multiple simultaneous changes do not isolate their '
                'individual effects. Arbitrary gain/coverage thresholds do not prove a method useless. '
                'A negative result for one key or proxy estimator cannot exclude simpler keys or other '
                'estimators. Distribution shift alone does not establish feature harm, a prediction ceiling '
                'or an optimal strategy. Mark such claims as unverified questions, not observations. '
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
        finally:
            # A timeout can leave only part of a plan measured. Do not reserve the unfinished checks forever.
            try:
                values = json.loads((node / 'measurements.json').read_text(encoding='utf-8'))['values']
            except (OSError, json.JSONDecodeError):
                values = {}
            completed = {(row['phase'], measurement_key(request)) for request in row.get('analyses', [])
                         if request['name'] in values}
            self.measurement_sources = {key: source for key, source in self.measurement_sources.items()
                                        if source != row['node_id'] or key in completed}


    def stage(self, row, stage):
        row['stage'] = stage
        self.persist()
        self.refresh()
        self.agents.log('research_stage', research_node=row['node_id'], stage=stage)

    def collect(self, done):
        for task in [task for task in done if task in self.active]:
            row = self.active.pop(task)
            self.agents.release_calls(row['node_id'])
            try:
                result = task.result()
            except (Exception, asyncio.CancelledError) as error:
                result = {'status': 'cancelled' if isinstance(error, asyncio.CancelledError) else 'failed',
                          'error': (str(error) or 'Research cancelled').replace(self.agents.settings.api_key, '[REDACTED]')}
            row.update(stage=result['status'], **{k: v for k, v in result.items() if k != 'status'})
            if result['status'] == 'completed':
                self.failures = 0
                review = result['review']
                path = self.directory / 'research' / 'nodes' / row['node_id'] / 'measurements.json'
                measured = json.loads(path.read_text(encoding='utf-8'))['values'] if path.exists() else {}
                successful = [request for request in row.get('analyses', []) if request['name'] in measured
                    and (not isinstance(measured[request['name']], dict) or
                         measured[request['name']].get('status') not in {'failed', 'unavailable'})]
                row['measured_directions'] = sorted(directions(successful))
                for request in row.get('analyses', []):
                    facts = measured.get(request['name'], {})
                    gain = facts.get('improvement') if isinstance(facts, dict) else None
                    if request['kind'] not in {'feature', 'ablation'} or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
                        continue
                    supporting = [item for item in review['insights'] if request['name'] in item['evidence_keys']]
                    assessment = max(supporting, key=importance_score) if supporting else {}
                    recipes = {recipe['name']: recipe for recipe in [*request.get('baseline_features', []), *request['features']]}
                    candidate = {'research_node': row['node_id'], 'features': list(recipes.values()),
                        'columns': request.get('baseline_columns') or request.get('columns', []),
                        'metric': facts['candidate']['metric'], 'improvement': gain,
                        'relative_improvement': gain / max(abs(facts['baseline']['value']), 1e-12),
                        'rows': facts.get('rows_used'), 'validation_rows': facts['candidate'].get('inner_validation_rows', 0),
                        'filters': request.get('filters', []), 'validation': request.get('validation'),
                        'importance': assessment.get('importance', 'medium'),
                        'confidence': assessment.get('confidence', 'medium'),
                        'scope': 'Internal proxy holdout; not outer-validation evidence'}
                    identity = measurement_key(request)
                    candidate['measurement_key'] = identity
                    configuration = facts['candidate'].get('recipe_configuration')
                    targets = self.shared.get('preparation', {}).get('data', {}).get('target_columns')
                    if configuration and not request.get('filters') and configuration['target_columns'] == targets:
                        from .builds import register_eda
                        candidate['build_id'] = 'eda_' + identity[:16]
                        register_eda(self.directory, candidate['build_id'], configuration,
                            {'research_node': row['node_id'], 'measurement': request['name'],
                             'metric': candidate['metric'], 'proxy_score': facts['candidate']['value'],
                             'improvement': gain, 'inner_split_sha256': facts['candidate']['inner_split_sha256']})
                    if request['kind'] == 'ablation' and not candidate.get('build_id'):
                        continue
                    self.feature_candidates = [item for item in self.feature_candidates if item.get('measurement_key') != identity]
                    self.feature_candidates.append(candidate)
                self.questions = list(dict.fromkeys([q for q in self.questions if q not in review['resolves']]
                                                   + review['open_questions']))
                importance = max(review['insights'], key=importance_score).get('importance', 'medium') if review['insights'] else 'medium'
                for question in review['open_questions']:
                    self.question_metadata[question] = {'importance': importance, 'direction': row.get('direction'),
                                                        'research_node': row['node_id']}
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
                    support = [request for request in successful if request['name'] in insight['evidence_keys']]
                    families = directions(support)
                    direction = row.get('direction') if row.get('direction') in families else next(iter(sorted(families)), None)
                    self.insights.append({**insight, 'id': f"{row['node_id']}:{index}",
                        'node_id': row['node_id'], 'phase': row['phase'], 'active': True,
                        'direction': direction,
                        'evidence': {key: evidence_summary(measured[key]) for key in insight['evidence_keys'] if key in measured},
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
