"""An independent research tree and evidence-backed context shared with Baker."""

import asyncio
import json
import re
import time
import traceback

from .agents import JSONAgents
from .contracts import ResearchPlan, ResearchReview
from .context import research_profile_view
from .data import check_code, file_hash
from .storage import atomic_bytes, fingerprint, save


RESEARCH_RULES = (
    'Investigate the supplied dataset to inform validation, features and model choices. '
    'Use measured evidence, not dataset-name guesses. Do not search for hidden labels. '
    'Structure phase supplies fit WITHOUT row-level targets: defer label-dependent questions to fit phase. '
    'Marginal profiles are available; investigate schema, meaning, '
    'quality, duplicates, groups and train/test shift. Fit phase supplies labelled fit only; '
    'validation/test never have targets. A date or group-like column alone does not determine the split. '
    'Do not change data, split, scorer, preparation or submissions. '
)


def question_key(phase, question):
    # Preserve numeric thresholds and comparison operators when normalizing prose.
    normalized = ' '.join(re.findall(r'\w+(?:\.\w+)*|!=|[<>=+*/%\-]+', question.casefold()))
    return fingerprint({'phase': phase, 'question': normalized})


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
        self.used = set()
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
            # A question/program and its review each need an API call. Preserve model-search headroom.
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
        context = {**self.shared, 'phase': row['phase'], 'node_id': row['node_id'],
                   'model_feedback': self.model_feedback,
                   'profile': research_profile_view(json.loads((self.directory / 'research' / 'profile.json').read_text(encoding='utf-8'))),
                   'target_access': {'fit': row['phase'] == 'fit', 'validation': False, 'test': False},
                   'focus': row['focus'],
                   'timeout_seconds': self.options.research_timeout}
        previous, error = None, None
        own_questions = set()
        try:
            for attempt in range(1, min(3, self.options.max_code_attempts) + 1):
                # Other researchers may have finished while this node was being repaired.
                context['data_research'] = self.view()
                context['history'] = [{**{k: item[k] for k in ('node_id', 'phase', 'stage', 'question', 'method')
                                         if k in item}, 'outcome': item.get('review', {}).get('outcome')}
                                      for item in self.nodes if item is not row]
                context['pending'] = [{k: item[k] for k in ('node_id', 'focus', 'question', 'method') if k in item}
                                      for item in self.active.values() if item is not row]
                request = {**context, 'previous_code': previous, 'execution_error': error}
                save(node / 'attempts' / f'{attempt:02d}' / 'request.json', request)
                plan = await client.ask(ResearchPlan, RESEARCH_RULES +
                    'Propose ONE specific question and falsifiable hypothesis, plus a short Python module run(state). '
                    'Explain which decision the answer changes. Use previous research; avoid duplicate or pending questions. '
                    'A different expected answer or reworded hypothesis does not make the same measurement new. '
                    'Compare population, columns, statistic and comparison with history and pending work. '
                    'Use already measured facts by their research IDs instead of recomputing them. '
                    'For a follow-up, name its parent and explain in method exactly which missing population, '
                    'statistic or comparison is new and why existing measurements cannot answer the question. '
                    'A new name, opposite prediction or changed conclusion alone is not a new investigation. '
                    'parent_id may reference a completed research node, or be null for an independent question. '
                    'state contains fit, validation, test, preparation, auxiliary and profile. '
                    'Use only supplied tables and installed libraries. No file access, printing, network, '
                    'subprocesses, installs, pretrained models or expensive model search. '
                    'Return a small dict of named measurements using native JSON values; avoid raw rows. '
                    'State question, hypothesis, method and decision briefly, one concrete point each. '
                    'Repair the same question after code errors; choose a new question if it is already tested. '
                    'Do not replace an unavailable measurement with a different statistic and call it equivalent.\n' +
                    json.dumps(request, ensure_ascii=False, separators=(',', ':')), 'data_researcher', code=True)
                previous = plan.code
                atomic_bytes(node / 'attempts' / f'{attempt:02d}' / 'probe.py', plan.code.encode('utf-8'))
                try:
                    check_code(plan.code, ('run',))
                    parents = {r['node_id'] for r in self.nodes if r['stage'] == 'completed'}
                    if plan.parent_id is not None and plan.parent_id not in parents:
                        raise ValueError('Research parent must be a completed node')
                    key = question_key(row['phase'], plan.question)
                    if key in self.used and key not in own_questions:
                        raise ValueError('This research question was already tested; choose a new question')
                    self.used.add(key)
                    own_questions.add(key)  # Code repairs may keep this node's own question.
                    atomic_bytes(node / 'probe.py', plan.code.encode('utf-8'))
                    start = {'node_id': row['node_id'], 'phase': row['phase'], 'context_revision': row['context_revision'],
                             **plan.model_dump(exclude={'code'}), 'code_sha256': file_hash(node / 'probe.py'),
                             'preparation_sha256': file_hash(self.directory / 'preparation.json')}
                    save(node / 'start.json', start)
                    row.update({k: start[k] for k in ('parent_id', 'question', 'hypothesis', 'method', 'decision')})
                    self.stage(row, 'execution')
                    await self.process(['-m', 'agent.research_worker', '--root', str(self.directory), '--node', str(node)],
                                       node, node / 'attempts' / f'{attempt:02d}' / 'execution.log',
                                       self.agents.deadline, self.options.research_timeout)
                    measured = json.loads((node / 'measurements.json').read_text(encoding='utf-8'))
                    if (measured['code_sha256'], measured['preparation_sha256']) != (start['code_sha256'], start['preparation_sha256']):
                        raise ValueError('Measurements do not match the recorded research program')
                    break
                except (ValueError, TimeoutError) as problem:
                    error = str(problem)[-4000:]
                    save(node / 'attempts' / f'{attempt:02d}' / 'error.json', {'error': error})
                    if attempt == min(3, self.options.max_code_attempts):
                        raise
                    self.stage(row, 'planning')
            self.stage(row, 'review')
            review_context = {'plan': start, 'code': plan.code, 'measurements': measured,
                              'data_research': self.view()}
            review = await client.ask(ResearchReview, RESEARCH_RULES +
                'Evaluate the hypothesis using ONLY recorded measurements. Distinguish observation from interpretation. '
                'Check that the executed code actually computes the claimed quantities from its inputs; '
                'hardcoded claims are not measured facts about the data. '
                'Check populations, units, denominators and input scope before interpreting comparisons. '
                'If the required labels or quantities are unavailable, mark that claim inconclusive; '
                'a proxy or a class-to-global comparison does not measure a class-to-class distribution shift. '
                'Repetition and correlation alone do not prove a sentinel, a data error or a causal mechanism. '
                'A failed univariate test or low outlier coverage does not establish absence of predictive '
                'signal or class inseparability; interactions and other distributions remain untested. '
                'Do not reverse conditional probabilities: a feature shared by all rows of one class '
                'need not identify that class among all rows with the feature. State the measured scope only. '
                'Every insight must cite exact top-level keys in measurements.values. Do not invent causes, '
                'model scores or statistical significance. Return 1-2 short summary sentences and at most '
                '3 genuinely new, decision-relevant insights; zero is valid for repeated or inconclusive work. '
                'Use one concise statement and implication per insight. Do not repeat known facts or raw tables. '
                'Give at most 3 short unresolved questions. Aim for about 450 characters in summary, '
                '350 per statement, 250 per implication and 300 per open question. '
                'These are brevity guidelines; preserve essential evidence and qualifications. '
                'Use supersedes only for existing insight IDs contradicted or refined by this evidence; '
                'use resolves to close previously open questions that these measurements answer.\n' +
                json.dumps(review_context, ensure_ascii=False, separators=(',', ':')), 'data_reviewer',
                validation={'measurements': measured['values'], 'insight_ids': [r['id'] for r in self.insights],
                            'open_questions': self.questions})
            return {'status': 'completed', 'review': review.model_dump(), 'duration_seconds': time.monotonic() - started}
        except (Exception, asyncio.CancelledError) as problem:
            save(node / 'error.json', {'error': str(problem), 'traceback': traceback.format_exc()})
            return {'status': 'cancelled' if isinstance(problem, asyncio.CancelledError) else 'failed',
                    'error': (str(problem) or 'Research cancelled').replace(self.agents.settings.api_key, '[REDACTED]')[-4000:],
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
