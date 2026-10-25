"""Full-controller fault injection: no paid API, real generated workers and output files."""
import asyncio
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from openai import AsyncOpenAI
from alfa_agent import agents, runner
from alfa_agent.config import Settings
from alfa_agent.contracts import CodeReview
from alfa_agent.qwen import QwenError
from tests.test_pipeline import fixture, PROGRAM

class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, folder, case):
        task, answers = fixture(folder)
        sent = []
        def handler(req):
            body = json.loads(req.content)
            title = json.loads(body['messages'][0]['content'].split('Schema: ', 1)[1])['title']
            sent.append(title)
            ctx = json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1]) if title in ('Hypotheses', 'Program') else {}
            if title == 'Program' and case == 'baseline_repair_http401' and ctx['start']['node_id'] == 'node_000':
                return httpx.Response(401, json={'error': {'message': 'injected'}})
            answer = copy.deepcopy(answers.get(title, {}))
            if title == 'Hypotheses':
                answer = {'hypotheses': []}
                for i, purpose in enumerate(ctx['requested_purposes']):
                    h = copy.deepcopy(answers['Hypothesis'])
                    h['pipeline']['model']['strategy'] = 'median' if i == 0 else 'mean_plus_one'
                    answer['hypotheses'].append({'purpose': purpose, 'parent_id': ctx['parent']['node_id'], 'hypothesis': h})
            if title == 'Program' and ctx['start']['node_id'] != 'node_000':
                strategy = ctx['start']['hypothesis']['pipeline']['model']['strategy']
                answer['code'] = PROGRAM.replace('fit[column].mean()', 'fit[column].median()' if strategy == 'median' else 'fit[column].mean() + 1')
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'mock',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(answer)}}]})
        def client(**kw):
            return AsyncOpenAI(**kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        original_create, original_wait, original_process = asyncio.create_task, asyncio.wait, runner.process
        injected = False
        def create(coro, **kw):
            start = coro.cr_frame.f_locals.get('start') if getattr(coro, 'cr_frame', None) else None
            t = original_create(coro, **kw)
            if getattr(coro, 'cr_code', None) and coro.cr_code.co_name == 'experiment' and case == 'cancel_before_start' and start.node_id == 'node_001':
                t.cancel()
            return t
        async def wait(tasks, **kw):
            nonlocal injected
            done, pending = await original_wait(tasks, **kw)
            if not injected and case in ('controller_error', 'controller_cancel', 'cleanup_recovers_winner'):
                ready = [t for t in done if t.get_coro().cr_code.co_name == 'experiment' and not t.cancelled() and t.exception() is None]
                if ready and ready[0].result().status == 'completed':
                    # Controller fails before collecting the completed baseline: only cleanup can publish it.
                    injected = True
                    if case == 'controller_cancel':
                        raise asyncio.CancelledError('injected controller cancellation')
                    raise RuntimeError('injected controller error')
            return done, pending
        async def process(arguments, *args):
            if '--refit' in arguments:
                if case == 'refit_error':
                    raise RuntimeError('injected refit failure')
                if case == 'refit_cancel':
                    raise asyncio.CancelledError('injected refit cancellation')
            return await original_process(arguments, *args)
        def template(preparation, threads):
            return ('def fit_model(fit, preparation):\n raise ValueError("injected controller fit error")', answers['Hypothesis'])
        with patch.object(agents, 'AsyncOpenAI', side_effect=client), patch.object(asyncio, 'create_task', side_effect=create), \
             patch.object(asyncio, 'wait', side_effect=wait), patch.object(runner, 'process', side_effect=process), \
             patch.object(runner, 'template', side_effect=template):
            root, report = await runner.run(task, runner.Options(time_budget=40, experiment_timeout=8, max_nodes=3,
                executors=2, autopilot=case == 'baseline_repair_http401', output_dir=folder/'artifacts'),
                Settings('fake', 'http://mock/v1', 'mock'))
        return root, report, sent

    async def test_cancel_before_start_is_one_failed_node_and_finalizes_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report, _ = await self.exercise(Path(tmp), 'cancel_before_start')
            self.assertEqual(report.stage, 'complete')
            self.assertEqual(report.failed_nodes, 1)
            self.assertEqual(json.loads((root/'nodes/node_001/finish.json').read_text())['status'], 'failed')
            self.assertEqual(json.loads((root/'finalization.json').read_text())['status'], 'completed')
            self.assertTrue((Path(tmp)/'result.tsv').exists())
            self.assertEqual(report.running_nodes, [])
            self.assertFalse(report.planning)

    async def test_controller_failure_and_cancellation_recover_uncollected_winner(self):
        for case in ('controller_error', 'controller_cancel'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root, report, _ = await self.exercise(Path(tmp), case)
                self.assertEqual((report.status, report.stage, report.completed_nodes), ('partial', 'complete', 1))
                self.assertTrue(report.errors)
                self.assertEqual(json.loads((root/'finalization.json').read_text())['status'], 'completed')
                self.assertTrue((Path(tmp)/'result.tsv').exists())
                self.assertEqual(report.running_nodes, [])

    async def test_initial_baseline_llm_error_can_start_new_baseline_without_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, report, sent = await self.exercise(Path(tmp), 'baseline_repair_http401')
            self.assertIn('Hypothesis', sent)
            self.assertTrue((Path(tmp)/'result.tsv').exists(), report.errors)
            self.assertEqual(report.stage, 'complete')
            self.assertGreaterEqual(report.completed_nodes, 1)
            self.assertEqual(json.loads((root/'finalization.json').read_text())['status'], 'completed')
            self.assertNotIn('', report.errors)

    async def test_refit_exception_or_cancellation_preserves_valid_output(self):
        for case in ('refit_error', 'refit_cancel'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root, report, _ = await self.exercise(Path(tmp), case)
                self.assertEqual(report.stage, 'complete')
                self.assertNotEqual(report.status, 'failed')
                self.assertTrue((Path(tmp)/'result.tsv').exists())
                self.assertEqual(json.loads((root/'finalization.json').read_text())['status'], 'failed')
                self.assertTrue(any('Final refit failed' in x for x in report.warnings))

class APIRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_quota_recovers_or_stops_shared_clients_after_three_attempts(self):
        for failures in (2, 99):
            requests, sleeps = [], []
            def handler(req):
                requests.append(1)
                if len(requests) <= failures:
                    return httpx.Response(429, json={'error': {'type': 'insufficient_quota', 'message': 'Budget has been exceeded'}})
                return httpx.Response(200, json={'id':'test', 'object':'chat.completion', 'created':0, 'model':'mock',
                    'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{"status":"pass"}'}}]})
            def client(**kw):
                return AsyncOpenAI(**kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            async def sleep(seconds): sleeps.append(seconds)
            agent = agents.JSONAgents(Settings('fake', 'http://mock/v1', 'mock'), time.monotonic()+60)
            with patch.object(agents, 'AsyncOpenAI', side_effect=client), patch.object(asyncio, 'sleep', side_effect=sleep):
                if failures == 2:
                    result = await agent.ask(CodeReview, 'Review', 'reviewer')
                    self.assertEqual(result.status, 'pass')
                    self.assertFalse(agent.budget.get('api_error'))
                else:
                    with self.assertRaises(QwenError): await agent.ask(CodeReview, 'Review', 'reviewer')
                    shared = agents.JSONAgents(agent.settings, agent.deadline, shared=agent)
                    with self.assertRaises(QwenError): await shared.ask(CodeReview, 'Review', 'reviewer')
                    self.assertTrue(agent.budget['api_error'])
            self.assertEqual(len(requests), 3)
            self.assertEqual(sleeps, [2, 4])

    async def test_temporary_rate_limit_is_bounded_and_does_not_block_future_calls(self):
        requests = []
        def handler(req):
            requests.append(1)
            return httpx.Response(429, json={'error': {'message': 'Too many requests'}}, headers={'Retry-After':'1'})
        def client(**kw):
            return AsyncOpenAI(**kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        async def sleep(seconds): self.assertEqual(seconds, 1)
        agent = agents.JSONAgents(Settings('fake', 'http://mock/v1', 'mock'), time.monotonic()+60)
        with patch.object(agents, 'AsyncOpenAI', side_effect=client), patch.object(asyncio, 'sleep', side_effect=sleep):
            with self.assertRaises(QwenError): await agent.ask(CodeReview, 'Review', 'reviewer')
        self.assertEqual(len(requests), 3)
        self.assertFalse(agent.budget.get('api_error'))

class DeadlineAndExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_whole_request_timeout_and_gate_deadline(self):
        for blocked_gate in (False, True):
            with self.subTest(blocked_gate=blocked_gate):
                started_requests = []
                async def handler(req):
                    started_requests.append(req)
                    # MockTransport has no socket inactivity timeout: only the whole-request deadline can stop it.
                    await asyncio.sleep(5)
                    raise AssertionError('Request should have been cancelled')
                def client(**kw):
                    return AsyncOpenAI(**kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
                agent = agents.JSONAgents(Settings('fake', 'http://mock/v1', 'mock', timeout_seconds=.05),
                    time.monotonic()+.12, concurrency=1)
                if blocked_gate:
                    await agent.gate.acquire()
                began = time.monotonic()
                with patch.object(agents, 'AsyncOpenAI', side_effect=client):
                    with self.assertRaises(QwenError):
                        await agent.ask(CodeReview, 'Review', 'reviewer', attempts=1)
                self.assertLess(time.monotonic()-began, .8)
                self.assertEqual(len(started_requests), 0 if blocked_gate else 1)
                self.assertTrue(agent.calls[-1].error.startswith('таймаут'))
                if blocked_gate:
                    agent.gate.release()
                self.assertFalse(agent.gate.locked())

    async def test_cancelled_extraction_role_produces_completed_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            task, answers = fixture(folder)
            agent = agents.JSONAgents(Settings('fake', 'http://mock/v1', 'mock'), time.monotonic()+5)
            async def ask(schema, prompt, role, **kw):
                if role == 'problem':
                    raise asyncio.CancelledError('injected role cancellation')
                return schema.model_validate(answers[schema.__name__])
            with patch.object(agent, 'ask', side_effect=ask):
                report = await agents.extract_task(task, folder/'run', agent.settings, agents=agent)
            saved = json.loads((folder/'run/report.json').read_text())
            self.assertEqual((report.status, report.stage), ('failed', 'complete'))
            self.assertIn('problem: CancelledError', saved['errors'][0])
            self.assertEqual(saved['stop_reason'], 'llm_error')

    async def test_cancelled_extraction_controller_drains_roles_and_saves_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            task, _ = fixture(folder)
            agent = agents.JSONAgents(Settings('fake', 'http://mock/v1', 'mock'), time.monotonic()+5)
            started, drained, ready = [], [], asyncio.Event()
            async def ask(schema, prompt, role, **kw):
                started.append(role)
                if len(started) == 4:
                    ready.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.append(role)
            with patch.object(agent, 'ask', side_effect=ask):
                controller = asyncio.create_task(agents.extract_task(task, folder/'run', agent.settings, agents=agent))
                await asyncio.wait_for(ready.wait(), 2)
                controller.cancel()
                report = await controller
            self.assertEqual(sorted(started), sorted(drained))
            self.assertEqual((report.status, report.stage), ('failed', 'complete'))
            self.assertIn('CancelledError', report.errors[0])
            self.assertEqual(json.loads((folder/'run/report.json').read_text())['status'], 'failed')
