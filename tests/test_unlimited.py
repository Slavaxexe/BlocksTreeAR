"""Unbounded resource budgets must work without Infinity in saved contracts."""
import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI
from pydantic import TypeAdapter

from agent.__main__ import main
from agent.agents import JSONAgents
from agent.config import Settings
from agent.contracts import NodeID, ProblemFacts, RunReport
from agent.runner import Options, process, remaining_seconds, run


class UnlimitedTests(unittest.IsolatedAsyncioTestCase):
    def test_node_ids_allow_more_than_one_thousand_experiments(self):
        self.assertEqual(TypeAdapter(NodeID).validate_python('node_1000'), 'node_1000')

    def test_zero_environment_limits_disable_client_caps(self):
        settings = Settings.from_env(Path('missing.env'), {'OPENAI_API_KEY': 'fake',
            'OPENAI_BASE_URL': 'https://example.test/v1', 'OPENAI_MODEL': 'model',
            'QWEN_TIMEOUT_SECONDS': '0', 'QWEN_MAX_TOKENS': '0'})
        settings.validate()
        self.assertIsNone(settings.timeout_seconds)
        self.assertIsNone(settings.max_tokens)
        self.assertIsNone(remaining_seconds(math.inf))

    def test_cli_defaults_are_unlimited_and_baseline_still_means_one_node(self):
        for command, expected_nodes in [('search', 0), ('baseline', 1)]:
            settings = Settings('fake', 'https://example.test/v1', 'model')
            mocked = AsyncMock(return_value=(Path('artifacts/test'), RunReport(status='ready', task_sha256='0' * 64)))
            with patch('agent.__main__.Settings.from_env', return_value=settings), \
                 patch('agent.runner.run', mocked), redirect_stdout(io.StringIO()):
                main([command])
            options = mocked.call_args.args[1]
            self.assertEqual(options.max_experiments, expected_nodes)
            self.assertEqual((options.time_budget, options.experiment_timeout,
                              options.max_llm_calls, options.max_code_attempts), (0, 0, 0, 0))
            self.assertEqual((options.parallel, options.parallel_agents), (4, 16))

    async def test_unlimited_api_request_and_diagnostics_contain_no_infinity(self):
        settings = Settings('fake', 'https://example.test/v1', 'model')
        def handler(request):
            body = json.loads(request.content)
            self.assertNotIn('max_tokens', body)
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
                'created': 0, 'model': 'model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': '{"evidence":{}}'}}]})
        def client(**kwargs):
            self.assertIsNone(kwargs['timeout'])
            return AsyncOpenAI(api_key='fake', base_url=settings.base_url, max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with tempfile.TemporaryDirectory() as temporary:
            agents = JSONAgents(settings, math.inf, max_calls=0)
            agents.log_directory = Path(temporary)
            with patch('agent.agents.AsyncOpenAI', side_effect=client):
                await agents.ask(ProblemFacts, 'Extract.', 'problem')
            self.assertTrue(math.isinf(agents.remaining_calls))
            for path in Path(temporary).rglob('*.json'):
                raw = path.read_text()
                self.assertNotIn('Infinity', raw)
                json.loads(raw)

    async def test_unlimited_child_process_has_no_wait_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            await process(['-c', 'print("completed")'], root, root / 'child.log', math.inf, None)
            self.assertIn('completed', (root / 'child.log').read_text())

    async def test_large_explicit_budgets_are_not_rejected_by_old_hard_caps(self):
        options = Options(time_budget=20000, experiment_timeout=20000,
                          max_experiments=2001, max_llm_calls=20000)
        with tempfile.TemporaryDirectory() as temporary:
            options = replace(options, output_dir=Path(temporary) / 'run')
            with patch('agent.runner.extract_task', AsyncMock(return_value=RunReport(status='failed', task_sha256='0' * 64))):
                _, report = await run(Path(temporary) / 'task.md', options,
                                     Settings('fake', 'https://example.test/v1', 'model'))
            self.assertEqual(report.status, 'failed')
