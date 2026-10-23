"""Regressions reproduced against submission_u; no live API requests."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd
from openai import AsyncOpenAI

from alfa_agent.agents import JSONAgents, implementation_context
from alfa_agent.config import Settings
from alfa_agent.contracts import CodeReview, OutputBinding
from alfa_agent.data import check_code, file_hash, inventory, module
from alfa_agent.library_docs import DOCS
from alfa_agent.qwen import QwenError
from alfa_agent.worker import validate_submission
from tests.test_pipeline import PipelineTests


class LoaderAndFilesTests(unittest.TestCase):
    def test_dataclass_future_and_same_size_source_repair(self):
        code = ('from __future__ import annotations\nfrom dataclasses import dataclass\n'
                '@dataclass\nclass Predictor:\n value: float\n'
                'def fit_model(fit, preparation):\n return Predictor(1)\n')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'experiment.py'
            path.write_text(code)
            check_code(code, ('fit_model',))
            first = module(path, file_hash(path))
            self.assertIs(sys.modules[first.__name__], first)
            self.assertEqual(first.fit_model(None, None).value, 1)
            path.write_text(code.replace('Predictor(1)', 'Predictor(2)'))
            second = module(path, file_hash(path))
            self.assertNotEqual(first.__name__, second.__name__)
            self.assertEqual(second.fit_model(None, None).value, 2)
            path.write_text('raise RuntimeError("invalid generated module")')
            name = 'generated_experiment_' + file_hash(path)[:12]
            with self.assertRaises(RuntimeError):
                module(path, file_hash(path))
            self.assertNotIn(name, sys.modules)

    def test_os_path_is_allowed(self):
        check_code('import os\ndef fit_model(fit, preparation):\n return os.path.join("a", "b")', ('fit_model',))

    def test_inventory_has_no_silent_cutoff_or_reserved_data_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(170):
                (root / f'{index:03}.txt').touch()
            for directory in ('tmp', 'outputs', 'artifacts', 'current_run'):
                (root / directory).mkdir()
                (root / directory / 'train.csv').touch()
            (root / '.env').write_text('secret')
            paths = {item['path'] for item in inventory(root, exclude=(root / 'current_run',))}
            self.assertTrue({'tmp/train.csv', 'outputs/train.csv', 'artifacts/train.csv'} <= paths)
            self.assertEqual(len(paths), 173)
            self.assertNotIn('.env', paths)
            self.assertNotIn('current_run/train.csv', paths)

    def test_csv_values_and_exact_identifier_order(self):
        binding = OutputBinding(filename='submission.csv', format='CSV', columns=['id', 'prediction'])
        test = pd.DataFrame({'id': ['001', '002']})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'submission.csv'
            for rows in ['001,\n002,2\n', '999,1\n002,2\n', '002,1\n001,2\n', '1,1\n2,2\n']:
                path.write_text('id,prediction\n' + rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    validate_submission(path, binding, 2, test=test, id_columns=['id'])
            for rows in ['001,0.5\n002,-1\n', '001,Normal\n002,Attack\n', '001,NA\n002,None\n',
                         '001,nan\n002,inf\n']:  # Literal class names must not be reinterpreted as numbers.
                path.write_text('id,prediction\n' + rows)
                validate_submission(path, binding, 2, test=test, id_columns=['id'])

    def test_autogluon_weight_is_documented_as_constructor_parameter(self):
        constructor, fit = DOCS['autogluon'].split('.fit(', 1)
        self.assertIn("sample_weight='balance_weight'", constructor)
        self.assertNotIn('sample_weight=', fit)

    def test_prompt_examples_follow_actual_thread_budget(self):
        for threads in (2, 3):
            context = implementation_context({'start': {'hypothesis': {'model': 'AutoGluon CatBoost'}},
                'preparation': {}, 'runtime_limits': {'executors': 6, 'threads_per_executor': threads}})
            self.assertEqual(context['runtime_limits']['executors'], 6)
            self.assertIn(f'num_cpus={threads}', context['library_notes']['autogluon'])
            self.assertIn(f'thread_count={threads}', context['library_notes']['catboost'])
            self.assertIn(f'thread_count={threads}', context['toolkit']['example'])


class LLMRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_json_mode_and_truncation_change_request(self):
        for case in ('empty', 'truncated', 'unsupported_larger_limit'):
            bodies = []
            def handler(request):
                body = json.loads(request.content)
                bodies.append(body)
                n = len(bodies)
                if case == 'unsupported_larger_limit' and n == 2:
                    return httpx.Response(400, json={'error': {'message': 'token limit unsupported'}})
                truncated = n == 1 and case != 'empty'
                content = '' if n == 1 and case == 'empty' else '{"status":' if truncated else '{"status":"pass"}'
                return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'mock',
                    'choices': [{'index': 0, 'finish_reason': 'length' if truncated else 'stop',
                                 'message': {'role': 'assistant', 'content': content}}]})
            def client(**kwargs):
                return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            agents = JSONAgents(Settings('fake', 'http://proxy/v1', 'mock', max_tokens=1024), time.monotonic() + 10)
            with patch('alfa_agent.agents.AsyncOpenAI', side_effect=client):
                result = await agents.ask(CodeReview, 'Review', 'preparation_reviewer')
            self.assertEqual(result.status, 'pass')
            self.assertIn('response_format', bodies[0])
            self.assertNotIn('response_format', bodies[1])
            if case != 'empty':
                self.assertEqual(bodies[1]['max_tokens'], 2048)
                self.assertTrue(all(m['role'] != 'assistant' for m in bodies[1]['messages']))
            if case == 'unsupported_larger_limit':
                self.assertEqual(bodies[2]['max_tokens'], 1024)
                self.assertEqual(len(bodies), 3)

    async def test_unavailable_preparation_reviewer_keeps_valid_runtime(self):
        original = JSONAgents.review
        for error in (QwenError('unavailable fake-secret'), TimeoutError()):
            async def review(agent, context, **kwargs):
                if context['kind'] == 'preparation':
                    raise error
                return await original(agent, context, **kwargs)
            with tempfile.TemporaryDirectory() as temporary, patch.object(JSONAgents, 'review', review):
                fixture = PipelineTests('test_empty_and_truncated_llm_answers_are_repaired')
                (root, report), _ = await fixture.exercise(Path(temporary))
                self.assertEqual(report.status, 'ready', report.errors)
                self.assertEqual(report.completed_nodes, 1)
                self.assertTrue((Path(temporary) / 'result.tsv').is_file())
                self.assertTrue(any('review unavailable' in message for message in report.warnings))
                self.assertFalse(any('fake-secret' in message for message in report.warnings))
