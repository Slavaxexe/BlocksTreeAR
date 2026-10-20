"""Duplicate-plan replacement through mocked HTTP, without external model requests."""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from agent.agents import JSONAgents, plan_key
from agent.config import Settings
from agent.contracts import Proposal


def plan(trees=100, purpose='improve', rationale='Measure the proposed tree count'):
    return {'purpose': purpose, 'parent_id': 'node_004',
            'hypothesis': {'rationale': rationale, 'expected_effect': 'Evaluate the new setting',
                'pipeline': {'data': {}, 'preprocessing': {},
                             'model': {'type': 'random_forest', 'n_estimators': trees}, 'training': {}}},
            'assembly': {'edit_block': 'model', 'blocks': [{
                'name': 'model', 'source_node': None, 'description': 'Fit a forest and predict',
                'requires': ['fit', 'validation', 'test', 'preparation'],
                'provides': ['validation_predictions', 'test_predictions']}]}}


class PlanningTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, replies, *, purposes=None, old=True, max_calls=10):
        contexts = []
        def handler(request):
            body = json.loads(request.content)
            contexts.append(json.loads(body['messages'][1]['content'].rsplit('\n', 1)[1]))
            reply = replies[min(len(contexts) - 1, len(replies) - 1)]
            if isinstance(reply, int):
                return httpx.Response(reply, json={'error': {'message': 'temporary failure'}})
            return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion', 'created': 0,
                'model': 'model', 'choices': [{'index': 0, 'message': {'role': 'assistant',
                'content': json.dumps({'proposals': reply})}, 'finish_reason': 'stop'}]})
        def client(**kwargs):
            return AsyncOpenAI(api_key='fake', base_url='https://qwen.example/v1', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        purposes = purposes or ['improve']
        context = {'count': len(purposes), 'purposes': purposes, 'leader_id': 'node_004',
                   'available_parents': ['node_004'], 'module_catalog': {},
                   'used': {plan_key(Proposal(**plan())): 'node_004'} if old else {}}
        agents = JSONAgents(Settings('fake', 'https://qwen.example/v1', 'model'), time.monotonic()+10, max_calls)
        with tempfile.TemporaryDirectory() as temporary, patch('agent.agents.AsyncOpenAI', side_effect=client):
            result = await agents.propose(context, log_directory=Path(temporary))
            logs = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(Path(temporary).glob('*.json'))]
        return result, contexts, logs, agents

    async def test_duplicate_feedback_identifies_node_and_archives_rejected_plan(self):
        result, contexts, logs, _ = await self.exercise([[plan(rationale='A different explanation')], [plan(200)]])
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.proposals[0].hypothesis.pipeline.model['n_estimators'], 200)
        feedback = contexts[1]['rejected_plans'][0]
        self.assertEqual(feedback['duplicate_of'], 'node_004')
        self.assertEqual(feedback['proposal']['hypothesis']['pipeline']['model']['n_estimators'], 100)
        self.assertEqual(logs[0]['proposals'][0]['hypothesis']['rationale'], 'A different explanation')
        self.assertEqual(logs[0]['rejected'][0]['duplicate_of'], 'node_004')
        self.assertNotIn('used', contexts[0])

    async def test_keeps_valid_sibling_and_requests_only_the_rejected_role(self):
        result, contexts, _, _ = await self.exercise(
            [[plan(), plan(300, 'explore')], [plan(200)]], purposes=['improve', 'explore'])
        self.assertEqual(contexts[1]['purposes'], ['improve'])
        self.assertEqual(contexts[1]['count'], 1)
        self.assertEqual(len(contexts[1]['accepted_proposals']), 1)
        self.assertEqual({p.hypothesis.pipeline.model['n_estimators'] for p in result.proposals}, {200, 300})

    async def test_duplicate_within_one_batch_ignores_changed_rationale(self):
        result, contexts, _, _ = await self.exercise(
            [[plan(), plan(purpose='explore', rationale='Same pipeline in other words')], [plan(300, 'explore')]],
            purposes=['improve', 'explore'], old=False)
        self.assertEqual(contexts[1]['rejected_plans'][0]['duplicate_of'], 'accepted_proposal_1')
        self.assertEqual(len(result.proposals), 2)

    async def test_repeated_duplicates_return_empty_batch_without_fatal_error(self):
        result, contexts, logs, _ = await self.exercise([[plan()]])
        self.assertEqual(result.proposals, [])
        self.assertEqual(len(contexts), 2)
        self.assertEqual(len(logs), 2)
        self.assertIsNone(result.error)

    async def test_call_limit_prevents_a_replacement_request(self):
        result, contexts, _, agents = await self.exercise([[plan()]], max_calls=1)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(result.proposals, [])
        self.assertEqual(agents.remaining_calls, 0)

    async def test_failed_replacement_does_not_discard_an_accepted_sibling(self):
        result, _, logs, _ = await self.exercise([[plan(), plan(300, 'explore')], 503], purposes=['improve', 'explore'])
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.proposals[0].purpose, 'explore')
        self.assertIn('HTTP 503', result.error)
        self.assertIn('error', logs[-1])
