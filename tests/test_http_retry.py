"""Offline regression coverage for the HTTP fix ported from slava."""
import asyncio
from email.utils import formatdate
import json
import time
import unittest
from unittest.mock import patch

import httpx
from openai import APIStatusError, AsyncOpenAI
from pydantic import BaseModel

from agent.agents import JSONAgents
from agent.config import Settings
from agent.qwen import QwenError, QwenTemporaryError, api_error_details


class Reply(BaseModel):
    ok: bool


class HttpRetryTests(unittest.IsolatedAsyncioTestCase):
    def agents(self):
        return JSONAgents(Settings('offline-key', 'http://llm-proxy/v1', 'model', 5, 128),
                          time.monotonic() + 30)

    def test_retry_after_formats_and_quota(self):
        for headers, expected in [({'retry-after': '7'}, 7), ({'retry-after-ms': '1500'}, 1.5),
                                  ({'retry-after': formatdate(time.time()+10, usegmt=True)}, 10)]:
            response = httpx.Response(429, headers=headers, request=httpx.Request('POST', 'http://proxy'))
            details = api_error_details(APIStatusError('hidden', response=response, body={'error': {'message': 'busy'}}))
            self.assertEqual(details['kind'], 'rate_limit')
            self.assertAlmostEqual(details['retry_after_seconds'], expected, delta=1)
        details = api_error_details(APIStatusError('hidden', response=response,
                                   body={'error': {'code': 'insufficient_quota', 'message': 'private-body'}}))
        self.assertEqual(details['kind'], 'quota_exceeded')
        self.assertNotIn('private-body', json.dumps(details))

    async def test_rate_limit_waits_and_then_succeeds(self):
        agents, seen, pauses = self.agents(), [], []

        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(429, headers={'retry-after': '2'}, json={'error': {'message': 'busy'}})
            return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 0,
                'model': 'model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '{"ok":true}'}}]})

        async def pause(delay):
            pauses.append(delay)
            agents.api_retry_at = 0

        factory = lambda **kwargs: AsyncOpenAI(**kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with patch('agent.agents.AsyncOpenAI', side_effect=factory), patch('agent.agents.asyncio.sleep', side_effect=pause):
            reply = await agents.ask(Reply, 'offline', 'metric')
        self.assertTrue(reply.ok)
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(pauses), 1)
        self.assertEqual(agents.budget['requests'], 2)

    async def test_quota_stops_other_roles_without_another_request(self):
        agents, seen = self.agents(), []
        child = JSONAgents(agents.settings, agents.deadline, shared=agents)

        def handler(request):
            seen.append(request)
            return httpx.Response(429, json={'error': {'code': 'insufficient_quota', 'message': 'private-body'}})

        factory = lambda **kwargs: AsyncOpenAI(**kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with patch('agent.agents.AsyncOpenAI', side_effect=factory):
            for role in (agents, child):
                with self.assertRaises(QwenError) as caught:
                    await role.ask(Reply, 'offline', 'metric')
                self.assertNotIn('private-body', str(caught.exception))
        self.assertEqual(len(seen), 1)
        self.assertEqual(agents.budget['requests'], 1)

    async def test_shared_cooldown_respects_deadline_without_spending_call(self):
        agents = self.agents()
        child = JSONAgents(agents.settings, agents.deadline, shared=agents)
        agents.api_retry_at = time.monotonic() + 120
        with self.assertRaises(QwenTemporaryError):
            await child.ask(Reply, 'offline', 'metric')
        self.assertEqual(agents.budget['requests'], 0)


if __name__ == '__main__':
    unittest.main()
