"""Offline tests for model configuration and HTTP requests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agent.__main__ import main
from agent.config import ConfigurationError, Settings
from agent.llm import LLMError, ask_llm


class ConfigurationTests(unittest.TestCase):
    def test_dotenv_and_environment_precedence_without_mutating_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "OPENAI_API_KEY=test-secret\nOPENAI_BASE_URL=https://qwen.example/v1\n"
                "OPENAI_MODEL=file-model\n", encoding="utf-8"
            )
            original = dict(os.environ)
            config = Settings.from_env(path, {"OPENAI_MODEL": "environment-model"})
            config.validate()
            self.assertEqual(config.model, "environment-model")
            self.assertEqual(dict(os.environ), original)
            self.assertNotIn("test-secret", repr(config))
            self.assertNotIn("test-secret", json.dumps(config.public_status()))

    def test_generic_settings_and_legacy_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('LLM_TIMEOUT_SECONDS=400\nLLM_MAX_TOKENS=100\n', encoding='utf-8')
            settings = Settings.from_env(path, {'QWEN_TIMEOUT_SECONDS': '25',
                'LLM_MAX_TOKENS': '200', 'QWEN_MAX_TOKENS': '300',
                'LLM_API_PROFILE': 'qwen', 'LLM_TOOLS_ENABLED': 'false'})
            self.assertEqual((settings.timeout_seconds, settings.max_tokens), (25, 200))
            self.assertEqual(settings.api_profile, 'qwen')
            self.assertFalse(settings.tools_enabled)
            defaults = Settings.from_env(Path(directory) / 'missing', {})
            self.assertEqual(defaults.api_profile, 'standard')
            self.assertEqual(defaults.reasoning_options(True), {})

    def test_missing_endpoint_is_rejected_instead_of_using_default_openai_host(self):
        with self.assertRaisesRegex(ConfigurationError, "OPENAI_BASE_URL"):
            Settings("test-secret", "", "qwen-model").validate()

    def test_http_internal_addresses_and_url_components_are_allowed(self):
        for url in ["http://llm-proxy:8000/v1", "http://10.0.0.5:8000/v1",
                    "http://remote.example/v1", "https://user:secret@qwen.example/v1",
                    "https://qwen.example/v1?token=placeholder", "https://qwen.example/v1#section"]:
            with self.subTest(url=url):
                Settings("test-secret", url, "qwen-model").validate()

    def test_invalid_endpoint_format_is_rejected(self):
        for url in ["not-a-url", "ftp://proxy/v1", "http://proxy:invalid/v1", "http:///v1"]:
            with self.subTest(url=url), self.assertRaises(ConfigurationError):
                Settings("test-secret", url, "qwen-model").validate()

    def test_invalid_numeric_limits_are_rejected(self):
        for timeout, tokens in [(0, 10), (float("nan"), 10), (float("inf"), 10), (10, 0)]:
            with self.subTest(timeout=timeout, tokens=tokens), self.assertRaises(ConfigurationError):
                Settings("test-secret", "https://qwen.example/v1", "qwen-model", timeout, tokens).validate()


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings("test-secret", "https://qwen.example/v1", "qwen-model", 5, 128)

    def test_existing_run_is_not_overwritten_and_cli_has_no_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / 'task.md'
            task.write_text('A task')
            output = root / 'run'
            output.mkdir()
            marker = output / 'report.json'
            marker.write_text('{"status":"keep"}')
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch('agent.__main__.Settings.from_env', return_value=self.settings), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                result = main(['extract', '--task', str(task), '--output-dir', str(output)])
            self.assertEqual(result, 2)
            self.assertEqual(marker.read_text(), '{"status":"keep"}')
            self.assertNotIn('Traceback', stderr.getvalue())
            self.assertNotIn('test-secret', stdout.getvalue()+stderr.getvalue())

    def test_chat_request_uses_explicit_host_model_auth_and_token_limit(self):
        seen = []

        def handler(request):
            seen.append(request)
            self.assertEqual(str(request.url), "https://qwen.example/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-secret")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "qwen-model")
            self.assertEqual(body["max_tokens"], 128)
            self.assertEqual(body["messages"][-1], {"role": "user", "content": "hello"})
            return httpx.Response(200, json={
                "id": "chat-test", "object": "chat.completion", "created": 0,
                "model": "qwen-model", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "готов"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            })

        answer = ask_llm(self.settings, "hello", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertEqual(len(seen), 1)
        self.assertEqual(answer.text, "готов")
        self.assertEqual(answer.usage["total_tokens"], 4)

    def test_http_errors_are_not_retried_and_do_not_echo_provider_body(self):
        for code in [401, 429, 500]:
            seen = []

            def handler(request):
                seen.append(request)
                return httpx.Response(code, json={"error": {"message": "test-secret", "type": "test"}})

            with self.subTest(status=code):
                with self.assertRaises(LLMError) as caught:
                    ask_llm(self.settings, "hello", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
                self.assertIn(str(code), str(caught.exception))
                self.assertNotIn("test-secret", str(caught.exception))
                self.assertEqual(len(seen), 1)

    def test_timeout_has_a_controlled_error(self):
        def handler(request):
            raise httpx.ReadTimeout("test-secret", request=request)

        with self.assertRaises(LLMError) as caught:
            ask_llm(self.settings, "hello", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertIn("таймаута", str(caught.exception))
        self.assertNotIn("test-secret", str(caught.exception))

    def test_empty_response_is_not_reported_as_success(self):
        def handler(request):
            return httpx.Response(200, json={"id": "chat-test", "object": "chat.completion",
                "created": 0, "model": "qwen-model", "choices": []})

        with self.assertRaises(LLMError):
            ask_llm(self.settings, "hello", http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_check_config_is_offline_and_hides_the_key(self):
        out = io.StringIO()
        config = Settings("test-secret", "", "qwen-model")
        with patch("agent.__main__.Settings.from_env", return_value=config), \
             patch("agent.__main__.ask_llm") as ask, redirect_stdout(out):
            result = main(["check-config"])
        self.assertEqual(result, 2)
        self.assertFalse(json.loads(out.getvalue())["ready"])
        self.assertNotIn("test-secret", out.getvalue())
        ask.assert_not_called()

    def test_cli_handles_provider_failure_without_traceback_or_secret(self):
        err = io.StringIO()
        with patch("agent.__main__.Settings.from_env", return_value=self.settings), \
             patch("agent.__main__.ask_llm", side_effect=LLMError("HTTP 401 test-secret")), \
             redirect_stderr(err):
            result = main(["ping"])
        self.assertEqual(result, 2)
        self.assertIn("HTTP 401", err.getvalue())
        self.assertNotIn("test-secret", err.getvalue())


class ModelRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_and_qwen_profiles_and_optional_tools(self):
        import time
        from openai import AsyncOpenAI
        from agent.agents import JSONAgents
        from agent.contracts import Analysis

        for profile, tools in [('standard', False), ('standard', True), ('qwen', True)]:
            with self.subTest(profile=profile, tools=tools):
                requests = []
                def handler(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    if profile == 'standard':
                        self.assertLessEqual(set(body), {'model', 'messages', 'tools', 'tool_choice', 'max_tokens'})
                    return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion',
                        'created': 0, 'model': 'chosen-model', 'choices': [{'index': 0,
                        'message': {'role': 'assistant', 'content': '{"summary":"Measured result","findings":[],"next_ideas":[]}'},
                        'finish_reason': 'stop'}]})
                def client(**kwargs):
                    return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
                settings = Settings('fake', 'https://model.example/v1', 'chosen-model', max_tokens=128,
                                    api_profile=profile, tools_enabled=tools)
                with patch('agent.agents.AsyncOpenAI', side_effect=client):
                    result = await JSONAgents(settings, time.monotonic() + 10, 2).ask(
                        Analysis, 'Review measured facts', 'analyzer', reference=True)
                self.assertEqual(result.summary, 'Measured result')
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]['model'], 'chosen-model')
                self.assertEqual(requests[0]['max_tokens'], 128)
                self.assertEqual('tools' in requests[0], tools)
                if profile == 'qwen':
                    self.assertTrue(requests[0]['enable_thinking'])
                    self.assertEqual(requests[0]['chat_template_kwargs']['reasoning_effort'], 'medium')


class ReferenceRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_tool_continuation_retries_as_plain_context_once(self):
        import time
        from openai import AsyncOpenAI
        from agent.agents import JSONAgents
        from agent.contracts import Program

        for empty, succeeds in [(None, True), ('\n\n', True), (None, False)]:
            with self.subTest(empty=empty, succeeds=succeeds):
                requests = []
                code = 'def run(state):\n    return {}'

                def handler(request):
                    requests.append(json.loads(request.content))
                    message = {'role': 'assistant', 'content': empty}
                    if len(requests) == 1:
                        message.update(content=None, reasoning_content='prior reasoning', tool_calls=[{
                            'id': 'ref', 'type': 'function', 'function': {'name': 'search_api',
                            'arguments': '{"query":"example"}'}}])
                    elif len(requests) == 3 and succeeds:
                        message['content'] = json.dumps({'code': code})
                    return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion',
                        'created': 0, 'model': 'model', 'choices': [{'index': 0, 'message': message,
                        'finish_reason': 'tool_calls' if len(requests) == 1 else 'stop'}]})

                def client(**kwargs):
                    return AsyncOpenAI(**kwargs,
                        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

                agents = JSONAgents(Settings('fake', 'http://qwen.example/v1', 'model'),
                                    time.monotonic() + 10, 10, node_id='node_001', block='fit')
                with patch('agent.agents.AsyncOpenAI', side_effect=client), \
                        patch('agent.agents.call_reference', return_value='{"matches":[{"symbol":"example.API"}]}'):
                    if succeeds:
                        result = await agents.ask(Program, 'original task', 'implementer', code=True, reference=True)
                        self.assertEqual(result.code, code)
                    else:
                        with self.assertRaises(LLMError):
                            await agents.ask(Program, 'original task', 'implementer', code=True, reference=True)
                expected_calls = 3 if succeeds else 5  # One tool reply and up to four invalid final replies.
                self.assertEqual(len(requests), expected_calls)
                final = requests[2]
                self.assertNotIn('tools', final)
                self.assertNotIn('tool_choice', final)
                self.assertEqual([m['role'] for m in final['messages']], ['system', 'user', 'user'])
                self.assertEqual(final['messages'][1]['content'], 'original task')
                self.assertIn('example.API', final['messages'][2]['content'])
                self.assertNotIn('prior reasoning', json.dumps(final['messages']))
                self.assertEqual(agents.reference_rounds[('node_001', 'fit')], 1)
                self.assertEqual(agents.budget['requests'], expected_calls)


if __name__ == "__main__":
    unittest.main()
