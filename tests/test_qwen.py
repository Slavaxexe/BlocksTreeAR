"""Offline tests: no organizer token, network calls, or paid requests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from alfa_agent.__main__ import main
from alfa_agent.config import ConfigurationError, Settings
from alfa_agent.qwen import QwenError, ask_qwen


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

    def test_missing_endpoint_is_rejected_instead_of_using_default_openai_host(self):
        with self.assertRaisesRegex(ConfigurationError, "OPENAI_BASE_URL"):
            Settings("test-secret", "", "qwen-model").validate()

    def test_platform_can_supply_only_key_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings.from_env(Path(directory) / '.env', {
                'OPENAI_API_KEY': 'fake-secret', 'OPENAI_BASE_URL': 'https://proxy.example/v1'})
            settings.validate()
            self.assertEqual(settings.model, 'qwen/qwen3.8-27b')
            self.assertEqual(settings.timeout_seconds, 300)

    def test_plain_http_credentials_and_query_are_rejected(self):
        for url in ["http://remote.example/v1", "https://user:secret@qwen.example/v1",
                    "https://qwen.example/v1?token=secret", "not-a-url"]:
            with self.subTest(url=url), self.assertRaises(ConfigurationError):
                Settings("test-secret", url, "qwen-model").validate()

    def test_invalid_numeric_limits_are_rejected(self):
        for timeout, tokens in [(0, 10), (float("nan"), 10), (float("inf"), 10), (10, 0)]:
            with self.subTest(timeout=timeout, tokens=tokens), self.assertRaises(ConfigurationError):
                Settings("test-secret", "https://qwen.example/v1", "qwen-model", timeout, tokens).validate()


class QwenTests(unittest.TestCase):
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
            with patch('alfa_agent.__main__.Settings.from_env', return_value=self.settings), \
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

        answer = ask_qwen(self.settings, "hello", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
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
                with self.assertRaises(QwenError) as caught:
                    ask_qwen(self.settings, "hello", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
                self.assertIn(str(code), str(caught.exception))
                self.assertNotIn("test-secret", str(caught.exception))
                self.assertEqual(len(seen), 1)

    def test_timeout_has_a_controlled_error(self):
        def handler(request):
            raise httpx.ReadTimeout("test-secret", request=request)

        with self.assertRaises(QwenError) as caught:
            ask_qwen(self.settings, "hello", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertIn("таймаута", str(caught.exception))
        self.assertNotIn("test-secret", str(caught.exception))

    def test_empty_response_is_not_reported_as_success(self):
        def handler(request):
            return httpx.Response(200, json={"id": "chat-test", "object": "chat.completion",
                "created": 0, "model": "qwen-model", "choices": []})

        with self.assertRaises(QwenError):
            ask_qwen(self.settings, "hello", http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_check_config_is_offline_and_hides_the_key(self):
        out = io.StringIO()
        config = Settings("test-secret", "", "qwen-model")
        with patch("alfa_agent.__main__.Settings.from_env", return_value=config), \
             patch("alfa_agent.__main__.ask_qwen") as ask, redirect_stdout(out):
            result = main(["check-config"])
        self.assertEqual(result, 2)
        self.assertFalse(json.loads(out.getvalue())["ready"])
        self.assertNotIn("test-secret", out.getvalue())
        ask.assert_not_called()

    def test_cli_handles_provider_failure_without_traceback_or_secret(self):
        err = io.StringIO()
        with patch("alfa_agent.__main__.Settings.from_env", return_value=self.settings), \
             patch("alfa_agent.__main__.ask_qwen", side_effect=QwenError("HTTP 401 test-secret")), \
             redirect_stderr(err):
            result = main(["ping"])
        self.assertEqual(result, 2)
        self.assertIn("HTTP 401", err.getvalue())
        self.assertNotIn("test-secret", err.getvalue())


if __name__ == "__main__":
    unittest.main()
