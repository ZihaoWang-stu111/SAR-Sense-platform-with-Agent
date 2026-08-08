"""外部调用治理单元测试。"""

import os
import unittest
from unittest.mock import patch

from utils.call_governance import (
    DEFAULT_DASHSCOPE_TIMEOUT_S,
    MAX_ATTEMPTS,
    call_with_retries,
    classify_error,
    get_timeout_s,
    resolve_chat_provider_model,
)


class _HttpError(Exception):
    def __init__(self, status_code, message="err", headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = type("R", (), {"status_code": status_code, "headers": headers or {}})()


class ClassifyErrorTest(unittest.TestCase):
    def test_retryable_status_codes(self):
        for code in (429, 500, 502, 503, 504):
            c = classify_error(_HttpError(code))
            self.assertTrue(c.retryable, code)
            self.assertEqual(c.status_code, code)

    def test_non_retryable_4xx(self):
        for code in (400, 401, 403, 404, 422):
            c = classify_error(_HttpError(code))
            self.assertFalse(c.retryable, code)

    def test_timeout_and_network(self):
        self.assertTrue(classify_error(TimeoutError("read timed out")).retryable)
        self.assertTrue(classify_error(ConnectionResetError("reset")).retryable)
        self.assertTrue(classify_error(RuntimeError("failed to connect")).retryable)

    def test_retry_after_parsed(self):
        c = classify_error(_HttpError(429, headers={"Retry-After": "3"}))
        self.assertEqual(c.retry_after_s, 3.0)


class CallWithRetriesTest(unittest.TestCase):
    def test_success_first_try(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        self.assertEqual(call_with_retries(fn, provider="tavily", model="search"), "ok")
        self.assertEqual(calls["n"], 1)

    def test_retries_then_success(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _HttpError(503, "busy")
            return "ok"

        with patch("utils.call_governance.time.sleep"):
            self.assertEqual(call_with_retries(fn, provider="dashscope", model="emb"), "ok")
        self.assertEqual(calls["n"], 3)

    def test_max_attempts_is_three(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise _HttpError(503, "busy")

        with patch("utils.call_governance.time.sleep"):
            with self.assertRaises(_HttpError):
                call_with_retries(fn, provider="tavily", model="search")
        self.assertEqual(calls["n"], MAX_ATTEMPTS)

    def test_auth_error_no_retry(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise _HttpError(401, "bad key")

        with self.assertRaises(_HttpError):
            call_with_retries(fn, provider="tavily", model="search")
        self.assertEqual(calls["n"], 1)

    def test_nested_call_does_not_multiply_retries(self):
        calls = {"n": 0}

        def inner():
            calls["n"] += 1
            raise _HttpError(503, "busy")

        def outer():
            return call_with_retries(inner, provider="ollama", model="q")

        with patch("utils.call_governance.time.sleep"):
            with self.assertRaises(_HttpError):
                call_with_retries(outer, provider="ollama", model="q")
        self.assertEqual(calls["n"], MAX_ATTEMPTS)


class TimeoutConfigTest(unittest.TestCase):
    def test_get_timeout_default_and_override(self):
        os.environ.pop("DASHSCOPE_TIMEOUT_S", None)
        self.assertEqual(
            get_timeout_s("DASHSCOPE_TIMEOUT_S", DEFAULT_DASHSCOPE_TIMEOUT_S),
            DEFAULT_DASHSCOPE_TIMEOUT_S,
        )
        os.environ["DASHSCOPE_TIMEOUT_S"] = "45"
        self.assertEqual(get_timeout_s("DASHSCOPE_TIMEOUT_S", 60.0), 45.0)
        os.environ["DASHSCOPE_TIMEOUT_S"] = "bad"
        self.assertEqual(get_timeout_s("DASHSCOPE_TIMEOUT_S", 60.0), 60.0)
        os.environ.pop("DASHSCOPE_TIMEOUT_S", None)

    def test_resolve_chat_provider_model_env(self):
        with patch.dict(
            os.environ,
            {"CHAT_PROVIDER": "Ollama", "CHAT_MODEL_NAME": "qwen-test"},
            clear=False,
        ):
            provider, model = resolve_chat_provider_model()
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "qwen-test")


if __name__ == "__main__":
    unittest.main()
