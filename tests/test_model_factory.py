"""Chat model factory unit tests."""

import os
import unittest
from unittest.mock import patch

from model.factory import ChatModelFactory


class ChatModelFactoryTest(unittest.TestCase):
    def test_openai_provider_prefers_compatible_service_configuration(self):
        env = {
            "CHAT_PROVIDER": "openai",
            "CHAT_MODEL_NAME": "mimo-v2.5-pro",
            "OPENAI_API_KEY": "system-key",
            "OPENAI_BASE_URL": "https://standard.invalid/v1",
            "OPENAI_TIMEOUT_S": "15",
            "OPENAI_COMPATIBLE_API_KEY": "compatible-key",
            "OPENAI_COMPATIBLE_BASE_URL": "https://compatible.invalid/v1",
            "OPENAI_COMPATIBLE_TIMEOUT_S": "45",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("model.factory.ChatOpenAI") as chat_openai:
                ChatModelFactory().generator()

        chat_openai.assert_called_once_with(
            model="mimo-v2.5-pro",
            api_key="compatible-key",
            base_url="https://compatible.invalid/v1",
            timeout=45.0,
            max_retries=0,
            use_responses_api=False,
        )

    def test_openai_provider_falls_back_to_standard_configuration(self):
        env = {
            "CHAT_PROVIDER": "openai",
            "CHAT_MODEL_NAME": "standard-model",
            "OPENAI_API_KEY": "standard-key",
            "OPENAI_BASE_URL": "https://standard.invalid/v1",
            "OPENAI_TIMEOUT_S": "30",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("model.factory.ChatOpenAI") as chat_openai:
                ChatModelFactory().generator()

        chat_openai.assert_called_once_with(
            model="standard-model",
            api_key="standard-key",
            base_url="https://standard.invalid/v1",
            timeout=30.0,
            max_retries=0,
            use_responses_api=False,
        )

    def test_openai_provider_rejects_compatible_key_without_base_url(self):
        env = {
            "CHAT_PROVIDER": "openai",
            "CHAT_MODEL_NAME": "mimo-v2.5-pro",
            "OPENAI_API_KEY": "system-key",
            "OPENAI_BASE_URL": "https://standard.invalid/v1",
            "OPENAI_COMPATIBLE_API_KEY": "compatible-key",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(
                ValueError, "OPENAI_COMPATIBLE_API_KEY and .*BASE_URL"
            ):
                ChatModelFactory().generator()

    def test_openai_provider_rejects_compatible_base_url_without_key(self):
        env = {
            "CHAT_PROVIDER": "openai",
            "CHAT_MODEL_NAME": "mimo-v2.5-pro",
            "OPENAI_API_KEY": "system-key",
            "OPENAI_BASE_URL": "https://standard.invalid/v1",
            "OPENAI_COMPATIBLE_BASE_URL": "https://compatible.invalid/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(
                ValueError, "OPENAI_COMPATIBLE_API_KEY and .*BASE_URL"
            ):
                ChatModelFactory().generator()

    def test_unknown_provider_raises_value_error(self):
        with patch.dict(
            os.environ,
            {"CHAT_PROVIDER": "unsupported", "CHAT_MODEL_NAME": "test-model"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported chat provider"):
                ChatModelFactory().generator()


if __name__ == "__main__":
    unittest.main()
