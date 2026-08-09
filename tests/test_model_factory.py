"""Chat model factory unit tests."""

import os
import unittest
from unittest.mock import patch

from model.factory import ChatModelFactory


class ChatModelFactoryTest(unittest.TestCase):
    def test_openai_provider_uses_compatible_client_configuration(self):
        env = {
            "CHAT_PROVIDER": "openai",
            "CHAT_MODEL_NAME": "mimo-v2.5-pro",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.invalid/v1",
            "OPENAI_TIMEOUT_S": "45",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("model.factory.ChatOpenAI") as chat_openai:
                ChatModelFactory().generator()

        chat_openai.assert_called_once_with(
            model="mimo-v2.5-pro",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            timeout=45.0,
            max_retries=0,
            use_responses_api=False,
        )

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
