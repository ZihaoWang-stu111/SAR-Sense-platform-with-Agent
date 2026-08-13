import asyncio
import unittest
from unittest.mock import AsyncMock, patch


class MemoryMiddlewareTest(unittest.TestCase):
    def test_appends_memory_after_system_prompt(self):
        from agent.tools.middleware import _append_memory_block

        prompt = _append_memory_block(
            "你是助手。", "## 用户长期记忆\n- [profile] 用户是研究生"
        )

        self.assertTrue(prompt.startswith("你是助手。"))
        self.assertIn("用户是研究生", prompt)
        self.assertEqual(_append_memory_block("你是助手。", ""), "你是助手。")


class MemoryChatRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_loads_before_agent_and_updates_after_persist(self):
        from services import chat_runner

        observed = {"agent_context": None, "updated": []}

        class FakeAgent:
            def execute_stream(self, *args, user_context, **kwargs):
                observed["agent_context"] = dict(user_context)
                yield "最终回答"

        class FakeMetrics:
            def start_conversation(self):
                return 1.0

            def end_conversation(self, *args, **kwargs):
                return None

        class InlineExecutor:
            def submit(self, task, *args, **kwargs):
                task(*args, **kwargs)

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def commit(self):
                return None

        with (
            patch.object(chat_runner, "get_agent", return_value=FakeAgent()),
            patch.object(chat_runner, "get_metrics", return_value=FakeMetrics()),
            patch.object(
                chat_runner, "get_agent_executor", return_value=InlineExecutor()
            ) as get_executor,
            patch.object(
                chat_runner,
                "_try_load_memory_context",
                return_value="## 用户长期记忆\n- [profile] 用户是研究生",
            ),
            patch.object(
                chat_runner,
                "_try_process_memory_turn",
                side_effect=lambda **kwargs: observed["updated"].append(kwargs),
            ),
            patch.object(chat_runner, "AsyncSessionLocal", return_value=FakeSession()),
            patch.object(chat_runner.conv_crud, "append_message", new=AsyncMock()),
        ):
            chat_runner.submit_chat_run(
                messages=[{"role": "user", "content": "请记住我做 SAR 检测"}],
                memory_user_message="请记住我做 SAR 检测",
                conversation_id="conv_1",
                user_context={"user_id": 7},
                user_id=7,
                event_queue=asyncio.Queue(),
                loop=asyncio.get_running_loop(),
            )
            await asyncio.sleep(0.05)

        self.assertIn("用户是研究生", observed["agent_context"]["memory_context"])
        self.assertEqual(observed["updated"][0]["assistant_content"], "最终回答")
        self.assertEqual(observed["updated"][0]["user_message"], "请记住我做 SAR 检测")
        self.assertEqual(get_executor.call_count, 1)


if __name__ == "__main__":
    unittest.main()
