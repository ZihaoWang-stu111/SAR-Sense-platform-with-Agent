import threading
import time
import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api import dependencies
from api.routers import chat
from services.agent_executor import AgentExecutor


class AgentExecutorTest(unittest.TestCase):
    def tearDown(self):
        dependencies.shutdown_agent_executor(wait=True)

    def test_max_workers_limits_concurrent_agent_tasks(self):
        executor = AgentExecutor(max_workers=2)
        release = threading.Event()
        state_lock = threading.Lock()
        running = 0
        max_running = 0
        two_tasks_started = threading.Event()

        def task():
            nonlocal running, max_running
            with state_lock:
                running += 1
                max_running = max(max_running, running)
                if running == 2:
                    two_tasks_started.set()
            release.wait(timeout=2)
            with state_lock:
                running -= 1

        futures = [executor.submit(task) for _ in range(4)]
        try:
            self.assertTrue(two_tasks_started.wait(timeout=1))
            time.sleep(0.05)
            self.assertEqual(max_running, 2)
        finally:
            release.set()
            for future in futures:
                future.result(timeout=2)
            executor.shutdown()

    def test_dependency_returns_singleton_and_recreates_after_shutdown(self):
        first = dependencies.get_agent_executor()

        self.assertIs(first, dependencies.get_agent_executor())

        dependencies.shutdown_agent_executor(wait=True)

        self.assertIsNot(first, dependencies.get_agent_executor())


class ChatAgentDispatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_submits_agent_work_to_shared_executor(self):
        class FakeAgent:
            def execute_stream(self, *args, **kwargs):
                yield "测试回答"

        class FakeMetrics:
            def start_conversation(self):
                return 1.0

            def end_conversation(self, *args, **kwargs):
                return None

        class InlineExecutor:
            def __init__(self):
                self.submissions = 0

            def submit(self, task):
                self.submissions += 1
                task()

        executor = InlineExecutor()
        payload = SimpleNamespace(
            message="测试问题",
            display_message=None,
            conversation_id=None,
            messages=[],
        )
        request = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
        )

        with (
            patch.object(chat, "rate_limit", new=AsyncMock()),
            patch.object(chat, "get_allowed_doc_ids", new=AsyncMock(return_value=None)),
            patch.object(chat, "get_agent", return_value=FakeAgent()),
            patch.object(chat, "get_metrics", return_value=FakeMetrics()),
            patch.object(
                chat,
                "get_agent_executor",
                return_value=executor,
                create=True,
            ),
        ):
            response = await chat.chat_stream(
                payload,
                request,
                user={"id": 1, "role": "admin"},
                db=AsyncMock(),
            )
            body = "".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(executor.submissions, 1)
        self.assertIn("event: status", body)
        self.assertIn("正在思考...", body)
        self.assertLess(body.index("event: status"), body.index("event: chunk"))
        self.assertIn("测试回答", body)


class AgentExecutorLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_app_shutdown_closes_agent_executor(self):
        app_module = importlib.import_module("api.app")
        app = app_module.create_app()

        with patch.object(app_module, "shutdown_agent_executor", create=True) as shutdown:
            for handler in app.router.on_shutdown:
                await handler()

        shutdown.assert_called_once_with(wait=True)


if __name__ == "__main__":
    unittest.main()
