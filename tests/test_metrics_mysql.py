import inspect
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from models import Base
from models.metrics import MetricEvent
from agent.metrics_collector import AgentMetrics
from agent.react_agent import ReactAgent
from agent.tools import middleware
from api import dependencies
from api.routers import chat as chat_api
from api.routers import metrics as metrics_api
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage


_IMPORT_ERROR = None
try:
    from repositories.metrics_repository import MetricsRepository
except (ImportError, AttributeError) as exc:
    _IMPORT_ERROR = exc


class PhaseFourComponentsTest(unittest.TestCase):
    def test_metrics_repository_is_available(self):
        self.assertIsNone(
            _IMPORT_ERROR,
            f"Phase 4 MetricsRepository is missing: {_IMPORT_ERROR}",
        )

    def test_legacy_metrics_crud_module_is_removed(self):
        crud_module = Path(__file__).resolve().parents[1] / "crud" / "metrics.py"
        self.assertFalse(crud_module.exists(), f"Legacy metrics CRUD still exists: {crud_module}")


@unittest.skipIf(_IMPORT_ERROR is not None, "Phase 4 MetricsRepository is not implemented yet")
class MetricsRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        database_path = Path(self.tempdir.name) / "metrics.sqlite3"
        self.engine = create_engine(f"sqlite+pysqlite:///{database_path}")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.repository = MetricsRepository(session_factory=self.session_factory)

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()
        self.tempdir.cleanup()

    def _add_event(self, *, created_at, **values):
        with self.session_factory.begin() as session:
            session.add(MetricEvent(created_at=created_at, **values))

    def test_history_survives_a_new_repository_instance(self):
        self.repository.record_event(
            user_id=42,
            event_type="tool_call",
            tool_name="detect_ships",
            success=True,
            duration_ms=125.0,
        )

        restarted_repository = MetricsRepository(session_factory=self.session_factory)
        metrics = restarted_repository.aggregate()

        self.assertEqual(metrics["total_tool_calls"], 1)
        with self.session_factory() as session:
            event = session.scalar(select(MetricEvent))
        self.assertEqual(event.user_id, 42)

    def test_aggregate_matches_existing_api_contract(self):
        now = datetime(2026, 7, 16, 9, 30, 0)
        events = [
            {
                "event_type": "tool_call",
                "tool_name": "detect_ships",
                "success": True,
                "duration_ms": 100.04,
                "created_at": now,
            },
            {
                "event_type": "tool_call",
                "tool_name": "web_search",
                "success": False,
                "duration_ms": 500.0,
                "created_at": now + timedelta(seconds=1),
            },
            {
                "event_type": "tool_call",
                "tool_name": "detect_ships",
                "success": True,
                "duration_ms": 300.06,
                "created_at": now + timedelta(seconds=2),
            },
            {
                "event_type": "llm_call",
                "created_at": now + timedelta(seconds=3),
            },
            {
                "event_type": "llm_call",
                "created_at": now + timedelta(seconds=4),
            },
            {
                "event_type": "conversation_timing",
                "duration_ms": 1000.0,
                "created_at": now + timedelta(seconds=5),
            },
            {
                "event_type": "conversation_timing",
                "duration_ms": 3000.0,
                "created_at": now + timedelta(seconds=6),
            },
            {
                "event_type": "tool_call",
                "tool_name": "legacy_unknown",
                "success": None,
                "duration_ms": 900.0,
                "created_at": now + timedelta(seconds=7),
            },
        ]
        with self.session_factory.begin() as session:
            session.add_all(MetricEvent(user_id=7, **event) for event in events)

        metrics = self.repository.aggregate(limit=2)

        self.assertEqual(
            set(metrics),
            {
                "conversation_rounds",
                "total_tool_calls",
                "overall_success_rate",
                "avg_tool_calls_per_round",
                "avg_response_time_s",
                "llm_call_count",
                "tool_stats",
                "recent_records",
            },
        )
        self.assertEqual(metrics["conversation_rounds"], 2)
        self.assertEqual(metrics["total_tool_calls"], 3)
        self.assertEqual(metrics["overall_success_rate"], 66.7)
        self.assertEqual(metrics["avg_tool_calls_per_round"], 1.5)
        self.assertEqual(metrics["avg_response_time_s"], 2.0)
        self.assertEqual(metrics["llm_call_count"], 2)
        self.assertEqual(
            metrics["tool_stats"],
            [
                {
                    "tool_name": "detect_ships",
                    "total": 2,
                    "success": 2,
                    "fail": 0,
                    "success_rate": 100.0,
                    "avg_duration_ms": 200.1,
                },
                {
                    "tool_name": "web_search",
                    "total": 1,
                    "success": 0,
                    "fail": 1,
                    "success_rate": 0.0,
                    "avg_duration_ms": 500.0,
                },
            ],
        )
        self.assertEqual(
            metrics["recent_records"],
            [
                {
                    "tool_name": "detect_ships",
                    "timestamp": "09:30:02",
                    "success": True,
                    "duration_ms": 300.1,
                },
                {
                    "tool_name": "web_search",
                    "timestamp": "09:30:01",
                    "success": False,
                    "duration_ms": 500.0,
                },
            ],
        )

    def test_empty_aggregate_preserves_legacy_defaults(self):
        self.assertEqual(
            self.repository.aggregate(),
            {
                "conversation_rounds": 0,
                "total_tool_calls": 0,
                "overall_success_rate": 100.0,
                "avg_tool_calls_per_round": 0,
                "avg_response_time_s": 0,
                "llm_call_count": 0,
                "tool_stats": [],
                "recent_records": [],
            },
        )

    def test_tool_stats_preserve_first_seen_order_when_totals_tie(self):
        now = datetime(2026, 7, 16, 10, 0, 0)
        self._add_event(
            user_id=1,
            event_type="tool_call",
            tool_name="z_first",
            success=True,
            duration_ms=10.0,
            created_at=now,
        )
        self._add_event(
            user_id=1,
            event_type="tool_call",
            tool_name="a_second",
            success=True,
            duration_ms=10.0,
            created_at=now + timedelta(seconds=1),
        )

        tool_names = [
            item["tool_name"] for item in self.repository.aggregate()["tool_stats"]
        ]

        self.assertEqual(tool_names, ["z_first", "a_second"])

    def test_recent_records_use_descending_id_when_timestamps_match(self):
        created_at = datetime(2026, 7, 16, 10, 30, 0)
        self._add_event(
            user_id=1,
            event_type="tool_call",
            tool_name="first",
            success=True,
            duration_ms=10.0,
            created_at=created_at,
        )
        self._add_event(
            user_id=1,
            event_type="tool_call",
            tool_name="second",
            success=True,
            duration_ms=20.0,
            created_at=created_at,
        )

        recent = self.repository.aggregate()["recent_records"]

        self.assertEqual(
            [record["tool_name"] for record in recent],
            ["second", "first"],
        )

    def test_reset_and_write_are_atomic_across_repositories(self):
        self.assertIn(
            "memory_reset_callback",
            inspect.signature(self.repository.reset).parameters,
        )
        writer_repository = MetricsRepository(session_factory=self.session_factory)
        metrics = AgentMetrics()
        metrics.reset()
        callback_started = threading.Event()
        writer_attempted = threading.Event()
        writer_finished = threading.Event()

        def reset_memory():
            callback_started.set()
            self.assertTrue(writer_attempted.wait(timeout=2))
            self.assertFalse(writer_finished.is_set())
            with self.session_factory() as session:
                count = session.scalar(select(func.count()).select_from(MetricEvent))
            self.assertEqual(count, 0)
            metrics.reset()

        def write_new_event():
            if not callback_started.wait(timeout=2):
                raise AssertionError("reset callback did not start")
            writer_attempted.set()
            metrics.record_tool_call(
                tool_name="new",
                user_id=2,
                success=True,
                duration_ms=2.0,
            )
            writer_finished.set()

        with patch.object(metrics, "_repository", writer_repository):
            metrics.record_tool_call(
                tool_name="old",
                user_id=1,
                success=True,
                duration_ms=1.0,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                reset_future = executor.submit(self.repository.reset, reset_memory)
                write_future = executor.submit(write_new_event)
                reset_future.result(timeout=5)
                write_future.result(timeout=5)

        with self.session_factory() as session:
            events = list(session.scalars(select(MetricEvent)).all())
        self.assertEqual([event.tool_name for event in events], ["new"])
        self.assertEqual(metrics.total_tool_calls, 1)
        self.assertEqual(metrics.get_recent_records()[0]["tool_name"], "new")
        metrics.reset()

    def test_reset_does_not_call_memory_callback_when_delete_fails(self):
        self.assertIn(
            "memory_reset_callback",
            inspect.signature(self.repository.reset).parameters,
        )
        failing_session_factory = MagicMock()
        failing_session_factory.begin.return_value.__enter__.side_effect = RuntimeError(
            "delete failed"
        )
        repository = MetricsRepository(session_factory=failing_session_factory)
        memory_reset = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "delete failed"):
            repository.reset(memory_reset)

        memory_reset.assert_not_called()

    def test_delete_all_commits_the_reset(self):
        self.repository.record_event(user_id=3, event_type="llm_call")

        deleted = self.repository.delete_all()

        self.assertEqual(deleted, 1)
        with self.session_factory() as session:
            count = session.scalar(select(func.count()).select_from(MetricEvent))
        self.assertEqual(count, 0)


class AgentMetricsPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.metrics = AgentMetrics()
        self.metrics.reset()

    def tearDown(self):
        self.metrics.reset()

    def test_record_methods_persist_the_real_user_id(self):
        with (
            patch.object(self.metrics._repository, "record_event") as record_event,
            patch("agent.metrics_collector.time.monotonic", side_effect=[10.0, 11.5]),
        ):
            self.metrics.record_tool_call("web_search", True, 12.5, user_id=21)
            self.metrics.record_llm_call(user_id=22)
            started_at = self.metrics.start_conversation()
            self.metrics.end_conversation(started_at, user_id=23)

        self.assertEqual(
            record_event.call_args_list,
            [
                call(
                    event_type="tool_call",
                    tool_name="web_search",
                    success=True,
                    duration_ms=12.5,
                    user_id=21,
                ),
                call(event_type="llm_call", user_id=22),
                call(
                    event_type="conversation_timing",
                    duration_ms=1500.0,
                    user_id=23,
                ),
            ],
        )

    def test_interleaved_conversation_timers_do_not_overwrite_each_other(self):
        with (
            patch.object(self.metrics._repository, "record_event") as record_event,
            patch(
                "agent.metrics_collector.time.monotonic",
                side_effect=[10.0, 20.0, 30.0, 50.0],
            ),
        ):
            first_started_at = self.metrics.start_conversation()
            second_started_at = self.metrics.start_conversation()
            self.metrics.end_conversation(first_started_at, user_id=31)
            self.metrics.end_conversation(second_started_at, user_id=32)

        self.assertEqual(first_started_at, 10.0)
        self.assertEqual(second_started_at, 20.0)
        self.assertEqual(self.metrics.conversation_rounds, 2)
        self.assertEqual(self.metrics.total_response_time_ms, 50000.0)
        self.assertEqual(
            record_event.call_args_list,
            [
                call(
                    event_type="conversation_timing",
                    duration_ms=20000.0,
                    user_id=31,
                ),
                call(
                    event_type="conversation_timing",
                    duration_ms=30000.0,
                    user_id=32,
                ),
            ],
        )

    def test_true_threads_keep_memory_and_persistence_in_one_lock_scope(self):
        barrier = threading.Barrier(3)
        persisted = []
        persisted_lock = threading.Lock()

        def capture_event(**event):
            with persisted_lock:
                persisted.append(
                    {
                        **event,
                        "shared_lock_owned": MetricsRepository._lock._is_owned(),
                        "memory_lock_owned": self.metrics._lock._is_owned(),
                    }
                )

        def run_conversation(user_id):
            started_at = self.metrics.start_conversation()
            barrier.wait(timeout=2)
            self.metrics.record_tool_call(
                "web_search",
                True,
                5.0,
                user_id=user_id,
            )
            self.metrics.record_llm_call(user_id=user_id)
            self.metrics.end_conversation(started_at, user_id=user_id)

        with patch.object(
            self.metrics._repository,
            "record_event",
            side_effect=capture_event,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(run_conversation, user_id)
                    for user_id in (41, 42)
                ]
                barrier.wait(timeout=2)
                for future in futures:
                    future.result(timeout=5)

        self.assertEqual(self.metrics.conversation_rounds, 2)
        self.assertEqual(self.metrics.total_tool_calls, 2)
        self.assertEqual(self.metrics.llm_call_count, 2)
        self.assertEqual(len(persisted), 6)
        self.assertEqual({event["user_id"] for event in persisted}, {41, 42})
        self.assertTrue(all(event["shared_lock_owned"] for event in persisted))
        self.assertTrue(all(event["memory_lock_owned"] for event in persisted))

    def test_persistence_failure_does_not_interrupt_memory_updates(self):
        with (
            patch.object(
                self.metrics._repository,
                "record_event",
                side_effect=RuntimeError("database unavailable"),
            ),
            patch("agent.metrics_collector.logger.warning") as warning,
        ):
            self.metrics.record_llm_call(user_id=51)

        self.assertEqual(self.metrics.llm_call_count, 1)
        warning.assert_called_once()
        self.assertIn("database unavailable", warning.call_args.args[0])


class MetricsMiddlewareTest(unittest.TestCase):
    def test_runtime_user_id_is_forwarded_for_tool_and_llm_events(self):
        fake_metrics = SimpleNamespace(
            record_tool_call=MagicMock(),
            record_llm_call=MagicMock(),
        )
        runtime = SimpleNamespace(context={"user_id": 77, "report": False})
        request = SimpleNamespace(
            tool_call={"name": "web_search", "args": {"query": "ships"}},
            runtime=runtime,
        )

        with patch.object(middleware, "AgentMetrics", return_value=fake_metrics):
            result = middleware.monitor_tool.wrap_tool_call(
                request, lambda _request: "result"
            )
            middleware.log_before_model.before_model(
                {"messages": [HumanMessage(content="hello")]},
                runtime,
            )

        self.assertEqual(result, "result")
        tool_args = fake_metrics.record_tool_call.call_args
        self.assertEqual(tool_args.args[:2], ("web_search", True))
        self.assertEqual(tool_args.kwargs["user_id"], 77)
        fake_metrics.record_llm_call.assert_called_once_with(user_id=77)

    def test_missing_runtime_user_id_defaults_to_seed_user(self):
        fake_metrics = SimpleNamespace(record_llm_call=MagicMock())
        runtime = SimpleNamespace(context={})

        with patch.object(middleware, "AgentMetrics", return_value=fake_metrics):
            middleware.log_before_model.before_model(
                {"messages": [HumanMessage(content="hello")]},
                runtime,
            )

        fake_metrics.record_llm_call.assert_called_once_with(user_id=1)

    def test_log_before_model_reports_approximate_context_tokens(self):
        fake_metrics = SimpleNamespace(record_llm_call=MagicMock())
        runtime = SimpleNamespace(context={"user_id": 7})

        with (
            patch.object(middleware, "AgentMetrics", return_value=fake_metrics),
            patch.object(middleware.logger, "info") as info,
        ):
            middleware.log_before_model.before_model(
                {"messages": [HumanMessage(content="hello")]},
                runtime,
            )

        log_message = info.call_args.args[0]
        self.assertIn("1条消息", log_message)
        self.assertRegex(log_message, r"约\d+ tokens")


class ReactAgentRuntimeContextTest(unittest.TestCase):
    def test_execute_stream_merges_user_context_into_langchain_runtime(self):
        captured = {}

        class FakeCompiledAgent:
            def stream(self, input_dict, *, stream_mode, context, config):
                captured["input_dict"] = input_dict
                captured["stream_mode"] = stream_mode
                captured["context"] = context
                captured["config"] = config
                yield {"messages": list(input_dict["messages"])}

        react_agent = ReactAgent.__new__(ReactAgent)
        react_agent.agent = FakeCompiledAgent()
        chat_pack = [{"role": "user", "content": "hello"}]

        chunks = list(
            react_agent.execute_stream(
                chat_pack,
                user_context={"user_id": 42, "role": "researcher"},
            )
        )

        self.assertEqual(chunks, [])
        self.assertEqual(captured["input_dict"], {"messages": chat_pack})
        self.assertEqual(captured["stream_mode"], "values")
        context = captured["context"]
        self.assertEqual(context["report"], False)
        self.assertEqual(context["user_id"], 42)
        self.assertEqual(context["role"], "researcher")
        # V3 子 Agent 桥接：请求级 RAG 来源 collector
        self.assertIn("_subagent_rag_results", context)
        self.assertEqual(context["_subagent_rag_results"], [])

    def test_execute_stream_applies_main_agent_recursion_limit(self):
        captured = {}

        class FakeCompiledAgent:
            def stream(self, input_dict, *, stream_mode, context, config):
                captured["config"] = config
                yield {"messages": list(input_dict["messages"])}

        react_agent = ReactAgent.__new__(ReactAgent)
        react_agent.agent = FakeCompiledAgent()

        list(react_agent.execute_stream([{"role": "user", "content": "hello"}]))

        self.assertEqual(captured["config"]["recursion_limit"], 25)


class MetricsAPITest(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_repository_dependency_is_a_lazy_singleton(self):
        original = dependencies._metrics_repository
        dependencies._metrics_repository = None
        fake_repository = object()
        try:
            with patch(
                "repositories.metrics_repository.MetricsRepository",
                return_value=fake_repository,
            ) as repository_class:
                first = dependencies.get_metrics_repository()
                second = dependencies.get_metrics_repository()
        finally:
            dependencies._metrics_repository = original

        self.assertIs(first, fake_repository)
        self.assertIs(second, fake_repository)
        repository_class.assert_called_once_with()

    async def test_get_returns_the_existing_json_contract_from_mysql_history(self):
        aggregate = {
            "conversation_rounds": 4,
            "total_tool_calls": 6,
            "overall_success_rate": 83.3,
            "avg_tool_calls_per_round": 1.5,
            "avg_response_time_s": 2.4,
            "llm_call_count": 8,
            "tool_stats": [{"tool_name": "web_search"}],
            "recent_records": [{"tool_name": "web_search"}],
        }
        repository = SimpleNamespace(aggregate=MagicMock(return_value=aggregate))

        async def inline_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        threadpool = AsyncMock(side_effect=inline_threadpool)

        with (
            patch.object(
                metrics_api,
                "get_metrics_repository",
                return_value=repository,
            ),
            patch.object(
                metrics_api,
                "run_in_threadpool",
                new=threadpool,
            ),
            patch.object(
                metrics_api,
                "get_metrics",
                side_effect=AssertionError("GET must not read in-memory metrics"),
            ),
        ):
            response = await metrics_api.get_metrics_data(
                user={"id": 9, "username": "reader"}
            )

        self.assertEqual(response, {"success": True, "metrics": aggregate})
        repository.aggregate.assert_called_once_with()
        threadpool.assert_awaited_once_with(repository.aggregate)

    async def test_reset_deletes_database_before_clearing_memory(self):
        order = []
        repository = SimpleNamespace(
            reset=MagicMock(
                side_effect=lambda callback: (order.append("db"), callback())
            ),
            delete_all=MagicMock(side_effect=lambda: order.append("legacy_delete")),
        )
        memory = SimpleNamespace(reset=MagicMock(side_effect=lambda: order.append("memory")))

        async def inline_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        threadpool = AsyncMock(side_effect=inline_threadpool)

        with (
            patch.object(
                metrics_api,
                "get_metrics_repository",
                return_value=repository,
            ),
            patch.object(metrics_api, "get_metrics", return_value=memory),
            patch.object(
                metrics_api,
                "run_in_threadpool",
                new=threadpool,
            ),
        ):
            response = await metrics_api.reset_metrics(
                admin={"id": 1, "username": "admin"}
            )

        self.assertEqual(order, ["db", "memory"])
        threadpool.assert_awaited_once_with(repository.reset, memory.reset)
        repository.delete_all.assert_not_called()
        self.assertEqual(
            response,
            {"success": True, "message": "Metrics reset successfully"},
        )

    async def test_reset_keeps_memory_when_database_delete_fails(self):
        repository = SimpleNamespace(
            reset=MagicMock(side_effect=RuntimeError("database unavailable")),
            delete_all=MagicMock(side_effect=RuntimeError("wrong reset method")),
        )
        memory = SimpleNamespace(reset=MagicMock())

        async def inline_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        threadpool = AsyncMock(side_effect=inline_threadpool)

        with (
            patch.object(
                metrics_api,
                "get_metrics_repository",
                return_value=repository,
            ),
            patch.object(metrics_api, "get_metrics", return_value=memory),
            patch.object(
                metrics_api,
                "run_in_threadpool",
                new=threadpool,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await metrics_api.reset_metrics(admin={"id": 1, "username": "admin"})

        memory.reset.assert_not_called()
        threadpool.assert_awaited_once_with(repository.reset, memory.reset)
        repository.delete_all.assert_not_called()

    def test_auth_dependencies_remain_unchanged(self):
        get_default = inspect.signature(metrics_api.get_metrics_data).parameters[
            "user"
        ].default
        reset_default = inspect.signature(metrics_api.reset_metrics).parameters[
            "admin"
        ].default

        self.assertIs(get_default.dependency, metrics_api.get_current_user)
        self.assertIs(reset_default.dependency, metrics_api.require_admin)


class ChatMetricsIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream_forwards_authenticated_user_id_to_metrics(self):
        agent_calls = []
        ended = threading.Event()

        class FakeAgent:
            def execute_stream(
                self,
                messages,
                conversation_id,
                *,
                user_context,
                on_step,
            ):
                agent_calls.append(
                    {
                        "messages": messages,
                        "conversation_id": conversation_id,
                        "user_context": user_context,
                        "on_step": on_step,
                    }
                )
                yield "answer"

        class FakeMetrics:
            def __init__(self):
                self.end_calls = []

            def start_conversation(self):
                return 123.0

            def end_conversation(self, started_at, user_id):
                self.end_calls.append((started_at, user_id))
                ended.set()

        fake_metrics = FakeMetrics()
        # 请求体解析已由 FastAPI + ChatStreamRequest 负责，直接构造 payload
        payload = chat_api.ChatStreamRequest(
            message="hello",
            messages=[],
            conversation_id=None,
        )
        request = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
        )
        user = {"id": 88, "username": "reader", "role": "researcher"}

        from services import chat_runner as chat_runner_mod

        with (
            patch.object(chat_runner_mod, "get_agent", return_value=FakeAgent()),
            patch.object(chat_runner_mod, "get_metrics", return_value=fake_metrics),
            patch.object(chat_runner_mod, "get_allowed_doc_ids", new=AsyncMock(return_value=None)),
            patch.object(chat_api, "rate_limit", new=AsyncMock()),
        ):
            response = await chat_api.chat_stream(
                payload=payload,
                request=request,
                user=user,
                db=SimpleNamespace(),
            )
            body_parts = []
            async for part in response.body_iterator:
                body_parts.append(part.decode() if isinstance(part, bytes) else part)

        self.assertIsInstance(response, StreamingResponse)
        self.assertIn("answer", "".join(body_parts))
        self.assertTrue(ended.wait(timeout=2))
        self.assertEqual(agent_calls[0]["user_context"]["user_id"], 88)
        self.assertEqual(fake_metrics.end_calls, [(123.0, 88)])


if __name__ == "__main__":
    unittest.main()
