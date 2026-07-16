"""Thread-safe in-memory agent metrics with best-effort SQL persistence."""

import threading
import time
from collections import defaultdict
from datetime import datetime

from crud.metrics import insert_metric_event
from services.metrics_store import MetricsStore


class AgentMetrics:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._lock = threading.RLock()
                    instance._legacy_timer = threading.local()
                    instance._reset()
                    cls._instance = instance
        return cls._instance

    def _reset(self):
        with self._lock:
            self.conversation_rounds = 0
            self.tool_call_records = []
            self.tool_call_counts = defaultdict(int)
            self.tool_success_counts = defaultdict(int)
            self.tool_fail_counts = defaultdict(int)
            self.tool_durations = defaultdict(list)
            self.llm_call_count = 0
            self.total_response_time_ms = 0.0
        if hasattr(self._legacy_timer, "started_at"):
            del self._legacy_timer.started_at

    def start_conversation(self) -> float:
        with MetricsStore._lock:
            with self._lock:
                started_at = time.monotonic()
                self.conversation_rounds += 1
                self._legacy_timer.started_at = started_at
                return started_at

    def end_conversation(
        self,
        started_at: float | None = None,
        user_id: int = 1,
    ) -> float | None:
        with MetricsStore._lock:
            with self._lock:
                if started_at is None:
                    started_at = getattr(self._legacy_timer, "started_at", None)
                if started_at is None:
                    return None

                duration_ms = max(0.0, (time.monotonic() - started_at) * 1000)
                self.total_response_time_ms += duration_ms
                if getattr(self._legacy_timer, "started_at", None) == started_at:
                    del self._legacy_timer.started_at

                insert_metric_event(
                    event_type="conversation_timing",
                    duration_ms=duration_ms,
                    user_id=user_id,
                )
                return duration_ms

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        user_id: int = 1,
    ) -> None:
        with MetricsStore._lock:
            with self._lock:
                record = {
                    "tool_name": tool_name,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "success": success,
                    "duration_ms": round(duration_ms, 1),
                }
                self.tool_call_counts[tool_name] += 1
                if success:
                    self.tool_success_counts[tool_name] += 1
                else:
                    self.tool_fail_counts[tool_name] += 1
                self.tool_durations[tool_name].append(duration_ms)
                self.tool_call_records.append(record)

                insert_metric_event(
                    event_type="tool_call",
                    tool_name=tool_name,
                    success=success,
                    duration_ms=duration_ms,
                    user_id=user_id,
                )

    def record_llm_call(self, user_id: int = 1) -> None:
        with MetricsStore._lock:
            with self._lock:
                self.llm_call_count += 1
                insert_metric_event(event_type="llm_call", user_id=user_id)

    def reset(self) -> None:
        self._reset()

    @property
    def total_tool_calls(self) -> int:
        with self._lock:
            return sum(self.tool_call_counts.values())

    @property
    def avg_tool_calls_per_round(self) -> float:
        with self._lock:
            if self.conversation_rounds == 0:
                return 0
            total_tool_calls = sum(self.tool_call_counts.values())
            return round(total_tool_calls / self.conversation_rounds, 1)

    @property
    def overall_success_rate(self) -> float:
        with self._lock:
            total_tool_calls = sum(self.tool_call_counts.values())
            if total_tool_calls == 0:
                return 100.0
            successful = sum(self.tool_success_counts.values())
            return round(successful / total_tool_calls * 100, 1)

    @property
    def avg_response_time_s(self) -> float:
        with self._lock:
            if self.conversation_rounds == 0:
                return 0
            return round(
                self.total_response_time_ms / self.conversation_rounds / 1000,
                1,
            )

    def get_tool_stats(self) -> list[dict]:
        with self._lock:
            stats = []
            for name, total in self.tool_call_counts.items():
                successful = self.tool_success_counts[name]
                failed = self.tool_fail_counts[name]
                durations = self.tool_durations[name]
                stats.append(
                    {
                        "tool_name": name,
                        "total": total,
                        "success": successful,
                        "fail": failed,
                        "success_rate": (
                            round(successful / total * 100, 1) if total else 100.0
                        ),
                        "avg_duration_ms": (
                            round(sum(durations) / len(durations), 1)
                            if durations
                            else 0
                        ),
                    }
                )
            stats.sort(key=lambda item: item["total"], reverse=True)
            return stats

    def get_recent_records(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(reversed(self.tool_call_records[-limit:]))
