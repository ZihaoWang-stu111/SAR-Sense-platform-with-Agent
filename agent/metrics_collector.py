import time
from collections import defaultdict
from datetime import datetime

from utils.logger_handler import logger


class AgentMetrics:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reset()
        return cls._instance

    def _reset(self):
        self.conversation_rounds = 0
        self.tool_call_records = []
        self.tool_call_counts = defaultdict(int)
        self.tool_success_counts = defaultdict(int)
        self.tool_fail_counts = defaultdict(int)
        self.tool_durations = defaultdict(list)
        self.llm_call_count = 0
        self.conversation_start_time = None
        self.total_response_time_ms = 0.0
        self._db_ready = None

    def _persist_event(self, event_type: str, tool_name=None, success=None, duration_ms=None):
        """写入 metric_events。DB 失败只 warning，不影响内存与主流程。"""
        if self._db_ready is None:
            try:
                from utils.db import init_db
                init_db()
                self._db_ready = True
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[metrics] init_db 失败，指标仅内存: {e}")
                self._db_ready = False
        if not self._db_ready:
            return
        try:
            from utils.db import get_conn
            with get_conn() as conn:
                conn.execute(
                    """INSERT INTO metric_events
                       (user_id, event_type, tool_name, success, duration_ms, created_at)
                       VALUES ('default_user', ?, ?, ?, ?, ?)""",
                    (event_type, tool_name,
                     1 if success else 0 if success is not None else None,
                     duration_ms,
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[metrics] DB 写事件失败（不影响内存）: {e}")

    def start_conversation(self):
        self.conversation_rounds += 1
        self.conversation_start_time = time.time()

    def end_conversation(self):
        if self.conversation_start_time:
            duration_ms = (time.time() - self.conversation_start_time) * 1000
            self.total_response_time_ms += duration_ms
            self.conversation_start_time = None
            self._persist_event("conversation_timing", duration_ms=duration_ms)

    def record_tool_call(self, tool_name: str, success: bool, duration_ms: float):
        self.tool_call_counts[tool_name] += 1
        if success:
            self.tool_success_counts[tool_name] += 1
        else:
            self.tool_fail_counts[tool_name] += 1
        self.tool_durations[tool_name].append(duration_ms)
        self.tool_call_records.append({
            "tool_name": tool_name,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "success": success,
            "duration_ms": round(duration_ms, 1)
        })
        self._persist_event("tool_call", tool_name=tool_name,
                            success=success, duration_ms=duration_ms)

    def record_llm_call(self):
        self.llm_call_count += 1
        self._persist_event("llm_call")

    def reset(self):
        # reset 只清内存；DB 历史保留（持久化的意义）
        self._reset()

    @property
    def total_tool_calls(self):
        return sum(self.tool_call_counts.values())

    @property
    def avg_tool_calls_per_round(self):
        if self.conversation_rounds == 0:
            return 0
        return round(self.total_tool_calls / self.conversation_rounds, 1)

    @property
    def overall_success_rate(self):
        total = self.total_tool_calls
        if total == 0:
            return 100.0
        return round(sum(self.tool_success_counts.values()) / total * 100, 1)

    @property
    def avg_response_time_s(self):
        if self.conversation_rounds == 0:
            return 0
        return round(self.total_response_time_ms / self.conversation_rounds / 1000, 1)

    def get_tool_stats(self):
        stats = []
        for name in self.tool_call_counts:
            total = self.tool_call_counts[name]
            success = self.tool_success_counts[name]
            fail = self.tool_fail_counts[name]
            durations = self.tool_durations[name]
            avg_duration = round(sum(durations) / len(durations), 1) if durations else 0
            stats.append({
                "tool_name": name,
                "total": total,
                "success": success,
                "fail": fail,
                "success_rate": round(success / total * 100, 1) if total else 100,
                "avg_duration_ms": avg_duration,
            })
        stats.sort(key=lambda x: x["total"], reverse=True)
        return stats

    def get_recent_records(self, limit: int = 50):
        return list(reversed(self.tool_call_records[-limit:]))