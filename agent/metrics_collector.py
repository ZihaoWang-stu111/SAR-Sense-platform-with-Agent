"""Agent 指标收集（内存实时统计 + 同步写 MySQL）。

设计：
- 内存：单例 + dict/list，/api/metrics 直接读，零延迟。
- 持久化：每次 record_* 同步写一条 metric_events 到 MySQL（pymysql 直连）。
  写失败只 warning 不抛，绝不影响 agent 执行。
- 不按用户隔离（暂时统一写 user_id=1=种子用户），二期可加 user_id 透传。
"""
import time
from collections import defaultdict
from datetime import datetime

from crud.metrics import insert_metric_event


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

    def start_conversation(self):
        self.conversation_rounds += 1
        self.conversation_start_time = time.time()

    def end_conversation(self):
        if self.conversation_start_time:
            duration_ms = (time.time() - self.conversation_start_time) * 1000
            self.total_response_time_ms += duration_ms
            self.conversation_start_time = None
            insert_metric_event(event_type="conversation_timing", duration_ms=duration_ms)

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
        insert_metric_event(
            event_type="tool_call", tool_name=tool_name,
            success=success, duration_ms=duration_ms,
        )

    def record_llm_call(self):
        self.llm_call_count += 1
        insert_metric_event(event_type="llm_call")

    def reset(self):
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
