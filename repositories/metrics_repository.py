import threading

from sqlalchemy import case, delete, func, select

from config.db_conf import SyncSessionLocal
from models.metrics import MetricEvent


class MetricsRepository:
    """Synchronous persistence and historical aggregation for agent metrics."""

    _lock = threading.RLock()

    def __init__(self, session_factory=SyncSessionLocal):
        self.session_factory = session_factory

    def record_event(
        self,
        user_id: int,
        event_type: str,
        tool_name: str | None = None,
        success: bool | None = None,
        duration_ms: float | None = None,
    ) -> None:
        with self._lock:
            with self.session_factory.begin() as session:
                session.add(
                    MetricEvent(
                        user_id=user_id,
                        event_type=event_type,
                        tool_name=tool_name,
                        success=success,
                        duration_ms=duration_ms,
                    )
                )

    def aggregate(self, limit: int = 50) -> dict:
        with self.session_factory() as session:
            conversation_rounds = self._event_count(session, "conversation_timing")
            valid_tool_filter = (
                MetricEvent.event_type == "tool_call",
                MetricEvent.success.is_not(None),
            )
            total_tool_calls = int(
                session.scalar(
                    select(func.count())
                    .select_from(MetricEvent)
                    .where(*valid_tool_filter)
                )
                or 0
            )
            llm_call_count = self._event_count(session, "llm_call")
            successful_tool_calls = session.scalar(
                select(func.count())
                .select_from(MetricEvent)
                .where(
                    MetricEvent.event_type == "tool_call",
                    MetricEvent.success.is_(True),
                )
            ) or 0
            avg_response_time_ms = session.scalar(
                select(func.avg(MetricEvent.duration_ms)).where(
                    MetricEvent.event_type == "conversation_timing"
                )
            )

            total_label = func.count(MetricEvent.id).label("total")
            success_label = func.sum(
                case((MetricEvent.success.is_(True), 1), else_=0)
            ).label("success")
            first_id_label = func.min(MetricEvent.id).label("first_id")
            tool_rows = session.execute(
                select(
                    MetricEvent.tool_name,
                    total_label,
                    success_label,
                    func.avg(MetricEvent.duration_ms).label("avg_duration_ms"),
                    first_id_label,
                )
                .where(*valid_tool_filter)
                .group_by(MetricEvent.tool_name)
                .order_by(total_label.desc(), first_id_label.asc())
            ).all()

            recent_events = list(
                session.scalars(
                    select(MetricEvent)
                    .where(*valid_tool_filter)
                    .order_by(MetricEvent.created_at.desc(), MetricEvent.id.desc())
                    .limit(limit)
                ).all()
            )

        tool_stats = []
        for row in tool_rows:
            total = int(row.total)
            successful = int(row.success or 0)
            tool_stats.append(
                {
                    "tool_name": row.tool_name,
                    "total": total,
                    "success": successful,
                    "fail": total - successful,
                    "success_rate": round(successful / total * 100, 1) if total else 100.0,
                    "avg_duration_ms": round(float(row.avg_duration_ms or 0), 1),
                }
            )

        recent_records = [
            {
                "tool_name": event.tool_name,
                "timestamp": event.created_at.strftime("%H:%M:%S"),
                "success": bool(event.success),
                "duration_ms": round(event.duration_ms or 0, 1),
            }
            for event in recent_events
        ]

        return {
            "conversation_rounds": conversation_rounds,
            "total_tool_calls": total_tool_calls,
            "overall_success_rate": (
                round(successful_tool_calls / total_tool_calls * 100, 1)
                if total_tool_calls
                else 100.0
            ),
            "avg_tool_calls_per_round": (
                round(total_tool_calls / conversation_rounds, 1)
                if conversation_rounds
                else 0
            ),
            "avg_response_time_s": (
                round(float(avg_response_time_ms) / 1000, 1)
                if avg_response_time_ms is not None
                else 0
            ),
            "llm_call_count": llm_call_count,
            "tool_stats": tool_stats,
            "recent_records": recent_records,
        }

    @staticmethod
    def _event_count(session, event_type: str) -> int:
        return int(
            session.scalar(
                select(func.count())
                .select_from(MetricEvent)
                .where(MetricEvent.event_type == event_type)
            )
            or 0
        )

    def delete_all(self) -> int:
        with self._lock:
            return self._delete_all()

    def _delete_all(self) -> int:
        with self.session_factory.begin() as session:
            result = session.execute(delete(MetricEvent))
            return int(result.rowcount or 0)

    def reset(self, memory_reset_callback) -> int:
        with self._lock:
            deleted = self._delete_all()
            memory_reset_callback()
            return deleted
