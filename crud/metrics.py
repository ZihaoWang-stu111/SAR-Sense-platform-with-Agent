"""Best-effort synchronous metric event writes for agent callbacks."""

from services.metrics_store import MetricsStore
from utils.logger_handler import logger


_metrics_store = MetricsStore()


def insert_metric_event(
    event_type: str,
    tool_name: str = None,
    success: bool = None,
    duration_ms: float = None,
    user_id: int = 1,
) -> None:
    """Insert one event without allowing DB failures to affect the agent."""
    try:
        _metrics_store.record_event(
            user_id=user_id,
            event_type=event_type,
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        logger.warning(f"[metrics] failed to write MySQL event (memory unaffected): {exc}")
