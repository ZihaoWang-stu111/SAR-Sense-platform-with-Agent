import logging

from fastapi import APIRouter

from api.dependencies import get_metrics

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])


@router.get("")
async def get_metrics_data():
    """Get observability metrics"""
    metrics = get_metrics()
    data = {
        'conversation_rounds': metrics.conversation_rounds,
        'total_tool_calls': metrics.total_tool_calls,
        'overall_success_rate': metrics.overall_success_rate,
        'avg_tool_calls_per_round': metrics.avg_tool_calls_per_round,
        'avg_response_time_s': metrics.avg_response_time_s,
        'llm_call_count': metrics.llm_call_count,
        'tool_stats': metrics.get_tool_stats(),
        'recent_records': metrics.get_recent_records()
    }
    return {'success': True, 'metrics': data}


@router.post("/reset")
async def reset_metrics():
    """Reset all metrics"""
    metrics = get_metrics()
    metrics.reset()
    return {'success': True, 'message': 'Metrics reset successfully'}
