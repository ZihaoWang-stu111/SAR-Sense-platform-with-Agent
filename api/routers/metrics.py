import logging

from fastapi import APIRouter, Depends

from api.auth import get_current_user, require_admin
from api.dependencies import get_metrics

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])


@router.get("")
async def get_metrics_data(user: dict = Depends(get_current_user)):
    """Get observability metrics（需登录：指标含全量工具调用记录，可能含其他用户查询片段）"""
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
async def reset_metrics(admin: dict = Depends(require_admin)):
    """Reset all metrics（限管理员：清空全量指标，影响可观测性）"""
    metrics = get_metrics()
    metrics.reset()
    return {'success': True, 'message': 'Metrics reset successfully'}
