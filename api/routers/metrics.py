import logging

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from api.auth import get_current_user, require_admin
from api.dependencies import get_metrics, get_metrics_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])


@router.get("")
async def get_metrics_data(user: dict = Depends(get_current_user)):
    """Get observability metrics（需登录：指标含全量工具调用记录，可能含其他用户查询片段）"""
    store = get_metrics_store()
    data = await run_in_threadpool(store.aggregate)
    return {'success': True, 'metrics': data}


@router.post("/reset")
async def reset_metrics(admin: dict = Depends(require_admin)):
    """Reset all metrics（限管理员：清空全量指标，影响可观测性）"""
    store = get_metrics_store()
    metrics = get_metrics()
    await run_in_threadpool(store.reset, metrics.reset)
    return {'success': True, 'message': 'Metrics reset successfully'}
