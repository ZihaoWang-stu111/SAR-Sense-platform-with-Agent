"""Shared singleton getters for API route handlers."""

import threading

_yolo_model = None
_agent = None
_agent_executor = None
_agent_executor_lock = threading.Lock()
_metrics = None
_metrics_repository = None


def get_yolo_model():
    """延迟加载 YOLO 模型。"""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        from utils.path_tool import get_abs_path
        model_path = get_abs_path("Detct_prdc/MBE-Net/weights/best.pt")
        _yolo_model = YOLO(model_path)
    return _yolo_model


def get_agent():
    """延迟加载 ReactAgent。"""
    global _agent
    if _agent is None:
        from agent.react_agent import ReactAgent
        _agent = ReactAgent()
    return _agent


def get_agent_executor():
    """返回全应用共用的有界 Agent 线程执行器。"""
    global _agent_executor
    if _agent_executor is None:
        with _agent_executor_lock:
            if _agent_executor is None:
                from services.agent_executor import AgentExecutor
                _agent_executor = AgentExecutor()
    return _agent_executor


def shutdown_agent_executor(*, wait: bool = True) -> None:
    """停止接收 Agent 任务，并按需等待已提交任务完成。"""
    global _agent_executor
    with _agent_executor_lock:
        executor = _agent_executor
        _agent_executor = None
    if executor is not None:
        executor.shutdown(wait=wait)


def get_vector_store():
    """延迟加载共享的 VectorStoreService。"""
    from rag.vector_store import get_vector_store_service
    return get_vector_store_service()


def get_metrics():
    """延迟加载 AgentMetrics。"""
    global _metrics
    if _metrics is None:
        from agent.metrics_collector import AgentMetrics
        _metrics = AgentMetrics()
    return _metrics


def get_metrics_repository():
    """延迟加载同步历史指标 Repository。"""
    global _metrics_repository
    if _metrics_repository is None:
        from repositories.metrics_repository import MetricsRepository
        _metrics_repository = MetricsRepository()
    return _metrics_repository
