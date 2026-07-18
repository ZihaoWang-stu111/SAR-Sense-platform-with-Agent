"""Shared singleton getters for API route handlers."""

_yolo_model = None
_agent = None
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
