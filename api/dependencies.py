"""Shared singleton getters for API route handlers."""

_yolo_model = None
_agent = None
_metrics = None
_metrics_store = None


def get_yolo_model():
    """Lazy load YOLO model"""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        from utils.path_tool import get_abs_path
        model_path = get_abs_path("Detct_prdc/MBE-Net/weights/best.pt")
        _yolo_model = YOLO(model_path)
    return _yolo_model


def get_agent():
    """Lazy load ReactAgent"""
    global _agent
    if _agent is None:
        from agent.react_agent import ReactAgent
        _agent = ReactAgent()
    return _agent


def get_vector_store():
    """Lazy load shared VectorStoreService."""
    from rag.vector_store import get_vector_store_service
    return get_vector_store_service()


def get_metrics():
    """Lazy load AgentMetrics"""
    global _metrics
    if _metrics is None:
        from agent.metrics_collector import AgentMetrics
        _metrics = AgentMetrics()
    return _metrics


def get_metrics_store():
    """Lazy load the synchronous historical metrics store."""
    global _metrics_store
    if _metrics_store is None:
        from services.metrics_store import MetricsStore
        _metrics_store = MetricsStore()
    return _metrics_store
