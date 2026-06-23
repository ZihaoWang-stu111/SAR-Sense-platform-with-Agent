from __future__ import annotations

import os
from collections.abc import Callable

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_core.documents import Document

from rag.parent_child_retriever import ParentChildResolver
from rag.reranker import BGERerankerService
from rag.vector_store import get_vector_store_service
from utils.config_handler import chroma_conf

_vs = None
_resolver = None
_reranker = None


def _get_vs():
    global _vs
    if _vs is None:
        _vs = get_vector_store_service()
    return _vs


def _get_resolver() -> ParentChildResolver:
    global _resolver
    if _resolver is None:
        vs = _get_vs()
        if not (vs.parent_child_enabled and vs.parent_docstore):
            raise RuntimeError("parent_child_enabled is required for PC pipelines")
        _resolver = ParentChildResolver(vs.parent_docstore, top_k_parents=999)
    return _resolver


def _get_reranker() -> BGERerankerService:
    global _reranker
    if _reranker is None:
        _reranker = BGERerankerService()
    return _reranker


def _pool(k: int) -> int:
    return max(k, int(chroma_conf.get("retrieve_k_children", 20)))


def _retrieve_children(query: str, k_pool: int, hybrid: bool) -> list[Document]:
    vs = _get_vs()
    if not hybrid:
        return vs.vector_store.as_retriever(search_kwargs={"k": k_pool}).invoke(query)

    engine = vs.hybrid_engine
    old_engine_k = engine.k
    old_bm25_k = engine.bm25_retriever.k if engine.bm25_retriever else None
    engine.k = k_pool
    if engine.bm25_retriever:
        engine.bm25_retriever.k = k_pool
    try:
        return engine.get_retriever(query).invoke(query)
    finally:
        engine.k = old_engine_k
        if engine.bm25_retriever and old_bm25_k is not None:
            engine.bm25_retriever.k = old_bm25_k


def pipeline_vector_only(query: str, k: int) -> list[Document]:
    return _retrieve_children(query, k_pool=k, hybrid=False)[:k]


def pipeline_hybrid_no_pc_no_rr(query: str, k: int) -> list[Document]:
    return _retrieve_children(query, k_pool=k, hybrid=True)[:k]


def pipeline_hybrid_pc_no_rr(query: str, k: int) -> list[Document]:
    children = _retrieve_children(query, k_pool=_pool(k), hybrid=True)
    return _get_resolver().resolve(children)[:k]


def pipeline_hybrid_no_pc_with_rr(query: str, k: int) -> list[Document]:
    children = _retrieve_children(query, k_pool=_pool(k), hybrid=True)
    return _get_reranker().rerank(query, children, score_threshold=0.0)[:k]


def pipeline_full(query: str, k: int) -> list[Document]:
    """Production order: hybrid children -> child rerank -> parent resolve."""
    children = _retrieve_children(query, k_pool=_pool(k), hybrid=True)
    reranked_children = _get_reranker().rerank(query, children, score_threshold=0.0)
    return _get_resolver().resolve(reranked_children)[:k]


PIPELINES: dict[str, Callable[[str, int], list[Document]]] = {
    "vector_only": pipeline_vector_only,
    "hybrid_no_pc_no_rr": pipeline_hybrid_no_pc_no_rr,
    "hybrid_pc_no_rr": pipeline_hybrid_pc_no_rr,
    "hybrid_no_pc_with_rr": pipeline_hybrid_no_pc_with_rr,
    "full": pipeline_full,
}


PIPELINE_TOGGLES = {
    "vector_only": {"label": "vector", "bm25": False, "pc": False, "rr": False},
    "hybrid_no_pc_no_rr": {"label": "vector+BM25", "bm25": True, "pc": False, "rr": False},
    "hybrid_pc_no_rr": {"label": "vector+BM25+PC", "bm25": True, "pc": True, "rr": False},
    "hybrid_no_pc_with_rr": {"label": "vector+BM25+RR", "bm25": True, "pc": False, "rr": True},
    "full": {"label": "full", "bm25": True, "pc": True, "rr": True},
}

