"""
5 个评测 pipeline 工厂，统一签名 (query: str, k: int) -> list[Document]。

设计要点
--------
- 复用 get_vector_store_service() 单例，不重建 Chroma 也不重建 BM25。
- 父子块 / Reranker 都是生产代码原样复用，仅在外层组合中开/关。
- Reranker 的 score_threshold 在 rerank() 调用时传 0.0，绕过门控，让消融公平。
- 每个 pipeline 都内部召回更宽的 pool，再切到 k，确保各 pipeline 都在
  "返回给用户的 top-k 文档"这一同一标准上比较。
"""
from __future__ import annotations

from langchain_core.documents import Document

from rag.vector_store import get_vector_store_service
from rag.parent_child_retriever import ParentChildResolver
from rag.reranker import BGERerankerService

_vs = None
_resolver = None
_reranker = None


def _get_vs():
    global _vs
    if _vs is None:
        _vs = get_vector_store_service()
    return _vs


def _get_resolver():
    """父子块解析器（评测要求生产配置已开启 parent_child_enabled）。"""
    global _resolver
    if _resolver is None:
        vs = _get_vs()
        if not (vs.parent_child_enabled and vs.parent_docstore):
            raise RuntimeError(
                "评测要求 chroma.yml 中 parent_child_enabled: true 且已重建索引。"
                "当前向量库未启用父子块，无法跑 PC 相关消融。"
            )
        # top_k_parents 设大，由外层 pipeline 切片到 k
        _resolver = ParentChildResolver(vs.parent_docstore, top_k_parents=999)
    return _resolver


def _get_reranker() -> BGERerankerService:
    """单例 reranker。rerank 只打分排序，截断由各 pipeline 用 [:k] 负责。"""
    global _reranker
    if _reranker is None:
        _reranker = BGERerankerService()
    return _reranker


def _retrieve_children(query: str, k_pool: int, hybrid: bool) -> list[Document]:
    """
    从 Chroma 召回原始子块（生产配置下向量库存的就是子块）。
    混合管线复用生产候选入口，避免评测与线上策略漂移。
    """
    vs = _get_vs()

    if not hybrid:
        return vs.vector_store.as_retriever(search_kwargs={"k": k_pool}).invoke(query)
    return vs.retrieve(query)


# ==================== 5 个 pipeline ====================

def pipeline_vector_only(query: str, k: int) -> list[Document]:
    """baseline：纯向量检索 top-k 子块。"""
    docs = _retrieve_children(query, k_pool=k, hybrid=False)
    return docs[:k]


def pipeline_hybrid_no_pc_no_rr(query: str, k: int) -> list[Document]:
    """混合检索 top-k 子块（看 BM25 加成）。"""
    docs = _retrieve_children(query, k_pool=k, hybrid=True)
    return docs[:k]


def pipeline_hybrid_pc_no_rr(query: str, k: int) -> list[Document]:
    """混合检索 → 父子块回表去重 → top-k 父块（看父子块单独贡献）。"""
    pool = max(15, k * 3)  # 子块召回后去重会缩水，多召回保证 ≥k 个父块
    docs = _retrieve_children(query, k_pool=pool, hybrid=True)
    parents = _get_resolver().resolve(docs)
    return parents[:k]


def pipeline_hybrid_no_pc_with_rr(query: str, k: int) -> list[Document]:
    """混合检索 → BGE Reranker → top-k（看 reranker 单独贡献）。"""
    pool = max(15, k * 3)  # rerank 需要更宽的候选池
    docs = _retrieve_children(query, k_pool=pool, hybrid=True)
    return _get_reranker().rerank(query, docs, score_threshold=0.0)[:k]


def pipeline_full(query: str, k: int) -> list[Document]:
    """生产配置：混合 → 父子块 → Reranker → top-k。"""
    pool = max(15, k * 3)
    docs = _retrieve_children(query, k_pool=pool, hybrid=True)
    parents = _get_resolver().resolve(docs)
    return _get_reranker().rerank(query, parents, score_threshold=0.0)[:k]


def pipeline_full_v2_rr_then_pc(query: str, k: int) -> list[Document]:
    """
    实验流水线：Reranker 提前到子块层（在父子块去重之前）。

    动机：评测发现原 'full' 把 reranker 用在父块（~240 字）上时 MRR 反而下降。
    假设是 BGE-CrossEncoder 对长上下文的"语义聚焦度"打分被父块覆盖宽广稀释。
    本 pipeline 把 rerank 提前到子块（~160 字），由 ParentChildResolver 按
    rerank 顺序去重 —— 父块自动继承子块的相关性排序。

    rerank 本身只打分排序不截断，子块全量进 resolve，最终在 parents[:k] 截断。
    """
    pool = max(15, k * 3)
    children = _retrieve_children(query, k_pool=pool, hybrid=True)
    if not children:
        return []
    reranked = _get_reranker().rerank(query, children, score_threshold=0.0)
    # resolve 按输入顺序去重，父块继承 rerank 顺序
    parents = _get_resolver().resolve(reranked)
    return parents[:k]


PIPELINES: dict[str, callable] = {
    "vector_only": pipeline_vector_only,
    "hybrid_no_pc_no_rr": pipeline_hybrid_no_pc_no_rr,
    "hybrid_pc_no_rr": pipeline_hybrid_pc_no_rr,
    "hybrid_no_pc_with_rr": pipeline_hybrid_no_pc_with_rr,
    "full": pipeline_full,
    "full_v2_rr_then_pc": pipeline_full_v2_rr_then_pc,
}
