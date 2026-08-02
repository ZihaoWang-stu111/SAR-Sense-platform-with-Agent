import os

# Windows 本地默认值；部署环境可在导入前显式覆盖。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from utils.logger_handler import logger


def resolve_reranker_source(model_name, cache_resolver=None):
    """优先解析本地目录或 Hugging Face 缓存，避免启动时重复联网探测。"""
    if os.path.exists(model_name):
        return model_name, True

    if cache_resolver is None:
        from huggingface_hub import snapshot_download

        cache_resolver = snapshot_download

    try:
        cached_path = cache_resolver(
            repo_id=model_name,
            local_files_only=True,
        )
        return cached_path, True
    except Exception:
        return model_name, False


class BGERerankerService:
    def __init__(self, model_name=None):
        # 环境变量优先；配置默认使用公开模型名，也允许覆盖成本地模型路径。
        if not model_name:
            from utils.config_handler import rag_conf
            model_name = os.getenv("RERANKER_MODEL_NAME") or rag_conf.get("reranker_model_name") or "BAAI/bge-reranker-base"

        model_source, local_only = resolve_reranker_source(model_name)
        logger.info(
            "正在加载重排模型: %s (%s)...",
            model_source,
            "本地" if local_only else "首次下载",
        )
        try:
            self.reranker = CrossEncoder(
                model_source,
                device='cpu',
                local_files_only=local_only,
            )
            logger.info("✅ BGE 重排模型加载完成")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
            raise

    def rerank(self, query: str, documents: list[Document], score_threshold: float = 0.3) -> list[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)

        # 不原地改入参：BM25 检索返回的是 retriever 长期持有的共享 Document 对象，
        # 并发查询下原地写 metadata 会相互覆盖。复制成新 Document（metadata 浅拷贝），
        # 把分写到副本上，确保 reranker 只读不写入参、杜绝跨查询共享写。
        scored_docs = []
        for doc, score in zip(documents, scores):
            meta = dict(doc.metadata)
            meta["rerank_score"] = float(score)
            scored_docs.append(Document(page_content=doc.page_content, metadata=meta))

        ranked_docs = sorted(scored_docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
        top_score = ranked_docs[0].metadata["rerank_score"]

        if top_score < score_threshold:
            logger.info(f"⚡ 重排最高分 {top_score:.4f} < 阈值 {score_threshold}，判定检索质量过低，返回空")
            return []

        logger.info(f"🎯 GPU 重排完成！最高得分: {top_score:.4f}")
        # rerank 只负责打分+排序+卡门；要几个由调用方截断（resolve 取父块 / [:k]）
        return ranked_docs
