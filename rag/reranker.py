import os

# 🌟 保留防崩咒语，防止 Windows 下的 OpenMP/MKL DLL 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# 🚨 注意：这里已经删除了禁用显卡的代码

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from utils.logger_handler import logger


class BGERerankerService:
    def __init__(self, model_name=None):
        # 三级回退：环境变量(容器注入) → 配置(rag.yml reranker_model_name，本地路径) → 公共模型名(任何机器可拉)
        # 这样本地开发继续用 rag.yml 里的 E:/... 路径（秒加载、禁联网），
        # Docker 里 compose 注入 RERANKER_MODEL_NAME=BAAI/bge-reranker-base 走 HF 下载。
        if not model_name:
            from utils.config_handler import rag_conf
            model_name = os.getenv("RERANKER_MODEL_NAME") or rag_conf.get("reranker_model_name") or "BAAI/bge-reranker-base"

        logger.info(f"正在加载重排模型: {model_name}...")
        try:
            # 本地路径存在则禁止联网；HF repo id 则允许下载
            self.reranker = CrossEncoder(
                model_name,
                device='cpu',
                local_files_only=os.path.exists(model_name),
            )
            logger.info("✅ BGE 重排模型加载完成")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
            raise e

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