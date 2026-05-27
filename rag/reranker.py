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
    # 🌟 核心修改 1：把默认参数改成你刚才下载的本地绝对路径
    # 记得前面的 'r' 一定要有，防止 Windows 路径里的反斜杠转义报错
    def __init__(self, model_name=r'E:/models/Xorbits/bge-reranker-base', top_n=3):
        logger.info(f"🚀 正在从本地加载重排模型 (准备启动 GPU 加速): {model_name}...")
        try:
            # 🌟 核心修改 2：加入 local_files_only=True，彻底禁止程序去联网！
            self.reranker = CrossEncoder(
                model_name,
                device='cpu',
                local_files_only=True
            )
            self.top_n = top_n
            logger.info("✅ BGE 重排模型已成功加载到 GPU！起飞！")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
            logger.error("👉 如果报错说找不到 CUDA 或请求 CPU 回退，说明你的环境里没有装 GPU 版的 PyTorch。")
            raise e

    def rerank(self, query: str, documents: list[Document], score_threshold: float = 0.3) -> list[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        ranked_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
        top_score = ranked_docs[0].metadata["rerank_score"]

        if top_score < score_threshold:
            logger.info(f"⚡ 重排最高分 {top_score:.4f} < 阈值 {score_threshold}，判定检索质量过低，返回空")
            return []

        logger.info(f"🎯 GPU 重排完成！最高得分: {top_score:.4f}")
        return ranked_docs[:self.top_n]