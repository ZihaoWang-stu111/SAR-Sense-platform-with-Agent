from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from utils.logger_handler import logger  # 复用你的日志模块


class DynamicHybridRetriever:
    """混合检索器：动态路由 BM25 + 向量检索"""

    def __init__(self, vector_store, k=3):
        self.vector_store = vector_store
        self.k = k
        # 实例化时，直接在内存中炼制好 BM25 备用
        self.bm25_retriever = self._build_bm25()

    def _build_bm25(self):
        """从 Chroma 提取所有切片，在内存中构建 BM25 检索器"""
        try:
            logger.info("正在内存中构建 BM25 检索树...")
            all_data = self.vector_store.get(include=['documents', 'metadatas'])

            docs = []
            for doc_content, meta in zip(all_data.get('documents', []), all_data.get('metadatas', [])):
                if doc_content:
                    docs.append(Document(page_content=doc_content, metadata=meta))

            if not docs:
                logger.warning("知识库为空，BM25 初始化跳过。")
                return None

            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = self.k
            logger.info("BM25 检索树构建完成！")
            return bm25_retriever

        except Exception as e:
            logger.error(f"初始化 BM25 检索器失败: {str(e)}", exc_info=True)
            return None

    def _get_dynamic_weights(self, query: str):
        """根据用户提问的长度和词密度，动态分配 [向量权重, BM25权重]"""
        default_vector_weight = 0.5
        default_bm25_weight = 0.5

        if not query:
            return [default_vector_weight, default_bm25_weight]

        query_length = len(query)
        query_words = len(query.split())

        # 核心 AI 预判逻辑
        if query_length > 50:
            # 长难句：大概率在描述场景，倾向于向量语义
            vector_weight = 0.7
            bm25_weight = 0.3
        elif query_length < 20:
            # 短句子：大概率在搜专有名词/型号，倾向于 BM25 字面匹配
            vector_weight = 0.3
            bm25_weight = 0.7
        else:
            vector_weight = default_vector_weight
            bm25_weight = default_bm25_weight

        # 如果英文单词密度高，进一步倾向于 BM25（防止专有名词被向量模糊掉）
        if query_words > 0:
            word_density = query_words / query_length
            if word_density > 0.1:
                bm25_weight = min(bm25_weight + 0.1, 0.7)
                vector_weight = max(vector_weight - 0.1, 0.3)

        return [vector_weight, bm25_weight]

    def get_retriever(self, query: str):
        """根据当前 query，返回定制化权重的混合检索器"""
        vector_retriever = self.vector_store.as_retriever(search_kwargs={'k': self.k})

        # 降级保护
        if not self.bm25_retriever:
            return vector_retriever

        # 获取动态权重，假设返回的是 [0.5, 0.5]
        weights = self._get_dynamic_weights(query)


        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=weights  # <--- 去掉外面的 [ ]
        )
        return ensemble_retriever
