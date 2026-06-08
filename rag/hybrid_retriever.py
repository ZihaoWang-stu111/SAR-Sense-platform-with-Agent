import re
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from utils.logger_handler import logger  # 复用你的日志模块


class DynamicHybridRetriever:
    """混合检索器：动态路由 BM25 + 向量检索"""

    DOMAIN_TERMS = [
        # SAR 舰船检测核心概念
        "SAR", "合成孔径雷达", "舰船检测", "目标检测", "SAR图像", "海洋监视",
        "散斑噪声", "小目标检测", "密集目标", "复杂背景", "低对比度", "多尺度",
        "虚警率", "误检率", "召回率", "精确率", "平均精度",

        # 数据集与场景
        "HRSID", "SSDD", "SAR-Ship-Dataset", "AIR-SARShip", "OpenSARShip",
        "高分辨率", "近岸", "远海", "港口区域", "岛屿周边", "海峡航道",
        "X波段", "C波段", "L波段", "HH极化", "VV极化", "VH极化", "HV极化",

        # 模型、模块和论文名
        "MBE-Net", "SFQ-Det", "YOLO", "YOLOv8", "YOLOv8n", "YOLO11", "YOLO11n",
        "Faster R-CNN", "SSD", "DETR", "CFAR", "FPN", "ACF-FPN", "EEWB", "LD-DEH",
        "Backbone", "Neck", "Head", "多分支", "边缘引导", "特征增强", "特征融合",

        # 指标和部署
        "mAP", "mAP50", "mAP50-95", "AP", "AP50", "AP_small", "AP_medium",
        "AP_large", "F1-Score", "IoU", "FPS", "模型大小", "推理速度", "边缘设备",

        # RAG/系统词
        "RAG", "LangChain", "Chroma", "ChromaDB", "向量库", "知识库", "BM25",
        "父子块", "语义分块", "混合检索",
    ]

    def __init__(self, vector_store, k=3):
        self.vector_store = vector_store
        self.k = k
        # 实例化时，直接在内存中炼制好 BM25 备用
        self.bm25_retriever = self._build_bm25()

    @classmethod
    def _register_domain_terms(cls):
        """给 jieba 注册项目专业词，避免中文术语被切碎。"""
        for term in cls.DOMAIN_TERMS:
            jieba.add_word(term)

    @classmethod
    def _preprocess_for_bm25(cls, text: str) -> list[str]:
        """中英文混合 BM25 分词，兼顾中文术语、英文型号和数字指标。"""
        if not text:
            return []

        cls._register_domain_terms()
        normalized = text.lower()

        # 保留英文、型号和指标，如 sar、mbe-net、sfq-det、map50-95、yolov8n、v2.1。
        ascii_tokens = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*%?", normalized)

        chinese_tokens = [
            token.strip().lower()
            for token in jieba.lcut(normalized)
            if token.strip()
        ]

        tokens = []
        seen = set()
        for token in ascii_tokens + chinese_tokens:
            if not re.search(r"[\u4e00-\u9fffa-z0-9]", token):
                continue
            # 过滤纯单字中文，降低 BM25 噪声；英文缩写、数字指标仍保留。
            if re.fullmatch(r"[\u4e00-\u9fff]", token):
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)

        return tokens

    def _build_bm25(self):
        """从 Chroma 提取所有切片，在内存中构建 BM25 检索器"""
        try:
            logger.info("正在内存中构建 BM25 检索树...")
            all_data = self.vector_store.get(include=['documents', 'metadatas'])

            docs = []
            for doc_content, meta in zip(all_data.get('documents', []), all_data.get('metadatas', [])):
                if doc_content:
                    docs.append(Document(page_content=doc_content, metadata=meta or {}))

            if not docs:
                logger.warning("知识库为空，BM25 初始化跳过。")
                return None

            bm25_retriever = BM25Retriever.from_documents(
                docs,
                preprocess_func=self._preprocess_for_bm25
            )
            bm25_retriever.k = self.k
            logger.info("BM25 检索树构建完成！")
            return bm25_retriever

        except Exception as e:
            logger.error(f"初始化 BM25 检索器失败: {str(e)}", exc_info=True)
            return None

    def rebuild_bm25(self):
        """文档变更后重建 BM25 索引"""
        self.bm25_retriever = self._build_bm25()

    def _get_dynamic_weights(self, query: str):
        """根据中英文混合查询特征，动态分配 [向量权重, BM25权重]。"""
        if not query:
            return [0.5, 0.5]

        query = query.strip()
        normalized = query.lower()
        query_length = len(query)
        tokens = self._preprocess_for_bm25(query)

        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
        has_ascii = bool(re.search(r"[a-zA-Z0-9]", query))
        has_number = bool(re.search(r"\d", query))

        exact_patterns = [
            r"\bmap\b", r"\bmap50\b", r"\bmap50-95\b", r"\bap50\b",
            r"\bmbe[-_]?net\b", r"\bsfq[-_]?det\b", r"\byolo\w*\b",
            r"\bcfar\b", r"\bsar\b", r"\bhrsid\b", r"\bssdd\b",
            r"\bfaster\s*r[- ]?cnn\b", r"\bssd\b", r"\bdetr\b",
            r"\bhh\b", r"\bvv\b", r"\bvh\b", r"\bhv\b",
            r"召回率", r"精确率", r"准确率", r"平均精度", r"误检率", r"虚警率",
            r"小目标", r"密集目标", r"极化", r"波段", r"场景id",
        ]
        has_exact_term = any(re.search(pattern, normalized) for pattern in exact_patterns)

        if query_length >= 60 and not has_exact_term:
            # 长描述更像语义问答，向量语义更可靠。
            return [0.7, 0.3]

        if has_exact_term or has_number:
            # 型号、指标、数字和专有名词更适合关键词精确召回。
            return [0.35, 0.65]

        if query_length <= 20 and len(tokens) <= 5:
            # 很短的问题通常是在搜关键词。
            return [0.4, 0.6]

        if has_chinese and has_ascii:
            # 中英文混合问题通常既需要语义，也需要术语精确匹配。
            return [0.45, 0.55]

        return [0.5, 0.5]

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
