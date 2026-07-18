import os
import pickle
import re
import warnings
from threading import Lock

from pydantic import ConfigDict

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from utils.file_handler import get_file_hash
from utils.logger_handler import logger  # 复用你的日志模块


class FilteredBM25Retriever(BaseRetriever):
    source: BM25Retriever
    allowed_doc_ids: set | None = None
    active_chunk_ids: set | None = None
    k: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        allowed = self.allowed_doc_ids
        if allowed is not None and not allowed:
            return []
        active = self.active_chunk_ids
        if active is not None and not active:
            return []

        processed_query = self.source.preprocess_func(query)
        scores = self.source.vectorizer.get_scores(processed_query)
        ranked = []
        for score, doc in zip(scores, self.source.docs):
            metadata = doc.metadata or {}
            if allowed is not None and metadata.get("doc_id") not in allowed:
                continue
            if active is not None and metadata.get("chunk_id") not in active:
                continue
            ranked.append((float(score), doc))

        ranked.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, doc in ranked[: self.k]:
            metadata = dict(doc.metadata or {})
            metadata["bm25_score"] = score
            results.append(Document(page_content=doc.page_content, metadata=metadata, id=doc.id))
        return results


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

    def __init__(
        self,
        vector_store,
        k=3,
        manifest_path=None,
        bm25_cache_path=None,
        fingerprint_provider=None,
        knowledge_repository=None,
        active_chunk_ids_provider=None,
    ):
        self.vector_store = vector_store
        self.k = k
        # 优先使用注入的数据库指纹判断 BM25 缓存是否有效。
        # manifest_path 仅作为旧调用方的兼容回退路径。
        self.manifest_path = manifest_path
        if fingerprint_provider is None and knowledge_repository is not None:
            fingerprint_provider = knowledge_repository.fingerprint
        self.fingerprint_provider = fingerprint_provider
        if active_chunk_ids_provider is None and knowledge_repository is not None:
            active_chunk_ids_provider = knowledge_repository.active_chunk_ids
        self.active_chunk_ids_provider = active_chunk_ids_provider
        self.bm25_cache_path = bm25_cache_path
        self._bm25_lock = Lock()
        # 实例化时，直接在内存中炼制好 BM25 备用
        self.bm25_retriever = self._build_bm25()

    def _get_current_fingerprint(self):
        if self.fingerprint_provider is not None:
            return self.fingerprint_provider()
        if self.manifest_path:
            return get_file_hash(self.manifest_path)
        return None

    def _get_active_chunk_ids(self):
        if self.active_chunk_ids_provider is None:
            return None
        return set(self.active_chunk_ids_provider())

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

    def _build_bm25(self, force=False):
        """构建/加载 BM25 检索器。
        force=False（默认）：load pkl 优先，pkl 存在且知识库指纹相符则直接 load（省几十秒重建）。
        force=True：强制重建（知识库变更后），重建后覆盖 pkl。
        整个 load-or-rebuild 在 self._bm25_lock 内原子执行，防并发双重重建。
        """
        with self._bm25_lock:
            current_fingerprint = self._get_current_fingerprint()

            # load 优先：pkl 存在且指纹相符则直接用
            if not force and self.bm25_cache_path and current_fingerprint is not None:
                cached = self._load_bm25_cache(current_fingerprint)
                if cached is not None:
                    logger.info("BM25 从 pkl 加载完成（跳过重建）")
                    return cached

            # 重建：从 Chroma 拉全量文档 + 分词 + 建树
            bm25_retriever = self._rebuild_bm25_from_chroma()
            if bm25_retriever is None:
                return None  # 知识库空

            # 重建后持久化（带当前知识库指纹，供下次 load 比对）
            if self.bm25_cache_path and current_fingerprint is not None:
                self._persist_bm25_cache(bm25_retriever, current_fingerprint)
            return bm25_retriever

    def _load_bm25_cache(self, current_fingerprint):
        """load pkl：指纹相符且 preprocess_func 引用未漂移则返回 bm25，否则 None。"""
        if not os.path.exists(self.bm25_cache_path):
            return None
        try:
            with open(self.bm25_cache_path, "rb") as f:
                data = pickle.load(f)
            cached_fingerprint = data.get(
                "knowledge_fingerprint",
                data.get("manifest_hash"),
            )
            if cached_fingerprint != current_fingerprint:
                logger.info("BM25 pkl 指纹不符（知识库已变更），重建")
                return None
            bm25 = data.get("bm25")
            if bm25 is None:
                return None
            # 防止反序列化后分词函数引用漂移，避免使用错误的分词器。
            # 比较 __qualname__ 字符串：pickle 往返序列化后绑定方法身份会变化，但底层函数名称不变。
            actual_func = getattr(bm25.preprocess_func, "__func__", bm25.preprocess_func)
            if getattr(actual_func, "__qualname__", "") != "DynamicHybridRetriever._preprocess_for_bm25":
                logger.warning("BM25 pkl 的 preprocess_func 引用漂移，重建")
                return None
            bm25.k = self.k
            return bm25
        except Exception as e:
            logger.warning(f"BM25 pkl 加载失败，回退重建: {e}")
            return None

    def _persist_bm25_cache(self, bm25_retriever, knowledge_fingerprint):
        """pickle 持久化 BM25 索引 + 知识库指纹。"""
        try:
            os.makedirs(os.path.dirname(self.bm25_cache_path) or ".", exist_ok=True)
            with open(self.bm25_cache_path, "wb") as f:
                pickle.dump(
                    {
                        "bm25": bm25_retriever,
                        "knowledge_fingerprint": knowledge_fingerprint,
                    },
                    f,
                )
            logger.info("BM25 索引已持久化")
        except Exception as e:
            logger.warning(f"BM25 pkl 持久化失败（不影响运行）: {e}")

    def _rebuild_bm25_from_chroma(self):
        """从 Chroma 提取所有切片，在内存中构建 BM25 检索器。"""
        try:
            logger.info("正在内存中构建 BM25 检索树...")
            all_data = self.vector_store.get(include=['documents', 'metadatas'])

            active_chunk_ids = self._get_active_chunk_ids()
            if active_chunk_ids is not None and not active_chunk_ids:
                logger.info("No active knowledge chunks; BM25 build skipped")
                return None

            docs = []
            for doc_content, meta in zip(all_data.get('documents', []), all_data.get('metadatas', [])):
                metadata = meta or {}
                if doc_content and (
                    active_chunk_ids is None
                    or metadata.get("chunk_id") in active_chunk_ids
                ):
                    docs.append(Document(page_content=doc_content, metadata=metadata))

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
        """文档变更后重建 BM25 索引（强制重建并覆盖 pkl）"""
        self.bm25_retriever = self._build_bm25(force=True)

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

    def retrieve(self, query: str, allowed_doc_ids=None) -> list[Document]:
        if allowed_doc_ids is not None:
            allowed_doc_ids = set(allowed_doc_ids)
            if not allowed_doc_ids:
                return []
        active_chunk_ids = self._get_active_chunk_ids()
        if active_chunk_ids is not None and not active_chunk_ids:
            return []
        return self.get_retriever(
            query,
            allowed_doc_ids=allowed_doc_ids,
            active_chunk_ids=active_chunk_ids,
        ).invoke(query)

    def get_retriever(self, query: str, allowed_doc_ids=None, active_chunk_ids=None):
        """根据当前 query，返回定制化权重的混合检索器"""
        search_kwargs = {"k": self.k}
        if active_chunk_ids is None:
            active_chunk_ids = self._get_active_chunk_ids()

        filters = []
        if active_chunk_ids is not None:
            filters.append({"chunk_id": {"$in": sorted(active_chunk_ids)}})
        if allowed_doc_ids is not None:
            filters.append({"doc_id": {"$in": sorted(allowed_doc_ids)}})
        if len(filters) == 1:
            search_kwargs["filter"] = filters[0]
        elif filters:
            search_kwargs["filter"] = {"$and": filters}
        vector_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)

        # 降级保护
        if not self.bm25_retriever:
            return vector_retriever

        # 获取动态权重，假设返回的是 [0.5, 0.5]
        weights = self._get_dynamic_weights(query)


        bm25_retriever = self.bm25_retriever
        if allowed_doc_ids is not None or active_chunk_ids is not None:
            bm25_retriever = FilteredBM25Retriever(
                source=self.bm25_retriever,
                allowed_doc_ids=(
                    None if allowed_doc_ids is None else set(allowed_doc_ids)
                ),
                active_chunk_ids=(
                    None if active_chunk_ids is None else set(active_chunk_ids)
                ),
                k=self.k,
            )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights  # <--- 去掉外面的 [ ]
        )
        return ensemble_retriever
