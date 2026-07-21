import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger_handler import logger


class DocumentChunker:
    """负责普通、语义、MinerU 结构化及父子块分块。"""

    def __init__(self, config: dict, embedding_model):
        """读取分块配置，并初始化普通块、父块和子块使用的 splitter。"""
        self.config = config
        self.embedding_model = embedding_model
        default_size = config["chunk_size"]
        default_overlap = config["chunk_overlap"]
        self.pdf_splitter = self._build_splitter(
            config.get("pdf_chunk_size", default_size),
            config.get("pdf_chunk_overlap", default_overlap),
        )
        self.txt_splitter = self._build_splitter(
            config.get("txt_chunk_size", default_size),
            config.get("txt_chunk_overlap", default_overlap),
        )
        self.default_splitter = self._build_splitter(default_size, default_overlap)
        self.child_splitter = self._build_splitter(
            config.get("child_chunk_size", 120),
            config.get("child_chunk_overlap", 30),
        )
        self.semantic_enabled = config.get("semantic_chunking_enabled", True)
        self.semantic_threshold = config.get("semantic_threshold", 0.5)
        self.semantic_min_size = config.get("semantic_min_chunk_size", 100)
        self.semantic_max_size = config.get("semantic_max_chunk_size", 800)
        self.semantic_min_size_en = config.get("semantic_min_chunk_size_en", 350)
        self.semantic_max_size_en = config.get("semantic_max_chunk_size_en", 1500)
        self.semantic_cjk_ratio = config.get("semantic_cjk_ratio_threshold", 0.2)

    def initial_chunk_method(self, parent_child_enabled: bool) -> str:
        """根据当前配置返回预计使用的分块方式，便于入库记录状态。"""
        if parent_child_enabled:
            return "parent_child_semantic" if self.semantic_enabled else "parent_child_fixed"
        return "semantic" if self.semantic_enabled else "fixed"

    def build_chunks(
        self,
        documents,
        *,
        file_path,
        doc_id,
        file_hash,
        file_type,
        id_namespace,
    ):
        """父子块关闭时的备用路径：生成普通块并直接写入 Chroma。"""
        # 语义分块失败或被关闭时，退回按文件类型选择固定分块器。
        chunks = self._semantic_split(documents) if self.semantic_enabled else None
        chunk_method = "semantic"
        if chunks is None:
            chunks = self._get_splitter(file_path).split_documents(documents)
            chunk_method = "fixed"
        if not chunks:
            raise ValueError("document produced no chunks")
        enriched, chunk_ids = self._enrich_chunks(
            chunks,
            doc_id,
            file_hash,
            file_path,
            file_type,
            id_namespace=id_namespace,
        )
        return enriched, chunk_ids, chunk_method

    def build_parent_child_chunks(
        self,
        documents,
        *,
        file_path,
        doc_id,
        file_hash,
        file_type,
        id_namespace,
    ):
        """先生成父块，再把父块转换为可召回的子块和父块记录。"""
        parent_docs = None
        is_mineru_structured = bool(
            documents and documents[0].metadata.get("mineru_structured")
        )
        if is_mineru_structured and self.config.get("mineru_structured_split", True):
            parent_docs = self._structured_split(documents, doc_id)
            chunk_method = "mineru_structured"
        elif self.semantic_enabled:
            parent_docs = self._semantic_split(documents)
            chunk_method = "parent_child_semantic"
        else:
            # 未启用语义分块时，下面统一退回固定长度父块。
            chunk_method = "parent_child_fixed"

        if parent_docs is None:
            # 语义 embedding 失败时也走固定分块，保证入库仍可完成。
            parent_docs = self._get_splitter(file_path).split_documents(documents)
            chunk_method = "parent_child_fixed"
        if not parent_docs:
            raise ValueError("document produced no parent chunks")

        chunks, child_ids, parent_ids, parent_records = self._make_parent_child_chunks(
            parent_docs,
            doc_id,
            file_hash,
            file_path,
            file_type,
            id_namespace=id_namespace,
        )
        return chunks, child_ids, parent_ids, parent_records, chunk_method

    def _build_splitter(self, chunk_size, chunk_overlap):
        """根据块大小、重叠长度和分隔符创建固定分块器。"""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.config["separators"],
            length_function=len,
        )

    def _get_splitter(self, read_path: str) -> RecursiveCharacterTextSplitter:
        """语义分块不可用时，按文件类型选择固定分块器。"""
        if read_path.endswith(".txt"):
            return self.txt_splitter
        if read_path.endswith(".pdf"):
            return self.pdf_splitter
        return self.default_splitter

    @staticmethod
    def _cosine_sim(a, b):
        """计算两个 embedding 向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _cjk_ratio(text):
        """计算文本中 CJK 字符在中文和拉丁字母中的占比。"""
        cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        latin = sum(1 for char in text if char.isascii() and char.isalpha())
        base = cjk + latin
        return cjk / base if base else 1.0

    def _semantic_split(self, documents):
        """根据相邻句子的 embedding 相似度寻找语义断点并合并成块。"""
        joined = "".join(document.page_content for document in documents)
        if self._cjk_ratio(joined) < self.semantic_cjk_ratio:
            min_size = self.semantic_min_size_en
            max_size = self.semantic_max_size_en
        else:
            min_size = self.semantic_min_size
            max_size = self.semantic_max_size

        sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=min_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", "：", ":"],
        )
        sentences = sentence_splitter.split_documents(documents)
        if len(sentences) <= 1:
            return documents

        try:
            embeddings = self.embedding_model.embed_documents(
                [sentence.page_content for sentence in sentences]
            )
        except Exception as exc:
            logger.warning(f"语义分块嵌入失败，回退到固定分块: {exc}")
            return None

        boundaries = [
            index
            for index in range(len(embeddings) - 1)
            if self._cosine_sim(embeddings[index], embeddings[index + 1])
            < self.semantic_threshold
        ]
        chunks = []
        start = 0
        for end in boundaries + [len(sentences) - 1]:
            base_meta = sentences[start].metadata if sentences[start].metadata else {}
            merged = "\n".join(
                sentences[index].page_content for index in range(start, end + 1)
            )
            if len(merged) <= max_size:
                chunks.append(Document(page_content=merged, metadata=base_meta))
            else:
                pieces = []
                piece_length = 0
                for index in range(start, end + 1):
                    text = sentences[index].page_content
                    if piece_length + len(text) > max_size and pieces:
                        chunks.append(
                            Document(page_content="\n".join(pieces), metadata=base_meta)
                        )
                        pieces = [text]
                        piece_length = len(text)
                    else:
                        pieces.append(text)
                        piece_length += len(text)
                if pieces:
                    chunks.append(Document(page_content="\n".join(pieces), metadata=base_meta))
            start = end + 1

        final_chunks = []
        pending = None
        for chunk in chunks:
            if len(chunk.page_content) < min_size:
                if pending:
                    pending = Document(
                        page_content=pending.page_content + "\n" + chunk.page_content,
                        metadata=pending.metadata,
                    )
                else:
                    pending = chunk
            elif pending:
                final_chunks.append(
                    Document(
                        page_content=pending.page_content + "\n" + chunk.page_content,
                        metadata=chunk.metadata if chunk.metadata else pending.metadata,
                    )
                )
                pending = None
            else:
                final_chunks.append(chunk)

        if pending and final_chunks:
            final_chunks[-1] = Document(
                page_content=final_chunks[-1].page_content + "\n" + pending.page_content,
                metadata=final_chunks[-1].metadata,
            )
        elif pending:
            final_chunks.append(pending)

        result = final_chunks if final_chunks else sentences
        logger.info(
            f"语义分块完成: {len(sentences)}个句子 → {len(boundaries)}个断点 → {len(result)}个chunk"
        )
        return result

    @staticmethod
    def _enrich_chunks(
        chunks,
        doc_id,
        file_hash,
        file_path,
        file_type,
        id_namespace=None,
    ):
        """为普通分块补齐 metadata，并生成可写入 Chroma 的 chunk ID。"""
        filename = os.path.basename(file_path)
        enriched_chunks = []
        chunk_ids = []
        for index, chunk in enumerate(chunks):
            chunk_id = (
                f"{id_namespace}:{index:04d}"
                if id_namespace
                else f"{doc_id}_{index}"
            )
            metadata = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "file_hash": file_hash,
                "source": file_path,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": index,
            }
            if "page" in chunk.metadata:
                metadata["page"] = chunk.metadata["page"]
            enriched_chunks.append(
                Document(page_content=chunk.page_content, metadata=metadata)
            )
            chunk_ids.append(chunk_id)
        return enriched_chunks, chunk_ids

    def _structured_split(self, documents, doc_id):
        """按 MinerU 类型拆分正文、表格和公式，形成结构化父块。"""
        text_buffer = []
        parents = []
        table_sequence = 0

        def flush_text():
            """将缓存的连续正文切块、标记后追加到父块列表。"""
            if not text_buffer:
                return
            chunks = self._semantic_split(list(text_buffer))
            if chunks is None:
                # 正文语义分块失败时，使用 PDF 固定分块器兜底。
                chunks = self.pdf_splitter.split_documents(list(text_buffer))
            for chunk in chunks:
                metadata = dict(chunk.metadata or {})
                metadata["chunk_type"] = "text"
                metadata["mineru_structured"] = True
                parents.append(Document(page_content=chunk.page_content, metadata=metadata))
            text_buffer.clear()

        for document in documents:
            mineru_type = document.metadata.get("mineru_type", "text")
            if mineru_type == "table":
                flush_text()
                table_sequence += 1
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "chunk_type": "table",
                        "table_id": f"{doc_id}:table:{table_sequence:03d}",
                        "page": document.metadata.get("page"),
                    }
                )
                parents.append(Document(page_content=document.page_content, metadata=metadata))
            elif mineru_type == "equation":
                flush_text()
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "chunk_type": "equation",
                        "page": document.metadata.get("page"),
                    }
                )
                parents.append(Document(page_content=document.page_content, metadata=metadata))
            else:
                text_buffer.append(document)

        flush_text()
        return parents

    def _make_parent_child_chunks(
        self,
        parent_docs,
        doc_id,
        file_hash,
        file_path,
        file_type,
        id_namespace=None,
    ):
        """为每个父块生成父块记录、子块 Document 以及对应 ID。"""
        filename = os.path.basename(file_path)
        child_chunks = []
        child_ids = []
        parent_ids = []
        parent_records = {}

        for parent_index, parent in enumerate(parent_docs):
            namespace = id_namespace or doc_id
            parent_id = f"{namespace}:parent:{parent_index:03d}"
            parent_ids.append(parent_id)
            parent_metadata = {
                "doc_id": doc_id,
                "parent_id": parent_id,
                "chunk_type": "parent",
                "parent_index": parent_index,
                "file_hash": file_hash,
                "source": file_path,
                "filename": filename,
                "file_type": file_type,
            }
            if "page" in parent.metadata:
                parent_metadata["page"] = parent.metadata["page"]
            for key in ("mineru_type", "table_id", "mineru_structured"):
                if key in parent.metadata:
                    parent_metadata[key] = parent.metadata[key]

            parent_records[parent_id] = {
                "page_content": parent.page_content,
                "metadata": parent_metadata,
            }
            if parent.metadata.get("chunk_type") in ("table", "equation"):
                children = [parent]
            else:
                children = self.child_splitter.split_documents([parent]) or [parent]

            for child_index, child in enumerate(children):
                child_id = f"{parent_id}:child:{child_index:03d}"
                child_metadata = {
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "chunk_id": child_id,
                    "chunk_type": "child",
                    "parent_index": parent_index,
                    "child_index": child_index,
                    "chunk_index": child_index,
                    "file_hash": file_hash,
                    "source": file_path,
                    "filename": filename,
                    "file_type": file_type,
                }
                if "page" in parent_metadata:
                    child_metadata["page"] = parent_metadata["page"]
                for key in ("mineru_type", "table_id", "mineru_structured"):
                    if key in parent_metadata:
                        child_metadata[key] = parent_metadata[key]
                child_chunks.append(
                    Document(page_content=child.page_content, metadata=child_metadata)
                )
                child_ids.append(child_id)

        return child_chunks, child_ids, parent_ids, parent_records
