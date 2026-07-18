from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
import os
from threading import Lock
from utils.logger_handler import logger
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type, \
    get_file_hash
from rag.hybrid_retriever import DynamicHybridRetriever
from repositories.knowledge_repository import KnowledgeRepository
from repositories.parent_chunk_repository import ParentChunkRepository

_vector_store_service = None
_vector_store_lock = Lock()


def get_vector_store_service():
    """返回 RAG 和知识库管理共用的 VectorStoreService 实例。"""
    global _vector_store_service

    if _vector_store_service is None:
        with _vector_store_lock:
            if _vector_store_service is None:
                _vector_store_service = VectorStoreService()

    return _vector_store_service


class VectorStoreService:
    def __init__(self):
        self.knowledge_repository = KnowledgeRepository()
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"])

        )

        self.pdf_splitter = self._build_spliter(chunk_size=chroma_conf.get("pdf_chunk_size", chroma_conf["chunk_size"]),
                                                chunk_overlap=chroma_conf.get("pdf_chunk_overlap", chroma_conf["chunk_overlap"]))
        self.txt_splitter = self._build_spliter(chunk_size=chroma_conf.get("txt_chunk_size", chroma_conf["chunk_size"]),
                                                chunk_overlap=chroma_conf.get("txt_chunk_overlap", chroma_conf["chunk_overlap"]))
        self.default_splitter = self._build_spliter(chunk_size=chroma_conf.get("chunk_size"),
                                                    chunk_overlap=chroma_conf.get("chunk_overlap"))

        retrieve_k = chroma_conf.get("retrieve_k_children", 15)
        self.hybrid_engine = DynamicHybridRetriever(
            vector_store=self.vector_store,
            k=retrieve_k,
            manifest_path=None,
            bm25_cache_path=get_abs_path(chroma_conf.get("bm25_cache_path", "runtime/bm25_index.pkl")),
            knowledge_repository=self.knowledge_repository,
            active_chunk_ids_provider=self.knowledge_repository.active_chunk_ids,
        )

        # 父子块检索配置
        self.parent_child_enabled = chroma_conf.get("parent_child_enabled", False)
        self.parent_docstore = None
        if self.parent_child_enabled:
            self.parent_docstore = ParentChunkRepository()
            self.child_splitter = self._build_spliter(
                chunk_size=chroma_conf.get("child_chunk_size", 120),
                chunk_overlap=chroma_conf.get("child_chunk_overlap", 30),
            )

        # 语义断点分块配置
        self.semantic_enabled = chroma_conf.get("semantic_chunking_enabled", True)
        self.semantic_threshold = chroma_conf.get("semantic_threshold", 0.5)
        self.semantic_min_size = chroma_conf.get("semantic_min_chunk_size", 100)
        self.semantic_max_size = chroma_conf.get("semantic_max_chunk_size", 800)
        # 英文（拉丁文本）用更大的字符阈值：一句英文≈100字符，沿用中文阈值会让单句成块
        self.semantic_min_size_en = chroma_conf.get("semantic_min_chunk_size_en", 350)
        self.semantic_max_size_en = chroma_conf.get("semantic_max_chunk_size_en", 1500)
        self.semantic_cjk_ratio = chroma_conf.get("semantic_cjk_ratio_threshold", 0.2)

        # 知识库清单注册表（替代原 md5.text）
    def get_retriever(self, query: str):
        return self.hybrid_engine.get_retriyever(query)

    def retrieve(self, query: str, allowed_doc_ids=None):
        return self.hybrid_engine.retrieve(query, allowed_doc_ids=allowed_doc_ids)

    @property
    def manifest(self):
        """返回由 MySQL 生成的、兼容旧调用方的最新知识库清单。"""
        return self.knowledge_repository.as_manifest()

    def _build_spliter(self, chunk_size, chunk_overlap):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=chroma_conf["separators"],
            length_function=len
        )
    def _get_splitter(self, read_path: str) -> RecursiveCharacterTextSplitter:
        """根据文件类型选择更合适的切块器。"""
        if read_path.endswith(".txt"):
            return self.txt_splitter
        if read_path.endswith(".pdf"):
            return self.pdf_splitter
        return self.default_splitter

    @staticmethod
    def _cosine_sim(a, b):
        """计算两个向量的余弦相似度（纯 Python，无需 numpy）。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _cjk_ratio(text):
        """CJK 字符在中英文字符中的占比，用于判定文本主语言。"""
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        base = cjk + latin
        return cjk / base if base else 1.0

    def _semantic_split(self, documents):
        """基于相邻句子 embedding 相似度的语义断点分块。"""
        # 第0步：按主语言选阈值（英文用更大字符阈值，避免单句英文成块）
        joined = "".join(d.page_content for d in documents)
        if self._cjk_ratio(joined) < self.semantic_cjk_ratio:
            min_size, max_size = self.semantic_min_size_en, self.semantic_max_size_en
        else:
            min_size, max_size = self.semantic_min_size, self.semantic_max_size

        # 第1步：切分为句子级单元
        sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=min_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", "：", ":"]
        )
        sentences = sentence_splitter.split_documents(documents)
        if len(sentences) <= 1:
            return documents

        # 第2步：批量计算 embedding
        texts = [d.page_content for d in sentences]
        try:
            embeddings = embed_model.embed_documents(texts)
        except Exception as e:
            logger.warning(f"语义分块嵌入失败，回退到固定分块: {e}")
            return None

        # 第3步：计算相邻句子相似度，找到语义断点
        boundaries = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_sim(embeddings[i], embeddings[i + 1])
            if sim < self.semantic_threshold:
                boundaries.append(i)

        # 第4步：按断点合并句子为 chunk（chunk_index 由 _enrich_chunks 统一分配）
        chunks = []
        start = 0
        all_ends = boundaries + [len(sentences) - 1]

        for end in all_ends:
            segment = [sentences[j].page_content for j in range(start, end + 1)]
            merged = "\n".join(segment)

            base_meta = sentences[start].metadata if sentences[start].metadata else {}
            if len(merged) <= max_size:
                chunks.append(Document(page_content=merged, metadata=base_meta))
            else:
                # 超长段落在句子边界做子切分
                sub_texts = []
                sub_len = 0
                for j in range(start, end + 1):
                    seg = sentences[j].page_content
                    if sub_len + len(seg) > max_size and sub_texts:
                        chunks.append(Document(page_content="\n".join(sub_texts), metadata=base_meta))
                        sub_texts = [seg]
                        sub_len = len(seg)
                    else:
                        sub_texts.append(seg)
                        sub_len += len(seg)
                if sub_texts:
                    chunks.append(Document(page_content="\n".join(sub_texts), metadata=base_meta))

            start = end + 1

        # 第5步：合并过小的 chunk 到相邻 chunk（保留首个 chunk 的 metadata）
        final_chunks = []
        pending = None
        for ch in chunks:
            if len(ch.page_content) < min_size:
                if pending:
                    pending = Document(
                        page_content=pending.page_content + "\n" + ch.page_content,
                        metadata=pending.metadata
                    )
                else:
                    pending = ch
            else:
                if pending:
                    merged_text = pending.page_content + "\n" + ch.page_content
                    # 保留较大 chunk 的 metadata（含 page 等有用字段）
                    merged_meta = ch.metadata if ch.metadata else pending.metadata
                    final_chunks.append(Document(page_content=merged_text, metadata=merged_meta))
                    pending = None
                else:
                    final_chunks.append(ch)
        if pending and final_chunks:
            final_chunks[-1] = Document(
                page_content=final_chunks[-1].page_content + "\n" + pending.page_content,
                metadata=final_chunks[-1].metadata
            )
        elif pending:
            final_chunks.append(pending)

        result = final_chunks if final_chunks else sentences
        logger.info(
            f"语义分块完成: {len(sentences)}个句子 → {len(boundaries)}个断点 → {len(result)}个chunk"
        )
        return result

    def _enrich_chunks(
        self,
        chunks,
        doc_id,
        file_hash,
        file_path,
        file_type,
        id_namespace=None,
    ):
        """为 chunk 批量注入完整 metadata（doc_id / chunk_id / file_hash / source / filename / file_type / chunk_index）
           并生成稳定 chunk_id 序列"""
        filename = os.path.basename(file_path)
        enriched_chunks = []
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            cid = f"{id_namespace}:{i:04d}" if id_namespace else f"{doc_id}_{i}"
            meta = {
                "doc_id": doc_id,
                "chunk_id": cid,
                "file_hash": file_hash,
                "source": file_path,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
            }
            # PDF 文档如果有 page 信息，保留
            if "page" in chunk.metadata:
                meta["page"] = chunk.metadata["page"]

            enriched_chunks.append(Document(
                page_content=chunk.page_content,
                metadata=meta
            ))
            chunk_ids.append(cid)

        return enriched_chunks, chunk_ids

    def _structured_split(self, documents, doc_id):
        """MinerU content_list 重组后的结构感知分块。
        - 文本块 → 攒一批交给原 _semantic_split（只切正文，不处理表格/公式）
        - 表格/公式 → 原子保留，整块作为 parent，page_content 带 caption
        - 表格生成 table_id 便于溯源
        不做跨页合并/大表切分/章节栈（YAGNI，召回不行再加）。"""
        text_buffer = []
        parents = []
        table_seq = 0

        def flush_text():
            if not text_buffer:
                return
            chunks = self._semantic_split(list(text_buffer))
            if chunks is None:  # embedding 失败回退
                chunks = self.pdf_splitter.split_documents(list(text_buffer))
            for ch in chunks:
                meta = dict(ch.metadata or {})
                meta["chunk_type"] = "text"
                meta["mineru_structured"] = True
                parents.append(Document(page_content=ch.page_content, metadata=meta))
            text_buffer.clear()

        for doc in documents:
            mtype = doc.metadata.get("mineru_type", "text")
            page = doc.metadata.get("page")

            if mtype == "table":
                flush_text()
                table_seq += 1
                meta = dict(doc.metadata)
                meta["chunk_type"] = "table"
                meta["table_id"] = f"{doc_id}:table:{table_seq:03d}"
                meta["page"] = page
                parents.append(Document(page_content=doc.page_content, metadata=meta))
            elif mtype == "equation":
                flush_text()
                meta = dict(doc.metadata)
                meta["chunk_type"] = "equation"
                meta["page"] = page
                parents.append(Document(page_content=doc.page_content, metadata=meta))
            else:
                text_buffer.append(doc)

        flush_text()
        return parents

    def _build_parent_child_chunks(
        self,
        parent_docs,
        doc_id,
        file_hash,
        file_path,
        file_type,
        id_namespace=None,
    ):
        """父块写入 docstore，子块写入 Chroma。"""
        filename = os.path.basename(file_path)
        child_chunks = []
        child_ids = []
        parent_ids = []
        parent_records = {}

        for parent_index, parent in enumerate(parent_docs):
            namespace = id_namespace or doc_id
            parent_id = f"{namespace}:parent:{parent_index:03d}"
            parent_ids.append(parent_id)

            parent_meta = {
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
                parent_meta["page"] = parent.metadata["page"]
            # 透传 MinerU 结构化字段（table_id 溯源 / mineru_type 识别原子块）
            for k in ("mineru_type", "table_id", "mineru_structured"):
                if k in parent.metadata:
                    parent_meta[k] = parent.metadata[k]

            parent_records[parent_id] = {
                "page_content": parent.page_content,
                "metadata": parent_meta,
            }

            # 表格和公式是原子块，不进行字符切分，整块作为 child（child_splitter 的空字符串兜底会切碎 HTML）
            if parent.metadata.get("chunk_type") in ("table", "equation"):
                children = [parent]
            else:
                children = self.child_splitter.split_documents([parent])
                if not children:
                    children = [parent]

            for child_index, child in enumerate(children):
                child_id = f"{parent_id}:child:{child_index:03d}"
                child_meta = {
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
                if "page" in parent_meta:
                    child_meta["page"] = parent_meta["page"]
                for k in ("mineru_type", "table_id", "mineru_structured"):
                    if k in parent_meta:
                        child_meta[k] = parent_meta[k]

                child_chunks.append(Document(page_content=child.page_content, metadata=child_meta))
                child_ids.append(child_id)

        return child_chunks, child_ids, parent_ids, parent_records

    @staticmethod
    def _snapshot_document(record, manifest_entry):
        return {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "file_hash": record.file_hash,
            "storage_key": getattr(record, "storage_key", None),
            "file_type": getattr(record, "file_type", None),
            "chunk_method": getattr(record, "chunk_method", None),
            "chunk_ids": list(getattr(record, "chunk_ids", None) or []),
            "chunk_count": getattr(record, "chunk_count", 0) or 0,
            "parent_ids": list((manifest_entry or {}).get("parent_ids") or []),
            "parent_count": getattr(record, "parent_count", None),
            "child_count": getattr(record, "child_count", None),
            "allowed_roles": list(getattr(record, "allowed_roles", None) or []),
            "updated_by": getattr(record, "updated_by", None),
            "ingested_at": getattr(record, "ingested_at", None),
        }

    def _cleanup_staged_generation(self, child_ids, parent_ids):
        if child_ids:
            try:
                self.vector_store.delete(ids=child_ids)
            except Exception as exc:
                logger.warning(f"Maintenance orphan in staged Chroma children: {exc}")
        if parent_ids and self.parent_docstore:
            try:
                self.parent_docstore.delete_many(parent_ids)
            except Exception as exc:
                logger.warning(f"Maintenance orphan in staged parent chunks: {exc}")

    @staticmethod
    def _initial_chunk_method(parent_child_enabled, semantic_enabled):
        if parent_child_enabled:
            return "parent_child_semantic" if semantic_enabled else "parent_child_fixed"
        return "semantic" if semantic_enabled else "fixed"

    def _delete_original_file(self, record, file_path=None):
        data_dir = os.path.abspath(get_abs_path(chroma_conf["data_path"]))
        candidate = file_path or getattr(record, "storage_key", None) or record.filename
        candidate = os.path.abspath(
            candidate if os.path.isabs(candidate) else os.path.join(data_dir, candidate)
        )
        try:
            inside_data_dir = os.path.commonpath([data_dir, candidate]) == data_dir
        except ValueError:
            inside_data_dir = False
        if not inside_data_dir:
            logger.warning(f"Refusing to delete knowledge file outside data directory: {candidate}")
            return False
        if os.path.isfile(candidate):
            os.remove(candidate)
            return True
        return False

    def _delete_document_record(
        self,
        record,
        *,
        delete_file=False,
        file_path=None,
        rebuild_bm25=True,
    ):
        chunk_ids = list(getattr(record, "chunk_ids", None) or [])
        self.knowledge_repository.mark_deleting(record.doc_id)
        if chunk_ids:
            self.vector_store.delete(ids=chunk_ids)
        if self.parent_docstore:
            self.parent_docstore.delete_by_doc_id(record.doc_id)
        if delete_file:
            self._delete_original_file(record, file_path=file_path)
        self.knowledge_repository.delete(record.doc_id)
        if rebuild_bm25:
            self.hybrid_engine.rebuild_bm25()
        return len(chunk_ids)

    def delete_document(
        self,
        filename,
        delete_file=False,
        file_path=None,
        _rebuild_bm25=True,
    ):
        record = self.knowledge_repository.get_by_filename(filename)
        if record is None:
            logger.warning(f"Knowledge document not found: filename={filename}")
            return 0
        return self._delete_document_record(
            record,
            delete_file=delete_file,
            file_path=file_path,
            rebuild_bm25=_rebuild_bm25,
        )

    def delete_document_by_doc_id(
        self,
        doc_id,
        delete_file=False,
        file_path=None,
        _rebuild_bm25=True,
    ):
        record = self.knowledge_repository.get_by_doc_id(doc_id)
        if record is None:
            logger.warning(f"Knowledge document not found: doc_id={doc_id}")
            return 0
        return self._delete_document_record(
            record,
            delete_file=delete_file,
            file_path=file_path,
            rebuild_bm25=_rebuild_bm25,
        )

    def load_document(
        self,
        file_paths=None,
        allowed_roles=None,
        updated_by=None,
        return_details=False,
    ):
        """使用 MySQL 元数据和安全的版本替换流程完成文件入库。"""

        def get_file_documents(read_path):
            if read_path.lower().endswith(".txt"):
                return text_loader(read_path)
            if read_path.lower().endswith(".pdf"):
                return pdf_loader(read_path)
            return []

        data_dir = os.path.abspath(get_abs_path(chroma_conf["data_path"]))

        def get_storage_key(read_path):
            resolved = os.path.abspath(read_path)
            try:
                if os.path.commonpath([data_dir, resolved]) == data_dir:
                    return os.path.relpath(resolved, data_dir).replace(os.sep, "/")
            except ValueError:
                pass
            return os.path.basename(resolved)

        allowed_types = tuple(chroma_conf["allow_knowledge_file_type"])
        if file_paths is None:
            allow_files_path = list(
                listdir_with_allowed_type(
                    get_abs_path(chroma_conf["data_path"]),
                    allowed_types,
                )
            )
            known_paths = {os.path.abspath(path) for path in allow_files_path}
            for record in self.knowledge_repository.list_active():
                storage_key = getattr(record, "storage_key", None) or record.filename
                candidate = os.path.abspath(
                    storage_key
                    if os.path.isabs(storage_key)
                    else os.path.join(data_dir, storage_key)
                )
                try:
                    inside_data_dir = os.path.commonpath([data_dir, candidate]) == data_dir
                except ValueError:
                    inside_data_dir = False
                if (
                    inside_data_dir
                    and candidate not in known_paths
                    and candidate.lower().endswith(allowed_types)
                    and os.path.isfile(candidate)
                ):
                    allow_files_path.append(candidate)
                    known_paths.add(candidate)
            cleanup_missing = True
        else:
            allow_files_path = []
            for path in file_paths:
                if not path or not path.lower().endswith(allowed_types):
                    continue
                resolved = os.path.abspath(path if os.path.isabs(path) else get_abs_path(path))
                if os.path.isfile(resolved):
                    allow_files_path.append(resolved)
            cleanup_missing = False

        new_count = 0
        updated_count = 0
        skipped_count = 0
        file_details = []

        for path in allow_files_path:
            filename = os.path.basename(path)
            storage_key = get_storage_key(path)
            file_hash = get_file_hash(path)
            file_type = path.rsplit(".", 1)[-1].lower() if "." in path else "unknown"
            if file_hash is None:
                logger.error(f"Failed to hash knowledge file: {filename}")
                file_details.append(
                    {
                        "filename": filename,
                        "path": path,
                        "status": "failed",
                        "success": False,
                        "storage_key": storage_key,
                        "previous_storage_key": None,
                        "error": "file hash failed",
                    }
                )
                continue

            existing = self.knowledge_repository.get_by_filename(filename)
            duplicate = self.knowledge_repository.get_by_hash(file_hash)
            if (
                existing is not None
                and existing.status == "active"
                and existing.file_hash == file_hash
            ):
                skipped_count += 1
                file_details.append(
                    {
                        "filename": filename,
                        "path": path,
                        "doc_id": existing.doc_id,
                        "status": "same",
                        "success": True,
                        "storage_key": storage_key,
                        "previous_storage_key": existing.storage_key,
                        "error": None,
                    }
                )
                continue
            if (
                existing is None
                and duplicate is not None
                and duplicate.status == "active"
            ):
                skipped_count += 1
                file_details.append(
                    {
                        "filename": filename,
                        "path": path,
                        "doc_id": duplicate.doc_id,
                        "status": "duplicate",
                        "success": True,
                        "storage_key": storage_key,
                        "previous_storage_key": duplicate.storage_key,
                        "error": None,
                    }
                )
                continue

            status = (
                "UPDATED"
                if existing is not None and existing.status == "active"
                else "NEW"
            )
            doc_id = existing.doc_id if existing is not None else file_hash[:16]
            previous = None
            if existing is not None and existing.status == "active":
                previous = self._snapshot_document(
                    existing,
                    self.manifest.get(filename),
                )

            roles = (
                list(allowed_roles)
                if allowed_roles is not None
                else list(getattr(existing, "allowed_roles", None) or [])
            )
            generation = f"{doc_id}:gen:{file_hash[:12]}"
            chunk_method = self._initial_chunk_method(
                self.parent_child_enabled,
                self.semantic_enabled,
            )
            staged_child_ids = []
            staged_parent_ids = []
            ingestion_started = False

            try:
                if previous is None:
                    self.knowledge_repository.begin_ingestion(
                        doc_id=doc_id,
                        filename=filename,
                        file_hash=file_hash,
                        storage_key=storage_key,
                        file_type=file_type,
                        chunk_method=chunk_method,
                    )
                    ingestion_started = True
                documents = get_file_documents(path)
                if not documents:
                    raise ValueError("document content is empty")

                parent_count = None
                child_count = None
                if self.parent_child_enabled:
                    parent_docs = None
                    is_mineru_structured = bool(
                        documents and documents[0].metadata.get("mineru_structured")
                    )
                    if is_mineru_structured and chroma_conf.get("mineru_structured_split", True):
                        parent_docs = self._structured_split(documents, doc_id)
                        chunk_method = "mineru_structured"
                    elif self.semantic_enabled:
                        parent_docs = self._semantic_split(documents)
                    if parent_docs is None:
                        parent_docs = self._get_splitter(path).split_documents(documents)
                        chunk_method = "parent_child_fixed"
                    if not parent_docs:
                        raise ValueError("document produced no parent chunks")

                    (
                        enriched_chunks,
                        staged_child_ids,
                        staged_parent_ids,
                        parent_records,
                    ) = self._build_parent_child_chunks(
                        parent_docs,
                        doc_id,
                        file_hash,
                        path,
                        file_type,
                        id_namespace=generation,
                    )
                    self.parent_docstore.save_batch(parent_records)
                    parent_count = len(staged_parent_ids)
                    child_count = len(staged_child_ids)
                else:
                    split_documents = None
                    if self.semantic_enabled:
                        split_documents = self._semantic_split(documents)
                    if split_documents is None:
                        split_documents = self._get_splitter(path).split_documents(documents)
                        chunk_method = "fixed"
                    if not split_documents:
                        raise ValueError("document produced no chunks")
                    enriched_chunks, staged_child_ids = self._enrich_chunks(
                        split_documents,
                        doc_id,
                        file_hash,
                        path,
                        file_type,
                        id_namespace=generation,
                    )

                self.vector_store.add_documents(
                    enriched_chunks,
                    ids=staged_child_ids,
                )
                self.knowledge_repository.activate_document(
                    doc_id=doc_id,
                    filename=filename,
                    file_hash=file_hash,
                    storage_key=storage_key,
                    file_type=file_type,
                    chunk_method=chunk_method,
                    chunk_count=len(staged_child_ids),
                    chunk_ids=staged_child_ids,
                    parent_count=parent_count,
                    child_count=child_count,
                    allowed_roles=roles,
                    updated_by=updated_by,
                )

                if previous is not None:
                    try:
                        old_child_ids = [
                            chunk_id
                            for chunk_id in previous["chunk_ids"]
                            if chunk_id not in staged_child_ids
                        ]
                        if old_child_ids:
                            self.vector_store.delete(ids=old_child_ids)
                        if previous["parent_ids"] and self.parent_docstore:
                            self.parent_docstore.delete_many(previous["parent_ids"])
                    except Exception as exc:
                        logger.warning(
                            f"Maintenance orphan from previous generation for {filename}: {exc}"
                        )

                if status == "NEW":
                    new_count += 1
                else:
                    updated_count += 1
                file_details.append(
                    {
                        "filename": filename,
                        "path": path,
                        "doc_id": doc_id,
                        "status": status.lower(),
                        "success": True,
                        "storage_key": storage_key,
                        "previous_storage_key": (
                            previous["storage_key"] if previous is not None else None
                        ),
                        "error": None,
                    }
                )
            except Exception as exc:
                logger.error(f"Knowledge ingestion failed for {filename}: {exc}", exc_info=True)
                self._cleanup_staged_generation(staged_child_ids, staged_parent_ids)
                if ingestion_started:
                    try:
                        self.knowledge_repository.mark_failed(doc_id, str(exc))
                    except Exception as mark_exc:
                        logger.warning(f"Failed to mark {filename} as failed: {mark_exc}")
                file_details.append(
                    {
                        "filename": filename,
                        "path": path,
                        "doc_id": doc_id,
                        "status": "failed",
                        "success": False,
                        "storage_key": storage_key,
                        "previous_storage_key": (
                            previous["storage_key"] if previous is not None else None
                        ),
                        "error": str(exc),
                    }
                )

        removed_count = 0
        if cleanup_missing:
            current_filenames = {os.path.basename(path) for path in allow_files_path}
            stale_records = [
                record
                for record in self.knowledge_repository.list_active()
                if record.filename not in current_filenames
            ]
            for record in stale_records:
                self._delete_document_record(
                    record,
                    rebuild_bm25=False,
                )
                removed_count += 1

        if new_count or updated_count or removed_count:
            try:
                self.hybrid_engine.rebuild_bm25()
            except Exception as exc:
                logger.warning(f"BM25 rebuild deferred after knowledge update: {exc}")
        counts = (new_count, updated_count, skipped_count, removed_count)
        if return_details:
            return {
                "new_count": new_count,
                "updated_count": updated_count,
                "skipped_count": skipped_count,
                "removed_count": removed_count,
                "files": file_details,
            }
        return counts

    def load_documents(
        self,
        file_paths,
        allowed_roles=None,
        updated_by=None,
        return_details=False,
    ):
        if file_paths is None:
            return self.load_document(
                file_paths=None,
                allowed_roles=allowed_roles,
                updated_by=updated_by,
                return_details=return_details,
            )
        paths = list(file_paths or [])
        aggregate = {
            "new_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
            "removed_count": 0,
            "files": [],
        }
        for path in paths:
            try:
                result = self.load_document(
                    file_paths=[path],
                    allowed_roles=allowed_roles,
                    updated_by=updated_by,
                    return_details=True,
                )
            except Exception as exc:
                logger.error(
                    f"Knowledge ingestion failed before processing {path}: {exc}",
                    exc_info=True,
                )
                result = {
                    "new_count": 0,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "removed_count": 0,
                    "files": [
                        {
                            "filename": os.path.basename(path),
                            "path": path,
                            "status": "failed",
                            "success": False,
                            "storage_key": None,
                            "previous_storage_key": None,
                            "error": str(exc),
                        }
                    ],
                }

            for key in ("new_count", "updated_count", "skipped_count", "removed_count"):
                aggregate[key] += result[key]
            aggregate["files"].extend(result.get("files") or [])

        if return_details:
            return aggregate
        return (
            aggregate["new_count"],
            aggregate["updated_count"],
            aggregate["skipped_count"],
            aggregate["removed_count"],
        )





if __name__ == '__main__':
    vs = get_vector_store_service()
    vs.load_document()
