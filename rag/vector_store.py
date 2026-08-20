from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from utils.path_tool import get_abs_path
import os
from threading import Lock
from utils.logger_handler import logger
from utils.file_handler import text_loader, pdf_loader, get_file_hash
from rag.hybrid_retriever import DynamicHybridRetriever
from rag.document_chunker import DocumentChunker
from repositories.knowledge_repository import KnowledgeRepository
from repositories.parent_chunk_repository import ParentChunkRepository
from rag.chroma_health import check_collection_health

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
        """初始化知识库元数据、分块、向量库和混合检索组件。"""
        self.knowledge_repository = KnowledgeRepository()
        self.chunker = DocumentChunker(chroma_conf, embed_model)
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
            collection_configuration={
                "hnsw": {
                    "batch_size": chroma_conf.get("hnsw_batch_size", 100),
                    "sync_threshold": chroma_conf.get("hnsw_sync_threshold", 100),
                }
            },
        )
        health = check_collection_health(self.vector_store._collection)
        logger.info(
            "Chroma 向量健康检查通过: count=%s, dimension=%s",
            health["count"],
            health["dimension"],
        )

        retrieve_k = chroma_conf.get("retrieve_k_children", 15)
        self.hybrid_engine = DynamicHybridRetriever(
            vector_store=self.vector_store,
            k=retrieve_k,
            rerank_candidate_k=chroma_conf.get("rerank_candidate_k", 60),
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

    def retrieve(self, query: str, allowed_doc_ids=None):
        """将查询和权限范围交给混合检索器。"""
        return self.hybrid_engine.retrieve(query, allowed_doc_ids=allowed_doc_ids)

    @staticmethod
    def _snapshot_document(record):
        """保存旧 generation 清理所需的子块 ID、父块 ID 和文件路径。"""
        chunk_ids = list(getattr(record, "chunk_ids", None) or [])
        parent_ids = list(dict.fromkeys(
            chunk_id.rsplit(":child:", 1)[0]
            for chunk_id in chunk_ids
            if ":child:" in chunk_id
        ))
        return {
            "chunk_ids": chunk_ids,
            "parent_ids": parent_ids,
            "storage_key": getattr(record, "storage_key", None),
        }

    def _cleanup_staged_generation(self, child_ids, parent_ids):
        """入库失败时删除本次已经写入的 Chroma 子块和父块。"""
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

    def _delete_original_file(self, record, file_path=None):
        """在 data 目录范围内安全删除文档原始文件。"""
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
        """执行删除文档的底层流程，并按需重建 BM25。"""
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
        """按文件名查找并删除知识库文档。"""
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
        """按稳定 doc_id 查找并删除知识库文档。"""
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

    @staticmethod
    def _file_result(
        filename,
        path,
        status,
        *,
        success,
        doc_id=None,
        storage_key=None,
        previous_storage_key=None,
        error=None,
    ):
        """统一的单文件入库结果结构，供 API 展示和批量汇总使用。"""
        return {
            "filename": filename,
            "path": path,
            "doc_id": doc_id,
            "status": status,
            "success": success,
            "storage_key": storage_key,
            "previous_storage_key": previous_storage_key,
            "error": error,
        }

    @staticmethod
    def _load_file_documents(read_path):
        """根据文件扩展名选择 TXT 或 PDF 解析器。"""
        if read_path.lower().endswith(".txt"):
            return text_loader(read_path)
        if read_path.lower().endswith(".pdf"):
            return pdf_loader(read_path)
        return []

    @staticmethod
    def _storage_key_for(path, data_dir):
        """将文件绝对路径转换为相对 data 目录的可移植路径。"""
        resolved = os.path.abspath(path)
        try:
            if os.path.commonpath([data_dir, resolved]) == data_dir:
                return os.path.relpath(resolved, data_dir).replace(os.sep, "/")
        except ValueError:
            pass
        return os.path.basename(resolved)

    @staticmethod
    def _resolve_path(file_path, allowed_types):
        """过滤扩展名并解析为实际存在的绝对路径；不合格返回 None。"""
        if not file_path or not str(file_path).lower().endswith(allowed_types):
            return None
        resolved = os.path.abspath(
            file_path if os.path.isabs(file_path) else get_abs_path(file_path)
        )
        return resolved if os.path.isfile(resolved) else None

    def _dedup_result(self, existing, duplicate, *, filename, path, storage_key, file_hash):
        """命中去重（同名同内容 same / 异名同内容 duplicate）时返回跳过结果，否则 None。"""
        if (
            existing is not None
            and existing.status == "active"
            and existing.file_hash == file_hash
        ):
            return self._file_result(
                filename,
                path,
                "same",
                success=True,
                doc_id=existing.doc_id,
                storage_key=storage_key,
                previous_storage_key=existing.storage_key,
            )
        if (
            existing is None
            and duplicate is not None
            and duplicate.status == "active"
        ):
            return self._file_result(
                filename,
                path,
                "duplicate",
                success=True,
                doc_id=duplicate.doc_id,
                storage_key=storage_key,
                previous_storage_key=duplicate.storage_key,
            )
        return None

    def _cleanup_previous_generation(self, previous, staged_child_ids, filename):
        """新版本 activate 之后清理旧代残留块；失败只告警，active 数据不受影响。"""
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

    def _ingest_one(self, path, data_dir, *, allowed_roles, updated_by):
        """单文件入库全流程：去重 → 分块 → 写 Chroma → activate → 清旧代。

        hash 失败与入库过程异常都返回 status=failed 的结果 dict，不向外抛；
        入库开始前的数据库查询异常向上传播，由 load_documents 的批量兜底捕获
        （与拆分前语义一致）。
        """
        filename = os.path.basename(path)
        storage_key = self._storage_key_for(path, data_dir)
        file_hash = get_file_hash(path)
        file_type = path.rsplit(".", 1)[-1].lower() if "." in path else "unknown"
        if file_hash is None:
            logger.error(f"Failed to hash knowledge file: {filename}")
            return self._file_result(
                filename,
                path,
                "failed",
                success=False,
                storage_key=storage_key,
                error="file hash failed",
            )

        existing = self.knowledge_repository.get_by_filename(filename)
        duplicate = self.knowledge_repository.get_by_hash(file_hash)
        skipped = self._dedup_result(
            existing,
            duplicate,
            filename=filename,
            path=path,
            storage_key=storage_key,
            file_hash=file_hash,
        )
        if skipped is not None:
            return skipped

        status = (
            "updated"
            if existing is not None and existing.status == "active"
            else "new"
        )
        doc_id = existing.doc_id if existing is not None else file_hash[:16]
        previous = None
        if existing is not None and existing.status == "active":
            previous = self._snapshot_document(existing)

        roles = (
            list(allowed_roles)
            if allowed_roles is not None
            else list(getattr(existing, "allowed_roles", None) or [])
        )
        generation = f"{doc_id}:gen:{file_hash[:12]}"
        chunk_method = self.chunker.initial_chunk_method(
            self.parent_child_enabled,
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
            documents = self._load_file_documents(path)
            if not documents:
                raise ValueError("document content is empty")

            parent_count = None
            child_count = None
            if self.parent_child_enabled:
                (
                    enriched_chunks,
                    staged_child_ids,
                    staged_parent_ids,
                    parent_records,
                    chunk_method,
                ) = self.chunker.build_parent_child_chunks(
                    documents,
                    file_path=path,
                    doc_id=doc_id,
                    file_hash=file_hash,
                    file_type=file_type,
                    id_namespace=generation,
                )
                self.parent_docstore.save_batch(parent_records)
                parent_count = len(staged_parent_ids)
                child_count = len(staged_child_ids)
            else:
                # 父子块关闭时走普通 chunk 备用路径，块会直接写入 Chroma。
                (
                    enriched_chunks,
                    staged_child_ids,
                    chunk_method,
                ) = self.chunker.build_chunks(
                    documents,
                    file_path=path,
                    doc_id=doc_id,
                    file_hash=file_hash,
                    file_type=file_type,
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

            # 顺序不可交换：先 activate 新版本、后清旧代，中途失败最多留孤儿，不丢 active 数据。
            if previous is not None:
                self._cleanup_previous_generation(previous, staged_child_ids, filename)

            return self._file_result(
                filename,
                path,
                status,
                success=True,
                doc_id=doc_id,
                storage_key=storage_key,
                previous_storage_key=(
                    previous["storage_key"] if previous is not None else None
                ),
            )
        except Exception as exc:
            logger.error(f"Knowledge ingestion failed for {filename}: {exc}", exc_info=True)
            self._cleanup_staged_generation(staged_child_ids, staged_parent_ids)
            if ingestion_started:
                try:
                    self.knowledge_repository.mark_failed(doc_id, str(exc))
                except Exception as mark_exc:
                    logger.warning(f"Failed to mark {filename} as failed: {mark_exc}")
            return self._file_result(
                filename,
                path,
                "failed",
                success=False,
                doc_id=doc_id,
                storage_key=storage_key,
                previous_storage_key=(
                    previous["storage_key"] if previous is not None else None
                ),
                error=str(exc),
            )

    def load_document(
        self,
        file_path,
        allowed_roles=None,
        updated_by=None,
        return_details=False,
    ):
        """入库单个文件。不扫描 data/，也不因磁盘缺文件删库记录。"""
        data_dir = os.path.abspath(get_abs_path(chroma_conf["data_path"]))
        allowed_types = tuple(chroma_conf["allow_knowledge_file_type"])
        resolved = self._resolve_path(file_path, allowed_types)
        if resolved is None:
            detail = self._file_result(
                os.path.basename(file_path or ""),
                file_path,
                "failed",
                success=False,
                error="unsupported type or file not found",
            )
        else:
            detail = self._ingest_one(
                resolved,
                data_dir,
                allowed_roles=allowed_roles,
                updated_by=updated_by,
            )

        status = detail["status"]
        new_count = 1 if status == "new" else 0
        updated_count = 1 if status == "updated" else 0
        skipped_count = 1 if status in ("same", "duplicate") else 0

        if new_count or updated_count:
            try:
                self.hybrid_engine.rebuild_bm25()
            except Exception as exc:
                logger.warning(f"BM25 rebuild deferred after knowledge update: {exc}")

        if return_details:
            return {
                "new_count": new_count,
                "updated_count": updated_count,
                "skipped_count": skipped_count,
                "removed_count": 0,
                "files": [detail],
            }
        return (new_count, updated_count, skipped_count, 0)

    def load_documents(
        self,
        file_paths,
        allowed_roles=None,
        updated_by=None,
        return_details=False,
    ):
        """逐个调用单文件入库流程，并汇总本批次处理结果。"""
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
                    path,
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
