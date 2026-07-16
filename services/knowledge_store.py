import hashlib
import json
from datetime import datetime

from sqlalchemy import select

from config.db_conf import SyncSessionLocal
from models.knowledge import KnowledgeDocument, ParentChunk


class KnowledgeStore:
    """知识库元数据的同步数据库访问层。"""

    def __init__(self, session_factory=SyncSessionLocal):
        self.session_factory = session_factory

    def get_by_doc_id(self, doc_id: str) -> KnowledgeDocument | None:
        return self._get_one(KnowledgeDocument.doc_id == doc_id)

    def get_by_filename(self, filename: str) -> KnowledgeDocument | None:
        return self._get_one(KnowledgeDocument.filename == filename)

    def get_by_hash(self, file_hash: str | None) -> KnowledgeDocument | None:
        if file_hash is None:
            return None
        return self._get_one(KnowledgeDocument.file_hash == file_hash)

    def _get_one(self, predicate) -> KnowledgeDocument | None:
        with self.session_factory() as session:
            return session.scalar(select(KnowledgeDocument).where(predicate))

    def list_active(self) -> list[KnowledgeDocument]:
        with self.session_factory() as session:
            return list(
                session.scalars(
                    select(KnowledgeDocument)
                    .where(KnowledgeDocument.status == "active")
                    .order_by(KnowledgeDocument.filename)
                ).all()
            )

    def begin_ingestion(
        self,
        *,
        doc_id: str,
        filename: str,
        file_hash: str | None = None,
        storage_key: str | None = None,
        file_type: str | None = None,
        chunk_method: str | None = None,
        allowed_roles: list[str] | None = None,
        updated_by: int | None = None,
    ) -> KnowledgeDocument:
        now = datetime.now()
        with self.session_factory() as session:
            try:
                doc = session.scalar(
                    select(KnowledgeDocument)
                    .where(KnowledgeDocument.doc_id == doc_id)
                    .with_for_update()
                )
                if doc is None:
                    doc = KnowledgeDocument(doc_id=doc_id, filename=filename)
                    session.add(doc)

                doc.filename = filename
                doc.file_hash = file_hash
                if storage_key is not None:
                    doc.storage_key = storage_key
                if file_type is not None:
                    doc.file_type = file_type
                if chunk_method is not None:
                    doc.chunk_method = chunk_method
                if allowed_roles is not None:
                    doc.allowed_roles = list(allowed_roles)
                if updated_by is not None:
                    doc.updated_by = updated_by
                doc.chunk_count = 0
                doc.chunk_ids = []
                doc.parent_count = None
                doc.child_count = None
                doc.status = "processing"
                doc.ingested_at = None
                doc.error_message = None
                doc.updated_at = now

                session.commit()
                session.refresh(doc)
                return doc
            except Exception:
                session.rollback()
                raise

    def mark_active(
        self,
        doc_id: str,
        *,
        chunk_count: int,
        chunk_ids: list[str],
        chunk_method: str | None = None,
        parent_count: int | None = None,
        child_count: int | None = None,
        ingested_at: datetime | None = None,
    ) -> KnowledgeDocument:
        now = datetime.now()
        with self.session_factory() as session:
            try:
                doc = self._require_document(session, doc_id)
                doc.chunk_count = chunk_count
                doc.chunk_ids = list(chunk_ids)
                if chunk_method is not None:
                    doc.chunk_method = chunk_method
                doc.parent_count = parent_count
                doc.child_count = child_count
                doc.status = "active"
                doc.ingested_at = ingested_at or now
                doc.error_message = None
                doc.updated_at = now
                session.commit()
                session.refresh(doc)
                return doc
            except Exception:
                session.rollback()
                raise

    def mark_failed(self, doc_id: str, error_message: str) -> KnowledgeDocument:
        return self._set_status(doc_id, "failed", error_message=error_message)

    def mark_deleting(self, doc_id: str) -> KnowledgeDocument:
        return self._set_status(doc_id, "deleting")

    def _set_status(
        self,
        doc_id: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> KnowledgeDocument:
        with self.session_factory() as session:
            try:
                doc = self._require_document(session, doc_id)
                doc.status = status
                doc.error_message = error_message
                doc.updated_at = datetime.now()
                session.commit()
                session.refresh(doc)
                return doc
            except Exception:
                session.rollback()
                raise

    @staticmethod
    def _require_document(session, doc_id: str) -> KnowledgeDocument:
        doc = session.scalar(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.doc_id == doc_id)
            .with_for_update()
        )
        if doc is None:
            raise KeyError(f"Knowledge document not found: {doc_id}")
        return doc

    def delete(self, doc_id: str) -> bool:
        with self.session_factory() as session:
            try:
                doc = session.scalar(
                    select(KnowledgeDocument).where(KnowledgeDocument.doc_id == doc_id)
                )
                if doc is None:
                    return False
                session.delete(doc)
                session.commit()
                return True
            except Exception:
                session.rollback()
                raise

    def as_manifest(self) -> dict[str, dict]:
        with self.session_factory() as session:
            docs = list(
                session.scalars(
                    select(KnowledgeDocument)
                    .where(KnowledgeDocument.status == "active")
                    .order_by(KnowledgeDocument.filename)
                ).all()
            )
            parent_ids_by_doc: dict[str, list[str]] = {}
            if docs:
                parent_rows = session.execute(
                    select(ParentChunk.doc_id, ParentChunk.parent_id)
                    .where(ParentChunk.doc_id.in_([doc.doc_id for doc in docs]))
                    .order_by(ParentChunk.doc_id, ParentChunk.parent_index)
                ).all()
                for parent_doc_id, parent_id in parent_rows:
                    parent_ids_by_doc.setdefault(parent_doc_id, []).append(parent_id)

            manifest = {}
            for doc in docs:
                entry = {
                    "doc_id": doc.doc_id,
                    "file_hash": doc.file_hash,
                    "chunk_count": doc.chunk_count,
                    "chunk_ids": list(doc.chunk_ids or []),
                    "chunk_method": doc.chunk_method,
                    "file_type": doc.file_type,
                    "ingested_at": (
                        doc.ingested_at.strftime("%Y-%m-%dT%H:%M:%S")
                        if doc.ingested_at
                        else None
                    ),
                    "status": doc.status,
                }
                if doc.parent_count is not None:
                    entry.update(
                        {
                            "parent_ids": parent_ids_by_doc.get(doc.doc_id, []),
                            "parent_count": doc.parent_count,
                            "child_count": doc.child_count,
                        }
                    )
                manifest[doc.filename] = entry
            return manifest

    def fingerprint(self) -> str:
        with self.session_factory() as session:
            rows = session.execute(
                select(
                    KnowledgeDocument.doc_id,
                    KnowledgeDocument.file_hash,
                    KnowledgeDocument.chunk_count,
                    KnowledgeDocument.ingested_at,
                )
                .where(KnowledgeDocument.status == "active")
                .order_by(KnowledgeDocument.doc_id)
            ).all()

        payload = [
            [
                doc_id,
                file_hash,
                chunk_count,
                ingested_at.isoformat(timespec="microseconds") if ingested_at else None,
            ]
            for doc_id, file_hash, chunk_count, ingested_at in rows
        ]
        serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
