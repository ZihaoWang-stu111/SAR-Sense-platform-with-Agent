from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from models.knowledge import ParentChunk


class ParentChunkRepository:
    """基于 SQLAlchemy 的父块存储，支持 MySQL 和 SQLite upsert。"""

    def __init__(self, session_factory=None):
        if session_factory is None:
            from config.db_conf import SyncSessionLocal

            session_factory = SyncSessionLocal
        self.session_factory = session_factory

    def save(self, parent_id: str, page_content: str, metadata: dict) -> None:
        self.save_batch(
            {
                parent_id: {
                    "page_content": page_content,
                    "metadata": metadata,
                }
            }
        )

    def save_batch(self, records: dict[str, dict]) -> None:
        if not records:
            return

        now = datetime.now()
        rows = [self._record_to_row(parent_id, record, now) for parent_id, record in records.items()]
        with self.session_factory() as session:
            try:
                dialect_name = session.get_bind().dialect.name
                if dialect_name == "mysql":
                    statement = mysql_insert(ParentChunk.__table__).values(rows)
                    statement = statement.on_duplicate_key_update(
                        doc_id=statement.inserted.doc_id,
                        parent_index=statement.inserted.parent_index,
                        page_content=statement.inserted.page_content,
                        metadata=statement.inserted.metadata,
                        updated_at=statement.inserted.updated_at,
                    )
                    session.execute(statement)
                elif dialect_name == "sqlite":
                    statement = sqlite_insert(ParentChunk.__table__).values(rows)
                    statement = statement.on_conflict_do_update(
                        index_elements=[ParentChunk.parent_id],
                        set_={
                            "doc_id": statement.excluded.doc_id,
                            "parent_index": statement.excluded.parent_index,
                            "page_content": statement.excluded.page_content,
                            "metadata": statement.excluded.metadata,
                            "updated_at": statement.excluded.updated_at,
                        },
                    )
                    session.execute(statement)
                else:
                    self._merge_rows(session, rows)
                session.commit()
            except Exception:
                session.rollback()
                raise

    @staticmethod
    def _record_to_row(parent_id: str, record: dict, now: datetime) -> dict:
        metadata = dict(record.get("metadata") or {})
        doc_id = metadata.get("doc_id")
        if not doc_id:
            raise ValueError(f"Parent chunk {parent_id} is missing metadata.doc_id")

        parent_index = metadata.get("parent_index")
        if parent_index is None:
            try:
                parent_index = int(parent_id.rsplit(":", 1)[-1])
            except ValueError as exc:
                raise ValueError(
                    f"Parent chunk {parent_id} is missing metadata.parent_index"
                ) from exc

        return {
            "parent_id": parent_id,
            "doc_id": doc_id,
            "parent_index": int(parent_index),
            "page_content": record["page_content"],
            "metadata": metadata,
            "created_at": now,
            "updated_at": now,
        }

    @staticmethod
    def _merge_rows(session, rows: list[dict]) -> None:
        for row in rows:
            chunk = session.get(ParentChunk, row["parent_id"])
            if chunk is None:
                session.add(
                    ParentChunk(
                        parent_id=row["parent_id"],
                        doc_id=row["doc_id"],
                        parent_index=row["parent_index"],
                        page_content=row["page_content"],
                        metadata_json=row["metadata"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )
                continue
            chunk.doc_id = row["doc_id"]
            chunk.parent_index = row["parent_index"]
            chunk.page_content = row["page_content"]
            chunk.metadata_json = row["metadata"]
            chunk.updated_at = row["updated_at"]

    @staticmethod
    def _as_record(chunk: ParentChunk) -> dict:
        return {
            "page_content": chunk.page_content,
            "metadata": dict(chunk.metadata_json or {}),
        }

    def get(self, parent_id: str) -> dict | None:
        with self.session_factory() as session:
            chunk = session.get(ParentChunk, parent_id)
            return self._as_record(chunk) if chunk is not None else None

    def get_many(self, parent_ids: list[str]) -> dict[str, dict]:
        if not parent_ids:
            return {}
        with self.session_factory() as session:
            chunks = session.scalars(
                select(ParentChunk).where(ParentChunk.parent_id.in_(parent_ids))
            ).all()
            records = {chunk.parent_id: self._as_record(chunk) for chunk in chunks}
            return {
                parent_id: records[parent_id]
                for parent_id in parent_ids
                if parent_id in records
            }

    def delete_many(self, parent_ids: list[str]) -> int:
        if not parent_ids:
            return 0
        return self._delete_where(ParentChunk.parent_id.in_(parent_ids))

    def delete_by_doc_id(self, doc_id: str) -> int:
        return self._delete_where(ParentChunk.doc_id == doc_id)

    def _delete_where(self, predicate) -> int:
        with self.session_factory() as session:
            try:
                result = session.execute(delete(ParentChunk).where(predicate))
                session.commit()
                return max(result.rowcount or 0, 0)
            except Exception:
                session.rollback()
                raise

    def count(self) -> int:
        with self.session_factory() as session:
            return session.scalar(select(func.count()).select_from(ParentChunk)) or 0
