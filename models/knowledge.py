from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    __table_args__ = (
        Index("idx_kdoc_doc_id", "doc_id", unique=True),
        Index("idx_kdoc_filename", "filename", unique=True),
        Index("idx_kdoc_file_hash", "file_hash", unique=True),
        Index("idx_kdoc_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_hash: Mapped[Optional[str]] = mapped_column(String(128))
    file_type: Mapped[Optional[str]] = mapped_column(String(32))
    storage_key: Mapped[Optional[str]] = mapped_column(String(512))
    chunk_method: Mapped[Optional[str]] = mapped_column(String(64))
    chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    parent_count: Mapped[Optional[int]] = mapped_column(Integer)
    child_count: Mapped[Optional[int]] = mapped_column(Integer)
    allowed_roles: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="active")
    ingested_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    __table_args__ = (
        Index("idx_parent_chunk_doc_id", "doc_id"),
    )

    parent_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_content: Mapped[str] = mapped_column(
        Text().with_variant(MEDIUMTEXT(), "mysql"),
        nullable=False,
    )
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )
