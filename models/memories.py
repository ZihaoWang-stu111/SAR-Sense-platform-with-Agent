"""用户长期记忆 ORM；MySQL 是正文事实源。"""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class UserMemory(Base):
    __tablename__ = "user_memories"
    __table_args__ = (
        Index("idx_memory_user", "user_id"),
        Index("idx_memory_user_category", "user_id", "category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, comment="所属用户ID"
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="稳定用户事实")
    category: Mapped[str] = mapped_column(
        String(32), nullable=False, default="context", comment="profile/preference/context"
    )
    source_conv_id: Mapped[str | None] = mapped_column(String(64), comment="来源对话ID")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

