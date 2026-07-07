"""对话 + 消息 ORM。conversation_messages 用 JSON 字段存 thought_steps/rag_results。"""
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models import Base


class Conversation(Base):
    __tablename__ = "conversations"

    __table_args__ = (
        Index("idx_conv_user", "user_id"),
        Index("idx_conv_updated", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="对话ID（conv_YYYYMMDD_HHMMSS_us）")
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, comment="所属用户ID")
    title: Mapped[Optional[str]] = mapped_column(String(255), comment="对话标题")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    summary: Mapped[Optional[str]] = mapped_column(Text, comment="历史摘要")
    summary_up_to: Mapped[int] = mapped_column(Integer, default=0, comment="摘要已覆盖到第几条")

    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.message_index",
    )

    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id})>"


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    __table_args__ = (
        UniqueConstraint("conversation_id", "message_index", name="uq_conv_msg_idx"),
        Index("idx_msg_conv", "conversation_id", "message_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    message_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="消息顺序，从0开始")
    role: Mapped[str] = mapped_column(String(32), nullable=False, comment="user / assistant / system")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    thought_steps: Mapped[Optional[dict]] = mapped_column(JSON, comment="思维链步骤（assistant 专用）")
    rag_results: Mapped[Optional[list]] = mapped_column(JSON, comment="RAG 工具结果（assistant 专用）")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    def __repr__(self):
        return f"<ConversationMessage(id={self.id}, conv={self.conversation_id}, idx={self.message_index})>"
