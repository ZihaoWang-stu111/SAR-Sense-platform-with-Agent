"""指标事件表 ORM。AgentMetrics 写入但不按用户隔离（user_id 当前固定为种子用户=1）。"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class MetricEvent(Base):
    __tablename__ = "metric_events"

    __table_args__ = (
        Index("idx_metric_created", "created_at"),
        Index("idx_metric_type", "event_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, default=1, comment="预留用户ID（当前不隔离，统一写种子用户）")
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="tool_call / llm_call / conversation_timing")
    tool_name: Mapped[Optional[str]] = mapped_column(String(128))
    success: Mapped[Optional[bool]] = mapped_column(Boolean)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
