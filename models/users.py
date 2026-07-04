"""用户表 ORM。对齐参考方式但精简字段（仅保留登录刚需）。"""
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class User(Base):
    __tablename__ = "users"

    __table_args__ = (
        Index("username_UNIQUE", "username"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="用户ID")
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="用户名")
    password: Mapped[str] = mapped_column(String(255), nullable=False, comment="密码（bcrypt 哈希）")
    nickname: Mapped[Optional[str]] = mapped_column(String(50), comment="昵称")
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="guest", comment="角色")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"
