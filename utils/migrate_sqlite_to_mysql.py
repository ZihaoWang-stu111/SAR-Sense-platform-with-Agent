"""一次性从旧 SQLite (runtime/sar_sense.db) 把 admin、29 条对话、消息、指标迁到 MySQL。

仅在第一次切换时跑一次。跑完即可删除（连同 runtime/sar_sense.db 也可归档）。

用法：
    cd 项目根
    python -m utils.migrate_sqlite_to_mysql
"""
import asyncio
import json
import os
import sqlite3
from datetime import datetime

from sqlalchemy import select

from config.db_conf import AsyncSessionLocal, async_engine
from crud.users import create_user, get_user_by_username
from models import Base
from models import users as _users_mod  # noqa: F401 注册
from models import conversations as _conv_mod  # noqa: F401
from models import metrics as _metric_mod  # noqa: F401
from models.conversations import Conversation, ConversationMessage
from models.metrics import MetricEvent
from utils.path_tool import get_abs_path
from utils.security import hash_password


SQLITE_PATH = get_abs_path("runtime/sar_sense.db")


def _read_sqlite():
    """从旧 SQLite 读出 users/conversations/conversation_messages/metric_events 全部行。"""
    if not os.path.exists(SQLITE_PATH):
        print(f"[migrate] SQLite 文件不存在: {SQLITE_PATH}")
        return None
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    def fetch(sql):
        try:
            return [dict(r) for r in conn.execute(sql).fetchall()]
        except sqlite3.OperationalError as e:
            print(f"[migrate]   读取失败 {sql}: {e}")
            return []

    data = {
        "users": fetch("SELECT * FROM users"),
        "conversations": fetch("SELECT * FROM conversations"),
        "messages": fetch("SELECT * FROM conversation_messages ORDER BY conversation_id, message_index"),
        "metric_events": fetch("SELECT * FROM metric_events"),
    }
    conn.close()
    print(f"[migrate] SQLite: users={len(data['users'])}, conv={len(data['conversations'])}, "
          f"msg={len(data['messages'])}, metric={len(data['metric_events'])}")
    return data


def _to_dt(s):
    """SQLite 里时间字段是字符串，转 Python datetime。"""
    if not s:
        return datetime.now()
    if isinstance(s, datetime):
        return s
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return datetime.now()


async def _ensure_tables():
    """确保 MySQL 表已建（幂等）。"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _migrate(data: dict):
    """把数据写到 MySQL。旧用户的 SQLite id 是字符串（如 'default_user'），需要映射到 MySQL 的自增 id。"""
    user_id_map = {}  # 旧 id (字符串) -> 新 id (整数)
    async with AsyncSessionLocal() as session:
        # 1. users
        for u in data["users"]:
            existing = await get_user_by_username(session, u["username"])
            if existing:
                user_id_map[u["id"]] = existing.id
                print(f"[migrate] 用户 {u['username']} 已存在(id={existing.id})，跳过")
                continue
            new_user = await create_user(session, u["username"], u["password_hash"])
            await session.flush()
            user_id_map[u["id"]] = new_user.id
            print(f"[migrate] 用户 {u['username']}: {u['id']} -> id={new_user.id}")
        await session.commit()

        # 2. conversations
        for c in data["conversations"]:
            mapped_uid = user_id_map.get(c.get("user_id"))
            if not mapped_uid:
                print(f"[migrate]   对话 {c['id']} user_id={c.get('user_id')} 无映射，跳过")
                continue
            # 检查是否已存在
            exist = await session.execute(select(Conversation).where(Conversation.id == c["id"]))
            if exist.scalar_one_or_none():
                continue
            session.add(Conversation(
                id=c["id"],
                user_id=mapped_uid,
                title=c.get("title") or "新对话",
                created_at=_to_dt(c.get("created_at")),
                updated_at=_to_dt(c.get("updated_at")),
                summary=c.get("summary") or "",
                summary_up_to=int(c.get("summary_up_to") or 0),
            ))
        await session.commit()

        # 3. messages
        for m in data["messages"]:
            exist = await session.execute(
                select(ConversationMessage).where(
                    ConversationMessage.conversation_id == m["conversation_id"],
                    ConversationMessage.message_index == m["message_index"],
                )
            )
            if exist.scalar_one_or_none():
                continue
            # thought_steps 在 SQLite 里存的是 JSON 字符串
            ts = m.get("thought_steps")
            if ts and isinstance(ts, str):
                try:
                    ts = json.loads(ts)
                except (json.JSONDecodeError, TypeError):
                    ts = None
            session.add(ConversationMessage(
                conversation_id=m["conversation_id"],
                message_index=int(m["message_index"]),
                role=m["role"],
                content=m["content"],
                thought_steps=ts,
                created_at=_to_dt(m.get("created_at")),
            ))
        await session.commit()

        # 4. metric_events
        for e in data["metric_events"]:
            session.add(MetricEvent(
                user_id=user_id_map.get(e.get("user_id")) or 1,
                event_type=e["event_type"],
                tool_name=e.get("tool_name"),
                success=bool(e["success"]) if e.get("success") is not None else None,
                duration_ms=e.get("duration_ms"),
                created_at=_to_dt(e.get("created_at")),
            ))
        await session.commit()
    print("[migrate] 全部写入完成")


async def main():
    data = _read_sqlite()
    if not data:
        return
    await _ensure_tables()
    await _migrate(data)


if __name__ == "__main__":
    asyncio.run(main())
