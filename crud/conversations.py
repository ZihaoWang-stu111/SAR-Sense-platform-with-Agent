"""对话 + 消息 crud（异步）。所有方法都带 user_id 进行隔离。"""
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.conversations import Conversation, ConversationMessage


def _gen_conv_id() -> str:
    """带微秒的 conv_id，避免同秒主键冲突。"""
    return datetime.now().strftime("conv_%Y%m%d_%H%M%S_%f")


async def ensure_conversation_rag_results_column(db: AsyncSession) -> None:
    """ponytail: existing installs need one missing JSON column; no migration framework."""
    try:
        await db.execute(text("ALTER TABLE conversation_messages ADD COLUMN rag_results JSON NULL"))
    except Exception as e:
        if "Duplicate column name" not in str(e):
            raise


async def create_conversation(db: AsyncSession, user_id: int, first_message: str = "") -> str:
    conv_id = _gen_conv_id()
    title = (first_message[:20] + "...") if len(first_message) > 20 else (first_message or "新对话")
    now = datetime.now()
    conv = Conversation(
        id=conv_id, user_id=user_id, title=title,
        created_at=now, updated_at=now, summary="", summary_up_to=0,
    )
    db.add(conv)
    await db.flush()
    return conv_id


async def list_conversations(db: AsyncSession, user_id: int) -> list[dict]:
    stmt = (
        select(Conversation.id, Conversation.title, Conversation.updated_at)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    result = await db.execute(stmt)
    return [
        {"id": r.id, "title": r.title, "updated_at": r.updated_at.strftime("%Y-%m-%d %H:%M:%S") if r.updated_at else ""}
        for r in result.all()
    ]


async def load_conversation(db: AsyncSession, conv_id: str, user_id: int) -> dict:
    """越权或不存在返回空默认 dict（保持原 ConversationManager 语义）。"""
    stmt = select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if conv is None:
        return {"id": conv_id, "title": "新对话", "messages": []}

    msg_stmt = (
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conv_id)
        .order_by(ConversationMessage.message_index)
    )
    msg_result = await db.execute(msg_stmt)
    messages = []
    for m in msg_result.scalars().all():
        msg = {"role": m.role, "content": m.content}
        if m.thought_steps:
            msg["thought_steps"] = m.thought_steps
        if m.rag_results:
            msg["rag_results"] = m.rag_results
        messages.append(msg)
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.strftime("%Y-%m-%d %H:%M:%S") if conv.created_at else "",
        "updated_at": conv.updated_at.strftime("%Y-%m-%d %H:%M:%S") if conv.updated_at else "",
        "summary": conv.summary or "",
        "summary_up_to": conv.summary_up_to or 0,
        "messages": messages,
    }


async def append_message(
    db: AsyncSession, conv_id: str, user_id: int, role: str, content: str,
    thought_steps: Optional[list] = None, rag_results: Optional[list] = None,
) -> None:
    """先验证对话归属，再 INSERT 一行消息 + 更新 updated_at + 首条 user 消息时改 title。"""
    own_stmt = select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user_id)
    own_result = await db.execute(own_stmt)
    conv = own_result.scalar_one_or_none()
    if conv is None:
        return  # 越权或不存在 → no-op

    # 下一条 message_index
    max_stmt = (
        select(func.coalesce(func.max(ConversationMessage.message_index), -1))
        .where(ConversationMessage.conversation_id == conv_id)
    )
    next_idx = (await db.execute(max_stmt)).scalar_one() + 1

    db.add(ConversationMessage(
        conversation_id=conv_id, message_index=next_idx,
        role=role, content=content, thought_steps=thought_steps, rag_results=rag_results,
        created_at=datetime.now(),
    ))
    conv.updated_at = datetime.now()
    if role == "user" and next_idx == 0:
        conv.title = (content[:20] + "...") if len(content) > 20 else content


async def delete_conversation(db: AsyncSession, conv_id: str, user_id: int) -> None:
    """级联删消息（外键 ON DELETE CASCADE）。"""
    stmt = delete(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user_id)
    await db.execute(stmt)


async def update_summary(db: AsyncSession, conv_id: str, user_id: int, summary: str, summary_up_to: int) -> None:
    stmt = select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if conv:
        conv.summary = summary
        conv.summary_up_to = summary_up_to
        conv.updated_at = datetime.now()
