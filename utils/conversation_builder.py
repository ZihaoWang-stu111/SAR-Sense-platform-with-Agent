"""Build the message pack sent to the agent.

Conversation history may contain display-only fields such as thought_steps and
rag_results. This module intentionally keeps only role/content for the LLM.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from crud import conversations as conv_crud
from utils.logger_handler import logger


async def build_chat_pack(
    db: AsyncSession, conv_id: str, user_id: int, window_size: int = 10,
) -> list:
    """Build the agent input with a sliding window and incremental summary."""
    conv_data = await conv_crud.load_conversation(db, conv_id, user_id)
    all_messages = conv_data.get("messages", [])

    if len(all_messages) <= window_size:
        return [_clean_message(msg) for msg in all_messages]

    recent = [_clean_message(msg) for msg in all_messages[-window_size:]]
    for idx, msg in enumerate(recent):
        if msg.get("role") == "user":
            recent = recent[idx:]
            break
    older = all_messages[: len(all_messages) - len(recent)]

    summary = conv_data.get("summary", "")
    summary_up_to = conv_data.get("summary_up_to", 0)
    has_been_compressed = bool(summary)

    if not has_been_compressed or summary_up_to < len(older):
        new_messages = [_clean_message(msg) for msg in older[summary_up_to:]]
        logger.info(
            f"[记忆压缩] 对话 {conv_id} 需要增量压缩: "
            f"旧摘要覆盖 {summary_up_to} 条，新增 {len(new_messages)} 条，"
            f"当前需要覆盖 {len(older)} 条"
        )
        if summary:
            summary = _compress_summary_incremental(summary, new_messages)
        else:
            summary = _compress_messages(new_messages)
        await conv_crud.update_summary(db, conv_id, user_id, summary, len(older))
        logger.info(f"[记忆压缩] 对话 {conv_id} 摘要已更新")

    if summary:
        return [{
            "role": "system",
            "content": (
                "以下是本轮对话之前的历史摘要，仅用于理解上下文，"
                f"不要把它当作用户的新问题：\n{summary}"
            ),
        }] + recent

    return recent


def _clean_message(msg: dict) -> dict:
    return {"role": msg.get("role", "user"), "content": msg.get("content", "")}


def _compress_messages(messages: list) -> str:
    from model.factory import chat_model

    text = ""
    for msg in messages:
        role = "用户" if msg.get("role") == "user" else "助手"
        text += f"{role}: {msg.get('content', '')}\n"
    try:
        result = chat_model.invoke(
            f"请将以下对话压缩为200字摘要，保留场景ID和技术要点：\n\n{text}"
        )
        return result.content.strip()
    except Exception as e:
        logger.warning(f"[记忆压缩] LLM 失败，降级截断: {e}")
        return text[:500]


def _compress_summary_incremental(summary: str, new_messages: list) -> str:
    from model.factory import chat_model

    text = ""
    for msg in new_messages:
        role = "用户" if msg.get("role") == "user" else "助手"
        text += f"{role}: {msg.get('content', '')}\n"
    if not text.strip():
        return summary
    try:
        result = chat_model.invoke(
            "请将已有摘要和新增对话融合为新的200字摘要，保留场景ID和技术要点。"
            f"\n\n已有摘要：\n{summary}"
            f"\n\n新增对话：\n{text}"
        )
        return result.content.strip()
    except Exception as e:
        logger.warning(f"[记忆压缩] 增量LLM失败，降级拼接截断: {e}")
        return (summary + "\n" + text)[:500]
