"""Build the message pack sent to the agent.

Conversation history may contain display-only fields such as thought_steps and
rag_results. This module intentionally keeps only role/content for the LLM.
"""

import re

from sqlalchemy.ext.asyncio import AsyncSession

from crud import conversations as conv_crud
from utils.config_handler import rag_conf
from utils.logger_handler import logger


_COMPACTION = rag_conf.get("conversation_compaction", {})
_KEEP_FIRST_MESSAGES = max(0, int(_COMPACTION.get("keep_first_messages", 2)))
_KEEP_RECENT_MESSAGES = max(1, int(_COMPACTION.get("keep_recent_messages", 20)))
_COMPACT_BATCH_MESSAGES = max(1, int(_COMPACTION.get("compact_batch_messages", 8)))
_HISTORY_TOKEN_BUDGET = max(1, int(_COMPACTION.get("history_token_budget", 6000)))
_SUMMARY_MAX_CHARS = max(1, int(_COMPACTION.get("summary_max_chars", 500)))
_MESSAGE_TOKEN_OVERHEAD = 4
_SUMMARY_VERSION_PREFIX = "[middle-v1]\n"
_SUMMARY_REQUIREMENTS = (
    "按以下四项组织：用户目标/偏好；事实参数与场景ID/文件名；结论；未解决问题。"
    f"只保留明确内容，不推测，输出最多{_SUMMARY_MAX_CHARS}字符。"
)
_CJK_RE = re.compile(
    r"[\u1100-\u11ff\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U0002fa1f]"
)


async def build_chat_pack(db: AsyncSession, conv_id: str, user_id: int) -> list:
    """Build agent input while compacting only the oldest middle batch."""
    conv_data = await conv_crud.load_conversation(db, conv_id, user_id)
    all_messages = [_clean_message(msg) for msg in conv_data.get("messages", [])]
    if len(all_messages) <= _KEEP_FIRST_MESSAGES + _KEEP_RECENT_MESSAGES:
        return all_messages

    middle_end = len(all_messages) - _KEEP_RECENT_MESSAGES
    stored_summary = conv_data.get("summary") or ""
    summary = (
        stored_summary[len(_SUMMARY_VERSION_PREFIX):]
        if stored_summary.startswith(_SUMMARY_VERSION_PREFIX)
        else ""
    )
    stored_up_to = int(conv_data.get("summary_up_to") or 0)
    summary_up_to = (
        max(_KEEP_FIRST_MESSAGES, stored_up_to)
        if summary
        else _KEEP_FIRST_MESSAGES
    )
    pending = all_messages[summary_up_to:middle_end]

    if summary:
        current_history = [
            _summary_message(summary),
            *all_messages[:_KEEP_FIRST_MESSAGES],
            *pending,
            *all_messages[middle_end:],
        ]
    else:
        current_history = all_messages

    compact_count = 0
    if (
        pending
        and estimate_messages_tokens(current_history) > _HISTORY_TOKEN_BUDGET
    ):
        compact_count = len(pending)
    elif len(pending) >= _COMPACT_BATCH_MESSAGES:
        compact_count = _COMPACT_BATCH_MESSAGES

    if compact_count:
        batch = pending[:compact_count]
        logger.info(
            f"[记忆压缩] 对话 {conv_id} 压缩中段批次: "
            f"旧摘要覆盖到索引 {summary_up_to}，本批 {compact_count} 条"
        )
        if summary:
            new_summary = await _compress_summary_incremental(summary, batch)
        else:
            new_summary = await _compress_messages(batch)
        new_summary = (new_summary or "").strip()
        if new_summary:
            summary = new_summary
            summary_up_to += compact_count
            pending = pending[compact_count:]
            await conv_crud.update_summary(
                db,
                conv_id,
                user_id,
                _SUMMARY_VERSION_PREFIX + summary,
                summary_up_to,
            )
            logger.info(f"[记忆压缩] 对话 {conv_id} 摘要已更新")
        else:
            logger.warning(f"[记忆压缩] 对话 {conv_id} 摘要失败，保留原消息")

    if summary:
        return [
            _summary_message(summary),
            *all_messages[:_KEEP_FIRST_MESSAGES],
            *pending,
            *all_messages[middle_end:],
        ]

    return all_messages


def _clean_message(msg: dict) -> dict:
    return {"role": msg.get("role", "user"), "content": msg.get("content", "")}


def _estimate_message_tokens(msg: dict) -> int:
    content = msg.get("content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    cjk_chars = len(_CJK_RE.findall(content))
    return _MESSAGE_TOKEN_OVERHEAD + cjk_chars + (len(content) - cjk_chars + 3) // 4


def estimate_messages_tokens(messages: list) -> int:
    return sum(_estimate_message_tokens(message) for message in messages)


def fit_messages_to_budget(messages: list, token_budget: int | None = None) -> list:
    fitted = list(messages)
    budget = _HISTORY_TOKEN_BUDGET if token_budget is None else token_budget
    total = estimate_messages_tokens(fitted)
    if total <= budget or len(fitted) <= 1:
        return fitted

    current_index = len(fitted) - 1
    non_system = [
        index
        for index, message in enumerate(fitted)
        if message.get("role") != "system"
    ]
    first = non_system[:_KEEP_FIRST_MESSAGES]
    recent = non_system[-_KEEP_RECENT_MESSAGES:]
    first_set = set(first)
    recent_set = set(recent)
    removal_order = [
        *(
            index
            for index in non_system
            if index not in first_set
            and index not in recent_set
            and index != current_index
        ),
        *(
            index
            for index, message in enumerate(fitted)
            if message.get("role") == "system" and index != current_index
        ),
        *(
            index
            for index in recent
            if index not in first_set and index != current_index
        ),
        *(index for index in first if index != current_index),
    ]
    removed = set()
    for index in removal_order:
        if total <= budget:
            break
        removed.add(index)
        total -= _estimate_message_tokens(fitted[index])

    return [message for index, message in enumerate(fitted) if index not in removed]


def _summary_message(summary: str) -> dict:
    return {
        "role": "system",
        "content": (
            "以下是最初两条消息与近期原文之间的中段历史摘要，仅用于理解上下文。"
            "后续消息依次包含最初两条原文、尚未摘要的中段消息和近期原文；"
            f"不要把摘要当作用户的新问题：\n{summary}"
        ),
    }


async def _compress_messages(messages: list) -> str:
    # 改用 ainvoke：原 chat_model.invoke 是同步阻塞，在 async build_chat_pack 里
    # 会阻塞 FastAPI 事件循环（LLM 往返数秒，长对话压缩时其他请求全排队）
    from model.factory import chat_model

    text = ""
    for msg in messages:
        role = "用户" if msg.get("role") == "user" else "助手"
        text += f"{role}: {msg.get('content', '')}\n"
    try:
        result = await chat_model.ainvoke(
            f"请将以下对话压缩为结构化历史摘要。{_SUMMARY_REQUIREMENTS}"
            f"\n\n待摘要对话：\n{text}"
        )
        return result.content.strip()[:_SUMMARY_MAX_CHARS]
    except Exception as e:
        logger.warning(f"[记忆压缩] LLM 失败，保留原消息: {e}")
        return ""


async def _compress_summary_incremental(summary: str, new_messages: list) -> str:
    # 同 _compress_messages：ainvoke 避免 async 函数里同步阻塞 LLM
    from model.factory import chat_model

    text = ""
    for msg in new_messages:
        role = "用户" if msg.get("role") == "user" else "助手"
        text += f"{role}: {msg.get('content', '')}\n"
    if not text.strip():
        return summary[:_SUMMARY_MAX_CHARS]
    try:
        result = await chat_model.ainvoke(
            f"请融合已有摘要和新增对话为结构化历史摘要。{_SUMMARY_REQUIREMENTS}"
            f"\n\n已有摘要：\n{summary}"
            f"\n\n新增对话：\n{text}"
        )
        return result.content.strip()[:_SUMMARY_MAX_CHARS]
    except Exception as e:
        logger.warning(f"[记忆压缩] 增量LLM失败，保留原消息: {e}")
        return ""
