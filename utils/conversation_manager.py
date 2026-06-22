"""对话管理（MySQL + AsyncSession-backed）。

公开方法签名变化：所有方法新增 `db: AsyncSession` 和 `user_id: int` 必填参数，且全部 async。
内部委托给 crud/conversations.py 的异步函数。
build_chat_pack 的滑动窗口 + 增量摘要压缩逻辑保持原行为，仍同步调 chat_model.invoke。
"""
from sqlalchemy.ext.asyncio import AsyncSession

from crud import conversations as conv_crud
from utils.logger_handler import logger


class ConversationManager:
    """保留类形态以兼容旧调用习惯，但实际上是无状态的——所有数据都通过 db 参数传入。"""

    def __init__(self, storage_dir: str = None):
        # storage_dir 仅兼容旧调用（Streamlit），不再有任何作用
        self.storage_dir = storage_dir

    async def create_conversation(self, db: AsyncSession, user_id: int, first_message: str = "") -> str:
        return await conv_crud.create_conversation(db, user_id, first_message)

    async def list_conversations(self, db: AsyncSession, user_id: int) -> list[dict]:
        return await conv_crud.list_conversations(db, user_id)

    async def load_conversation(self, db: AsyncSession, conv_id: str, user_id: int) -> dict:
        return await conv_crud.load_conversation(db, conv_id, user_id)

    async def append_message(
        self, db: AsyncSession, conv_id: str, user_id: int,
        role: str, content: str, thought_steps: list = None,
    ) -> None:
        await conv_crud.append_message(db, conv_id, user_id, role, content, thought_steps)

    async def delete_conversation(self, db: AsyncSession, conv_id: str, user_id: int) -> None:
        await conv_crud.delete_conversation(db, conv_id, user_id)

    # ==================== 对话记忆压缩 ====================

    async def build_chat_pack(
        self, db: AsyncSession, conv_id: str, user_id: int, window_size: int = 10,
    ) -> list:
        """构建传给 Agent 的消息包：滑动窗口 + 增量摘要。返回 dict 列表。"""
        conv_data = await self.load_conversation(db, conv_id, user_id)
        all_messages = conv_data.get("messages", [])

        if len(all_messages) <= window_size:
            return [self._clean_message(msg) for msg in all_messages]

        recent = [self._clean_message(msg) for msg in all_messages[-window_size:]]
        # 对齐：确保 recent 以 user 消息开头
        for idx, msg in enumerate(recent):
            if msg.get("role") == "user":
                recent = recent[idx:]
                break
        older = all_messages[: len(all_messages) - len(recent)]

        summary = conv_data.get("summary", "")
        summary_up_to = conv_data.get("summary_up_to", 0)
        has_been_compressed = bool(summary)  # 有摘要才算压缩过（避免 summary_up_to=0 默认值导致误判）

        if not has_been_compressed or summary_up_to < len(older):
            new_messages = older[summary_up_to:]
            logger.info(
                f"[记忆压缩] 对话 {conv_id} 需要增量压缩: "
                f"旧摘要覆盖 {summary_up_to} 条，新增 {len(new_messages)} 条，"
                f"当前需覆盖 {len(older)} 条"
            )
            if summary:
                summary = self._compress_summary_incremental(summary, new_messages)
            else:
                summary = self._compress_messages(new_messages)
            await conv_crud.update_summary(db, conv_id, user_id, summary, len(older))
            logger.info(f"[记忆压缩] 对话 {conv_id} 摘要已更新")

        if summary:
            summary_msg = {
                "role": "system",
                "content": (
                    "以下是本轮对话之前的历史摘要，仅用于理解上下文，"
                    f"不要把它当作用户的新问题：\n{summary}"
                ),
            }
            return [summary_msg] + recent

        return recent

    @staticmethod
    def _clean_message(msg: dict) -> dict:
        """Keep only model-facing fields when building Agent input."""
        return {"role": msg.get("role", "user"), "content": msg.get("content", "")}

    def _compress_messages(self, messages: list) -> str:
        """用 LLM 将旧消息压缩为 200 字摘要（同步调用 chat_model）。"""
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

    def _compress_summary_incremental(self, summary: str, new_messages: list) -> str:
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
