import json
from datetime import datetime
from utils.logger_handler import logger
from utils.db import get_conn, init_db


class ConversationManager:
    """对话管理（SQLite-backed）。

    旧实现：每对话一个 JSON 文件全量重写。新实现：走 SQLite（runtime/sar_sense.db）。
    方法签名与返回结构保持不变，下游（API 路由、react_agent）无需改动。
    build_chat_pack 的滑动窗口 + 增量摘要压缩逻辑保持原行为。
    """

    def __init__(self, storage_dir: str = None):
        # storage_dir 保留参数兼容旧调用，不再用于读写
        self.storage_dir = storage_dir
        init_db()  # 幂等建表

    def create_conversation(self, first_message: str = "", *, user_id: str) -> str:
        # 带微秒，避免同秒创建主键冲突（原秒级 conv_YYYYMMDD_HHMMSS）
        conv_id = datetime.now().strftime("conv_%Y%m%d_%H%M%S_%f")
        title = (first_message[:20] + "...") if len(first_message) > 20 else (first_message or "新对话")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO conversations (id, user_id, title, created_at, updated_at, summary, summary_up_to)
                   VALUES (?, ?, ?, ?, ?, '', 0)""",
                (conv_id, user_id, title, now, now),
            )
        return conv_id

    def list_conversations(self, *, user_id: str) -> list[dict]:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id, title, updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            return [{"id": r["id"], "title": r["title"], "updated_at": r["updated_at"]} for r in rows]

    def load_conversation(self, conv_id: str, *, user_id: str) -> dict:
        with get_conn() as conn:
            conv = conn.execute(
                "SELECT * FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id)
            ).fetchone()
            if conv is None:
                # 不存在或越权都返回空默认，不区分（避免通过返回探测他人 conv_id）
                return {"id": conv_id, "title": "新对话", "messages": []}
            msgs = conn.execute(
                "SELECT role, content, thought_steps FROM conversation_messages "
                "WHERE conversation_id=? ORDER BY message_index",
                (conv_id,),
            ).fetchall()
            messages = []
            for m in msgs:
                msg = {"role": m["role"], "content": m["content"]}
                if m["thought_steps"]:
                    try:
                        msg["thought_steps"] = json.loads(m["thought_steps"])
                    except (json.JSONDecodeError, TypeError):
                        msg["thought_steps"] = []
                messages.append(msg)
            data = {
                "id": conv["id"],
                "title": conv["title"],
                "created_at": conv["created_at"],
                "updated_at": conv["updated_at"],
                "messages": messages,
            }
            if conv["summary"] is not None:
                data["summary"] = conv["summary"]
            if conv["summary_up_to"] is not None:
                data["summary_up_to"] = conv["summary_up_to"]
            return data

    def append_message(self, conv_id: str, role: str, content: str, thought_steps: list = None, *, user_id: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        thought_json = json.dumps(thought_steps, ensure_ascii=False) if thought_steps else None
        with get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id)
            ).fetchone()
            if row is None:
                # 越权对话不写入，不报错
                return
            max_idx = conn.execute(
                "SELECT COALESCE(MAX(message_index), -1) AS m FROM conversation_messages WHERE conversation_id=?",
                (conv_id,),
            ).fetchone()["m"]
            next_idx = max_idx + 1
            conn.execute(
                """INSERT INTO conversation_messages
                   (conversation_id, message_index, role, content, thought_steps, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conv_id, next_idx, role, content, thought_json, now),
            )
            # 首条 user 消息隐式重写 title
            if role == "user" and next_idx == 0:
                title = (content[:20] + "...") if len(content) > 20 else content
                conn.execute(
                    "UPDATE conversations SET updated_at=?, title=? WHERE id=?",
                    (now, title, conv_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?",
                    (now, conv_id),
                )

    def delete_conversation(self, conv_id: str, *, user_id: str):
        with get_conn() as conn:
            conn.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id))

    # ==================== 对话记忆压缩 ====================

    def build_chat_pack(self, conv_id: str, window_size: int = 10, *, user_id: str) -> list:
        """构建传给 Agent 的消息包，摘要拼到第一条消息 content 前面，返回 dict 列表"""
        conv_data = self.load_conversation(conv_id, user_id=user_id)
        all_messages = conv_data.get("messages", [])

        if len(all_messages) <= window_size:
            return [self._clean_message(msg) for msg in all_messages]

        recent = [self._clean_message(msg) for msg in all_messages[-window_size:]]
        # 对齐：确保 recent 以 user 消息开头
        for idx, msg in enumerate(recent):
            if msg.get("role") == "user":
                recent = recent[idx:]
                break
        older = all_messages[:len(all_messages) - len(recent)]

        summary = conv_data.get("summary", "")
        summary_up_to = conv_data.get("summary_up_to", 0)
        has_been_compressed = "summary_up_to" in conv_data

        if not has_been_compressed or summary_up_to < len(older):
            new_messages = older[summary_up_to:]
            logger.info(f"[记忆压缩] 对话 {conv_id} 需要增量压缩: "
                        f"旧摘要覆盖 {summary_up_to} 条，新增 {len(new_messages)} 条，当前需覆盖 {len(older)} 条")
            if summary:
                summary = self._compress_summary_incremental(summary, new_messages)
            else:
                summary = self._compress_messages(new_messages)
            self._update_summary(conv_id, summary, len(older))
            logger.info(f"[记忆压缩] 对话 {conv_id} 摘要已更新")

        if summary:
            summary_msg = {
                "role": "system",
                "content": f"以下是本轮对话之前的历史摘要，仅用于理解上下文，不要把它当作用户的新问题：\n{summary}"
            }
            return [summary_msg] + recent

        return recent

    def _update_summary(self, conv_id: str, summary: str, summary_up_to: int):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_conn() as conn:
            conn.execute(
                "UPDATE conversations SET summary=?, summary_up_to=?, updated_at=? WHERE id=?",
                (summary, summary_up_to, now, conv_id),
            )

    @staticmethod
    def _clean_message(msg: dict) -> dict:
        """Keep only model-facing fields when building Agent input."""
        return {
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        }

    def _compress_messages(self, messages: list) -> str:
        """用 LLM 将旧消息压缩为 200 字摘要"""
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
        """将已有摘要和新增旧消息融合为新的摘要，避免每次重压全部历史。"""
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
