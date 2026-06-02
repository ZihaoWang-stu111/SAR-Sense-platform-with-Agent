import os
import json
from datetime import datetime
from utils.logger_handler import logger


class ConversationManager:
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            from utils.path_tool import get_abs_path
            storage_dir = get_abs_path("conversations")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def create_conversation(self, first_message: str = "") -> str:
        conv_id = datetime.now().strftime("conv_%Y%m%d_%H%M%S")
        title = first_message[:20] + "..." if len(first_message) > 20 else first_message
        if not title:
            title = "新对话"
        conv_data = {
            "id": conv_id,
            "title": title,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": []
        }
        self._write(conv_id, conv_data)
        return conv_id

    def list_conversations(self) -> list[dict]:
        convs = []
        for fname in os.listdir(self.storage_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.storage_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                convs.append({
                    "id": data["id"],
                    "title": data["title"],
                    "updated_at": data["updated_at"]
                })
            except Exception:
                continue
        convs.sort(key=lambda x: x["updated_at"], reverse=True)
        return convs

    def load_conversation(self, conv_id: str) -> dict:
        path = os.path.join(self.storage_dir, f"{conv_id}.json")
        if not os.path.exists(path):
            return {"id": conv_id, "title": "新对话", "messages": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def append_message(self, conv_id: str, role: str, content: str, thought_steps: list = None):
        conv_data = self.load_conversation(conv_id)
        msg = {"role": role, "content": content}
        if thought_steps:
            msg["thought_steps"] = thought_steps
        conv_data["messages"].append(msg)
        conv_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if role == "user" and len(conv_data["messages"]) == 1:
            conv_data["title"] = content[:20] + "..." if len(content) > 20 else content
        self._write(conv_id, conv_data)

    def delete_conversation(self, conv_id: str):
        path = os.path.join(self.storage_dir, f"{conv_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def clear_all(self):
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(self.storage_dir, fname))

    def _write(self, conv_id: str, data: dict):
        path = os.path.join(self.storage_dir, f"{conv_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ==================== 对话记忆压缩 ====================

    def build_chat_pack(self, conv_id: str, window_size: int = 10) -> list:
        """构建传给 Agent 的消息包，摘要拼到第一条消息 content 前面，返回 dict 列表"""
        conv_data = self.load_conversation(conv_id)
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
            logger.info(f"[记忆压缩] 对话 {conv_id} 需要重新压缩: "
                        f"旧摘要覆盖 {summary_up_to} 条，当前需覆盖 {len(older)} 条")
            summary = self._compress_messages(older)
            conv_data["summary"] = summary
            conv_data["summary_up_to"] = len(older)
            self._write(conv_id, conv_data)
            logger.info(f"[记忆压缩] 对话 {conv_id} 摘要已更新")

        if summary:
            summary_msg = {
                "role": "system",
                "content": f"以下是本轮对话之前的历史摘要，仅用于理解上下文，不要把它当作用户的新问题：\n{summary}"
            }
            return [summary_msg] + recent

        return recent

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
