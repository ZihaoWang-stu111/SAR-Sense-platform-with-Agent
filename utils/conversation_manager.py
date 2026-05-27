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
