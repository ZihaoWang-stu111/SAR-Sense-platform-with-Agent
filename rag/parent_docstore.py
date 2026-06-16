import json
import os
from threading import Lock

from utils.logger_handler import logger


class ParentDocstore:
    """父块 JSON 持久化存储：向量库只索引子块，父块正文在此回表。"""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._lock = Lock()
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.store_path):
            self._data = {}
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"parent docstore 解析失败，重建空存储: {e}")
            self._data = {}

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def save(self, parent_id: str, page_content: str, metadata: dict) -> None:
        with self._lock:
            self._data[parent_id] = {
                "page_content": page_content,
                "metadata": metadata,
            }
            self._persist()

    def save_batch(self, records: dict[str, dict]) -> None:
        if not records:
            return
        with self._lock:
            self._data.update(records)
            self._persist()

    def get(self, parent_id: str) -> dict | None:
        with self._lock:
            return self._data.get(parent_id)

    def delete_many(self, parent_ids: list[str]) -> int:
        if not parent_ids:
            return 0
        removed = 0
        with self._lock:
            for parent_id in parent_ids:
                if parent_id in self._data:
                    del self._data[parent_id]
                    removed += 1
            if removed:
                self._persist()
        return removed

    def delete_by_doc_id(self, doc_id: str) -> int:
        with self._lock:
            to_delete = [
                pid for pid, record in self._data.items()
                if record.get("metadata", {}).get("doc_id") == doc_id
            ]
            for pid in to_delete:
                del self._data[pid]
            if to_delete:
                self._persist()
            return len(to_delete)

    def count(self) -> int:
        with self._lock:
            return len(self._data)
