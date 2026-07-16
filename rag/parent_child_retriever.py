from typing import Protocol

from langchain_core.documents import Document

from utils.logger_handler import logger


class ParentStoreProtocol(Protocol):
    def get(self, parent_id: str) -> dict | None:
        ...


class ParentChildResolver:
    """将混合检索命中的子块解析为去重后的父块。"""

    def __init__(self, parent_docstore: ParentStoreProtocol, top_k_parents: int = 6):
        self.parent_docstore = parent_docstore
        self.top_k_parents = top_k_parents

    def resolve(self, child_docs: list[Document], allowed_doc_ids=None) -> list[Document]:
        if not child_docs:
            return []

        allowed = None if allowed_doc_ids is None else set(allowed_doc_ids)
        if allowed is not None:
            if not allowed:
                return []
            child_docs = [doc for doc in child_docs if (doc.metadata or {}).get("doc_id") in allowed]
            if not child_docs:
                return []

        if not any(doc.metadata.get("parent_id") for doc in child_docs):
            return child_docs[:self.top_k_parents]

        parent_ids = []
        seen_parent_ids: set[str] = set()
        for child in child_docs:
            parent_id = (child.metadata or {}).get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_ids.append(parent_id)

        get_many = getattr(self.parent_docstore, "get_many", None)
        if callable(get_many):
            parent_records = get_many(parent_ids)
        else:
            parent_records = {
                parent_id: record
                for parent_id in parent_ids
                if (record := self.parent_docstore.get(parent_id)) is not None
            }

        emitted_parent_ids: set[str] = set()
        results: list[Document] = []

        for child in child_docs:
            child_meta = child.metadata or {}
            parent_id = child_meta.get("parent_id")
            if not parent_id:
                results.append(child)
                if len(results) >= self.top_k_parents:
                    break
                continue

            if parent_id in emitted_parent_ids:
                continue

            record = parent_records.get(parent_id)
            if not record:
                logger.warning(f"parent docstore 中未找到 {parent_id}，跳过")
                continue

            if allowed is not None and (record.get("metadata") or {}).get("doc_id") not in allowed:
                continue

            emitted_parent_ids.add(parent_id)
            parent_meta = dict(record.get("metadata") or {})
            parent_meta["match_child_id"] = child_meta.get("child_id", child_meta.get("chunk_id"))
            parent_meta["chunk_id"] = parent_id
            # 子块已按 rerank 分降序，首次命中的子块即该父块最高分，带到父块 meta 供展示/排序
            parent_meta["rerank_score"] = child_meta.get("rerank_score")

            results.append(Document(
                page_content=record["page_content"],
                metadata=parent_meta,
            ))

            if len(results) >= self.top_k_parents:
                break

        logger.info(f"父子块回表: {len(child_docs)} 个子块命中 → {len(results)} 个父块")
        return results
