"""用户长期记忆的同步数据库访问层。"""
from datetime import datetime

from sqlalchemy import delete, func, select

from models.memories import UserMemory

MEMORY_CATEGORIES = ("profile", "preference", "context")
_EVICTION_PRIORITY = ("context", "preference", "profile")


class MemoryRepository:
    def __init__(self, session_factory=None):
        if session_factory is None:
            from config.db_conf import SyncSessionLocal

            session_factory = SyncSessionLocal
        self.session_factory = session_factory

    @staticmethod
    def _to_dict(row: UserMemory) -> dict:
        return {
            "id": row.id,
            "user_id": row.user_id,
            "content": row.content,
            "category": row.category,
            "source_conv_id": row.source_conv_id,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def create(self, *, user_id, content, category, source_conv_id=None) -> dict:
        with self.session_factory() as session:
            row = UserMemory(
                user_id=user_id,
                content=content,
                category=category,
                source_conv_id=source_conv_id,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    def update(
        self,
        memory_id,
        *,
        user_id,
        content=None,
        category=None,
        source_conv_id=None,
    ) -> dict | None:
        with self.session_factory() as session:
            row = session.scalar(
                select(UserMemory)
                .where(UserMemory.id == memory_id, UserMemory.user_id == user_id)
                .with_for_update()
            )
            if row is None:
                return None
            if content is not None:
                row.content = content
            if category is not None:
                row.category = category
            if source_conv_id is not None:
                row.source_conv_id = source_conv_id
            row.updated_at = datetime.now()
            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    def get_many(self, memory_ids, *, user_id) -> dict[int, dict]:
        if not memory_ids:
            return {}
        with self.session_factory() as session:
            rows = session.scalars(
                select(UserMemory).where(
                    UserMemory.id.in_(memory_ids), UserMemory.user_id == user_id
                )
            ).all()
        by_id = {row.id: self._to_dict(row) for row in rows}
        return {memory_id: by_id[memory_id] for memory_id in memory_ids if memory_id in by_id}

    def list_by_category(self, user_id, category, *, limit) -> list[dict]:
        if limit <= 0:
            return []
        with self.session_factory() as session:
            rows = session.scalars(
                select(UserMemory)
                .where(UserMemory.user_id == user_id, UserMemory.category == category)
                .order_by(UserMemory.updated_at.desc(), UserMemory.id.desc())
                .limit(limit)
            ).all()
            return [self._to_dict(row) for row in rows]

    def delete_ids(self, memory_ids, *, user_id) -> list[int]:
        if not memory_ids:
            return []
        with self.session_factory() as session:
            existing = list(
                session.scalars(
                    select(UserMemory.id).where(
                        UserMemory.id.in_(memory_ids), UserMemory.user_id == user_id
                    )
                ).all()
            )
            if existing:
                session.execute(delete(UserMemory).where(UserMemory.id.in_(existing)))
            session.commit()
            return existing

    def evict_overflow(self, user_id, max_memories) -> list[int]:
        if max_memories < 0:
            return []
        with self.session_factory() as session:
            total = session.scalar(
                select(func.count()).select_from(UserMemory).where(UserMemory.user_id == user_id)
            ) or 0
            overflow = total - max_memories
            deleted = []
            for category in _EVICTION_PRIORITY:
                if overflow <= 0:
                    break
                rows = session.scalars(
                    select(UserMemory)
                    .where(UserMemory.user_id == user_id, UserMemory.category == category)
                    .order_by(UserMemory.updated_at, UserMemory.id)
                    .limit(overflow)
                ).all()
                for row in rows:
                    deleted.append(row.id)
                    session.delete(row)
                    overflow -= 1
            session.commit()
            return deleted
