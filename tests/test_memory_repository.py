import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base
from models.memories import UserMemory  # noqa: F401
from models.users import User  # noqa: F401
from repositories.memory_repository import MemoryRepository


class MemoryRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "memory.sqlite3"
        self.engine = create_engine(f"sqlite+pysqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.repository = MemoryRepository(session_factory=session_factory)

    def tearDown(self):
        self.engine.dispose()
        self.tempdir.cleanup()

    def test_update_and_delete_require_owner(self):
        memory = self.repository.create(
            user_id=2,
            content="用户研究光学遥感",
            category="profile",
        )

        self.assertIsNone(
            self.repository.update(
                memory["id"], user_id=1, content="用户研究 SAR 舰船检测"
            )
        )
        self.assertEqual(
            self.repository.delete_ids([memory["id"]], user_id=1),
            [],
        )

    def test_evicts_context_before_profile(self):
        profile = self.repository.create(
            user_id=1,
            content="用户是研究生",
            category="profile",
        )
        context = self.repository.create(
            user_id=1,
            content="用户正在准备一次演示",
            category="context",
        )

        deleted = self.repository.evict_overflow(1, max_memories=1)

        self.assertEqual(deleted, [context["id"]])
        self.assertEqual(
            list(self.repository.get_many([profile["id"]], user_id=1)),
            [profile["id"]],
        )


if __name__ == "__main__":
    unittest.main()
