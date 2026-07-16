import importlib.util
import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker


_IMPORT_ERROR = None
try:
    from config.db_conf import (
        ASYNC_DATABASE_URL,
        SYNC_DATABASE_URL,
        SyncSessionLocal,
        sync_engine,
    )
    from models import Base
    from models.knowledge import KnowledgeDocument, ParentChunk
    from rag.parent_docstore import MySQLParentDocstore
    from services.knowledge_store import KnowledgeStore
except (ImportError, AttributeError) as exc:
    _IMPORT_ERROR = exc


class PhaseOneComponentsTest(unittest.TestCase):
    def test_phase_one_components_are_available(self):
        self.assertIsNone(_IMPORT_ERROR, f"Phase 1 components are missing: {_IMPORT_ERROR}")


@unittest.skipIf(_IMPORT_ERROR is not None, "Phase 1 components are not implemented yet")
class MySQLKnowledgeStoreTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )
        self.store = KnowledgeStore(session_factory=self.session_factory)
        self.parent_store = MySQLParentDocstore(session_factory=self.session_factory)

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _begin_and_activate(self, *, doc_id="doc-1", filename="paper.pdf", file_hash="hash-1"):
        doc = self.store.begin_ingestion(
            doc_id=doc_id,
            filename=filename,
            file_hash=file_hash,
            storage_key=f"knowledge/{filename}",
            file_type="pdf",
            chunk_method="parent_child_fixed",
            allowed_roles=["admin", "analyst"],
            updated_by=7,
        )
        self.assertEqual(doc.status, "processing")
        return self.store.mark_active(
            doc_id,
            chunk_count=2,
            chunk_ids=[f"{doc_id}:child:0", f"{doc_id}:child:1"],
            parent_count=1,
            child_count=2,
        )

    def test_sync_database_configuration_matches_async_environment(self):
        self.assertEqual(
            make_url(SYNC_DATABASE_URL),
            make_url(ASYNC_DATABASE_URL).set(drivername="mysql+pymysql"),
        )
        self.assertIs(SyncSessionLocal.kw["bind"], sync_engine)

    def test_database_urls_preserve_special_character_credentials(self):
        username = "user:name"
        password = "p@ss/word"
        module_path = Path(__file__).parents[1] / "config" / "db_conf.py"
        spec = importlib.util.spec_from_file_location("_special_credentials_db_conf", module_path)
        module = importlib.util.module_from_spec(spec)

        with patch.dict(
            os.environ,
            {
                "MYSQL_USER": username,
                "MYSQL_PASSWORD": password,
                "MYSQL_HOST": "db.internal",
                "MYSQL_PORT": "3307",
                "MYSQL_DATABASE": "sar_test",
            },
        ):
            spec.loader.exec_module(module)

        try:
            for database_url in (module.ASYNC_DATABASE_URL, module.SYNC_DATABASE_URL):
                parsed = make_url(database_url)
                self.assertEqual(parsed.username, username)
                self.assertEqual(parsed.password, password)
                self.assertEqual(parsed.host, "db.internal")
                self.assertEqual(parsed.port, 3307)
                self.assertEqual(parsed.database, "sar_test")
                self.assertEqual(parsed.query, {"charset": "utf8mb4"})
        finally:
            module.async_engine.sync_engine.dispose()
            module.sync_engine.dispose()

    def test_models_compile_on_sqlite_and_define_unique_indexes(self):
        indexes = inspect(self.engine).get_indexes("knowledge_documents")
        unique_columns = {
            tuple(index["column_names"])
            for index in indexes
            if index["unique"]
        }

        self.assertIn(("doc_id",), unique_columns)
        self.assertIn(("filename",), unique_columns)
        self.assertIn(("file_hash",), unique_columns)

        self.store.begin_ingestion(doc_id="null-1", filename="one.txt", file_hash=None)
        self.store.begin_ingestion(doc_id="null-2", filename="two.txt", file_hash=None)
        self.assertIsNone(self.store.get_by_hash(None))

    def test_document_lifecycle_and_lookups(self):
        processing = self.store.begin_ingestion(
            doc_id="doc-1",
            filename="paper.pdf",
            file_hash="hash-1",
            storage_key="knowledge/paper.pdf",
            file_type="pdf",
            chunk_method="parent_child_fixed",
            allowed_roles=["admin"],
        )

        self.assertEqual(processing.status, "processing")
        self.assertEqual(processing.chunk_ids, [])
        self.assertIsNone(processing.ingested_at)
        self.assertEqual(self.store.get_by_filename("paper.pdf").doc_id, "doc-1")
        self.assertEqual(self.store.get_by_hash("hash-1").doc_id, "doc-1")

        failed = self.store.mark_failed("doc-1", "embedding failed")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error_message, "embedding failed")
        self.assertEqual(self.store.list_active(), [])

        processing_again = self.store.begin_ingestion(
            doc_id="doc-1",
            filename="paper.pdf",
            file_hash="hash-1",
            storage_key="knowledge/paper.pdf",
        )
        self.assertEqual(processing_again.status, "processing")
        self.assertIsNone(processing_again.error_message)

        active = self.store.mark_active(
            "doc-1",
            chunk_count=2,
            chunk_ids=["child-1", "child-2"],
            chunk_method="semantic",
            parent_count=1,
            child_count=2,
        )
        self.assertEqual(active.status, "active")
        self.assertEqual(active.chunk_method, "semantic")
        self.assertIsInstance(active.ingested_at, datetime)
        self.assertEqual([doc.doc_id for doc in self.store.list_active()], ["doc-1"])

        deleting = self.store.mark_deleting("doc-1")
        self.assertEqual(deleting.status, "deleting")
        self.assertTrue(self.store.delete("doc-1"))
        self.assertFalse(self.store.delete("doc-1"))
        self.assertIsNone(self.store.get_by_doc_id("doc-1"))

    def test_activate_document_atomically_upserts_complete_active_row(self):
        self._begin_and_activate(
            doc_id="doc-atomic",
            filename="paper.pdf",
            file_hash="old-hash",
        )

        activated = self.store.activate_document(
            doc_id="doc-atomic",
            filename="paper.pdf",
            file_hash="new-hash",
            storage_key=".knowledge_versions/v2/paper.pdf",
            file_type="pdf",
            chunk_method="parent_child_semantic",
            chunk_ids=["new-child-1", "new-child-2", "new-child-3"],
            chunk_count=3,
            parent_count=2,
            child_count=3,
            allowed_roles=["business"],
            updated_by=19,
        )

        self.assertEqual(activated.status, "active")
        self.assertEqual(activated.file_hash, "new-hash")
        self.assertEqual(activated.storage_key, ".knowledge_versions/v2/paper.pdf")
        self.assertEqual(activated.chunk_ids, ["new-child-1", "new-child-2", "new-child-3"])
        self.assertEqual(activated.chunk_count, 3)
        self.assertEqual(activated.parent_count, 2)
        self.assertEqual(activated.child_count, 3)
        self.assertEqual(activated.allowed_roles, ["business"])
        self.assertEqual(activated.updated_by, 19)
        self.assertIsNotNone(activated.ingested_at)

    def test_active_chunk_ids_excludes_processing_failed_and_deleting_rows(self):
        self._begin_and_activate(
            doc_id="active-doc",
            filename="active.pdf",
            file_hash="active-hash",
        )
        self.store.begin_ingestion(
            doc_id="processing-doc",
            filename="processing.pdf",
            file_hash="processing-hash",
        )
        self.store.begin_ingestion(
            doc_id="failed-doc",
            filename="failed.pdf",
            file_hash="failed-hash",
        )
        self.store.mark_failed("failed-doc", "failed")
        self._begin_and_activate(
            doc_id="deleting-doc",
            filename="deleting.pdf",
            file_hash="deleting-hash",
        )
        self.store.mark_deleting("deleting-doc")

        self.assertEqual(
            self.store.active_chunk_ids(),
            {"active-doc:child:0", "active-doc:child:1"},
        )

    def test_acl_updates_do_not_change_fingerprint(self):
        self._begin_and_activate()
        before = self.store.fingerprint()

        with self.session_factory.begin() as session:
            doc = session.scalar(
                select(KnowledgeDocument).where(KnowledgeDocument.doc_id == "doc-1")
            )
            doc.allowed_roles = ["admin"]
            doc.updated_by = 99
            doc.updated_at = doc.updated_at + timedelta(days=1)

        self.assertEqual(self.store.fingerprint(), before)

        self.store.begin_ingestion(doc_id="failed", filename="failed.txt", file_hash="failed-hash")
        self.store.mark_failed("failed", "parse error")
        self.assertEqual(self.store.fingerprint(), before)

        self.store.mark_active(
            "doc-1",
            chunk_count=3,
            chunk_ids=["child-1", "child-2", "child-3"],
        )
        self.assertNotEqual(self.store.fingerprint(), before)

    def test_fingerprint_tracks_active_document_technical_state(self):
        self._begin_and_activate()
        before = self.store.fingerprint()

        with self.session_factory.begin() as session:
            doc = session.scalar(
                select(KnowledgeDocument).where(KnowledgeDocument.doc_id == "doc-1")
            )
            doc.chunk_ids = ["doc-1:child:0", "doc-1:child:replacement"]

        after_chunk_identity_change = self.store.fingerprint()
        self.assertNotEqual(after_chunk_identity_change, before)

        with self.session_factory.begin() as session:
            doc = session.scalar(
                select(KnowledgeDocument).where(KnowledgeDocument.doc_id == "doc-1")
            )
            doc.filename = "renamed-paper.pdf"
            doc.chunk_method = "semantic"
            doc.parent_count = 2
            doc.child_count = 3

        self.assertNotEqual(self.store.fingerprint(), after_chunk_identity_change)

    def test_manifest_matches_legacy_shape_and_is_read_only(self):
        active = self._begin_and_activate()
        active = self.store.mark_active(
            "doc-1",
            chunk_count=2,
            chunk_ids=[
                "doc-1:parent:000:child:000",
                "doc-1:parent:000:child:001",
            ],
            parent_count=1,
            child_count=2,
        )
        self.parent_store.save(
            "doc-1:parent:000",
            "parent text",
            {
                "doc_id": "doc-1",
                "parent_id": "doc-1:parent:000",
                "parent_index": 0,
                "chunk_type": "parent",
            },
        )
        self.parent_store.save(
            "doc-1:old:parent:999",
            "orphan parent from an old generation",
            {
                "doc_id": "doc-1",
                "parent_id": "doc-1:old:parent:999",
                "parent_index": 999,
                "chunk_type": "parent",
            },
        )

        manifest = self.store.as_manifest()
        self.assertEqual(
            manifest["paper.pdf"],
            {
                "doc_id": "doc-1",
                "file_hash": "hash-1",
                "chunk_count": 2,
                "chunk_ids": [
                    "doc-1:parent:000:child:000",
                    "doc-1:parent:000:child:001",
                ],
                "chunk_method": "parent_child_fixed",
                "file_type": "pdf",
                "ingested_at": active.ingested_at.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": "active",
                "parent_ids": ["doc-1:parent:000"],
                "parent_count": 1,
                "child_count": 2,
            },
        )

        manifest["paper.pdf"]["chunk_ids"].append("mutated")
        manifest["paper.pdf"]["status"] = "failed"
        fresh_manifest = self.store.as_manifest()
        self.assertEqual(fresh_manifest["paper.pdf"]["status"], "active")
        self.assertEqual(
            fresh_manifest["paper.pdf"]["chunk_ids"],
            [
                "doc-1:parent:000:child:000",
                "doc-1:parent:000:child:001",
            ],
        )

    def test_parent_chunk_crud_get_many_and_batch_upsert(self):
        self.parent_store.save(
            "doc-1:parent:000",
            "old text",
            {"doc_id": "doc-1", "parent_id": "doc-1:parent:000", "parent_index": 0},
        )
        self.assertEqual(self.parent_store.count(), 1)
        self.assertEqual(
            self.parent_store.get("doc-1:parent:000")["page_content"],
            "old text",
        )

        self.parent_store.save_batch(
            {
                "doc-1:parent:000": {
                    "page_content": "updated text",
                    "metadata": {
                        "doc_id": "doc-1",
                        "parent_id": "doc-1:parent:000",
                        "parent_index": 0,
                        "page": 3,
                    },
                },
                "doc-1:parent:001": {
                    "page_content": "second text",
                    "metadata": {
                        "doc_id": "doc-1",
                        "parent_id": "doc-1:parent:001",
                        "parent_index": 1,
                    },
                },
                "doc-2:parent:000": {
                    "page_content": "other document",
                    "metadata": {
                        "doc_id": "doc-2",
                        "parent_id": "doc-2:parent:000",
                        "parent_index": 0,
                    },
                },
            }
        )

        requested_parent_ids = ["doc-1:parent:001", "missing", "doc-1:parent:000"]
        records = self.parent_store.get_many(requested_parent_ids)
        self.assertEqual(set(records), {"doc-1:parent:000", "doc-1:parent:001"})
        self.assertEqual(
            list(records),
            ["doc-1:parent:001", "doc-1:parent:000"],
        )
        self.assertEqual(records["doc-1:parent:000"]["page_content"], "updated text")
        self.assertEqual(records["doc-1:parent:000"]["metadata"]["page"], 3)
        self.assertEqual(self.parent_store.count(), 3)

        self.assertEqual(self.parent_store.delete_many(["doc-1:parent:001", "missing"]), 1)
        self.assertEqual(self.parent_store.delete_by_doc_id("doc-1"), 1)
        self.assertEqual(self.parent_store.count(), 1)
        self.assertEqual(self.parent_store.get_many([]), {})


if __name__ == "__main__":
    unittest.main()
