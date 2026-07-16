import json
import sqlite3
import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.orm import sessionmaker

from models import Base
from models.knowledge import KnowledgeDocument, ParentChunk
import utils.migrate_knowledge_mysql as migration
from utils.migrate_knowledge_mysql import (
    check_consistency,
    ensure_mysql_schema,
    migrate_json_to_mysql,
)


class KnowledgeMySQLMigrationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.manifest_path = root / "manifest.json"
        self.parent_path = root / "parent_docstore.json"
        self.engine = create_engine(f"sqlite:///{root / 'knowledge.db'}")
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

        self.manifest = {
            "alpha.pdf": {
                "doc_id": "doc-alpha",
                "file_hash": "hash-alpha",
                "storage_key": "data/alpha.pdf",
                "file_type": "pdf",
                "chunk_method": "parent_child",
                "chunk_ids": ["alpha-child-0", "alpha-child-1"],
                "chunk_count": 2,
                "parent_ids": [
                    "doc-alpha:parent:000",
                    "doc-alpha:parent:001",
                ],
                "parent_count": 2,
                "child_count": 2,
                "ingested_at": "2026-07-08T10:26:55",
                "status": "active",
            },
            "beta.txt": {
                "doc_id": "doc-beta",
                "file_hash": "hash-beta",
                "storage_key": "data/beta.txt",
                "file_type": "txt",
                "chunk_method": "semantic",
                "chunk_ids": ["beta-child-0"],
                "chunk_count": 1,
                "parent_count": 0,
                "child_count": 1,
                "ingested_at": "2026-07-09T11:00:00",
                "status": "active",
            },
        }
        self.parents = {
            "doc-alpha:parent:000": {
                "page_content": "first parent",
                "metadata": {"doc_id": "doc-alpha", "parent_index": 7},
            },
            "doc-alpha:parent:001": {
                "page_content": "second parent",
                "metadata": {"doc_id": "doc-alpha"},
            },
        }
        self._write_json(self.manifest_path, self.manifest)
        self._write_json(self.parent_path, self.parents)

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def _business_counts(self):
        with self.Session() as session:
            return (
                session.scalar(select(func.count()).select_from(KnowledgeDocument)),
                session.scalar(select(func.count()).select_from(ParentChunk)),
            )

    @staticmethod
    def _empty_report():
        return migration.CheckReport(
            missing_docs=[],
            extra_docs=[],
            parent_count_mismatches={},
            orphan_parent_ids=[],
            chroma_chunk_mismatches={},
            duplicate_filenames=[],
            duplicate_hashes=[],
            schema_errors=[],
        )

    def test_migration_is_idempotent_preserves_acl_and_updates_parent_blocks(self):
        original_created_at = datetime(2025, 1, 2, 3, 4, 5)
        with self.Session() as session:
            session.add(
                KnowledgeDocument(
                    doc_id="doc-alpha",
                    filename="alpha.pdf",
                    allowed_roles=["analyst"],
                    updated_by=42,
                    created_at=original_created_at,
                )
            )
            session.commit()

        first = migrate_json_to_mysql(
            self.Session, self.manifest_path, self.parent_path
        )
        self.assertEqual(first.documents_created, 1)
        self.assertEqual(first.documents_updated, 1)
        self.assertEqual(first.parents_upserted, 2)
        self.assertEqual(self._business_counts(), (2, 2))

        with self.Session() as session:
            alpha = session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.doc_id == "doc-alpha"
                )
            )
            beta = session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.doc_id == "doc-beta"
                )
            )
            first_parent = session.get(ParentChunk, "doc-alpha:parent:000")
            second_parent = session.get(ParentChunk, "doc-alpha:parent:001")
            self.assertEqual(alpha.allowed_roles, ["analyst"])
            self.assertEqual(alpha.updated_by, 42)
            self.assertEqual(alpha.created_at, original_created_at)
            self.assertEqual(alpha.storage_key, "data/alpha.pdf")
            self.assertEqual(alpha.chunk_ids, ["alpha-child-0", "alpha-child-1"])
            self.assertEqual(beta.allowed_roles, [])
            self.assertEqual(first_parent.parent_index, 7)
            self.assertEqual(second_parent.parent_index, 1)

        self.parents["doc-alpha:parent:000"]["page_content"] = "updated parent"
        self.parents["doc-alpha:parent:000"]["metadata"]["parent_index"] = 9
        self._write_json(self.parent_path, self.parents)
        second = migrate_json_to_mysql(
            self.Session, self.manifest_path, self.parent_path
        )

        self.assertEqual(second.documents_created, 0)
        self.assertEqual(second.documents_updated, 2)
        self.assertEqual(self._business_counts(), (2, 2))
        with self.Session() as session:
            parent = session.get(ParentChunk, "doc-alpha:parent:000")
            alpha = session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.doc_id == "doc-alpha"
                )
            )
            self.assertEqual(parent.page_content, "updated parent")
            self.assertEqual(parent.parent_index, 9)
            self.assertEqual(alpha.allowed_roles, ["analyst"])
            self.assertEqual(alpha.updated_by, 42)
            self.assertEqual(alpha.created_at, original_created_at)

    def test_check_finds_missing_orphan_and_chroma_mismatch_without_writes(self):
        migrate_json_to_mysql(self.Session, self.manifest_path, self.parent_path)
        with self.Session() as session:
            beta = session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.doc_id == "doc-beta"
                )
            )
            session.delete(beta)
            session.add(
                ParentChunk(
                    parent_id="missing-doc:parent:000",
                    doc_id="missing-doc",
                    parent_index=0,
                    page_content="orphan",
                    metadata_json={"doc_id": "missing-doc"},
                )
            )
            session.commit()

        before = self._business_counts()

        def fake_chroma_count(doc_id):
            return {"doc-alpha": 99, "doc-beta": 1}.get(doc_id, 0)

        report = check_consistency(
            self.Session,
            self.manifest_path,
            self.parent_path,
            chroma_count_provider=fake_chroma_count,
        )

        self.assertFalse(report.ok)
        self.assertIn("doc-beta", report.missing_docs)
        self.assertIn("missing-doc:parent:000", report.orphan_parent_ids)
        self.assertIn("doc-alpha", report.chroma_chunk_mismatches)
        self.assertEqual(before, self._business_counts())
        self.assertIsInstance(report.extra_docs, list)
        self.assertIsInstance(report.parent_count_mismatches, dict)
        self.assertIsInstance(report.duplicate_filenames, list)
        self.assertIsInstance(report.duplicate_hashes, list)

    def test_ensure_schema_creates_sqlite_tables_idempotently(self):
        engine = create_engine("sqlite:///:memory:")
        ensure_mysql_schema(engine)
        ensure_mysql_schema(engine)
        inspector = inspect(engine)
        self.assertIn("knowledge_documents", inspector.get_table_names())
        self.assertIn("parent_chunks", inspector.get_table_names())

    def test_check_reports_all_missing_required_indexes(self):
        required_names = {
            "idx_kdoc_doc_id",
            "idx_kdoc_filename",
            "idx_kdoc_file_hash",
            "idx_kdoc_status",
            "idx_parent_chunk_doc_id",
        }
        with self.engine.begin() as connection:
            for index_name in required_names:
                connection.exec_driver_sql(f'DROP INDEX "{index_name}"')

        report = check_consistency(
            self.Session,
            self.manifest_path,
            self.parent_path,
        )

        reported_names = {
            error["index"]
            for error in report.schema_errors
            if error["type"].startswith("missing_")
        }
        self.assertEqual(reported_names, required_names)

    def test_schema_reconciliation_recreates_all_indexes_idempotently(self):
        required_shapes = {
            "knowledge_documents": {
                "idx_kdoc_doc_id": (True, ["doc_id"]),
                "idx_kdoc_filename": (True, ["filename"]),
                "idx_kdoc_file_hash": (True, ["file_hash"]),
                "idx_kdoc_status": (False, ["status"]),
            },
            "parent_chunks": {
                "idx_parent_chunk_doc_id": (False, ["doc_id"]),
            },
        }
        with self.engine.begin() as connection:
            for indexes in required_shapes.values():
                for index_name in indexes:
                    connection.exec_driver_sql(f'DROP INDEX "{index_name}"')

        ensure_mysql_schema(self.engine)
        ensure_mysql_schema(self.engine)

        schema_inspector = inspect(self.engine)
        for table_name, expected in required_shapes.items():
            actual = {
                index["name"]: (
                    bool(index["unique"]),
                    index["column_names"],
                )
                for index in schema_inspector.get_indexes(table_name)
            }
            self.assertEqual(actual, expected)

    def test_check_and_preflight_reject_wrong_parent_index_definition(self):
        with self.engine.begin() as connection:
            connection.exec_driver_sql('DROP INDEX "idx_parent_chunk_doc_id"')
            connection.exec_driver_sql(
                "CREATE INDEX idx_parent_chunk_doc_id "
                "ON parent_chunks (parent_index)"
            )

        report = check_consistency(
            self.Session,
            self.manifest_path,
            self.parent_path,
        )
        errors = {
            error["index"]: error
            for error in report.schema_errors
            if error["type"].startswith("invalid_")
        }
        self.assertIn("idx_parent_chunk_doc_id", errors)
        self.assertEqual(
            errors["idx_parent_chunk_doc_id"]["actual"]["column_names"],
            ["parent_index"],
        )

        with self.assertRaises(migration.MigrationError) as raised:
            ensure_mysql_schema(self.engine)
        self.assertEqual(
            raised.exception.details["index"], "idx_parent_chunk_doc_id"
        )

    def test_chroma_provider_reads_sqlite_in_read_only_mode_by_doc_id(self):
        chroma_dir = Path(self.temp_dir.name) / "chroma"
        chroma_dir.mkdir()
        database_path = chroma_dir / "chroma.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.executescript(
                """
                CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);
                CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT);
                CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT);
                CREATE TABLE embedding_metadata (
                    id INTEGER, key TEXT, string_value TEXT
                );
                INSERT INTO collections VALUES ('collection-1', 'knowledge');
                INSERT INTO collections VALUES ('collection-2', 'other');
                INSERT INTO segments VALUES ('segment-1', 'collection-1');
                INSERT INTO segments VALUES ('segment-2', 'collection-2');
                INSERT INTO embeddings VALUES (1, 'segment-1');
                INSERT INTO embeddings VALUES (2, 'segment-1');
                INSERT INTO embeddings VALUES (3, 'segment-1');
                INSERT INTO embeddings VALUES (4, 'segment-2');
                INSERT INTO embedding_metadata VALUES (1, 'doc_id', 'doc-alpha');
                INSERT INTO embedding_metadata VALUES (2, 'doc_id', 'doc-alpha');
                INSERT INTO embedding_metadata VALUES (3, 'doc_id', 'doc-beta');
                INSERT INTO embedding_metadata VALUES (4, 'doc_id', 'doc-alpha');
                """
            )
            connection.commit()
        finally:
            connection.close()

        provider = migration.ChromaDocumentCountProvider(
            chroma_dir, "knowledge"
        )

        self.assertEqual(provider("doc-alpha"), 2)
        self.assertEqual(provider("doc-beta"), 1)
        self.assertEqual(provider("doc-missing"), 0)
        moved_path = chroma_dir / "moved.sqlite3"
        database_path.rename(moved_path)
        moved_path.rename(database_path)

    def test_chroma_provider_rejects_missing_database_without_creating_it(self):
        chroma_dir = Path(self.temp_dir.name) / "missing-chroma"

        with self.assertRaises(migration.MigrationError) as raised:
            migration.ChromaDocumentCountProvider(chroma_dir, "knowledge")

        self.assertEqual(raised.exception.code, "chroma_database_missing")
        self.assertFalse(chroma_dir.exists())

    def test_cli_check_injects_provider_without_schema_mutation(self):
        provider = object()
        report = self._empty_report()
        with (
            patch.object(migration, "check_consistency", return_value=report) as check,
            patch.object(migration, "ensure_mysql_schema") as ensure,
            patch("builtins.print"),
        ):
            exit_code = migration.main(
                ["--check"],
                session_factory=self.Session,
                chroma_count_provider=provider,
            )

        self.assertEqual(exit_code, 0)
        ensure.assert_not_called()
        check.assert_called_once_with(
            self.Session,
            "manifest.json",
            "parent_docstore.json",
            chroma_count_provider=provider,
        )

    def test_cli_check_builds_chroma_provider_with_path_override(self):
        provider = object()
        report = self._empty_report()
        with (
            patch.object(
                migration,
                "_create_chroma_count_provider",
                return_value=provider,
            ) as create_provider,
            patch.object(migration, "check_consistency", return_value=report),
            patch("builtins.print"),
        ):
            exit_code = migration.main(
                ["--check", "--chroma-path", "custom-chroma"],
                session_factory=self.Session,
            )

        self.assertEqual(exit_code, 0)
        create_provider.assert_called_once_with("custom-chroma")

    def test_check_reports_missing_schema_without_creating_tables(self):
        engine = create_engine("sqlite:///:memory:")
        self.addCleanup(engine.dispose)
        session_factory = sessionmaker(engine)

        report = check_consistency(
            session_factory,
            self.manifest_path,
            self.parent_path,
        )

        self.assertFalse(report.ok)
        self.assertTrue(report.schema_errors)
        self.assertEqual(inspect(engine).get_table_names(), [])

    def test_migration_rejects_duplicate_filenames_and_hashes_before_schema_changes(self):
        engine = create_engine("sqlite:///:memory:")
        self.addCleanup(engine.dispose)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE knowledge_documents ("
                    "id INTEGER PRIMARY KEY, doc_id VARCHAR(64), "
                    "filename VARCHAR(255), file_hash VARCHAR(128))"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO knowledge_documents "
                    "(doc_id, filename, file_hash) VALUES "
                    "('legacy-1', 'duplicate.pdf', 'duplicate-hash'), "
                    "('legacy-2', 'duplicate.pdf', 'duplicate-hash')"
                )
            )
        session_factory = sessionmaker(engine)

        with self.assertRaises(migration.MigrationError) as raised:
            migrate_json_to_mysql(
                session_factory,
                self.manifest_path,
                self.parent_path,
            )

        self.assertEqual(raised.exception.code, "duplicate_documents")
        self.assertEqual(
            raised.exception.details["duplicate_filenames"], ["duplicate.pdf"]
        )
        self.assertEqual(
            raised.exception.details["duplicate_hashes"], ["duplicate-hash"]
        )
        self.assertNotIn("parent_chunks", inspect(engine).get_table_names())

    def test_ensure_schema_rejects_duplicate_doc_ids_before_any_ddl(self):
        engine = create_engine("sqlite:///:memory:")
        self.addCleanup(engine.dispose)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE knowledge_documents ("
                    "id INTEGER PRIMARY KEY, doc_id VARCHAR(64), "
                    "filename VARCHAR(255), file_hash VARCHAR(128))"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO knowledge_documents "
                    "(doc_id, filename, file_hash) VALUES "
                    "('duplicate-doc', 'first.pdf', 'first-hash'), "
                    "('duplicate-doc', 'second.pdf', 'second-hash')"
                )
            )

        with (
            patch.object(migration.Base.metadata, "create_all") as create_all,
            self.assertRaises(migration.MigrationError) as raised,
        ):
            ensure_mysql_schema(engine)

        create_all.assert_not_called()
        self.assertEqual(raised.exception.code, "duplicate_documents")
        self.assertEqual(
            raised.exception.details["duplicate_doc_ids"], ["duplicate-doc"]
        )

    def test_ensure_schema_rejects_wrong_index_with_expected_name(self):
        engine = create_engine("sqlite:///:memory:")
        self.addCleanup(engine.dispose)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE knowledge_documents ("
                    "id INTEGER PRIMARY KEY, doc_id VARCHAR(64), "
                    "filename VARCHAR(255), file_hash VARCHAR(128))"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX idx_kdoc_doc_id "
                    "ON knowledge_documents (doc_id)"
                )
            )

        with (
            patch.object(migration.Base.metadata, "create_all") as create_all,
            self.assertRaises(migration.MigrationError) as raised,
        ):
            ensure_mysql_schema(engine)

        create_all.assert_not_called()
        self.assertEqual(raised.exception.code, "invalid_unique_index")
        self.assertEqual(raised.exception.details["index"], "idx_kdoc_doc_id")

    def test_ensure_schema_rejects_incomplete_existing_table_before_ddl(self):
        engine = create_engine("sqlite:///:memory:")
        self.addCleanup(engine.dispose)
        with engine.begin() as connection:
            connection.execute(
                text("CREATE TABLE knowledge_documents (id INTEGER PRIMARY KEY)")
            )

        with (
            patch.object(migration.Base.metadata, "create_all") as create_all,
            self.assertRaises(migration.MigrationError) as raised,
        ):
            ensure_mysql_schema(engine)

        create_all.assert_not_called()
        self.assertEqual(raised.exception.code, "incomplete_existing_schema")
        self.assertEqual(
            raised.exception.details["missing_columns"], ["doc_id", "filename"]
        )

    def test_migration_rolls_back_all_dml_when_parent_upsert_fails(self):
        self.parents["invalid-parent"] = {
            "page_content": "invalid",
            "metadata": {},
        }
        self._write_json(self.parent_path, self.parents)

        with self.assertRaises(ValueError):
            migrate_json_to_mysql(
                self.Session,
                self.manifest_path,
                self.parent_path,
            )

        self.assertEqual(self._business_counts(), (0, 0))

    def test_schema_ddl_error_explains_statement_level_retry(self):
        class FakeInspector:
            def get_table_names(self):
                return ["knowledge_documents"]

            def get_columns(self, table_name):
                names = {
                    "id",
                    "doc_id",
                    "filename",
                    *migration._MYSQL_MISSING_COLUMNS[table_name],
                }
                names.remove("storage_key")
                return [{"name": name} for name in names]

            def get_indexes(self, table_name):
                return [
                    {
                        "name": name,
                        "unique": True,
                        "column_names": [column],
                    }
                    for name, column in migration._MYSQL_UNIQUE_INDEXES[
                        table_name
                    ].items()
                ]

        def fail_ddl(sql):
            raise RuntimeError("simulated DDL failure")

        class EmptyResult:
            @staticmethod
            def scalars():
                return []

        read_connection = SimpleNamespace(
            execute=lambda statement: EmptyResult()
        )

        engine = SimpleNamespace(
            dialect=SimpleNamespace(name="mysql"),
            connect=lambda: nullcontext(read_connection),
            begin=lambda: nullcontext(
                SimpleNamespace(exec_driver_sql=fail_ddl)
            ),
        )
        with (
            patch.object(migration.Base.metadata, "create_all"),
            patch.object(migration, "inspect", return_value=FakeInspector()),
            self.assertRaises(migration.MigrationError) as raised,
        ):
            ensure_mysql_schema(engine)

        self.assertEqual(raised.exception.code, "schema_ddl_failed")
        self.assertEqual(raised.exception.details["atomicity"], "statement")
        self.assertTrue(raised.exception.details["retry_safe"])

    def test_cli_check_returns_one_for_consistency_mismatch(self):
        report = migration.CheckReport(
            missing_docs=["doc-missing"],
            extra_docs=[],
            parent_count_mismatches={},
            orphan_parent_ids=[],
            chroma_chunk_mismatches={},
            duplicate_filenames=[],
            duplicate_hashes=[],
        )
        output = StringIO()
        with (
            patch.object(migration, "check_consistency", return_value=report),
            patch("sys.stdout", output),
        ):
            exit_code = migration.main(
                ["--check"],
                session_factory=self.Session,
                chroma_count_provider=object(),
            )

        self.assertEqual(exit_code, 1)
        self.assertFalse(json.loads(output.getvalue())["ok"])

    def test_cli_check_returns_structured_error_for_provider_failure(self):
        error_output = StringIO()
        with (
            patch.object(
                migration,
                "_create_chroma_count_provider",
                side_effect=migration.MigrationError(
                    "chroma_database_missing",
                    "Chroma database does not exist",
                    {"path": "missing/chroma.sqlite3"},
                ),
            ),
            patch("sys.stderr", error_output),
        ):
            exit_code = migration.main(
                ["--check", "--chroma-path", "missing"],
                session_factory=self.Session,
            )

        payload = json.loads(error_output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["error"]["code"], "chroma_database_missing")


if __name__ == "__main__":
    unittest.main()
