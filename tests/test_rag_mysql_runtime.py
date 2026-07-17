import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from rag import parent_docstore as parent_docstore_module
from rag import vector_store as vector_module


class FakeKnowledgeRepository:
    def __init__(self, records=None, events=None):
        self.records = {record.doc_id: record for record in (records or [])}
        self.events = events if events is not None else []

    def get_by_doc_id(self, doc_id):
        return self.records.get(doc_id)

    def get_by_filename(self, filename):
        return next(
            (record for record in self.records.values() if record.filename == filename),
            None,
        )

    def get_by_hash(self, file_hash):
        return next(
            (record for record in self.records.values() if record.file_hash == file_hash),
            None,
        )

    def list_active(self):
        return [record for record in self.records.values() if record.status == "active"]

    def active_chunk_ids(self):
        return {
            chunk_id
            for record in self.list_active()
            for chunk_id in (record.chunk_ids or [])
        }

    def fingerprint(self):
        return "fake-fingerprint"

    def begin_ingestion(self, **kwargs):
        self.events.append(("begin", kwargs.copy()))
        record = self.records.get(kwargs["doc_id"])
        if record is None:
            record = make_record(doc_id=kwargs["doc_id"], filename=kwargs["filename"])
            self.records[record.doc_id] = record
        record.filename = kwargs["filename"]
        record.file_hash = kwargs.get("file_hash")
        record.storage_key = kwargs.get("storage_key")
        record.file_type = kwargs.get("file_type")
        record.chunk_method = kwargs.get("chunk_method")
        record.allowed_roles = list(kwargs.get("allowed_roles") or [])
        record.updated_by = kwargs.get("updated_by")
        record.chunk_ids = []
        record.chunk_count = 0
        record.parent_count = None
        record.child_count = None
        record.status = "processing"
        record.ingested_at = None
        record.error_message = None
        return record

    def mark_active(self, doc_id, **kwargs):
        self.events.append(("active", doc_id, kwargs.copy()))
        record = self.records[doc_id]
        record.chunk_ids = list(kwargs["chunk_ids"])
        record.chunk_count = kwargs["chunk_count"]
        record.chunk_method = kwargs.get("chunk_method", record.chunk_method)
        record.parent_count = kwargs.get("parent_count")
        record.child_count = kwargs.get("child_count")
        record.ingested_at = kwargs.get("ingested_at") or datetime.now()
        record.status = "active"
        record.error_message = None
        return record

    def activate_document(self, **kwargs):
        self.events.append(("activate", kwargs.copy()))
        record = self.records.get(kwargs["doc_id"])
        if record is None:
            record = make_record(doc_id=kwargs["doc_id"], filename=kwargs["filename"])
            self.records[record.doc_id] = record
        for field in (
            "filename",
            "file_hash",
            "storage_key",
            "file_type",
            "chunk_method",
            "chunk_count",
            "parent_count",
            "child_count",
            "updated_by",
        ):
            setattr(record, field, kwargs.get(field))
        record.chunk_ids = list(kwargs["chunk_ids"])
        record.allowed_roles = list(kwargs.get("allowed_roles") or [])
        record.status = "active"
        record.ingested_at = datetime.now()
        record.error_message = None
        return record

    def mark_failed(self, doc_id, error_message):
        self.events.append(("failed", doc_id, error_message))
        record = self.records[doc_id]
        record.status = "failed"
        record.error_message = error_message
        return record

    def mark_deleting(self, doc_id):
        self.events.append(("deleting", doc_id))
        self.records[doc_id].status = "deleting"
        return self.records[doc_id]

    def delete(self, doc_id):
        self.events.append(("document_delete", doc_id))
        return self.records.pop(doc_id, None) is not None

    def as_manifest(self):
        manifest = {}
        for record in self.list_active():
            entry = {
                "doc_id": record.doc_id,
                "file_hash": record.file_hash,
                "chunk_count": record.chunk_count,
                "chunk_ids": list(record.chunk_ids),
                "chunk_method": record.chunk_method,
                "file_type": record.file_type,
                "ingested_at": (
                    record.ingested_at.strftime("%Y-%m-%dT%H:%M:%S")
                    if record.ingested_at
                    else None
                ),
                "status": record.status,
            }
            if record.parent_count is not None:
                entry.update(
                    parent_ids=list(record.parent_ids),
                    parent_count=record.parent_count,
                    child_count=record.child_count,
                )
            manifest[record.filename] = entry
        return manifest


class FakeVectorStore:
    def __init__(self, events=None, fail_add=False, fail_delete_ids=None):
        self.events = events if events is not None else []
        self.fail_add = fail_add
        self.fail_delete_ids = set(fail_delete_ids or [])
        self.deleted_ids = []
        self.delete_attempts = []

    def add_documents(self, documents, ids):
        self.events.append(("chroma_add", list(ids)))
        if self.fail_add:
            raise RuntimeError("embedding failed")

    def delete(self, ids):
        self.delete_attempts.append(list(ids))
        if self.fail_delete_ids.intersection(ids):
            raise RuntimeError("chroma cleanup failed")
        self.deleted_ids.extend(ids)
        self.events.append(("chroma_delete", list(ids)))


class FakeParentDocstore:
    def __init__(self, events=None, fail_delete=False):
        self.events = events if events is not None else []
        self.saved = {}
        self.deleted_ids = []
        self.deleted_doc_ids = []
        self.fail_delete = fail_delete

    def save_batch(self, records):
        self.saved.update(records)
        self.events.append(("parent_save", list(records)))

    def delete_many(self, parent_ids):
        if self.fail_delete:
            raise RuntimeError("parent cleanup failed")
        self.deleted_ids.extend(parent_ids)
        self.events.append(("parent_delete", list(parent_ids)))
        for parent_id in parent_ids:
            self.saved.pop(parent_id, None)
        return len(parent_ids)

    def delete_by_doc_id(self, doc_id):
        if self.fail_delete:
            raise RuntimeError("parent cleanup failed")
        self.deleted_doc_ids.append(doc_id)
        self.events.append(("parent_delete_doc", doc_id))
        return 1


class FakeSplitter:
    def split_documents(self, documents):
        return list(documents)


class FakeHybridEngine:
    def __init__(self, events=None):
        self.events = events if events is not None else []

    def rebuild_bm25(self):
        self.events.append(("bm25_rebuild",))


def make_record(
    *,
    doc_id="doc-1",
    filename="paper.txt",
    file_hash="old-hash",
    chunk_ids=None,
    parent_ids=None,
    status="active",
    storage_key=None,
):
    parent_ids = list(parent_ids or [])
    return SimpleNamespace(
        doc_id=doc_id,
        filename=filename,
        file_hash=file_hash,
        storage_key=storage_key or filename,
        file_type="txt",
        chunk_method="fixed",
        chunk_ids=list(chunk_ids or [f"{doc_id}:old:0"]),
        chunk_count=len(chunk_ids or [f"{doc_id}:old:0"]),
        parent_ids=parent_ids,
        parent_count=len(parent_ids) if parent_ids else None,
        child_count=len(chunk_ids or [f"{doc_id}:old:0"]) if parent_ids else None,
        allowed_roles=["analyst"],
        status=status,
        ingested_at=datetime(2026, 1, 2, 3, 4, 5),
        error_message=None,
        updated_by=3,
    )


def make_service(repository, *, events=None, vector=None, parent_enabled=False):
    events = events if events is not None else []
    service = vector_module.VectorStoreService.__new__(vector_module.VectorStoreService)
    service.knowledge_repository = repository
    service.vector_store = vector or FakeVectorStore(events)
    service.hybrid_engine = FakeHybridEngine(events)
    service.parent_child_enabled = parent_enabled
    service.parent_docstore = FakeParentDocstore(events) if parent_enabled else None
    service.child_splitter = FakeSplitter()
    service.pdf_splitter = FakeSplitter()
    service.txt_splitter = FakeSplitter()
    service.default_splitter = FakeSplitter()
    service.semantic_enabled = False
    return service


class VectorStoreMySQLRuntimeTest(unittest.TestCase):
    def test_initialization_uses_mysql_repositories_and_dynamic_manifest(self):
        repository = FakeKnowledgeRepository()
        parent_store = FakeParentDocstore()
        hybrid = SimpleNamespace()
        config = {
            "collection_name": "test",
            "persist_directory": "unused-chroma",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "separators": ["\n"],
            "retrieve_k_children": 3,
            "bm25_cache_path": "unused.pkl",
            "parent_child_enabled": True,
            "child_chunk_size": 50,
            "child_chunk_overlap": 5,
            "semantic_chunking_enabled": False,
        }

        with (
            patch.object(vector_module, "chroma_conf", config),
            patch.object(vector_module, "Chroma", return_value=SimpleNamespace()),
            patch.object(
                vector_module,
                "KnowledgeRepository",
                return_value=repository,
            ),
            patch.object(
                vector_module,
                "ParentChunkRepository",
                return_value=parent_store,
            ),
            patch.object(vector_module, "DynamicHybridRetriever", return_value=hybrid) as retriever,
            patch.object(vector_module, "load_manifest", create=True) as load_manifest,
        ):
            service = vector_module.VectorStoreService()

        self.assertIs(service.knowledge_repository, repository)
        self.assertIs(service.parent_docstore, parent_store)
        self.assertFalse(hasattr(parent_docstore_module, "MySQLParentDocstore"))
        retriever.assert_called_once()
        self.assertIsNone(retriever.call_args.kwargs["manifest_path"])
        self.assertIs(
            retriever.call_args.kwargs["knowledge_repository"],
            repository,
        )
        self.assertEqual(
            retriever.call_args.kwargs["active_chunk_ids_provider"](),
            set(),
        )
        load_manifest.assert_not_called()

        self.assertEqual(service.manifest, {})
        repository.records["new-doc"] = make_record(doc_id="new-doc", filename="new.txt")
        self.assertIn("new.txt", service.manifest)

    def test_new_ingestion_writes_acl_only_when_activating(self):
        events = []
        repository = FakeKnowledgeRepository(events=events)
        service = make_service(repository, events=events)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.txt"
            path.write_text("ship detection", encoding="utf-8")
            with (
                patch.object(vector_module, "get_file_hash", return_value="a" * 64),
                patch.object(
                    vector_module,
                    "text_loader",
                    return_value=[Document(page_content="ship detection", metadata={})],
                ),
            ):
                result = service.load_documents(
                    [str(path)],
                    allowed_roles=["analyst"],
                    updated_by=42,
                )

        self.assertEqual(result, (1, 0, 0, 0))
        begin = next(event for event in events if event[0] == "begin")
        self.assertNotIn("allowed_roles", begin[1])
        self.assertNotIn("updated_by", begin[1])
        self.assertEqual(begin[1]["storage_key"], "paper.txt")
        activate = next(event for event in events if event[0] == "activate")
        self.assertEqual(activate[1]["allowed_roles"], ["analyst"])
        self.assertEqual(activate[1]["updated_by"], 42)
        self.assertLess(
            next(i for i, event in enumerate(events) if event[0] == "begin"),
            next(i for i, event in enumerate(events) if event[0] == "activate"),
        )
        self.assertEqual(repository.get_by_filename("paper.txt").status, "active")

    def test_batch_ingestion_contains_an_unexpected_failure_to_one_file(self):
        class FailingLookupRepository(FakeKnowledgeRepository):
            def get_by_filename(self, filename):
                if filename == "bad.txt":
                    raise RuntimeError("database unavailable")
                return super().get_by_filename(filename)

        repository = FailingLookupRepository()
        service = make_service(repository)

        with tempfile.TemporaryDirectory() as tmpdir:
            good = Path(tmpdir) / "good.txt"
            bad = Path(tmpdir) / "bad.txt"
            good.write_text("good", encoding="utf-8")
            bad.write_text("bad", encoding="utf-8")

            def fake_hash(path):
                return ("a" if os.path.basename(path) == "good.txt" else "b") * 64

            with (
                patch.object(vector_module, "get_file_hash", side_effect=fake_hash),
                patch.object(
                    vector_module,
                    "text_loader",
                    side_effect=lambda path: [
                        Document(page_content=os.path.basename(path), metadata={})
                    ],
                ),
            ):
                result = service.load_documents(
                    [str(good), str(bad)],
                    allowed_roles=["analyst"],
                    return_details=True,
                )

        self.assertEqual(result["new_count"], 1)
        self.assertEqual([item["status"] for item in result["files"]], ["new", "failed"])
        self.assertEqual(result["files"][1]["error"], "database unavailable")

    def test_single_ingestion_contains_an_unexpected_preprocessing_failure(self):
        class FailingLookupRepository(FakeKnowledgeRepository):
            def get_by_filename(self, filename):
                raise RuntimeError("database unavailable")

        service = make_service(FailingLookupRepository())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.txt"
            path.write_text("bad", encoding="utf-8")
            with patch.object(vector_module, "get_file_hash", return_value="b" * 64):
                result = service.load_documents([str(path)], return_details=True)

        self.assertEqual(result["new_count"], 0)
        self.assertEqual(result["files"][0]["status"], "failed")
        self.assertEqual(result["files"][0]["error"], "database unavailable")

    def test_failed_new_ingestion_marks_failed_and_cleans_staged_children_and_parents(self):
        events = []
        repository = FakeKnowledgeRepository(events=events)
        vector = FakeVectorStore(events, fail_add=True)
        service = make_service(
            repository,
            events=events,
            vector=vector,
            parent_enabled=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.txt"
            path.write_text("ship detection", encoding="utf-8")
            with (
                patch.object(vector_module, "get_file_hash", return_value="b" * 64),
                patch.object(
                    vector_module,
                    "text_loader",
                    return_value=[Document(page_content="ship detection", metadata={})],
                ),
            ):
                result = service.load_documents([str(path)], allowed_roles=["analyst"])

        self.assertEqual(result, (0, 0, 0, 0))
        record = repository.get_by_filename("paper.txt")
        self.assertEqual(record.status, "failed")
        staged_child_ids = next(event[1] for event in events if event[0] == "chroma_add")
        staged_parent_ids = next(event[1] for event in events if event[0] == "parent_save")
        self.assertEqual(vector.deleted_ids, staged_child_ids)
        self.assertEqual(service.parent_docstore.deleted_ids, staged_parent_ids)
        self.assertTrue(any(event[0] == "failed" for event in events))

    def test_failed_update_preserves_old_active_generation(self):
        events = []
        old = make_record(
            doc_id="stable-doc",
            filename="paper.txt",
            file_hash="old-hash",
            chunk_ids=["stable-doc:old:0", "stable-doc:old:1"],
        )
        repository = FakeKnowledgeRepository([old], events)
        vector = FakeVectorStore(events, fail_add=True)
        service = make_service(repository, events=events, vector=vector)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.txt"
            path.write_text("updated", encoding="utf-8")
            with (
                patch.object(vector_module, "get_file_hash", return_value="c" * 64),
                patch.object(
                    vector_module,
                    "text_loader",
                    return_value=[Document(page_content="updated", metadata={})],
                ),
            ):
                result = service.load_documents(
                    [str(path)],
                    allowed_roles=["admin"],
                    updated_by=99,
                )

        self.assertEqual(result, (0, 0, 0, 0))
        restored = repository.get_by_doc_id("stable-doc")
        self.assertEqual(restored.status, "active")
        self.assertEqual(restored.file_hash, "old-hash")
        self.assertEqual(restored.chunk_ids, ["stable-doc:old:0", "stable-doc:old:1"])
        self.assertNotIn("stable-doc:old:0", vector.deleted_ids)
        self.assertNotIn("stable-doc:old:1", vector.deleted_ids)
        self.assertFalse(any(event[0] == "begin" for event in events))
        self.assertFalse(any(event[0] == "failed" for event in events))
        self.assertFalse(any(event[0] == "activate" for event in events))

    def test_successful_update_activates_new_generation_before_old_cleanup(self):
        events = []
        old = make_record(
            doc_id="stable-doc",
            filename="paper.txt",
            file_hash="old-hash",
            chunk_ids=["stable-doc:old:0"],
        )
        repository = FakeKnowledgeRepository([old], events)
        service = make_service(repository, events=events)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.txt"
            path.write_text("updated", encoding="utf-8")
            with (
                patch.object(vector_module, "get_file_hash", return_value="d" * 64),
                patch.object(
                    vector_module,
                    "text_loader",
                    return_value=[Document(page_content="updated", metadata={})],
                ),
            ):
                result = service.load_documents([str(path)], allowed_roles=["analyst"])

        self.assertEqual(result, (0, 1, 0, 0))
        self.assertFalse(any(event[0] == "begin" for event in events))
        active_index = next(i for i, event in enumerate(events) if event[0] == "activate")
        old_delete_index = next(
            i
            for i, event in enumerate(events)
            if event == ("chroma_delete", ["stable-doc:old:0"])
        )
        self.assertLess(active_index, old_delete_index)
        self.assertNotEqual(
            repository.get_by_doc_id("stable-doc").chunk_ids,
            ["stable-doc:old:0"],
        )

    def test_cleanup_failures_leave_orphans_but_never_change_active_generation(self):
        events = []
        old = make_record(
            doc_id="stable-doc",
            filename="paper.txt",
            file_hash="old-hash",
            chunk_ids=["stable-doc:old:0"],
        )
        repository = FakeKnowledgeRepository([old], events)
        vector = FakeVectorStore(
            events,
            fail_add=True,
            fail_delete_ids={"stable-doc:gen:eeeeeeeeeeee:0000"},
        )
        service = make_service(repository, events=events, vector=vector)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.txt"
            path.write_text("updated", encoding="utf-8")
            with (
                patch.object(vector_module, "get_file_hash", return_value="e" * 64),
                patch.object(
                    vector_module,
                    "text_loader",
                    return_value=[Document(page_content="updated", metadata={})],
                ),
            ):
                service.load_documents([str(path)], allowed_roles=["business"])

        self.assertEqual(repository.active_chunk_ids(), {"stable-doc:old:0"})
        self.assertEqual(repository.get_by_doc_id("stable-doc").file_hash, "old-hash")
        self.assertIn(
            ["stable-doc:gen:eeeeeeeeeeee:0000"],
            vector.delete_attempts,
        )

    def test_old_cleanup_failure_keeps_new_generation_active(self):
        events = []
        old = make_record(
            doc_id="stable-doc",
            filename="paper.txt",
            file_hash="old-hash",
            chunk_ids=["stable-doc:old:0"],
        )
        repository = FakeKnowledgeRepository([old], events)
        vector = FakeVectorStore(events, fail_delete_ids={"stable-doc:old:0"})
        service = make_service(repository, events=events, vector=vector)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.txt"
            path.write_text("updated", encoding="utf-8")
            with (
                patch.object(vector_module, "get_file_hash", return_value="f" * 64),
                patch.object(
                    vector_module,
                    "text_loader",
                    return_value=[Document(page_content="updated", metadata={})],
                ),
            ):
                result = service.load_documents([str(path)], allowed_roles=["business"])

        self.assertEqual(result, (0, 1, 0, 0))
        self.assertEqual(
            repository.active_chunk_ids(),
            {"stable-doc:gen:ffffffffffff:0000"},
        )
        self.assertIn(["stable-doc:old:0"], vector.delete_attempts)

    def test_return_details_reports_version_storage_and_per_file_failure(self):
        events = []
        repository = FakeKnowledgeRepository(events=events)
        vector = FakeVectorStore(events, fail_add=True)
        service = make_service(repository, events=events, vector=vector)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            version_path = data_dir / ".knowledge_versions" / "version-1" / "paper.txt"
            version_path.parent.mkdir(parents=True)
            version_path.write_text("content", encoding="utf-8")
            with (
                patch.object(
                    vector_module,
                    "chroma_conf",
                    {
                        **vector_module.chroma_conf,
                        "data_path": str(data_dir),
                    },
                ),
                patch.object(vector_module, "get_file_hash", return_value="1" * 64),
                patch.object(
                    vector_module,
                    "text_loader",
                    return_value=[Document(page_content="content", metadata={})],
                ),
            ):
                result = service.load_documents(
                    [str(version_path)],
                    allowed_roles=["researcher"],
                    return_details=True,
                )

        self.assertEqual(result["new_count"], 0)
        self.assertEqual(result["files"][0]["status"], "failed")
        self.assertFalse(result["files"][0]["success"])
        self.assertEqual(
            result["files"][0]["storage_key"],
            ".knowledge_versions/version-1/paper.txt",
        )

    def test_full_scan_keeps_active_versioned_storage_file(self):
        events = []
        record = make_record(
            doc_id="stable-doc",
            filename="paper.txt",
            file_hash="old-hash",
            chunk_ids=["stable-doc:old:0"],
            storage_key=".knowledge_versions/version-1/paper.txt",
        )
        repository = FakeKnowledgeRepository([record], events)
        service = make_service(repository, events=events)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            version_path = data_dir / record.storage_key
            version_path.parent.mkdir(parents=True)
            version_path.write_text("active", encoding="utf-8")
            with (
                patch.object(
                    vector_module,
                    "chroma_conf",
                    {**vector_module.chroma_conf, "data_path": str(data_dir)},
                ),
                patch.object(vector_module, "get_file_hash", return_value="old-hash"),
            ):
                result = service.load_document()

        self.assertEqual(result, (0, 0, 1, 0))
        self.assertEqual(repository.get_by_doc_id("stable-doc").status, "active")
        self.assertNotIn(("document_delete", "stable-doc"), events)

    def test_delete_failure_leaves_document_deleting_for_retry(self):
        events = []
        record = make_record(
            doc_id="doc-delete",
            filename="paper.txt",
            chunk_ids=["child-1"],
        )
        repository = FakeKnowledgeRepository([record], events)
        vector = FakeVectorStore(events, fail_delete_ids={"child-1"})
        service = make_service(repository, events=events, vector=vector)

        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            service.delete_document_by_doc_id("doc-delete")

        self.assertEqual(repository.get_by_doc_id("doc-delete").status, "deleting")
        self.assertFalse(any(event[0] == "document_delete" for event in events))
        self.assertNotIn(("bm25_rebuild",), events)

    def test_delete_uses_database_metadata_rejects_traversal_and_rebuilds_bm25(self):
        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            outside = Path(tmpdir) / "outside.txt"
            outside.write_text("keep", encoding="utf-8")
            record = make_record(
                doc_id="doc-delete",
                filename="paper.txt",
                chunk_ids=["child-1", "child-2"],
                parent_ids=["parent-1"],
                storage_key="../outside.txt",
            )
            repository = FakeKnowledgeRepository([record], events)
            service = make_service(repository, events=events, parent_enabled=True)

            with patch.object(
                vector_module,
                "chroma_conf",
                {"data_path": str(data_dir)},
            ):
                deleted = service.delete_document_by_doc_id(
                    "doc-delete",
                    delete_file=True,
                )

            self.assertTrue(outside.exists())

        self.assertEqual(deleted, 2)
        self.assertEqual(service.vector_store.deleted_ids, ["child-1", "child-2"])
        self.assertEqual(service.parent_docstore.deleted_doc_ids, ["doc-delete"])
        self.assertIsNone(repository.get_by_doc_id("doc-delete"))
        self.assertEqual(events.count(("bm25_rebuild",)), 1)


if __name__ == "__main__":
    unittest.main()
