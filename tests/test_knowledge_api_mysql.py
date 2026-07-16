import io
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api.routers import knowledge


def make_doc(**overrides):
    values = {
        "doc_id": "doc-1",
        "filename": "paper.pdf",
        "file_hash": "hash-1",
        "storage_key": "paper.pdf",
        "file_type": "pdf",
        "chunk_method": "parent_child_fixed",
        "chunk_count": 4,
        "parent_count": 2,
        "child_count": 4,
        "allowed_roles": ["researcher"],
        "status": "active",
        "ingested_at": datetime(2026, 1, 2, 3, 4, 5),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeUpload:
    def __init__(self, filename, content=b"content"):
        self.filename = filename
        self.file = io.BytesIO(content)

    async def read(self):
        raise AssertionError("upload must be streamed from UploadFile.file")


class KnowledgeAPIWithMySQLTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.lock_keys = []

        @asynccontextmanager
        async def recording_lock(key, timeout=60):
            self.lock_keys.append((key, timeout))
            yield

        self.recording_lock = recording_lock

    async def test_list_uses_database_rows_without_reading_vector_manifest(self):
        list_docs = AsyncMock(return_value=[make_doc()])
        with (
            patch.object(knowledge, "list_visible_documents", list_docs),
            patch.object(
                knowledge,
                "get_vector_store",
                side_effect=AssertionError("vector manifest must not be read"),
            ),
        ):
            response = await knowledge.list_knowledge_files(
                user={"id": 8, "role": "researcher", "username": "reader"},
                db=SimpleNamespace(),
            )

        self.assertEqual(response["total_files"], 1)
        self.assertEqual(response["total_chunks"], 4)
        self.assertEqual(response["files"][0]["doc_id"], "doc-1")
        self.assertNotIn("allowed_roles", response["files"][0])

    async def test_upload_passes_acl_to_runtime_under_one_global_lock(self):
        calls = []
        threadpool_calls = []

        class VectorStore:
            def load_documents(
                self,
                paths,
                *,
                allowed_roles=None,
                updated_by=None,
                return_details=False,
            ):
                calls.append((list(paths), allowed_roles, updated_by, return_details))
                return {
                    "new_count": 1,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "removed_count": 0,
                    "files": [
                        {
                            "filename": "paper.pdf",
                            "status": "new",
                            "success": True,
                            "storage_key": ".knowledge_versions/version/paper.pdf",
                            "previous_storage_key": None,
                            "error": None,
                        }
                    ],
                }

        async def inline_threadpool(func, *args, **kwargs):
            threadpool_calls.append(func)
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    knowledge,
                    "chroma_conf",
                    {
                        "data_path": tmpdir,
                        "allow_knowledge_file_type": [".txt", ".pdf"],
                    },
                ),
                patch.object(knowledge, "get_abs_path", side_effect=lambda path: path),
                patch.object(knowledge, "get_vector_store", return_value=VectorStore()) as get_vector,
                patch.object(knowledge, "run_in_threadpool", side_effect=inline_threadpool),
                patch.object(knowledge, "rate_limit", new=AsyncMock()),
                patch.object(knowledge, "redis_lock", new=self.recording_lock),
                patch.object(
                    knowledge,
                    "upsert_document_acl",
                    new=AsyncMock(side_effect=AssertionError("post-ingestion ACL write is forbidden")),
                    create=True,
                ),
            ):
                response = await knowledge.upload_knowledge(
                    files=[FakeUpload("paper.pdf")],
                    visibility_mode="roles",
                    allowed_roles='["researcher"]',
                    admin={"id": 7, "role": "admin", "username": "admin"},
                    db=SimpleNamespace(),
                )

            version_files = list((Path(tmpdir) / ".knowledge_versions").glob("*/*"))
            self.assertEqual(len(version_files), 1)
            self.assertEqual(version_files[0].name, "paper.pdf")
            self.assertFalse((Path(tmpdir) / "paper.pdf").exists())

        self.assertEqual(response["new_count"], 1)
        self.assertTrue(response["success"])
        self.assertEqual(calls[0][1:], (["researcher"], 7, True))
        self.assertIs(threadpool_calls[0], get_vector)
        self.assertGreaterEqual(len(threadpool_calls), 3)
        self.assertEqual(self.lock_keys, [(knowledge.KNOWLEDGE_WRITE_LOCK_KEY, 600)])

    async def test_failed_upload_returns_success_false_and_removes_staged_version(self):
        class VectorStore:
            def __init__(self, data_dir):
                self.data_dir = data_dir

            def load_documents(self, paths, **kwargs):
                return {
                    "new_count": 0,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "removed_count": 0,
                    "files": [
                        {
                            "filename": "paper.pdf",
                            "status": "failed",
                            "success": False,
                            "storage_key": os.path.relpath(paths[0], self.data_dir),
                            "previous_storage_key": None,
                            "error": "embedding failed",
                        }
                    ],
                }

        async def inline_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    knowledge,
                    "chroma_conf",
                    {
                        "data_path": tmpdir,
                        "allow_knowledge_file_type": [".txt", ".pdf"],
                    },
                ),
                patch.object(knowledge, "get_abs_path", side_effect=lambda path: path),
                patch.object(knowledge, "get_vector_store", return_value=VectorStore(tmpdir)),
                patch.object(knowledge, "run_in_threadpool", side_effect=inline_threadpool),
                patch.object(knowledge, "rate_limit", new=AsyncMock()),
                patch.object(knowledge, "redis_lock", new=self.recording_lock),
            ):
                response = await knowledge.upload_knowledge(
                    files=[FakeUpload("paper.pdf", b"failed")],
                    visibility_mode="roles",
                    allowed_roles='["researcher"]',
                    admin={"id": 7, "role": "admin", "username": "admin"},
                    db=SimpleNamespace(),
                )

            remaining = [path for path in Path(tmpdir).rglob("*") if path.is_file()]

        self.assertFalse(response["success"])
        self.assertEqual(response["file_results"][0]["status"], "failed")
        self.assertEqual(remaining, [])

    async def test_unsupported_upload_is_reported_without_staging_a_file(self):
        class VectorStore:
            def load_documents(self, paths, **kwargs):
                self.paths = list(paths)
                return {
                    "new_count": 0,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "removed_count": 0,
                    "files": [],
                }

        async def inline_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        vector_store = VectorStore()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    knowledge,
                    "chroma_conf",
                    {
                        "data_path": tmpdir,
                        "allow_knowledge_file_type": [".txt", ".pdf"],
                    },
                ),
                patch.object(knowledge, "get_abs_path", side_effect=lambda path: path),
                patch.object(knowledge, "get_vector_store", return_value=vector_store),
                patch.object(knowledge, "run_in_threadpool", side_effect=inline_threadpool),
                patch.object(knowledge, "rate_limit", new=AsyncMock()),
                patch.object(knowledge, "redis_lock", new=self.recording_lock),
            ):
                response = await knowledge.upload_knowledge(
                    files=[FakeUpload("payload.exe", b"not allowed")],
                    visibility_mode="roles",
                    allowed_roles='["researcher"]',
                    admin={"id": 7, "role": "admin", "username": "admin"},
                    db=SimpleNamespace(),
                )
            remaining = [path for path in Path(tmpdir).rglob("*") if path.is_file()]

        self.assertFalse(response["success"])
        self.assertEqual(response["file_results"][0]["status"], "failed")
        self.assertIn("unsupported", response["file_results"][0]["error"])
        self.assertEqual(remaining, [])

    async def test_updated_upload_keeps_new_version_and_removes_previous_file(self):
        class VectorStore:
            def __init__(self, data_dir, previous_key):
                self.data_dir = data_dir
                self.previous_key = previous_key

            def load_documents(self, paths, **kwargs):
                storage_key = os.path.relpath(paths[0], self.data_dir).replace(os.sep, "/")
                return {
                    "new_count": 0,
                    "updated_count": 1,
                    "skipped_count": 0,
                    "removed_count": 0,
                    "files": [
                        {
                            "filename": "paper.pdf",
                            "path": paths[0],
                            "status": "updated",
                            "success": True,
                            "storage_key": storage_key,
                            "previous_storage_key": self.previous_key,
                            "error": None,
                        }
                    ],
                }

        async def inline_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = Path(tmpdir) / ".knowledge_versions" / "old" / "paper.pdf"
            old_path.parent.mkdir(parents=True)
            old_path.write_bytes(b"old")
            previous_key = os.path.relpath(old_path, tmpdir).replace(os.sep, "/")
            vector_store = VectorStore(tmpdir, previous_key)
            with (
                patch.object(
                    knowledge,
                    "chroma_conf",
                    {
                        "data_path": tmpdir,
                        "allow_knowledge_file_type": [".txt", ".pdf"],
                    },
                ),
                patch.object(knowledge, "get_abs_path", side_effect=lambda path: path),
                patch.object(knowledge, "get_vector_store", return_value=vector_store),
                patch.object(knowledge, "run_in_threadpool", side_effect=inline_threadpool),
                patch.object(knowledge, "rate_limit", new=AsyncMock()),
                patch.object(knowledge, "redis_lock", new=self.recording_lock),
            ):
                response = await knowledge.upload_knowledge(
                    files=[FakeUpload("paper.pdf", b"new")],
                    visibility_mode="roles",
                    allowed_roles='["researcher"]',
                    admin={"id": 7, "role": "admin", "username": "admin"},
                    db=SimpleNamespace(),
                )

            version_files = [path for path in Path(tmpdir).rglob("paper.pdf")]
            version_contents = [path.read_bytes() for path in version_files]

        self.assertTrue(response["success"])
        self.assertEqual(len(version_files), 1)
        self.assertEqual(version_contents, [b"new"])

    def test_stream_copy_failure_removes_partial_version_file(self):
        class ExplodingStream(io.BytesIO):
            def read(self, size=-1):
                data = super().read(size)
                if data:
                    raise OSError("copy failed")
                return data

        upload = SimpleNamespace(
            filename="paper.pdf",
            file=ExplodingStream(b"partial"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(OSError, "copy failed"):
                knowledge._copy_upload_to_version(upload, tmpdir)
            remaining = [path for path in Path(tmpdir).rglob("*") if path.is_file()]

        self.assertEqual(remaining, [])

    async def test_permission_update_uses_global_lock_without_rebuilding_indexes(self):
        updated = make_doc(allowed_roles=["business"])
        update_roles = AsyncMock(return_value=updated)
        db = SimpleNamespace(commit=AsyncMock(), refresh=AsyncMock())
        with (
            patch.object(knowledge, "update_allowed_roles", update_roles),
            patch.object(knowledge, "redis_lock", new=self.recording_lock),
            patch.object(
                knowledge,
                "get_vector_store",
                side_effect=AssertionError("ACL changes must not touch indexes"),
            ),
        ):
            response = await knowledge.update_knowledge_file_permissions(
                doc_id="doc-1",
                payload=SimpleNamespace(
                    visibility_mode="roles",
                    allowed_roles=["business"],
                ),
                admin={"id": 7, "role": "admin", "username": "admin"},
                db=db,
            )

        self.assertEqual(response["allowed_roles"], ["business"])
        self.assertEqual(self.lock_keys, [(knowledge.KNOWLEDGE_WRITE_LOCK_KEY, 60)])
        db.commit.assert_awaited_once_with()
        db.refresh.assert_awaited_once_with(updated)

    async def test_download_returns_same_404_for_missing_and_forbidden_documents(self):
        get_doc = AsyncMock(side_effect=[None, make_doc(allowed_roles=["researcher"])])
        with (
            patch.object(knowledge, "get_document_acl", get_doc, create=True),
            patch.object(
                knowledge,
                "get_vector_store",
                side_effect=AssertionError("download metadata must come from MySQL"),
            ),
        ):
            errors = []
            for role in ("researcher", "business"):
                with self.assertRaises(HTTPException) as raised:
                    await knowledge.download_knowledge_file(
                        "doc-1",
                        user={"id": 8, "role": role, "username": "reader"},
                        db=SimpleNamespace(),
                    )
                errors.append((raised.exception.status_code, raised.exception.detail))

        self.assertEqual(errors, [(404, "Document not found"), (404, "Document not found")])

    async def test_delete_uses_database_metadata_and_global_lock(self):
        deleted = []

        class VectorStore:
            def delete_document_by_doc_id(self, doc_id, delete_file=False):
                deleted.append((doc_id, delete_file))
                return 3

        async def inline_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch.object(
                knowledge,
                "get_document_acl",
                new=AsyncMock(return_value=make_doc()),
                create=True,
            ),
            patch.object(knowledge, "get_vector_store", return_value=VectorStore()),
            patch.object(knowledge, "run_in_threadpool", side_effect=inline_threadpool),
            patch.object(knowledge, "redis_lock", new=self.recording_lock),
            patch.object(
                knowledge,
                "delete_document_acl",
                new=AsyncMock(side_effect=AssertionError("runtime owns document deletion")),
                create=True,
            ),
        ):
            response = await knowledge.delete_knowledge_file(
                "doc-1",
                delete_file=True,
                _admin={"id": 7, "role": "admin", "username": "admin"},
                db=SimpleNamespace(),
            )

        self.assertEqual(deleted, [("doc-1", True)])
        self.assertEqual(response["filename"], "paper.pdf")
        self.assertEqual(self.lock_keys, [(knowledge.KNOWLEDGE_WRITE_LOCK_KEY, 600)])


if __name__ == "__main__":
    unittest.main()
