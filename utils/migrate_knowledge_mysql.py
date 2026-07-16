import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from sqlalchemy import func, inspect, or_, select

from models import Base
from models.knowledge import KnowledgeDocument, ParentChunk


class MigrationError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


class ChromaDocumentCountProvider:
    """Read per-document Chroma counts without opening Chroma in write mode."""

    _COUNT_SQL = """
        SELECT m.string_value, COUNT(*)
        FROM embedding_metadata m
        JOIN embeddings e ON e.id=m.id
        JOIN segments s ON s.id=e.segment_id
        JOIN collections c ON c.id=s.collection
        WHERE m.key=? AND c.name=?
        GROUP BY m.string_value
    """

    def __init__(self, chroma_path: str | Path, collection_name: str):
        path = Path(chroma_path)
        self.database_path = (
            path
            if path.name == "chroma.sqlite3" or path.suffix in {".sqlite3", ".db"}
            else path / "chroma.sqlite3"
        )
        if not self.database_path.is_file():
            raise MigrationError(
                "chroma_database_missing",
                "Chroma database does not exist",
                {"path": str(self.database_path)},
            )
        self.counts = self._read_counts(collection_name)

    def _read_counts(self, collection_name: str) -> dict[str, int]:
        uri = f"{self.database_path.resolve().as_uri()}?mode=ro"
        connection = None
        try:
            connection = sqlite3.connect(uri, uri=True)
            rows = connection.execute(
                self._COUNT_SQL,
                ("doc_id", collection_name),
            ).fetchall()
            return {
                str(doc_id): int(count)
                for doc_id, count in rows
                if doc_id is not None
            }
        except sqlite3.Error as exc:
            raise MigrationError(
                "chroma_query_failed",
                "Could not read Chroma document counts",
                {"path": str(self.database_path), "reason": str(exc)},
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def __call__(self, doc_id: str) -> int:
        return self.counts.get(doc_id, 0)


def _create_chroma_count_provider(
    chroma_path: str | Path | None = None,
) -> ChromaDocumentCountProvider:
    from utils.config_handler import chroma_conf

    persist_directory = chroma_path or chroma_conf["persist_directory"]
    collection_name = chroma_conf["collection_name"]
    return ChromaDocumentCountProvider(persist_directory, collection_name)


@dataclass(frozen=True)
class MigrationResult:
    documents_created: int
    documents_updated: int
    parents_upserted: int


@dataclass(frozen=True)
class CheckReport:
    missing_docs: list[str]
    extra_docs: list[str]
    parent_count_mismatches: dict[str, dict[str, int | None]]
    orphan_parent_ids: list[str]
    chroma_chunk_mismatches: dict[str, dict[str, int]]
    duplicate_filenames: list[str]
    duplicate_hashes: list[str]
    duplicate_doc_ids: list[str] = field(default_factory=list)
    schema_errors: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.missing_docs,
                self.extra_docs,
                self.parent_count_mismatches,
                self.orphan_parent_ids,
                self.chroma_chunk_mismatches,
                self.duplicate_filenames,
                self.duplicate_hashes,
                self.duplicate_doc_ids,
                self.schema_errors,
            )
        )


def _load_json_object(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=None)


def _parent_doc_id(parent_id: str, record: dict) -> str:
    metadata = record.get("metadata") or {}
    doc_id = metadata.get("doc_id")
    if doc_id:
        return str(doc_id)
    marker = ":parent:"
    if marker in parent_id:
        return parent_id.split(marker, 1)[0]
    raise ValueError(f"Parent chunk {parent_id} is missing metadata.doc_id")


def _parent_index(parent_id: str, record: dict) -> int:
    metadata = record.get("metadata") or {}
    if metadata.get("parent_index") is not None:
        return int(metadata["parent_index"])
    try:
        return int(parent_id.rsplit(":", 1)[-1])
    except ValueError as exc:
        raise ValueError(
            f"Parent chunk {parent_id} is missing metadata.parent_index"
        ) from exc


def _find_document(session, doc_id: str, filename: str) -> KnowledgeDocument | None:
    matches = list(
        session.scalars(
            select(KnowledgeDocument).where(
                or_(
                    KnowledgeDocument.doc_id == doc_id,
                    KnowledgeDocument.filename == filename,
                )
            )
        ).all()
    )
    if len(matches) > 1:
        raise ValueError(
            f"Manifest entry conflicts with separate doc_id and filename rows: "
            f"{doc_id}, {filename}"
        )
    return matches[0] if matches else None


def _existing_document_duplicates(session_factory) -> dict[str, list[str]]:
    duplicates = {
        "duplicate_doc_ids": [],
        "duplicate_filenames": [],
        "duplicate_hashes": [],
    }
    with session_factory() as session:
        schema_inspector = inspect(session.get_bind())
        if "knowledge_documents" not in schema_inspector.get_table_names():
            return duplicates
        columns = {
            column["name"]
            for column in schema_inspector.get_columns("knowledge_documents")
        }
        if "doc_id" in columns:
            duplicates["duplicate_doc_ids"] = _duplicate_values(
                session, KnowledgeDocument.doc_id
            )
        if "filename" in columns:
            duplicates["duplicate_filenames"] = _duplicate_values(
                session, KnowledgeDocument.filename
            )
        if "file_hash" in columns:
            duplicates["duplicate_hashes"] = _duplicate_values(
                session, KnowledgeDocument.file_hash
            )
    return duplicates


def _raise_for_document_duplicates(session_factory) -> None:
    duplicates = _existing_document_duplicates(session_factory)
    if any(duplicates.values()):
        raise MigrationError(
            "duplicate_documents",
            "Existing knowledge documents contain duplicate ids, filenames, or hashes",
            duplicates,
        )


def migrate_json_to_mysql(
    session_factory,
    manifest_path: str | Path,
    parent_path: str | Path,
) -> MigrationResult:
    manifest = _load_json_object(manifest_path)
    parents = _load_json_object(parent_path)
    _raise_for_document_duplicates(session_factory)
    documents_created = 0
    documents_updated = 0

    with session_factory() as session:
        try:
            for filename, raw_entry in manifest.items():
                entry = dict(raw_entry or {})
                doc_id = entry.get("doc_id")
                if not doc_id:
                    raise ValueError(f"Manifest entry {filename} is missing doc_id")

                document = _find_document(session, str(doc_id), str(filename))
                if document is None:
                    document = KnowledgeDocument(
                        doc_id=str(doc_id),
                        filename=str(filename),
                        allowed_roles=[],
                    )
                    session.add(document)
                    documents_created += 1
                else:
                    documents_updated += 1

                document.doc_id = str(doc_id)
                document.filename = str(filename)
                document.file_hash = entry.get("file_hash")
                document.file_type = entry.get("file_type")
                document.storage_key = entry.get("storage_key") or str(filename)
                document.chunk_method = entry.get("chunk_method")
                document.chunk_ids = list(entry.get("chunk_ids") or [])
                document.chunk_count = int(entry.get("chunk_count") or 0)
                document.parent_count = entry.get("parent_count")
                document.child_count = entry.get("child_count")
                document.ingested_at = _parse_datetime(entry.get("ingested_at"))
                document.status = entry.get("status") or "active"
                document.error_message = entry.get("error_message")
                document.updated_at = datetime.now()

            now = datetime.now()
            for parent_id, raw_record in parents.items():
                record = dict(raw_record or {})
                metadata = dict(record.get("metadata") or {})
                chunk = session.get(ParentChunk, str(parent_id))
                if chunk is None:
                    chunk = ParentChunk(
                        parent_id=str(parent_id),
                        created_at=now,
                    )
                    session.add(chunk)
                chunk.doc_id = _parent_doc_id(str(parent_id), record)
                chunk.parent_index = _parent_index(str(parent_id), record)
                chunk.page_content = str(record.get("page_content") or "")
                chunk.metadata_json = metadata
                chunk.updated_at = now

            session.commit()
        except Exception:
            session.rollback()
            raise

    return MigrationResult(
        documents_created=documents_created,
        documents_updated=documents_updated,
        parents_upserted=len(parents),
    )


def _duplicate_values(session, column) -> list[str]:
    rows = session.scalars(
        select(column)
        .where(column.is_not(None))
        .group_by(column)
        .having(func.count() > 1)
        .order_by(column)
    ).all()
    return [str(value) for value in rows]


def _manifest_duplicate_hashes(manifest: dict) -> list[str]:
    counts: dict[str, int] = {}
    for entry in manifest.values():
        file_hash = (entry or {}).get("file_hash")
        if file_hash:
            counts[str(file_hash)] = counts.get(str(file_hash), 0) + 1
    return sorted(value for value, count in counts.items() if count > 1)


def _chroma_counts(provider, doc_ids: list[str]) -> dict[str, int]:
    if provider is None:
        return {}
    if isinstance(provider, Mapping):
        return {doc_id: int(provider.get(doc_id, 0)) for doc_id in doc_ids}
    if hasattr(provider, "count_for_document"):
        return {
            doc_id: int(provider.count_for_document(doc_id)) for doc_id in doc_ids
        }
    if not isinstance(provider, Callable):
        raise TypeError("chroma_count_provider must be callable or a mapping")

    try:
        result = provider()
    except TypeError:
        return {doc_id: int(provider(doc_id)) for doc_id in doc_ids}
    if not isinstance(result, Mapping):
        raise TypeError("Zero-argument Chroma count provider must return a mapping")
    return {doc_id: int(result.get(doc_id, 0)) for doc_id in doc_ids}


def _schema_errors(bind) -> list[dict]:
    schema_inspector = inspect(bind)
    table_names = set(schema_inspector.get_table_names())
    errors: list[dict] = []
    required_tables = {
        KnowledgeDocument.__tablename__: KnowledgeDocument.__table__,
        ParentChunk.__tablename__: ParentChunk.__table__,
    }
    for table_name, table in required_tables.items():
        if table_name not in table_names:
            errors.append({"type": "missing_table", "table": table_name})
            continue
        existing_columns = {
            column["name"] for column in schema_inspector.get_columns(table_name)
        }
        for column in table.columns:
            if column.name not in existing_columns:
                errors.append(
                    {
                        "type": "missing_column",
                        "table": table_name,
                        "column": column.name,
                    }
                )

    for table_name, expected_indexes in _REQUIRED_INDEXES.items():
        if table_name not in table_names:
            continue
        existing_indexes = {
            index["name"]: index
            for index in schema_inspector.get_indexes(table_name)
        }
        for index_name, expected in expected_indexes.items():
            existing = existing_indexes.get(index_name)
            if existing is None:
                errors.append(
                    {
                        "type": (
                            "missing_unique_index"
                            if expected["unique"]
                            else "missing_index"
                        ),
                        "table": table_name,
                        "index": index_name,
                        "column_names": expected["column_names"],
                    }
                )
            elif (
                bool(existing.get("unique")) != expected["unique"]
                or existing.get("column_names") != expected["column_names"]
            ):
                errors.append(
                    {
                        "type": (
                            "invalid_unique_index"
                            if expected["unique"]
                            else "invalid_index"
                        ),
                        "table": table_name,
                        "index": index_name,
                        "expected": expected,
                        "actual": {
                            "unique": bool(existing.get("unique")),
                            "column_names": existing.get("column_names") or [],
                        },
                    }
                )
    return errors


def _empty_check_report(schema_errors: list[dict]) -> CheckReport:
    return CheckReport(
        missing_docs=[],
        extra_docs=[],
        parent_count_mismatches={},
        orphan_parent_ids=[],
        chroma_chunk_mismatches={},
        duplicate_filenames=[],
        duplicate_hashes=[],
        duplicate_doc_ids=[],
        schema_errors=schema_errors,
    )


def check_consistency(
    session_factory,
    manifest_path: str | Path,
    parent_path: str | Path,
    chroma_count_provider=None,
) -> CheckReport:
    manifest = _load_json_object(manifest_path)
    parents = _load_json_object(parent_path)
    expected_entries = {
        str(entry.get("doc_id")): dict(entry or {})
        for entry in manifest.values()
        if (entry or {}).get("doc_id")
    }
    expected_doc_ids = set(expected_entries)

    with session_factory() as session:
        schema_errors = _schema_errors(session.get_bind())
        if any(
            error["type"] in {"missing_table", "missing_column"}
            for error in schema_errors
        ):
            return _empty_check_report(schema_errors)
        documents = list(session.scalars(select(KnowledgeDocument)).all())
        parent_rows = list(session.scalars(select(ParentChunk)).all())
        duplicate_doc_ids = _duplicate_values(session, KnowledgeDocument.doc_id)
        duplicate_filenames = _duplicate_values(session, KnowledgeDocument.filename)
        duplicate_hashes = _duplicate_values(session, KnowledgeDocument.file_hash)

    db_documents = {document.doc_id: document for document in documents}
    db_doc_ids = set(db_documents)
    missing_docs = sorted(expected_doc_ids - db_doc_ids)
    extra_docs = sorted(db_doc_ids - expected_doc_ids)

    db_parent_counts: dict[str, int] = {}
    for parent in parent_rows:
        db_parent_counts[parent.doc_id] = db_parent_counts.get(parent.doc_id, 0) + 1

    json_parent_counts: dict[str, int] = {}
    json_orphans: list[str] = []
    for parent_id, raw_record in parents.items():
        doc_id = _parent_doc_id(str(parent_id), dict(raw_record or {}))
        json_parent_counts[doc_id] = json_parent_counts.get(doc_id, 0) + 1
        if doc_id not in db_doc_ids:
            json_orphans.append(str(parent_id))

    parent_count_mismatches: dict[str, dict[str, int | None]] = {}
    for doc_id, entry in expected_entries.items():
        expected = entry.get("parent_count")
        if expected is None and "parent_ids" in entry:
            expected = len(entry.get("parent_ids") or [])
        if expected is None:
            continue
        expected = int(expected)
        mysql_count = db_parent_counts.get(doc_id, 0)
        json_count = json_parent_counts.get(doc_id, 0)
        document_count = (
            db_documents[doc_id].parent_count if doc_id in db_documents else None
        )
        if (
            mysql_count != expected
            or json_count != expected
            or document_count != expected
        ):
            parent_count_mismatches[doc_id] = {
                "manifest": expected,
                "mysql": mysql_count,
                "parent_json": json_count,
                "document": document_count,
            }

    db_orphans = [
        parent.parent_id for parent in parent_rows if parent.doc_id not in db_doc_ids
    ]
    orphan_parent_ids = sorted(set(db_orphans + json_orphans))

    chroma_chunk_mismatches: dict[str, dict[str, int]] = {}
    chroma_counts = _chroma_counts(
        chroma_count_provider, sorted(expected_doc_ids)
    )
    for doc_id, actual in chroma_counts.items():
        expected = int(expected_entries[doc_id].get("chunk_count") or 0)
        if actual != expected:
            chroma_chunk_mismatches[doc_id] = {
                "manifest": expected,
                "chroma": actual,
            }

    duplicate_hashes = sorted(
        set(duplicate_hashes) | set(_manifest_duplicate_hashes(manifest))
    )
    return CheckReport(
        missing_docs=missing_docs,
        extra_docs=extra_docs,
        parent_count_mismatches=parent_count_mismatches,
        orphan_parent_ids=orphan_parent_ids,
        chroma_chunk_mismatches=chroma_chunk_mismatches,
        duplicate_filenames=duplicate_filenames,
        duplicate_hashes=duplicate_hashes,
        duplicate_doc_ids=duplicate_doc_ids,
        schema_errors=schema_errors,
    )


_MYSQL_MISSING_COLUMNS = {
    "knowledge_documents": {
        "file_hash": "VARCHAR(128) NULL",
        "file_type": "VARCHAR(32) NULL",
        "storage_key": "VARCHAR(512) NULL",
        "chunk_method": "VARCHAR(64) NULL",
        "chunk_ids": "JSON NULL",
        "chunk_count": "INTEGER NOT NULL DEFAULT 0",
        "parent_count": "INTEGER NULL",
        "child_count": "INTEGER NULL",
        "allowed_roles": "JSON NULL",
        "status": "VARCHAR(32) NOT NULL DEFAULT 'active'",
        "ingested_at": "DATETIME NULL",
        "error_message": "TEXT NULL",
        "updated_by": "INTEGER NULL",
        "created_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "updated_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
    },
    "parent_chunks": {
        "doc_id": "VARCHAR(64) NULL",
        "parent_index": "INTEGER NULL",
        "page_content": "MEDIUMTEXT NULL",
        "metadata": "JSON NULL",
        "created_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "updated_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
    },
}

_REQUIRED_INDEXES = {
    table.name: {
        index.name: {
            "unique": bool(index.unique),
            "column_names": [column.name for column in index.columns],
        }
        for index in sorted(table.indexes, key=lambda item: item.name)
    }
    for table in (KnowledgeDocument.__table__, ParentChunk.__table__)
}

# Kept as a compatibility view for callers that only need unique index columns.
_MYSQL_UNIQUE_INDEXES = {
    table_name: {
        index_name: definition["column_names"][0]
        for index_name, definition in indexes.items()
        if definition["unique"] and len(definition["column_names"]) == 1
    }
    for table_name, indexes in _REQUIRED_INDEXES.items()
}


def _engine_document_duplicates(
    sync_engine,
    schema_inspector,
    table_names: set[str],
) -> dict[str, list[str]]:
    duplicates = {
        "duplicate_doc_ids": [],
        "duplicate_filenames": [],
        "duplicate_hashes": [],
    }
    if "knowledge_documents" not in table_names:
        return duplicates
    existing_columns = {
        column["name"]
        for column in schema_inspector.get_columns("knowledge_documents")
    }
    targets = (
        ("doc_id", "duplicate_doc_ids", KnowledgeDocument.doc_id),
        ("filename", "duplicate_filenames", KnowledgeDocument.filename),
        ("file_hash", "duplicate_hashes", KnowledgeDocument.file_hash),
    )
    with sync_engine.connect() as connection:
        for column_name, result_key, column in targets:
            if column_name not in existing_columns:
                continue
            values = connection.execute(
                select(column)
                .where(column.is_not(None))
                .group_by(column)
                .having(func.count() > 1)
                .order_by(column)
            ).scalars()
            duplicates[result_key] = [str(value) for value in values]
    return duplicates


def _preflight_schema(sync_engine):
    schema_inspector = inspect(sync_engine)
    table_names = set(schema_inspector.get_table_names())
    columns_by_table = {
        table_name: {
            column["name"]
            for column in schema_inspector.get_columns(table_name)
        }
        for table_name in table_names
        if table_name in _MYSQL_MISSING_COLUMNS
    }
    required_existing_columns = {
        "knowledge_documents": {"id", "doc_id", "filename"},
        "parent_chunks": {"parent_id"},
    }
    for table_name, required_columns in required_existing_columns.items():
        if table_name not in table_names:
            continue
        missing_columns = sorted(
            required_columns - columns_by_table.get(table_name, set())
        )
        if missing_columns:
            raise MigrationError(
                "incomplete_existing_schema",
                f"Existing table {table_name} is missing required columns",
                {
                    "table": table_name,
                    "missing_columns": missing_columns,
                },
            )

    duplicates = _engine_document_duplicates(
        sync_engine, schema_inspector, table_names
    )
    if any(duplicates.values()):
        raise MigrationError(
            "duplicate_documents",
            "Existing knowledge documents contain duplicate ids, filenames, or hashes",
            duplicates,
        )

    indexes_by_table: dict[str, dict[str, dict]] = {}
    for table_name, expected_indexes in _REQUIRED_INDEXES.items():
        if table_name not in table_names:
            continue
        existing_indexes = {
            index["name"]: index
            for index in schema_inspector.get_indexes(table_name)
        }
        indexes_by_table[table_name] = existing_indexes
        for index_name, expected in expected_indexes.items():
            existing = existing_indexes.get(index_name)
            if existing is not None and (
                bool(existing.get("unique")) != expected["unique"]
                or existing.get("column_names") != expected["column_names"]
            ):
                raise MigrationError(
                    (
                        "invalid_unique_index"
                        if expected["unique"]
                        else "invalid_index"
                    ),
                    f"Index {index_name} exists with the wrong definition",
                    {
                        "table": table_name,
                        "index": index_name,
                        "expected": expected,
                        "actual": {
                            "unique": bool(existing.get("unique")),
                            "column_names": existing.get("column_names") or [],
                        },
                    },
                )
    return table_names, columns_by_table, indexes_by_table


def _schema_ddl_error(statement: str, exc: Exception) -> MigrationError:
    return MigrationError(
        "schema_ddl_failed",
        "MySQL applies DDL one statement at a time; completed idempotent steps "
        "are safe to rerun",
        {
            "statement": statement,
            "reason": str(exc),
            "atomicity": "statement",
            "retry_safe": True,
        },
    )


def _execute_schema_ddl(connection, statement: str) -> None:
    try:
        connection.exec_driver_sql(statement)
    except Exception as exc:
        raise _schema_ddl_error(statement, exc) from exc


def ensure_mysql_schema(sync_engine) -> None:
    """Bring the knowledge schema forward after a read-only full preflight.

    MySQL implicitly commits DDL per statement, so this cannot be one rollbackable
    transaction. Each ALTER/CREATE is guarded by inspection and is safe to rerun.
    """
    table_names, columns_by_table, indexes_by_table = _preflight_schema(sync_engine)
    try:
        Base.metadata.create_all(sync_engine)
    except Exception as exc:
        raise _schema_ddl_error("Base.metadata.create_all", exc) from exc

    with sync_engine.begin() as connection:
        if sync_engine.dialect.name == "mysql":
            for table_name, definitions in _MYSQL_MISSING_COLUMNS.items():
                if table_name not in table_names:
                    continue
                existing_columns = columns_by_table[table_name]
                for column_name, definition in definitions.items():
                    if column_name not in existing_columns:
                        _execute_schema_ddl(
                            connection,
                            f"ALTER TABLE `{table_name}` "
                            f"ADD COLUMN `{column_name}` {definition}",
                        )

        for table_name, indexes in _REQUIRED_INDEXES.items():
            if table_name not in table_names:
                continue
            existing_indexes = indexes_by_table.get(table_name, {})
            for index_name, definition in indexes.items():
                existing = existing_indexes.get(index_name)
                if existing is None:
                    unique_sql = "UNIQUE " if definition["unique"] else ""
                    columns_sql = ", ".join(
                        f"`{column_name}`"
                        for column_name in definition["column_names"]
                    )
                    _execute_schema_ddl(
                        connection,
                        f"CREATE {unique_sql}INDEX `{index_name}` "
                        f"ON `{table_name}` ({columns_sql})",
                    )


def _report_payload(report: CheckReport) -> dict:
    return {"ok": report.ok, **asdict(report)}


def main(
    argv: list[str] | None = None,
    *,
    session_factory=None,
    sync_engine=None,
    chroma_count_provider=None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate knowledge manifest and parent chunks into MySQL"
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--parent-docstore", default="parent_docstore.json")
    parser.add_argument("--chroma-path")
    args = parser.parse_args(argv)

    try:
        if session_factory is None:
            from config.db_conf import SyncSessionLocal

            session_factory = SyncSessionLocal

        if args.check:
            provider = chroma_count_provider or _create_chroma_count_provider(
                args.chroma_path
            )
            report = check_consistency(
                session_factory,
                args.manifest,
                args.parent_docstore,
                chroma_count_provider=provider,
            )
            print(json.dumps(_report_payload(report), ensure_ascii=False, indent=2))
            return 0 if report.ok else 1

        _raise_for_document_duplicates(session_factory)
        if sync_engine is None:
            from config.db_conf import sync_engine as configured_sync_engine

            sync_engine = configured_sync_engine
        ensure_mysql_schema(sync_engine)
        result = migrate_json_to_mysql(
            session_factory,
            args.manifest,
            args.parent_docstore,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0
    except MigrationError as exc:
        print(
            json.dumps(
                {"ok": False, "error": exc.as_dict()},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "migration_failed",
                        "message": str(exc),
                        "details": {},
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
