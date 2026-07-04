import argparse
import asyncio

from sqlalchemy import select

from config.db_conf import AsyncSessionLocal, async_engine
from crud.knowledge_acl import upsert_document_acl
from crud.users import ensure_user_role_column, get_user_by_username, update_user_role
from models import Base
from models import knowledge as _knowledge_mod  # noqa: F401
from models import users as _users_mod  # noqa: F401
from models.knowledge import KnowledgeDocument
from utils.config_handler import chroma_conf
from utils.file_handler import load_manifest
from utils.path_tool import get_abs_path
from utils.rbac import ROLE_ADMIN


def _load_manifest() -> dict:
    return load_manifest(get_abs_path(chroma_conf["manifest_store"]))


async def _ensure_tables() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def migrate() -> None:
    manifest = _load_manifest()
    await _ensure_tables()
    async with AsyncSessionLocal() as db:
        await ensure_user_role_column(db)
        admin = await get_user_by_username(db, "admin")
        if admin and admin.role != ROLE_ADMIN:
            await update_user_role(db, admin, ROLE_ADMIN)

        for filename, entry in manifest.items():
            await upsert_document_acl(
                db,
                doc_id=entry.get("doc_id"),
                filename=filename,
                file_hash=entry.get("file_hash"),
                file_type=entry.get("file_type"),
                chunk_count=entry.get("chunk_count") or 0,
                parent_count=entry.get("parent_count"),
                child_count=entry.get("child_count"),
                allowed_roles=[],
                status=entry.get("status") or "active",
            )
        await db.commit()
    print(f"[migrate_rag_acl] migrated {len(manifest)} manifest documents")


async def check() -> int:
    manifest = _load_manifest()
    await _ensure_tables()
    async with AsyncSessionLocal() as db:
        await ensure_user_role_column(db)
        result = await db.execute(select(KnowledgeDocument.doc_id))
        existing = {row[0] for row in result.all()}
        expected = {entry.get("doc_id") for entry in manifest.values() if entry.get("doc_id")}
        missing = sorted(expected - existing)
        extra = sorted(existing - expected)
        admin = await get_user_by_username(db, "admin")

    print(f"[migrate_rag_acl] manifest={len(expected)} acl={len(existing)} missing={len(missing)} extra={len(extra)}")
    if missing:
        print("[migrate_rag_acl] missing:", ", ".join(missing))
    if extra:
        print("[migrate_rag_acl] extra:", ", ".join(extra))
    if not admin or admin.role != ROLE_ADMIN:
        print("[migrate_rag_acl] admin role is not ready")
        return 1
    return 0 if not missing else 1


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return await check()
    await migrate()
    return await check()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
