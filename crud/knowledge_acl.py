from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.knowledge import KnowledgeDocument
from utils.rbac import ROLE_ADMIN, validate_allowed_roles


async def get_document_acl(db: AsyncSession, doc_id: str) -> KnowledgeDocument | None:
    result = await db.execute(select(KnowledgeDocument).where(KnowledgeDocument.doc_id == doc_id))
    return result.scalar_one_or_none()


async def list_visible_documents(db: AsyncSession, role: str) -> list[KnowledgeDocument]:
    result = await db.execute(
        select(KnowledgeDocument)
        .where(KnowledgeDocument.status == "active")
        .order_by(KnowledgeDocument.updated_at.desc())
    )
    docs = list(result.scalars().all())
    if role == ROLE_ADMIN:
        return docs
    return [doc for doc in docs if role in (doc.allowed_roles or [])]


async def get_allowed_doc_ids(db: AsyncSession, role: str) -> set[str] | None:
    if role == ROLE_ADMIN:
        return None
    docs = await list_visible_documents(db, role)
    return {doc.doc_id for doc in docs}


async def update_allowed_roles(
    db: AsyncSession,
    doc_id: str,
    allowed_roles: list[str],
    updated_by: int | None = None,
) -> KnowledgeDocument | None:
    doc = await get_document_acl(db, doc_id)
    if doc is None:
        return None
    doc.allowed_roles = validate_allowed_roles(allowed_roles)
    doc.updated_by = updated_by
    doc.updated_at = datetime.now()
    await db.flush()
    await db.refresh(doc)
    return doc
