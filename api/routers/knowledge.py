import json
import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user, require_admin
from api.dependencies import get_vector_store
from config.db_conf import get_db
from crud.knowledge_acl import (
    delete_document_acl,
    list_visible_documents,
    update_allowed_roles,
    upsert_document_acl,
)
from schemas.knowledge import UpdateDocumentPermissionsRequest
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.rbac import is_admin, validate_allowed_roles

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"])


def _parse_allowed_roles(visibility_mode: str, raw_roles) -> list[str]:
    if visibility_mode == "admin_only":
        return []
    if visibility_mode != "roles":
        raise HTTPException(status_code=400, detail="invalid visibility_mode")

    if isinstance(raw_roles, str):
        try:
            roles = json.loads(raw_roles or "[]")
        except json.JSONDecodeError:
            roles = [role.strip() for role in raw_roles.split(",") if role.strip()]
    else:
        roles = raw_roles or []

    try:
        roles = validate_allowed_roles(roles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not roles:
        raise HTTPException(status_code=400, detail="allowed_roles required when visibility_mode=roles")
    return roles


def _manifest_by_doc_id(manifest: dict) -> dict:
    return {
        entry.get("doc_id"): (filename, entry)
        for filename, entry in manifest.items()
        if entry.get("doc_id")
    }


def _document_payload(filename: str, entry: dict, allowed_roles: list[str], can_manage: bool) -> dict:
    payload = {
        "name": filename,
        "doc_id": entry.get("doc_id"),
        "file_type": entry.get("file_type"),
        "chunk_count": entry.get("chunk_count"),
        "parent_count": entry.get("parent_count"),
        "child_count": entry.get("child_count"),
        "chunk_method": entry.get("chunk_method"),
        "status": entry.get("status"),
        "ingested_at": entry.get("ingested_at"),
        "file_hash": entry.get("file_hash"),
        "can_manage": can_manage,
    }
    if can_manage:
        payload["allowed_roles"] = allowed_roles
    return payload


@router.post("/upload")
async def upload_knowledge(
    files: list[UploadFile] = File(...),
    visibility_mode: str = Form("admin_only"),
    allowed_roles: str = Form("[]"),
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    roles = _parse_allowed_roles(visibility_mode, allowed_roles)
    data_dir = get_abs_path(chroma_conf["data_path"])
    os.makedirs(data_dir, exist_ok=True)

    uploaded_files = []
    uploaded_paths = []
    for file in files:
        if not file.filename:
            continue
        filename = os.path.basename(file.filename)
        file_path = os.path.join(data_dir, filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        uploaded_files.append(filename)
        uploaded_paths.append(file_path)

    vector_store = get_vector_store()
    new_count, updated_count, skipped_count, removed_count = vector_store.load_documents(uploaded_paths)

    for filename in uploaded_files:
        entry = vector_store.manifest.get(filename)
        if not entry or not entry.get("doc_id"):
            continue
        await upsert_document_acl(
            db,
            doc_id=entry.get("doc_id"),
            filename=filename,
            file_hash=entry.get("file_hash"),
            file_type=entry.get("file_type"),
            chunk_count=entry.get("chunk_count") or 0,
            parent_count=entry.get("parent_count"),
            child_count=entry.get("child_count"),
            allowed_roles=roles,
            status=entry.get("status") or "active",
            updated_by=admin.get("id"),
        )

    return {
        "success": True,
        "uploaded_files": uploaded_files,
        "new_count": new_count,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "removed_count": removed_count,
        "message": f"uploaded {len(uploaded_files)} files, new {new_count}, updated {updated_count}, skipped {skipped_count}",
    }


@router.get("/files")
async def list_knowledge_files(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    vector_store = get_vector_store()
    manifest = vector_store.manifest
    can_manage = is_admin(user)
    visible_docs = await list_visible_documents(db, user.get("role", "guest"))
    acl_by_doc_id = {doc.doc_id: doc for doc in visible_docs}
    visible_doc_ids = None if can_manage else set(acl_by_doc_id)

    files = []
    for filename, entry in manifest.items():
        doc_id = entry.get("doc_id")
        if not doc_id:
            continue
        if visible_doc_ids is not None and doc_id not in visible_doc_ids:
            continue
        acl = acl_by_doc_id.get(doc_id)
        files.append(_document_payload(filename, entry, acl.allowed_roles if acl else [], can_manage))

    return {
        "success": True,
        "files": files,
        "total_files": len(files),
        "total_chunks": sum(file.get("chunk_count") or 0 for file in files),
    }


@router.patch("/files/{doc_id}/permissions")
async def update_knowledge_file_permissions(
    doc_id: str,
    payload: UpdateDocumentPermissionsRequest,
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    roles = _parse_allowed_roles(payload.visibility_mode, payload.allowed_roles)
    doc = await update_allowed_roles(db, doc_id, roles, updated_by=admin.get("id"))
    if doc is None:
        vector_store = get_vector_store()
        target = _manifest_by_doc_id(vector_store.manifest).get(doc_id)
        if target is None:
            raise HTTPException(status_code=404, detail="Document not found")
        filename, entry = target
        doc = await upsert_document_acl(
            db,
            doc_id=doc_id,
            filename=filename,
            file_hash=entry.get("file_hash"),
            file_type=entry.get("file_type"),
            chunk_count=entry.get("chunk_count") or 0,
            parent_count=entry.get("parent_count"),
            child_count=entry.get("child_count"),
            allowed_roles=roles,
            status=entry.get("status") or "active",
            updated_by=admin.get("id"),
        )
    return {"success": True, "doc_id": doc.doc_id, "allowed_roles": doc.allowed_roles or []}


@router.get("/files/{doc_id}/download")
async def download_knowledge_file(
    doc_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    vector_store = get_vector_store()
    target = _manifest_by_doc_id(vector_store.manifest).get(doc_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if not is_admin(user):
        visible_docs = await list_visible_documents(db, user.get("role", "guest"))
        if doc_id not in {doc.doc_id for doc in visible_docs}:
            raise HTTPException(status_code=404, detail="Document not found")

    filename, _entry = target
    data_dir = os.path.abspath(get_abs_path(chroma_conf["data_path"]))
    file_path = os.path.abspath(os.path.join(data_dir, filename))
    if not file_path.startswith(data_dir + os.sep) or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(file_path, filename=filename)


@router.delete("/files/{doc_id}")
async def delete_knowledge_file(
    doc_id: str,
    delete_file: bool = True,
    _admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    vector_store = get_vector_store()
    target = _manifest_by_doc_id(vector_store.manifest).get(doc_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    filename, _entry = target
    deleted_chunks = vector_store.delete_document_by_doc_id(doc_id, delete_file=delete_file)
    await delete_document_acl(db, doc_id)

    return {
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "deleted_chunks": deleted_chunks,
        "deleted_file": delete_file,
        "message": f"deleted {filename}, removed {deleted_chunks} chunks",
    }
