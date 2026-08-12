import json
import logging
import os
import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user, require_admin
from api.dependencies import get_vector_store
from config.db_conf import get_db
from crud.knowledge_acl import (
    get_document_acl,
    list_visible_documents,
    update_allowed_roles,
)
from repositories.parent_chunk_repository import ParentChunkRepository
from schemas.knowledge import (
    KnowledgeDocumentResponse,
    KnowledgeEvidenceResponse,
    KnowledgeFilesResponse,
    UpdateDocumentPermissionsRequest,
    UpdateDocumentPermissionsResponse,
)
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.rbac import is_admin, validate_allowed_roles
from utils.traffic_control import rate_limit, redis_lock

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"])
KNOWLEDGE_WRITE_LOCK_KEY = "lock:knowledge-base:write"
KNOWLEDGE_VERSIONS_DIR = ".knowledge_versions"
parent_chunk_repository = ParentChunkRepository()


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


def _to_document_response(document, can_manage: bool) -> KnowledgeDocumentResponse:
    response = KnowledgeDocumentResponse.model_validate(document)
    return response.model_copy(
        update={
            "can_manage": can_manage,
            "allowed_roles": list(document.allowed_roles or []) if can_manage else None,
        }
    )


def _document_file_path(document) -> str | None:
    data_dir = os.path.abspath(get_abs_path(chroma_conf["data_path"]))
    storage_key = document.storage_key or document.filename
    file_path = os.path.abspath(
        storage_key if os.path.isabs(storage_key) else os.path.join(data_dir, storage_key)
    )
    try:
        if os.path.commonpath([data_dir, file_path]) != data_dir:
            return None
    except ValueError:
        return None
    return file_path if os.path.isfile(file_path) else None


def _can_read_document(document, user: dict) -> bool:
    return bool(
        document
        and document.status == "active"
        and (
            is_admin(user)
            or user.get("role", "guest") in (document.allowed_roles or [])
        )
    )


def _is_active_parent(document, parent_id: str) -> bool:
    prefix = f"{parent_id}:child:"
    return any(
        chunk_id == parent_id or chunk_id.startswith(prefix)
        for chunk_id in (document.chunk_ids or [])
    )


def _copy_upload_to_version(upload: UploadFile, data_dir: str) -> tuple[str, str]:
    filename = os.path.basename(upload.filename or "")
    if not filename:
        raise ValueError("upload filename is empty")
    version_dir = os.path.join(data_dir, KNOWLEDGE_VERSIONS_DIR, uuid.uuid4().hex)
    os.makedirs(version_dir, exist_ok=False)
    destination = os.path.join(version_dir, filename)
    try:
        upload.file.seek(0)
        with open(destination, "wb") as output:
            shutil.copyfileobj(upload.file, output, length=1024 * 1024)
    except Exception:
        if os.path.exists(destination):
            os.remove(destination)
        try:
            os.rmdir(version_dir)
        except OSError:
            pass
        raise
    return filename, destination


def _remove_stored_file(data_dir: str, path_or_key: str | None) -> bool:
    if not path_or_key:
        return False
    data_dir = os.path.abspath(data_dir)
    candidate = os.path.abspath(
        path_or_key
        if os.path.isabs(path_or_key)
        else os.path.join(data_dir, path_or_key)
    )
    try:
        if os.path.commonpath([data_dir, candidate]) != data_dir:
            return False
    except ValueError:
        return False
    if not os.path.isfile(candidate):
        return False
    os.remove(candidate)
    version_root = os.path.join(data_dir, KNOWLEDGE_VERSIONS_DIR)
    parent = os.path.dirname(candidate)
    if parent != version_root:
        try:
            os.rmdir(parent)
        except OSError:
            pass
    return True


@router.post("/upload")
async def upload_knowledge(
    files: list[UploadFile] = File(...),
    visibility_mode: str = Form(...),
    allowed_roles: str = Form("[]"),
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    await rate_limit(f"user:{admin['id']}:upload", 5, 60)
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    roles = _parse_allowed_roles(visibility_mode, allowed_roles)
    data_dir = os.path.abspath(get_abs_path(chroma_conf["data_path"]))
    uploaded_files = []
    uploaded_paths = []
    copy_failures = []

    async with redis_lock(KNOWLEDGE_WRITE_LOCK_KEY, timeout=600):
        vector_store = await run_in_threadpool(get_vector_store)

        for upload in files:
            if not upload.filename:
                continue
            filename = os.path.basename(upload.filename)
            uploaded_files.append(filename)
            allowed_types = tuple(chroma_conf["allow_knowledge_file_type"])
            if not filename.lower().endswith(allowed_types):
                copy_failures.append(
                    {
                        "filename": filename,
                        "path": None,
                        "status": "failed",
                        "success": False,
                        "storage_key": None,
                        "previous_storage_key": None,
                        "error": "unsupported knowledge file type",
                    }
                )
                continue
            try:
                _filename, version_path = await run_in_threadpool(
                    _copy_upload_to_version,
                    upload,
                    data_dir,
                )
                uploaded_paths.append(version_path)
            except Exception as exc:
                logger.error("Failed to stage knowledge upload %s: %s", filename, exc)
                copy_failures.append(
                    {
                        "filename": filename,
                        "path": None,
                        "status": "failed",
                        "success": False,
                        "storage_key": None,
                        "previous_storage_key": None,
                        "error": str(exc),
                    }
                )

        if uploaded_paths:
            runtime_result = await run_in_threadpool(
                vector_store.load_documents,
                uploaded_paths,
                allowed_roles=roles,
                updated_by=admin.get("id"),
                return_details=True,
            )
        else:
            runtime_result = {
                "new_count": 0,
                "updated_count": 0,
                "skipped_count": 0,
                "removed_count": 0,
                "files": [],
            }

        staged_by_filename = {
            os.path.basename(path): path
            for path in uploaded_paths
        }
        runtime_files = list(runtime_result.get("files") or [])
        for result in runtime_files:
            staged_path = result.get("path") or staged_by_filename.get(result.get("filename"))
            status = result.get("status")
            if not result.get("success") or status in {"same", "duplicate"}:
                await run_in_threadpool(_remove_stored_file, data_dir, staged_path)
                continue
            if status == "updated":
                previous_storage_key = result.get("previous_storage_key")
                if (
                    previous_storage_key
                    and previous_storage_key != result.get("storage_key")
                ):
                    try:
                        await run_in_threadpool(
                            _remove_stored_file,
                            data_dir,
                            previous_storage_key,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Maintenance orphan in previous knowledge file %s: %s",
                            previous_storage_key,
                            exc,
                        )

    file_results = copy_failures + runtime_files
    new_count = runtime_result["new_count"]
    updated_count = runtime_result["updated_count"]
    skipped_count = runtime_result["skipped_count"]
    removed_count = runtime_result["removed_count"]
    success = bool(file_results) and all(
        result.get("success") for result in file_results
    )

    return {
        "success": success,
        "uploaded_files": uploaded_files,
        "new_count": new_count,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "removed_count": removed_count,
        "file_results": file_results,
        "message": f"uploaded {len(uploaded_files)} files, new {new_count}, updated {updated_count}, skipped {skipped_count}",
    }


@router.get(
    "/files",
    response_model=KnowledgeFilesResponse,
)
async def list_knowledge_files(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeFilesResponse:
    can_manage = is_admin(user)
    visible_docs = await list_visible_documents(db, user.get("role", "guest"))
    files = [_to_document_response(document, can_manage) for document in visible_docs]

    return KnowledgeFilesResponse(
        files=files,
        total_files=len(files),
        total_chunks=sum(file.chunk_count or 0 for file in files),
    )


@router.patch(
    "/files/{doc_id}/permissions",
    response_model=UpdateDocumentPermissionsResponse,
)
async def update_knowledge_file_permissions(
    doc_id: str,
    payload: UpdateDocumentPermissionsRequest,
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> UpdateDocumentPermissionsResponse:
    roles = _parse_allowed_roles(payload.visibility_mode, payload.allowed_roles)
    async with redis_lock(KNOWLEDGE_WRITE_LOCK_KEY, timeout=60):
        doc = await update_allowed_roles(db, doc_id, roles, updated_by=admin.get("id"))
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        await db.commit()
        await db.refresh(doc)
    return UpdateDocumentPermissionsResponse(
        doc_id=doc.doc_id,
        allowed_roles=doc.allowed_roles or [],
    )


@router.get(
    "/evidence/{parent_id}",
    response_model=KnowledgeEvidenceResponse,
)
async def get_knowledge_evidence(
    parent_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeEvidenceResponse:
    record = await run_in_threadpool(parent_chunk_repository.get, parent_id)
    metadata = dict((record or {}).get("metadata") or {})
    doc_id = metadata.get("doc_id")
    document = await get_document_acl(db, doc_id) if doc_id else None

    if not _can_read_document(document, user) or not _is_active_parent(
        document,
        parent_id,
    ):
        raise HTTPException(status_code=404, detail="Evidence not found")

    raw_page = metadata.get("page")
    try:
        page = int(raw_page) if raw_page not in (None, "", "-") else None
    except (TypeError, ValueError):
        page = None

    download_url = None
    if _document_file_path(document) is not None:
        download_url = f"/api/knowledge/files/{document.doc_id}/download"

    return KnowledgeEvidenceResponse(
        filename=document.filename,
        page=page,
        content=record["page_content"],
        doc_id=document.doc_id,
        download_url=download_url,
    )


@router.get("/files/{doc_id}/download")
async def download_knowledge_file(
    doc_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    document = await get_document_acl(db, doc_id)
    if not _can_read_document(document, user):
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = _document_file_path(document)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(file_path, filename=document.filename)


@router.delete("/files/{doc_id}")
async def delete_knowledge_file(
    doc_id: str,
    delete_file: bool = True,
    _admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    async with redis_lock(KNOWLEDGE_WRITE_LOCK_KEY, timeout=600):
        document = await get_document_acl(db, doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        filename = document.filename
        vector_store = await run_in_threadpool(get_vector_store)
        deleted_chunks = await run_in_threadpool(
            vector_store.delete_document_by_doc_id,
            doc_id,
            delete_file=delete_file,
        )

    return {
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "deleted_chunks": deleted_chunks,
        "deleted_file": delete_file,
        "message": f"deleted {filename}, removed {deleted_chunks} chunks",
    }
