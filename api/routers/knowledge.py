import logging
import os
import traceback

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from api.dependencies import get_vector_store
from api.auth import get_current_user, require_admin

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"])


@router.post("/upload")
async def upload_knowledge(
    files: list[UploadFile] = File(...),
    _admin: dict = Depends(require_admin),
):
    """Upload documents to knowledge base（仅 admin）"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail='No files provided')

        from utils.config_handler import chroma_conf
        from utils.path_tool import get_abs_path

        data_dir = get_abs_path(chroma_conf["data_path"])
        uploaded_files = []

        for file in files:
            if file.filename:
                filename = os.path.basename(file.filename)
                file_path = os.path.join(data_dir, filename)
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                uploaded_files.append(filename)

        vector_store = get_vector_store()
        new_count, updated_count, skipped_count, removed_count = vector_store.load_document()

        return {
            'success': True,
            'uploaded_files': uploaded_files,
            'new_count': new_count,
            'updated_count': updated_count,
            'removed_count': removed_count,
            'message': f'成功上传 {len(uploaded_files)} 个文件，新增 {new_count}、更新 {updated_count}、清理 {removed_count}'
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def list_knowledge_files(_user: dict = Depends(get_current_user)):
    """List knowledge base files — 从 manifest 读，展示 chunk_count/doc_id/status 等结构化信息。
    所有登录用户都能看（知识库内容公开），但只有 admin 能改。"""
    try:
        vector_store = get_vector_store()
        manifest = vector_store.manifest

        files = []
        for filename, entry in manifest.items():
            files.append({
                'name': filename,
                'doc_id': entry.get('doc_id'),
                'file_type': entry.get('file_type'),
                'chunk_count': entry.get('chunk_count'),
                'chunk_method': entry.get('chunk_method'),
                'status': entry.get('status'),
                'ingested_at': entry.get('ingested_at'),
                'file_hash': entry.get('file_hash'),
            })

        return {
            'success': True,
            'files': files,
            'total_files': len(files),
            'total_chunks': sum(e.get('chunk_count', 0) for e in manifest.values())
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{doc_id}")
async def delete_knowledge_file(
    doc_id: str,
    delete_file: bool = True,
    _admin: dict = Depends(require_admin),
):
    """Delete one knowledge document by doc_id（仅 admin）。"""
    try:
        vector_store = get_vector_store()
        target = None
        for filename, entry in vector_store.manifest.items():
            if entry.get("doc_id") == doc_id:
                target = {
                    "filename": filename,
                    "chunk_count": entry.get("chunk_count", 0),
                }
                break

        if target is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        deleted_chunks = vector_store.delete_document_by_doc_id(doc_id, delete_file=delete_file)

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": target["filename"],
            "deleted_chunks": deleted_chunks,
            "deleted_file": delete_file,
            "message": f"已删除 {target['filename']}，清理 {deleted_chunks} 个 chunk",
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
