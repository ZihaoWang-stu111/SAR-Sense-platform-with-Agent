import logging
import os
import tempfile
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from PIL import Image

from api.auth import get_current_user
from services.upload_store import (
    save_upload, get_upload_path, IMAGE_EXTS, IMAGE_MIMES, MAX_UPLOAD_BYTES, MAX_IMAGE_PIXELS,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["files"])


@router.post("/extract-file", dependencies=[Depends(get_current_user)])
async def extract_file(
    file: UploadFile = File(...),
):
    """提取上传文件内容 / 暂存上传图片。

    - 图片：经 MIME + magic bytes + 像素上限校验后存到 data/uploads，
      返回不透明 upload_id（供 Agent 调 detect_ships），不返回任何服务端路径。
    - 文档：temp + 提取文本 + 用完即清，返回 content。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail='No file selected')

    content_bytes = await file.read()
    if len(content_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='文件过大，最大 10MB')

    # basename 防路径穿越：客户端可能传 "../../evil.py" 之类文件名
    safe_name = os.path.basename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower()

    if ext in IMAGE_EXTS:
        # MIME 白名单（不信扩展名）
        if file.content_type and file.content_type not in IMAGE_MIMES:
            raise HTTPException(status_code=400, detail='不支持的图片类型')
        # magic bytes 校验：PIL 打开 + verify，防扩展名伪造 / 损坏文件 / 解码炸弹
        try:
            img = Image.open(BytesIO(content_bytes))
            w, h = img.size
            if w * h > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail='图片像素过大')
            img.verify()
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail='图片格式无效或损坏')

        upload_id = save_upload(content_bytes, ext)
        return {
            'success': True,
            'filename': file.filename,
            'upload_id': upload_id,
        }

    # 文档：temp + 提取文本 + 清理
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, safe_name)
    with open(temp_path, "wb") as f:
        f.write(content_bytes)
    try:
        # 延迟导入：agent_tools 模块级会实例化 RagSummarizeService（触发 vector store + BM25 + BGE 加载），
        # 放函数内避免启动时同步阻塞；只在真正需要提取文档时才触发
        from agent.tools.agent_tools import extract_file_content
        content = extract_file_content.invoke({"file_path": temp_path})
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            logger.warning(f"Failed to remove temp file: {temp_path}")

    return {
        'success': True,
        'content': content,
        'filename': file.filename,
    }


@router.get("/image/{upload_id}", dependencies=[Depends(get_current_user)])
async def get_upload_image(upload_id: str):
    """按 upload_id 返回上传的图片字节。

    鉴权后访问，防匿名枚举他人上传。detect_ships 工具画完检测图存 upload_store 后，
    前端用此端点按 upload_id 拉图渲染成回答下方卡片。
    """
    path = get_upload_path(upload_id)
    if path is None:
        raise HTTPException(status_code=404, detail="文件不存在或已过期")
    return FileResponse(path)
