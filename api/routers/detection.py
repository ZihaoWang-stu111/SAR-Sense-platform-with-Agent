import logging
import os
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from starlette.concurrency import run_in_threadpool
from PIL import Image

from api.auth import get_current_user
from services.detection_service import detect_ships_from_bytes
from services.upload_store import IMAGE_EXTS, IMAGE_MIMES, MAX_UPLOAD_BYTES, MAX_IMAGE_PIXELS
from utils.traffic_control import rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["detection"])


@router.post("/detect")
async def detect_ships(image: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """SAR ship detection endpoint

    上传安全：鉴权 + 10MB 大小限制 + 扩展名/MIME 白名单 + PIL magic bytes 校验 + 50MP 像素上限。
    """
    await rate_limit(f"user:{user['id']}:detect", 5, 60)
    logger.info(f"Detection request received: {image.filename}")

    content = await image.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='文件过大，最大 10MB')

    safe_name = os.path.basename(image.filename or "upload.png")
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail='仅支持图片格式')
    if image.content_type and image.content_type not in IMAGE_MIMES:
        raise HTTPException(status_code=400, detail='不支持的图片类型')

    # magic bytes 校验 + 像素上限：防扩展名伪造 / 损坏文件 / 解码炸弹
    try:
        img = Image.open(BytesIO(content))
        w, h = img.size
        if w * h > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=400, detail='图片像素过大')
        img.verify()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail='图片格式无效或损坏')

    return await run_in_threadpool(detect_ships_from_bytes, content, image.filename)
