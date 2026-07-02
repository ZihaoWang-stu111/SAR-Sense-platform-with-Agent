import logging

from fastapi import APIRouter, UploadFile, File
from starlette.concurrency import run_in_threadpool

from services.detection_service import detect_ships_from_bytes

logger = logging.getLogger(__name__)
router = APIRouter(tags=["detection"])


@router.post("/detect")
async def detect_ships(image: UploadFile = File(...)):
    """SAR ship detection endpoint"""
    logger.info(f"Detection request received: {image.filename}")
    content = await image.read()
    return await run_in_threadpool(detect_ships_from_bytes, content, image.filename)
