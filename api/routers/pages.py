from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["pages"])


@router.get("/")
async def index():
    return FileResponse("index.html")


@router.get("/index.html")
async def index_html():
    return FileResponse("index.html")


@router.get("/detection.html")
async def detection_page():
    return FileResponse("detection.html")


@router.get("/chat.html")
async def chat_page():
    return FileResponse("chat.html")


@router.get("/knowledge.html")
async def knowledge_page():
    return FileResponse("knowledge.html")


@router.get("/metrics.html")
async def metrics_page():
    return FileResponse("metrics.html")
