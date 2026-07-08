from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["pages"])

# 禁用 HTML 缓存的 HTTP 响应头
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0"
}


@router.get("/")
async def index():
    return FileResponse("index.html", headers=NO_CACHE_HEADERS)


@router.get("/index.html")
async def index_html():
    return FileResponse("index.html", headers=NO_CACHE_HEADERS)


@router.get("/detection.html")
async def detection_page():
    return FileResponse("detection.html", headers=NO_CACHE_HEADERS)


@router.get("/chat.html")
async def chat_page():
    return FileResponse("chat.html", headers=NO_CACHE_HEADERS)


@router.get("/knowledge.html")
async def knowledge_page():
    return FileResponse("knowledge.html", headers=NO_CACHE_HEADERS)


@router.get("/metrics.html")
async def metrics_page():
    return FileResponse("metrics.html", headers=NO_CACHE_HEADERS)


@router.get("/login.html")
async def login_page():
    return FileResponse("login.html", headers=NO_CACHE_HEADERS)
