import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.dependencies import get_agent, get_metrics
from api.auth import router as auth_router
from api.routers import (
    pages,
    detection,
    chat,
    conversations,
    knowledge,
    metrics,
    files,
    health,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="SAR-Sense API", version="2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/css", StaticFiles(directory="css"), name="css")
    app.mount("/js", StaticFiles(directory="js"), name="js")
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

    @app.on_event("startup")
    async def startup_event():
        """Startup: 1) 建表 + 种子用户  2) 后台预加载重对象"""
        # 1. 异步建表（幂等）：从 models 包导入所有模型注册到同一 metadata
        try:
            from config.db_conf import async_engine
            from models import Base
            from models import users, conversations, metrics  # noqa: F401 注册 ORM
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("DB 表已建（或已存在）")
            # 2. 种子用户：admin/admin123（幂等）
            from config.db_conf import AsyncSessionLocal
            from crud.users import get_user_by_username, create_user, count_users
            from utils.security import hash_password
            async with AsyncSessionLocal() as session:
                if (await count_users(session)) == 0:
                    await create_user(session, "admin", hash_password("admin123"))
                    await session.commit()
                    logger.info("已创建种子用户 admin/admin123")
        except Exception as e:
            logger.warning(f"DB 建表/种子用户失败: {e}")

        # 3. 后台预加载（同步对象，不阻塞 startup）
        def preload():
            logger.info("Pre-loading components...")
            try:
                get_agent()
                logger.info("Agent loaded")
            except Exception as e:
                logger.warning(f"Failed to pre-load agent: {e}")

            try:
                get_metrics()
                logger.info("Metrics loaded")
            except Exception as e:
                logger.warning(f"Failed to pre-load metrics: {e}")

            logger.info("Pre-loading complete")

        threading.Thread(target=preload, daemon=True).start()

    app.include_router(pages.router)
    app.include_router(auth_router)
    app.include_router(detection.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(conversations.router, prefix="/api/conversations")
    app.include_router(knowledge.router, prefix="/api/knowledge")
    app.include_router(metrics.router, prefix="/api/metrics")
    app.include_router(files.router, prefix="/api")
    app.include_router(health.router, prefix="/api")

    return app


app = create_app()
