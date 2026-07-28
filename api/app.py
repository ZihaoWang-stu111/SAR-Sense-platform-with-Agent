import asyncio
import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.dependencies import get_agent, get_metrics, shutdown_agent_executor
from api.auth import router as auth_router
from api.routers import (
    admin,
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

    # 全局异常处理器（路由删 try/except 后由这里兜底）
    from utils.exception_handlers import register_exception_handlers
    register_exception_handlers(app)

    @app.on_event("startup")
    async def startup_event():
        """Startup: 1) 建表 + 种子用户  2) 后台预加载重对象"""
        # 1. 异步建表（幂等）：从 models 包导入所有模型注册到同一 metadata
        try:
            from config.db_conf import async_engine
            from models import Base
            from models import users, conversations, metrics, knowledge  # noqa: F401 注册 ORM
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("DB 表已建（或已存在）")
            # 2. 种子用户：admin/admin123（幂等）
            from config.db_conf import AsyncSessionLocal
            from crud.users import (
                count_users,
                create_user,
                ensure_user_role_column,
                get_user_by_username,
                update_user_role,
            )
            from crud.conversations import ensure_conversation_rag_results_column
            from utils.security import hash_password
            from utils.rbac import ROLE_ADMIN
            async with AsyncSessionLocal() as session:
                await ensure_user_role_column(session)
                await ensure_conversation_rag_results_column(session)
                if (await count_users(session)) == 0:
                    await create_user(session, "admin", hash_password("admin123"), role=ROLE_ADMIN)
                    logger.info("已创建种子用户 admin/admin123")
                else:
                    admin_user = await get_user_by_username(session, "admin")
                    if admin_user and admin_user.role != ROLE_ADMIN:
                        await update_user_role(session, admin_user, ROLE_ADMIN)
                        logger.info("已修正 admin 用户角色")
                await session.commit()
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

    async def shutdown_event():
        """等待已提交的 Agent 任务收尾，再关闭共享线程池。"""
        await asyncio.to_thread(shutdown_agent_executor, wait=True)

    app.add_event_handler("shutdown", shutdown_event)

    app.include_router(pages.router)
    app.include_router(auth_router)
    app.include_router(admin.router, prefix="/api/admin")
    app.include_router(detection.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(conversations.router, prefix="/api/conversations")
    app.include_router(knowledge.router, prefix="/api/knowledge")
    app.include_router(metrics.router, prefix="/api/metrics")
    app.include_router(files.router, prefix="/api")
    app.include_router(health.router, prefix="/api")

    return app


app = create_app()
