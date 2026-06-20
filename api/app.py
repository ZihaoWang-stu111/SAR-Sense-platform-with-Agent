import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.dependencies import get_agent, get_conv_manager, get_metrics
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
        """Pre-load components on startup to avoid slow first request"""
        def preload():
            logger.info("Pre-loading components...")
            try:
                get_agent()
                logger.info("Agent loaded")
            except Exception as e:
                logger.warning(f"Failed to pre-load agent: {e}")

            try:
                get_conv_manager()
                logger.info("Conversation manager loaded")
            except Exception as e:
                logger.warning(f"Failed to pre-load conversation manager: {e}")

            try:
                get_metrics()
                logger.info("Metrics loaded")
            except Exception as e:
                logger.warning(f"Failed to pre-load metrics: {e}")

            logger.info("Pre-loading complete")

        threading.Thread(target=preload, daemon=True).start()

    app.include_router(pages.router)
    app.include_router(detection.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(conversations.router, prefix="/api/conversations")
    app.include_router(knowledge.router, prefix="/api/knowledge")
    app.include_router(metrics.router, prefix="/api/metrics")
    app.include_router(files.router, prefix="/api")
    app.include_router(health.router, prefix="/api")

    return app


app = create_app()
