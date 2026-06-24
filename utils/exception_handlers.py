"""全局异常处理器：路由删掉 try/except 样板，异常统一在这里兜底。

注册顺序：子类在前，父类在后（HTTPException → IntegrityError → SQLAlchemyError → Exception）。
响应格式保持本项目现状：失败统一 {"detail": "..."}，前端 data.detail 解析不变。
"""
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from utils.logger_handler import logger


async def http_exception_handler(request: Request, exc: HTTPException):
    """业务异常（401/403/404/409 等）：原样返回，保持 {detail} 格式。"""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def integrity_error_handler(request: Request, exc: IntegrityError):
    """DB 完整性约束冲突：转友好提示，不泄漏 SQL 原文。"""
    msg = str(exc.orig) if exc.orig else str(exc)
    low = msg.lower()
    if "duplicate entry" in low or "unique" in low or "username_unique" in low:
        detail = "用户名已存在"
    elif "foreign key" in low or "cannot delete" in low:
        detail = "关联数据不存在，无法删除"
    else:
        detail = "数据约束冲突，请检查输入"
    logger.warning(f"[IntegrityError] {request.url.path}: {msg}")
    return JSONResponse(status_code=400, content={"detail": detail})


async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    """其他 DB 错误：500，日志记详情，前端只见通用提示。"""
    logger.error(f"[DB Error] {request.url.path}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "数据库操作失败，请稍后重试"})


async def general_exception_handler(request: Request, exc: Exception):
    """兜底：任何未捕获异常，500，不泄漏堆栈给前端。"""
    logger.error(f"[Unhandled] {request.url.path}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})


def register_exception_handlers(app: FastAPI):
    """注册全局异常处理器。子类在前，父类在后。"""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(IntegrityError, integrity_error_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)
