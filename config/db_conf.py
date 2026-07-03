"""异步 MySQL 引擎 + AsyncSession + get_db 依赖（对齐参考方式）。

库：sar_sense（root/root），utf8mb4。
业务路由通过 Depends(get_db) 注入 AsyncSession；crud 函数全部 async。
"""
import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# 通过环境变量注入连接信息：本地默认 localhost/root/root（不设 env 照常工作），
# Docker 部署时由 compose 注入 MYSQL_HOST=db 等。
ASYNC_DATABASE_URL = (
    f"mysql+aiomysql://{os.getenv('MYSQL_USER', 'root')}:{os.getenv('MYSQL_PASSWORD', 'root')}"
    f"@{os.getenv('MYSQL_HOST', 'localhost')}:{os.getenv('MYSQL_PORT', '3306')}"
    f"/{os.getenv('MYSQL_DATABASE', 'sar_sense')}?charset=utf8mb4"
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,         # 生产环境关掉 SQL 日志；调试时改 True
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=False, # aiomysql 的 ping() 签名与 SQLAlchemy 期望不符，关闭。改用 pool_recycle 规避 wait_timeout
    pool_recycle=3600,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    """FastAPI 依赖：yield 一个 AsyncSession，正常退出 commit，异常 rollback，finally close。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
