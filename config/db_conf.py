"""MySQL 同步/异步引擎与 FastAPI 请求级 AsyncSession。"""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# db_conf 可能先于 config_handler 被导入，需独立确保本地 .env 已加载。
load_dotenv()

_MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
if not _MYSQL_PASSWORD:
    raise RuntimeError("必须通过环境变量 MYSQL_PASSWORD 配置数据库密码")

# 用户名和地址可采用本地开发默认值，密码必须显式注入。
_DATABASE_OPTIONS = {
    "username": os.getenv("MYSQL_USER", "root"),
    "password": _MYSQL_PASSWORD,
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "database": os.getenv("MYSQL_DATABASE", "sar_sense"),
    "query": {"charset": "utf8mb4"},
}
ASYNC_DATABASE_URL = URL.create("mysql+aiomysql", **_DATABASE_OPTIONS)
SYNC_DATABASE_URL = URL.create("mysql+pymysql", **_DATABASE_OPTIONS)

_POOL_OPTIONS = {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_recycle": 3600,
}

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,         # 生产环境关掉 SQL 日志；调试时改 True
    pool_pre_ping=False, # aiomysql 的 ping() 签名与 SQLAlchemy 期望不符，关闭。改用 pool_recycle 规避 wait_timeout
    **_POOL_OPTIONS,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

sync_engine = create_engine(
    SYNC_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    **_POOL_OPTIONS,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
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
