"""SQLite 连接 + 建表（极简，标准库 sqlite3）。

db 文件：runtime/sar_sense.db
get_conn() 做成上下文管理器：保证连接用完即关，避免泄漏（sqlite3 的 with conn 只管事务不管关闭）。
不做降级机制——SQLite 是本地文件，几乎不会不可用。
"""
import os
import sqlite3
from contextlib import contextmanager

from utils.path_tool import get_abs_path

DB_PATH = get_abs_path("runtime/sar_sense.db")


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)


@contextmanager
def get_conn():
    """SQLite 连接上下文管理器。

    内部：建连 → row_factory=Row → 开 WAL + 外键 → yield conn
    收尾：正常 commit，异常 rollback，finally close。
    业务层统一 `with get_conn() as conn: conn.execute(...)`。
    """
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# 建表 DDL —— 3 张表，均带 user_id（默认 default_user，为未来登录预留）
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS conversations (
        id            TEXT PRIMARY KEY,
        user_id       TEXT DEFAULT 'default_user',
        title         TEXT,
        created_at    TEXT,
        updated_at    TEXT,
        summary       TEXT,
        summary_up_to INTEGER DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)",

    """CREATE TABLE IF NOT EXISTS conversation_messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        message_index   INTEGER NOT NULL,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        thought_steps   TEXT,
        created_at      TEXT,
        UNIQUE(conversation_id, message_index),
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )""",
    "CREATE INDEX IF NOT EXISTS idx_msg_conv ON conversation_messages(conversation_id, message_index)",

    """CREATE TABLE IF NOT EXISTS metric_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     TEXT DEFAULT 'default_user',
        event_type  TEXT NOT NULL,
        tool_name   TEXT,
        success     INTEGER,
        duration_ms REAL,
        created_at  TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_metric_created ON metric_events(created_at)",

    """CREATE TABLE IF NOT EXISTS users (
        id            TEXT PRIMARY KEY,
        username      TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL
    )""",
]

# 种子用户：id='default_user'，使现有 29 条 conversations.user_id='default_user' 天然归属它，零迁移。
SEED_USER_ID = "default_user"
SEED_USERNAME = "admin"
SEED_PASSWORD = "admin123"  # 仅开发用，生产环境应改


def _ensure_seed_user(conn):
    """幂等创建种子用户（仅当 users 表为空时）。"""
    row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    if row["c"] == 0:
        from datetime import datetime
        from utils.security import hash_password
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (SEED_USER_ID, SEED_USERNAME, hash_password(SEED_PASSWORD), now),
        )


def init_db():
    """幂等建表 + 确保种子用户。可重复调用。"""
    with get_conn() as conn:
        for stmt in _SCHEMA:
            conn.execute(stmt)
        _ensure_seed_user(conn)
