"""metric_events 同步写入（pymysql 直连，不走 async session）。

LangChain middleware 是同步函数，调 AgentMetrics.record_*；这里走最简的同步 pymysql。
连接每次写入开一个，写完关——单 worker 单用户场景够用，求职项目不必上同步连接池。
"""
import pymysql
from utils.logger_handler import logger


def _conn():
    return pymysql.connect(
        host="localhost", port=3306,
        user="root", password="root",
        database="sar_sense", charset="utf8mb4",
        autocommit=True,
    )


def insert_metric_event(event_type: str, tool_name: str = None,
                        success: bool = None, duration_ms: float = None,
                        user_id: int = 1) -> None:
    """同步插入一条 metric_event。失败只 warning 不抛，不影响 agent 流程。"""
    try:
        conn = _conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO metric_events
                       (user_id, event_type, tool_name, success, duration_ms, created_at)
                       VALUES (%s, %s, %s, %s, %s, NOW())""",
                    (user_id, event_type, tool_name,
                     None if success is None else int(success),
                     duration_ms),
                )
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"[metrics] 写 MySQL 失败（不影响内存）: {e}")
