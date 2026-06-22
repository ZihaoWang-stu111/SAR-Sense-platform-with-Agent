"""conversations/*.json -> SQLite 一次性迁移脚本。

用法：
    python -m utils.migrate_sqlite             # 幂等 upsert 迁移
    python -m utils.migrate_sqlite --check     # 种子对齐检查
    python -m utils.migrate_sqlite --rebuild   # 只重建 JSON 种子对话，不动 SQLite 中新对话/指标

关键语义（迁移后新对话只写 SQLite，不再回写 JSON）：
- --check 只核对 JSON 种子 conv_id 在 SQLite 存在且消息数一致；SQLite 多出的对话只提示不失败。
- --rebuild 只删 JSON 种子对应的 conv_id，不删 SQLite 中其它对话，不清 metric_events。
"""
import argparse
import glob
import json
import os
import sys

from utils.db import get_conn, init_db
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

DEFAULT_USER_ID = "default_user"


def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"读取 {path} 失败: {e}")
        return None


def _seed_conversations():
    """返回 [(conv_id, data), ...]，从 conversations/*.json 读。"""
    conv_dir = get_abs_path("conversations")
    if not os.path.isdir(conv_dir):
        return []
    seeds = []
    for fpath in glob.glob(os.path.join(conv_dir, "*.json")):
        data = _load_json(fpath)
        if data and "id" in data:
            seeds.append((data["id"], data))
    return seeds


def _upsert_conversation(conn, conv_id, data):
    conn.execute(
        """INSERT INTO conversations (id, user_id, title, created_at, updated_at, summary, summary_up_to)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             user_id=excluded.user_id, title=excluded.title,
             created_at=excluded.created_at, updated_at=excluded.updated_at,
             summary=excluded.summary, summary_up_to=excluded.summary_up_to""",
        (conv_id, DEFAULT_USER_ID,
         data.get("title") or "新对话",
         data.get("created_at") or "",
         data.get("updated_at") or "",
         data.get("summary") or "",
         data.get("summary_up_to", 0) or 0),
    )


def _upsert_message(conn, conv_id, idx, msg, fallback_created_at):
    thought = msg.get("thought_steps")
    thought_json = json.dumps(thought, ensure_ascii=False) if thought else None
    conn.execute(
        """INSERT INTO conversation_messages
             (conversation_id, message_index, role, content, thought_steps, created_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(conversation_id, message_index) DO UPDATE SET
             role=excluded.role, content=excluded.content,
             thought_steps=excluded.thought_steps, created_at=excluded.created_at""",
        (conv_id, idx, msg.get("role", "user"), msg.get("content", ""),
         thought_json, fallback_created_at),
    )


def run_migrate():
    init_db()
    seeds = _seed_conversations()
    if not seeds:
        logger.info("[migrate] conversations/ 无 JSON 种子，跳过")
        return

    conv_n = 0
    msg_n = 0
    with get_conn() as conn:
        for conv_id, data in seeds:
            _upsert_conversation(conn, conv_id, data)
            conv_n += 1
            created_at = data.get("updated_at") or ""
            for idx, msg in enumerate(data.get("messages", [])):
                _upsert_message(conn, conv_id, idx, msg, created_at)
                msg_n += 1
    logger.info(f"[migrate] 完成：conversations={conv_n}, conversation_messages={msg_n}")


def run_check() -> int:
    """种子对齐检查。exit 0=种子全对齐，1=有种子缺失或消息数不符。"""
    init_db()
    seeds = _seed_conversations()
    seed_ids = {cid for cid, _ in seeds}
    ok = True

    with get_conn() as conn:
        # SQLite 现有 conv_id 集合
        db_ids = {row[0] for row in conn.execute("SELECT id FROM conversations")}

        # 种子对齐
        for conv_id, data in seeds:
            json_msg_count = len(data.get("messages", []))
            row = conn.execute(
                "SELECT id, (SELECT COUNT(*) FROM conversation_messages WHERE conversation_id=?) AS msg_n "
                "FROM conversations WHERE id=?",
                (conv_id, conv_id),
            ).fetchone()
            if row is None:
                print(f"[check] FAIL 种子 {conv_id} 在 SQLite 中缺失")
                ok = False
            elif row["msg_n"] != json_msg_count:
                print(f"[check] FAIL 种子 {conv_id} 消息数不符 JSON={json_msg_count} SQLite={row['msg_n']}")
                ok = False
            else:
                print(f"[check] OK 种子 {conv_id} 对齐 (messages={json_msg_count})")

        # SQLite 多出的对话（迁移后新建）——只提示不失败
        extra = db_ids - seed_ids
        if extra:
            print(f"[check] INFO SQLite 中有 {len(extra)} 个非种子对话（迁移后新建），不计为失败: {list(extra)[:5]}")

    if ok:
        print("[check] [OK] 全部种子对齐")
        return 0
    else:
        print("[check] [FAIL] 存在种子缺失或消息数不符")
        return 1


def run_rebuild():
    """只重建 JSON 种子对话：删种子 conv_id → 重新导入。不动 SQLite 其它对话，不清 metric_events。"""
    init_db()
    seeds = _seed_conversations()
    if not seeds:
        logger.info("[rebuild] conversations/ 无 JSON 种子，跳过")
        return

    seed_ids = [cid for cid, _ in seeds]
    with get_conn() as conn:
        # 只删种子对应的 conv（CASCADE 连带删其消息）
        conn.executemany("DELETE FROM conversations WHERE id=?", [(cid,) for cid in seed_ids])
        conv_n = 0
        msg_n = 0
        for conv_id, data in seeds:
            _upsert_conversation(conn, conv_id, data)
            conv_n += 1
            created_at = data.get("updated_at") or ""
            for idx, msg in enumerate(data.get("messages", [])):
                _upsert_message(conn, conv_id, idx, msg, created_at)
                msg_n += 1
    logger.info(f"[rebuild] 完成：重建种子 conversations={conv_n}, messages={msg_n}（未动 SQLite 其它对话与 metric_events）")


def main():
    parser = argparse.ArgumentParser(description="conversations JSON -> SQLite 迁移")
    parser.add_argument("--check", action="store_true", help="种子对齐检查")
    parser.add_argument("--rebuild", action="store_true", help="只重建 JSON 种子对话")
    args = parser.parse_args()

    if args.check:
        sys.exit(run_check())
    elif args.rebuild:
        run_rebuild()
    else:
        run_migrate()


if __name__ == "__main__":
    main()
