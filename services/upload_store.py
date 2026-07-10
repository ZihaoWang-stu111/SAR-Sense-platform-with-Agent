"""上传文件存储抽象层。

当前实现：本地磁盘 data/uploads/<user_id>/（按用户隔离，防越权读取他人图片）。
生产替换：把 save_upload / get_upload_path 改成 S3/OSS 调用即可，
路由层（files.py）和工具层（detect_ships）无需改动--这正是抽象的意义。

设计要点：
- 对外只暴露不透明 upload_id（如 img_<hex>），绝不暴露文件系统路径；
- detect_ships 工具入参是 upload_id 而非路径，LLM 无路径可注入；
- 上传记录按 user_id 隔离：get_upload_path 必传 user_id，只在自己目录找，
  拿到他人 upload_id 也读不到（归属校验）。
"""
import os
import time
import uuid

from utils.path_tool import get_abs_path

# 上传文件持久化目录（按 user_id 子目录隔离）
UPLOAD_DIR = get_abs_path("data/uploads")

# 图片扩展名白名单
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 图片 MIME 白名单
IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}

# 上传文件大小上限（字节）-- files.py / detection.py 共用
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB

# 图片像素上限--防 PIL 解码炸弹，files.py / detection_service.py 共用
MAX_IMAGE_PIXELS = 50_000_000  # 50MP

# 旧上传超过该时长（秒）就在下次上传时顺手清掉
UPLOAD_MAX_AGE_SECONDS = 3600


def save_upload(content_bytes: bytes, ext: str, user_id) -> str:
    """保存上传文件，返回不透明 upload_id。

    文件存到 UPLOAD_DIR/<user_id>/<upload_id><ext>，绑定所有者；
    读取时 get_upload_path 必须带同一 user_id 才能找到。
    生产环境把这里换成 S3 put_object 即可，调用方不变。
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    _cleanup_old_uploads()
    upload_id = f"img_{uuid.uuid4().hex}"
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    # 文件名 = {upload_id}{ext}，不含原始文件名（防文件名注入）
    stored_path = os.path.join(user_dir, f"{upload_id}{ext}")
    with open(stored_path, "wb") as f:
        f.write(content_bytes)
    return upload_id


def get_upload_path(upload_id: str, user_id) -> str | None:
    """按 upload_id 在 UPLOAD_DIR/<user_id>/ 查找文件路径。

    user_id 是归属校验：只在调用方自己的目录找，拿到他人 upload_id 也读不到。
    找不到返回 None。生产环境把这里换成 S3 get_object 即可。
    """
    if not isinstance(upload_id, str) or not upload_id:
        return None
    # 防御：upload_id 应为纯标识（img_xxx），含路径分隔符一律拒绝
    if os.path.basename(upload_id) != upload_id:
        return None
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    if not os.path.isdir(user_dir):
        return None
    for fname in os.listdir(user_dir):
        base, _ = os.path.splitext(fname)
        if base == upload_id:
            return os.path.join(user_dir, fname)
    return None


def _cleanup_old_uploads(max_age_seconds: int = UPLOAD_MAX_AGE_SECONDS) -> None:
    """顺手清掉超过 max_age 的旧上传，避免目录无限增长。

    遍历所有用户子目录。生产环境用 S3 lifecycle 规则或定时任务替代。
    """
    try:
        now = time.time()
        if not os.path.isdir(UPLOAD_DIR):
            return
        for user_dir_name in os.listdir(UPLOAD_DIR):
            user_dir = os.path.join(UPLOAD_DIR, user_dir_name)
            if not os.path.isdir(user_dir):
                continue
            for fname in os.listdir(user_dir):
                fpath = os.path.join(user_dir, fname)
                if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > max_age_seconds:
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
    except OSError:
        pass
