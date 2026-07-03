"""上传文件存储抽象层。

当前实现：本地磁盘 data/uploads/。
生产替换：把 save_upload / get_upload_path 改成 S3/OSS 调用即可，
路由层（files.py）和工具层（detect_ships）无需改动——这正是抽象的意义。

设计要点：
- 对外只暴露不透明 upload_id（如 img_<hex>），绝不暴露文件系统路径；
- detect_ships 工具入参是 upload_id 而非路径，LLM 无路径可注入；
- get_upload_path 只在受控目录内做精确 id 匹配，不接受路径字符。
"""
import os
import time
import uuid

from utils.path_tool import get_abs_path

# 上传文件持久化目录
UPLOAD_DIR = get_abs_path("data/uploads")

# 图片扩展名白名单
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 图片 MIME 白名单
IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}

# 上传文件大小上限（字节）—— files.py / detection.py 共用
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB

# 图片像素上限——防 PIL 解码炸弹，files.py / detection_service.py 共用
MAX_IMAGE_PIXELS = 50_000_000  # 50MP

# 旧上传超过该时长（秒）就在下次上传时顺手清掉
UPLOAD_MAX_AGE_SECONDS = 3600


def save_upload(content_bytes: bytes, ext: str) -> str:
    """保存上传文件，返回不透明 upload_id。

    生产环境把这里换成 S3 put_object 即可，调用方不变。
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    _cleanup_old_uploads()
    upload_id = f"img_{uuid.uuid4().hex}"
    # 文件名 = {upload_id}{ext}，不含原始文件名（防文件名注入）
    stored_path = os.path.join(UPLOAD_DIR, f"{upload_id}{ext}")
    with open(stored_path, "wb") as f:
        f.write(content_bytes)
    return upload_id


def get_upload_path(upload_id: str) -> str | None:
    """按 upload_id 查找本地存储的文件路径。找不到返回 None。

    upload_id 是不透明标识；本函数只在 UPLOAD_DIR 内做精确匹配，
    拒绝任何含路径分隔符的输入，因此无路径注入面。
    生产环境把这里换成 S3 get_object 即可。
    """
    if not isinstance(upload_id, str) or not upload_id:
        return None
    # 防御：upload_id 应为纯标识（img_xxx），含路径分隔符一律拒绝
    if os.path.basename(upload_id) != upload_id:
        return None
    if not os.path.isdir(UPLOAD_DIR):
        return None
    for fname in os.listdir(UPLOAD_DIR):
        base, _ = os.path.splitext(fname)
        if base == upload_id:
            return os.path.join(UPLOAD_DIR, fname)
    return None


def _cleanup_old_uploads(max_age_seconds: int = UPLOAD_MAX_AGE_SECONDS) -> None:
    """顺手清掉超过 max_age 的旧上传，避免目录无限增长。

    生产环境用 S3 lifecycle 规则或定时任务替代，更可靠。
    """
    try:
        now = time.time()
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > max_age_seconds:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except OSError:
        pass
