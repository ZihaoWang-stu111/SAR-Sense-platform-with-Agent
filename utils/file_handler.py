import hashlib
import json
from datetime import datetime
from utils.logger_handler import logger
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# ==================== 文件 Hash ====================

def get_file_hash(file_path):
    """计算文件 SHA-256 hash（替代原 MD5，用于文件级去重）"""
    if not os.path.exists(file_path):
        logger.error(f"{file_path} 路径下的文件不存在")
        return None
    if not os.path.isfile(file_path):
        logger.error(f"{file_path} 下的文件类型不可处理")
        return None
    sha256_obj = hashlib.sha256()
    chunk_size = 4096
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256_obj.update(chunk)
        return sha256_obj.hexdigest()
    except Exception as e:
        logger.error(f"计算 {file_path} 失败，{e}")
        return None


# ==================== Manifest 管理 ====================

def load_manifest(manifest_path):
    """加载 manifest.json → Python dict（O(1) 查询）"""
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, 'r', encoding='UTF-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"manifest.json 解析失败，重建空 manifest: {e}")
        return {}


def save_manifest(manifest_path, manifest):
    """持久化 manifest.json"""
    with open(manifest_path, 'w', encoding='UTF-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def check_file_status(manifest, filename, file_hash):
    """判断文件状态：NEW / UPDATED / SAME / DUPLICATE
    DUPLICATE: 同名不在 manifest 中，但 hash 与其他文件相同 → 内容重复"""
    if filename not in manifest:
        # 全局 hash 去重：同名不存在，但内容可能和别的文件一样
        for other_name, entry in manifest.items():
            if entry.get("file_hash") == file_hash:
                return "DUPLICATE"
        return "NEW"
    if manifest[filename]["file_hash"] != file_hash:
        return "UPDATED"
    return "SAME"


def update_manifest_entry(manifest, filename, doc_id, file_hash,
                          chunk_count, chunk_ids, chunk_method, file_type):
    """更新/新增 manifest 条目"""
    manifest[filename] = {
        "doc_id": doc_id,
        "file_hash": file_hash,
        "chunk_count": chunk_count,
        "chunk_ids": chunk_ids,
        "chunk_method": chunk_method,
        "file_type": file_type,
        "ingested_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "active"
    }
    return manifest


# ==================== 原有函数（不变） ====================

def listdir_with_allowed_type(path, allowed_types):

    if not os.path.isdir(path):
        logger.error(f"你所上传的路径{path}非文件夹")
        return allowed_types
    files = []

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)


def pdf_loader(file_path, passwd=None):
    return PyPDFLoader(file_path, password=passwd).load()


def text_loader(file_path):
    return TextLoader(file_path, encoding="UTF-8").load()