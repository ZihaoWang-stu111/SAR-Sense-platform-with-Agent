import hashlib
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
    """解析 PDF → list[Document]。

    优先走 MinerU（结构化 Markdown，公式→LaTeX、表格→HTML，扫描页 OCR）；
    失败/未启用/加密 PDF 自动回退 PyPDFLoader，上传永不中断。
    """
    # 加密 PDF：MinerU 不支持密码，直接走 PyPDFLoader。
    if passwd:
        return PyPDFLoader(file_path, password=passwd).load()

    from utils.mineru_client import _mineru_enabled, parse_pdf_to_documents
    if _mineru_enabled():
        try:
            docs = parse_pdf_to_documents(file_path)
            if docs:
                return docs
            logger.warning(f"MinerU 返回空，回退 PyPDFLoader: {file_path}")
        except Exception as e:
            logger.warning(f"MinerU 解析失败，回退 PyPDFLoader: {file_path} - {e}")

    return PyPDFLoader(file_path, password=passwd).load()


def text_loader(file_path):
    return TextLoader(file_path, encoding="UTF-8").load()
