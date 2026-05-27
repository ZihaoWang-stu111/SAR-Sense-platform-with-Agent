import hashlib
from utils.logger_handler import logger
import os
from langchain_community.document_loaders import PyPDFLoader,TextLoader

def get_file_md5_hex(file_path):

    if not os.path.exists(file_path):
        logger.error(f"{file_path}路径下的文件不存在")
        return
    if not os.path.isfile(file_path):
        logger.error(f"{file_path}下的文件类型不可处理")
        return
    md5_obj = hashlib.md5()
    chunk_size = 4096
    try:
        with open(file_path,'rb') as f:
            while chunk:=f.read(chunk_size):
                md5_obj.update(chunk)

            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算{file_path}失败，{e}")
        return None

def listdir_with_allowed_type(path, allowed_types):

    if not os.path.isdir(path):
        logger.error(f"你所上传的路径{path}非文件夹")
        return allowed_types
    files = []

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path,f))
    return tuple(files)

def pdf_loader(file_path, passwd = None):
    return PyPDFLoader(file_path, password=passwd).load()
def text_loader(file_path):
    return TextLoader(file_path,encoding="UTF-8").load()