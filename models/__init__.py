"""所有 ORM 模型共用的 Base（DeclarativeBase）。

参考方式里每个 model 文件各自定义 Base，会导致 metadata 分裂——表创建时
Base.metadata.create_all 只能拿到一份 metadata。我们用单一 Base 让所有模型注册到同一 metadata。
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
