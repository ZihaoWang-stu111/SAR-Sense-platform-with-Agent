# 迁移到 MySQL 后的父子块运行说明

这篇文档解释当前版本父子块的运行链路：父块如何生成、子块如何生成、父块如何保存到 MySQL、子块如何保存到 Chroma，以及检索时如何从子块回到父块。

## 1. 迁移后，父块和子块分别在哪里

迁移前，父块正文保存在根目录的 parent_docstore.json 中，ParentDocstore 会在启动时把整个 JSON 加载到内存。

迁移后：

~~~text
父块正文和 metadata -> MySQL 的 parent_chunks 表
子块正文和 metadata -> Chroma collection
子块向量           -> Chroma 内部 HNSW 索引
文档状态和 child_ids -> MySQL 的 knowledge_documents 表
~~~

当前运行时使用 repositories/parent_chunk_repository.py 的 ParentChunkRepository。

rag/parent_docstore.py 仍然存在，但它是旧 JSON 实现，主要用于迁移兼容，不是当前运行时的父块存储。

## 2. 先区分四种数据

代码里的 parent_docs、parent_records、child_chunks、record 容易混淆，它们不是同一个东西。

### 2.1 parent_docs

parent_docs 是内存中的 LangChain Document 列表，是父块生成阶段的结果，还没有写入数据库。

~~~python
parent_docs = [
    Document(
        page_content="一段完整上下文",
        metadata={"page": 2, "chunk_type": "text"},
    )
]
~~~

### 2.2 parent_records

parent_records 是准备写入 MySQL 的普通 Python 字典：

~~~python
{
    "doc_id:gen:xxx:parent:000": {
        "page_content": "父块正文",
        "metadata": {
            "doc_id": "doc_id",
            "parent_id": "doc_id:gen:xxx:parent:000",
            "parent_index": 0,
        },
    }
}
~~~

它不是 ORM 对象，也不是子块，只是 ParentChunkRepository.save_batch() 的输入。

### 2.3 child_chunks

child_chunks 是准备写入 Chroma 的 LangChain Document 列表：

~~~python
Document(
    page_content="父块中的一小段文字",
    metadata={
        "doc_id": "doc_id",
        "parent_id": "doc_id:gen:xxx:parent:000",
        "child_id": "doc_id:gen:xxx:parent:000:child:000",
        "chunk_id": "doc_id:gen:xxx:parent:000:child:000",
    },
)
~~~

它会被 embedding，然后交给 Chroma.add_documents()。

### 2.4 ParentChunk ORM 对象

从 MySQL 查询出来的是 SQLAlchemy ORM 对象：

~~~python
chunk = session.get(ParentChunk, parent_id)
~~~

Repository 会把 ORM 对象转换为普通字典，再交给 Resolver：

~~~python
{
    "page_content": chunk.page_content,
    "metadata": dict(chunk.metadata_json or {}),
}
~~~

因此有两次边界转换：

~~~text
业务字典 -> 数据库 row：_record_to_row()
ORM 对象 -> RAG 字典：_as_record()
~~~

## 3. 父子块 ID 如何关联

当前 ID 使用文档、生成版本、父块序号、子块序号组成：

~~~text
doc_id:gen:<file_hash前12位>:parent:<父块序号>:child:<子块序号>
~~~

例如：

~~~text
parent_id = fedfa33ad2f8dce3:gen:9c3a1b2d4e5f:parent:003
child_id  = fedfa33ad2f8dce3:gen:9c3a1b2d4e5f:parent:003:child:001
~~~

子块 metadata 中的 parent_id 指向父块。当前 child_id 和 chunk_id 通常是同一个值，用来兼容旧代码。

## 4. DocumentChunker：只负责切分，不负责存储

文件：rag/document_chunker.py

DocumentChunker 把解析后的 Documents 转换为：

~~~text
父块
  -> parent_records
  -> child_chunks
~~~

它不直接操作 MySQL，也不直接操作 Chroma。

### 4.1 __init__()

构造函数读取 chroma.yml 的分块配置，准备 PDF、TXT、默认固定切分器、子块切分器和语义分块参数。

子块切分器类似：

~~~python
self.child_splitter = self._build_splitter(
    config.get("child_chunk_size", 120),
    config.get("child_chunk_overlap", 30),
)
~~~

当前策略：

~~~text
第一层：语义分块或 MinerU 结构化分块，生成父块
第二层：固定长度切分父块，生成子块
~~~

父块切成子块时不会再次做语义分块。语义分块发生在父块生成阶段，子块阶段使用固定长度切分。

### 4.2 initial_chunk_method()

这个函数只返回分块方式字符串，不负责实际切分：

~~~python
def initial_chunk_method(self, parent_child_enabled: bool) -> str:
    if parent_child_enabled:
        return (
            "parent_child_semantic"
            if self.semantic_enabled
            else "parent_child_fixed"
        )
    return "semantic" if self.semantic_enabled else "fixed"
~~~

它用于记录 knowledge_documents.chunk_method。实际使用 MinerU 结构化分块时，入库流程会把方式改成 mineru_structured。

### 4.3 build_parent_child_chunks()

这是父子块模式入口：

~~~python
child_chunks, child_ids, parent_ids, parent_records = (
    self.chunker.build_parent_child_chunks(...)
)
~~~

选择顺序：

~~~text
1. MinerU 结构化文档 -> _structured_split()
2. 开启语义分块     -> _semantic_split()
3. 语义失败或关闭    -> 固定长度 splitter
4. 得到父块后        -> _make_parent_child_chunks()
~~~

四个返回值：

| 返回值 | 用途 |
|---|---|
| child_chunks | 写入 Chroma |
| child_ids | Chroma 向量 ID |
| parent_ids | 更新或失败时清理父块 |
| parent_records | 写入 MySQL |

### 4.4 _make_parent_child_chunks()

这个函数同时准备两套数据。

先生成父块 ID：

~~~python
parent_id = f"{namespace}:parent:{parent_index:03d}"
~~~

再创建父块 metadata：

~~~python
parent_metadata = {
    "doc_id": doc_id,
    "parent_id": parent_id,
    "chunk_type": "parent",
    "parent_index": parent_index,
    "file_hash": file_hash,
    "source": file_path,
    "filename": filename,
    "file_type": file_type,
}
~~~

页码和 MinerU 字段也会保留，例如 page、mineru_type、table_id、mineru_structured。

准备 parent_records：

~~~python
parent_records[parent_id] = {
    "page_content": parent.page_content,
    "metadata": parent_metadata,
}
~~~

切分子块：

~~~python
if parent.metadata.get("chunk_type") in ("table", "equation"):
    children = [parent]
else:
    children = self.child_splitter.split_documents([parent]) or [parent]
~~~

表格和公式作为原子块，不被普通字符切分器切碎。

最后为每个子块补充 parent_id、child_id、chunk_id 等 metadata，并返回：

~~~python
return child_chunks, child_ids, parent_ids, parent_records
~~~

这里没有 child_records，不是遗漏：

~~~text
parent_records -> save_batch()        -> MySQL
child_chunks   -> add_documents(...)  -> Chroma
~~~

两个存储后端需要的输入格式不同。

## 5. ParentChunk ORM 模型

文件：models/knowledge.py

核心字段：

~~~python
class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    parent_id = mapped_column(String(128), primary_key=True)
    doc_id = mapped_column(String(64), nullable=False)
    parent_index = mapped_column(Integer, nullable=False)
    page_content = mapped_column(Text, nullable=False)
    metadata_json = mapped_column("metadata", JSON, default=dict)
    created_at = mapped_column(...)
    updated_at = mapped_column(...)
~~~

关键点：

~~~text
Python 属性：metadata_json
数据库列名：metadata
~~~

这样可以避免和 SQLAlchemy 模型基类中的 metadata 名称冲突。

## 6. ParentChunkRepository：父块数据库边界

文件：repositories/parent_chunk_repository.py

上层只需要调用：

~~~python
parent_repository.save_batch(parent_records)
parent_repository.get_many(parent_ids)
parent_repository.delete_by_doc_id(doc_id)
~~~

### 6.1 __init__()

~~~python
def __init__(self, session_factory=None):
    if session_factory is None:
        from config.db_conf import SyncSessionLocal
        session_factory = SyncSessionLocal
    self.session_factory = session_factory
~~~

正常运行不传 session_factory，因此使用 SyncSessionLocal 连接 MySQL。测试可以传 SQLite session factory，所以保留 SQLite 方言分支，但生产运行仍是 MySQL。

### 6.2 save() 和 save_batch()

save() 把单条数据包装成一个字典，复用 save_batch()：

~~~python
def save(self, parent_id, page_content, metadata):
    self.save_batch({
        parent_id: {
            "page_content": page_content,
            "metadata": metadata,
        }
    })
~~~

save_batch() 的流程：

~~~text
1. 空输入直接返回
2. _record_to_row() 转换业务字典
3. 根据数据库方言执行 upsert
4. commit；失败 rollback
~~~

upsert 的意思：

~~~text
parent_id 不存在 -> INSERT
parent_id 已存在 -> UPDATE
~~~

MySQL 使用 ON DUPLICATE KEY UPDATE，SQLite 测试分支使用 ON CONFLICT DO UPDATE。

### 6.3 _record_to_row()

这里的 record 就是 parent_records[parent_id] 的值：

~~~python
record = {
    "page_content": "父块正文",
    "metadata": {
        "doc_id": "doc_id",
        "parent_index": 0,
    },
}
~~~

它是普通 Python 字典，不是 ORM 对象。

函数检查 doc_id，读取 parent_index，补充时间，并组装成数据库 row：

~~~python
{
    "parent_id": parent_id,
    "doc_id": doc_id,
    "parent_index": int(parent_index),
    "page_content": record["page_content"],
    "metadata": metadata,
    "created_at": now,
    "updated_at": now,
}
~~~

### 6.4 mysql_insert(ParentChunk.__table__)

这句话表示：

~~~text
根据 ParentChunk 对应的 parent_chunks 表，
生成一条 MySQL INSERT 语句。
~~~

ParentChunk.__table__ 是 SQLAlchemy 的表对象，不是某一行数据。on_duplicate_key_update() 表示主键已存在时更新。

### 6.5 _merge_rows()

没有专用方言 upsert 时，走普通 ORM 逻辑：

~~~python
chunk = session.get(ParentChunk, row["parent_id"])

if chunk is None:
    session.add(ParentChunk(...))
else:
    chunk.page_content = row["page_content"]
    chunk.metadata_json = row["metadata"]
~~~

它是 Repository 的兼容兜底，不是另一套父子块业务流程。

### 6.6 _as_record()、get()、get_many()

_as_record() 把 ORM 对象转换成 Resolver 使用的字典：

~~~python
def _as_record(chunk: ParentChunk) -> dict:
    return {
        "page_content": chunk.page_content,
        "metadata": dict(chunk.metadata_json or {}),
    }
~~~

get() 查询一个父块。get_many() 使用 parent_id.in_(parent_ids) 一次查询多个父块，避免每个子块单独 get() 造成 N+1 查询。

get_many() 还会按输入 parent_ids 的顺序重排结果，尽量保留 rerank 的相关性顺序。

### 6.7 delete_many()、delete_by_doc_id()、count()

delete_many(parent_ids) 用于更新时删除旧版本父块。

delete_by_doc_id(doc_id) 用于删除整个文档的所有父块。

二者都复用 _delete_where()，统一处理删除、commit、rollback 和返回行数。

count() 只用于统计和检查，不参与正常召回。

## 7. VectorStoreService 如何连接两个存储

文件：rag/vector_store.py

初始化时：

~~~python
self.parent_docstore = ParentChunkRepository()
~~~

变量名仍叫 parent_docstore，是为了兼容旧调用方；实际对象已经是 MySQL Repository。

### 7.1 正常入库顺序

~~~text
1. 读取 PDF/TXT
2. 解析成 Documents
3. DocumentChunker 生成 parent_records 和 child_chunks
4. parent_records 写入 MySQL parent_chunks
5. child_chunks 写入 Chroma 并生成 embedding
6. knowledge_documents 发布 active 记录
7. 更新时清理旧 child_ids 和 parent_ids
8. 重建 BM25
~~~

对应调用关系：

~~~python
child_chunks, child_ids, parent_ids, parent_records = (
    self.chunker.build_parent_child_chunks(...)
)

self.parent_docstore.save_batch(parent_records)
self.vector_store.add_documents(child_chunks, ids=child_ids)

self.knowledge_repository.activate_document(
    doc_id=doc_id,
    chunk_ids=child_ids,
    parent_count=len(parent_ids),
    child_count=len(child_ids),
    ...,
)
~~~

### 7.2 _cleanup_staged_generation()

staged_child_ids 是本次新生成的 Chroma 子块 ID，staged_parent_ids 是本次新生成的 MySQL 父块 ID。

如果入库失败：

~~~python
self.vector_store.delete(ids=child_ids)
self.parent_docstore.delete_many(parent_ids)
~~~

它只清理本次失败的新数据，不碰旧 active 版本。

mark_failed() 只更新 knowledge_documents.status：

~~~text
_cleanup_staged_generation() -> 清理实体数据
mark_failed()                -> 更新文档状态
~~~

### 7.3 _delete_document_record()

这里的 record 是 KnowledgeDocument ORM 文档对象，不是 parent_records 字典。

删除过程：

~~~text
1. 标记 deleting
2. 按 chunk_ids 删除 Chroma 子块
3. 按 doc_id 删除 MySQL 父块
4. 可选删除原始文件
5. 删除 knowledge_documents 记录
6. 可选重建 BM25
~~~

因此项目中的 record 要结合上下文看：

~~~text
_delete_document_record(record)
    record = KnowledgeDocument ORM 对象

save_batch(records)
    records = parent_id -> 父块普通字典
~~~

## 8. ParentChildResolver：从子块回到父块

文件：rag/parent_child_retriever.py

总链路：

~~~text
Chroma/BM25 召回子块
        |
        v
ParentChildResolver.resolve()
        |
        v
ParentChunkRepository.get_many()
        |
        v
返回去重后的父块 Document
~~~

### 8.1 ParentStoreProtocol

~~~python
class ParentStoreProtocol(Protocol):
    def get(self, parent_id: str) -> dict | None:
        ...
~~~

它定义父块存储的最小接口，使 Resolver 不依赖具体数据库。当前 Repository 额外提供 get_many()，所以正常运行优先使用批量查询。

### 8.2 resolve() 的八步

~~~text
1. 空结果直接返回
2. 先过滤无权限子块
3. 没有 parent_id 时返回普通块
4. 收集并去重 parent_ids
5. 批量从 MySQL 取父块
6. 按子块相关性顺序重建父块 Document
7. 回表后再次检查权限
8. 去重并截断到 top_k_parents
~~~

没有 parent_id 时，Document 本身就是最终结果，用于兼容普通块模式或旧数据。

一个父块可能命中多个子块，Resolver 按命中顺序去重：

~~~python
parent_ids = []
seen_parent_ids = set()

for child in child_docs:
    parent_id = (child.metadata or {}).get("parent_id")
    if parent_id and parent_id not in seen_parent_ids:
        seen_parent_ids.add(parent_id)
        parent_ids.append(parent_id)
~~~

### 8.3 批量回表和重建 Document

Resolver 优先调用 get_many()。回表后重建新的 LangChain Document：

~~~python
parent_meta = dict(record.get("metadata") or {})
parent_meta["match_child_id"] = child_meta.get(
    "child_id",
    child_meta.get("chunk_id"),
)
parent_meta["chunk_id"] = parent_id

results.append(
    Document(
        page_content=record["page_content"],
        metadata=parent_meta,
    )
)
~~~

含义：

~~~text
子块 parent_id     -> 决定查哪个父块
父块 page_content  -> 成为最终上下文正文
子块 child_id      -> 记录命中来源
~~~

父块回表后还要再次检查 allowed_doc_ids，因为父块正文马上要进入 LLM。最后使用 emitted_parent_ids 去重，并限制 top_k_parents。

## 9. RagSummarizeService：子块召回，父块汇总

文件：rag/rag_service.py

当前流程：

~~~text
query
  -> vector_store.retrieve()
  -> 候选子块
  -> reranker.rerank(query, child_docs)
  -> ParentChildResolver.resolve(scored_children)
  -> 父块正文交给 LLM
~~~

初始化 Resolver：

~~~python
self.parent_resolver = ParentChildResolver(
    parent_docstore=self.vector_store.parent_docstore,
    top_k_parents=self.final_k,
)
~~~

设计思想：

~~~text
向量/BM25 负责粗排子块
CrossEncoder 负责精排子块
Resolver 根据子块顺序选择父块
MySQL 提供父块完整正文
~~~

所以这是“子块负责命中和排序，父块负责提供上下文”。

## 10. 更新、失败和删除

更新会使用新的 generation namespace：

~~~text
旧版本：doc_id:gen:old_hash:parent:000:child:000
新版本：doc_id:gen:new_hash:parent:000:child:000
~~~

更新顺序：

~~~text
1. 计算新文件 SHA-256
2. 生成新父块和子块
3. 写新父块到 MySQL
4. 写新子块到 Chroma
5. knowledge_documents 指向新的 child_ids，并标记 active
6. 删除旧 child_ids
7. 删除旧 parent_ids
8. 重建 BM25
~~~

入库失败时，先清理本次新生成的父块和子块，再标记 failed。旧 active 版本不会被误删。

删除文档时清理：

~~~text
1. Chroma 中的 child_ids
2. MySQL parent_chunks 中 doc_id 对应的父块
3. MySQL knowledge_documents 文档记录
4. 可选删除原始文件
5. 重建 BM25
~~~

## 11. 为什么父块不直接放 Chroma

父块和子块职责不同：

| 对象 | 主要目的 | 存储 |
|---|---|---|
| 子块 | 短文本匹配、向量召回、BM25 召回 | Chroma |
| 父块 | 完整上下文、结构保留、回表 | MySQL |

父块通常较长，可能包含多个主题；子块更短，更适合匹配查询。

~~~text
子块：适合搜索
父块：适合阅读和生成
~~~

## 12. 推荐阅读顺序

1. models/knowledge.py 的 ParentChunk：看 MySQL 一行父块结构。
2. rag/document_chunker.py 的 build_parent_child_chunks()：看入口。
3. _make_parent_child_chunks()：看 ID、metadata 和两份输出。
4. repositories/parent_chunk_repository.py 的 _record_to_row()：看业务字典转数据库 row。
5. save_batch()：看 MySQL upsert。
6. rag/vector_store.py 的入库调用：看两个存储如何连接。
7. rag/rag_service.py 的 retriever_docs()：看召回、重排和回父块。
8. rag/parent_child_retriever.py 的 resolve()：看批量查询、去重和权限检查。
9. vector_store.py 的失败清理和删除逻辑：看更新和删除。

## 13. 一句话总结

~~~text
父块保存完整上下文，子块负责命中查询；
子块写入 Chroma，父块写入 MySQL；
RAG 先召回并重排子块，再按 parent_id 批量回表，
最后把去重后的父块交给 LLM。
~~~

