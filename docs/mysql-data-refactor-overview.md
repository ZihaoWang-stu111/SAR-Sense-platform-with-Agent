# 数据存储大改造说明

## 1. 改造目标

这次改造的核心不是把所有数据都塞进 MySQL，而是把**会变化、需要查询、需要保持一致的业务状态**从本地 JSON 和进程内存中移出来。

改造后形成了三个明确的存储职责：

```text
MySQL
  负责知识库元数据、父块正文、指标事件，以及原有的用户和对话数据

ChromaDB
  负责子块向量和子块检索所需的 metadata

文件系统
  负责原始 PDF/TXT 文件，以及知识库文件的版本化副本
```

其中，MySQL 是知识库运行状态的主数据源，ChromaDB 仍然是向量索引，不负责保存完整的知识库业务状态。

---

## 2. 改造前：JSON、Chroma 和内存各自维护一部分状态

### 2.1 知识库状态

改造前，一份知识库文件通常同时出现在三个位置：

```text
manifest.json
  记录 filename、doc_id、file_hash、chunk_ids、chunk_count 等文件级信息

parent_docstore.json
  记录 parent_id 对应的父块正文和 metadata

chroma_db/
  记录子块正文、metadata 和 embedding
```

一次入库大致是：

```text
扫描 data/
  -> 读取文件
  -> 分块
  -> 写 parent_docstore.json
  -> 写入 Chroma 子块和向量
  -> 重写 manifest.json
  -> 在内存中重建 BM25
```

这种方式能跑通 RAG，但有几个工程问题：

1. JSON 是整文件读写，追加或更新一个文件可能重写整个文件。
2. 文件元数据、父块和向量不在同一个结构化数据源中。
3. 进程中断时，JSON、Chroma 和 BM25 可能处于不同状态。
4. 业务代码、RAG 代码都可能直接操作存储细节。
5. 指标只在 `AgentMetrics` 单例的内存中，进程重启后统计归零。
6. 更新文件时，旧向量和新向量的替换过程没有清晰的“正在处理、已完成、失败”状态。

### 2.2 父子块检索

改造前，父块正文由 JSON 文件提供：

```python
record = parent_docstore.get(parent_id)
```

`ParentChildResolver` 根据子块 metadata 中的 `parent_id` 查询 JSON，然后把父块正文重新组装成 `Document`。

### 2.3 指标

改造前的调用链是：

```text
工具调用 / LLM 调用
  -> AgentMetrics 单例
  -> 修改内存计数器
  -> /api/metrics 直接读取当前进程内存
```

因此，重启服务、换 worker 或使用另一个进程后，历史指标不可见。

---

## 3. 改造后：MySQL 成为结构化状态中心

最终运行结构如下：

```text
                         +----------------------+
                         | knowledge_documents |
                         | 文档级状态、chunk ID |
                         +----------+-----------+
                                    |
                          KnowledgeRepository
                                    |
       +----------------------------+----------------------------+
       |                            |                            |
       v                            v                            v
  ChromaDB                    parent_chunks                BM25 cache
  子块向量                    父块正文与 metadata             运行时索引
```

指标链路则变成：

```text
工具 / LLM / 对话结束
  -> AgentMetrics
      -> 保留内存实时统计
      -> MetricsRepository
          -> metric_events
  -> /api/metrics 从 MySQL 聚合历史数据
```

这里没有把 ChromaDB 替换成 MySQL。向量检索仍然由 ChromaDB 完成，MySQL 负责告诉系统哪些文档和哪些 chunk 当前有效，以及如何取回父块正文。

---

## 4. 数据模型发生了什么变化

### 4.1 `knowledge_documents`：一行代表一个逻辑文档

对应模型是 [`models/knowledge.py`](../models/knowledge.py) 中的 `KnowledgeDocument`。

主要字段可以分成五组：

```text
身份字段
  id、doc_id、filename、file_hash、file_type

文件字段
  storage_key

分块字段
  chunk_method、chunk_ids、chunk_count、parent_count、child_count

权限和状态字段
  allowed_roles、status、updated_by、error_message

时间字段
  created_at、updated_at、ingested_at
```

其中最重要的是：

- `doc_id` 标识逻辑文档。
- `file_hash` 用于判断内容是否变化和识别重复文件。
- `chunk_ids` 记录当前有效版本对应的 Chroma 子块 ID。
- `status` 表示 `processing`、`active`、`failed` 或 `deleting` 等生命周期状态。
- `storage_key` 记录原始文件或版本文件在 `data/` 下的位置。
- `parent_count` 和 `child_count` 让 API 不必重新扫描向量库就能展示统计信息。
- `allowed_roles` 保存文件级角色权限。

因此，文件列表不再依赖扫描 `data/` 或解析 `manifest.json`，而是查询 MySQL 中 `status == "active"` 的记录。

### 4.2 `parent_chunks`：一行代表一个父块

对应模型仍在 [`models/knowledge.py`](../models/knowledge.py) 中，类名是 `ParentChunk`。

```text
parent_id       父块主键
doc_id          所属逻辑文档
parent_index    父块在文档中的顺序
page_content    父块完整正文
metadata        父块 metadata，JSON 列
created_at
updated_at
```

这里的 `page_content` 使用 MySQL 的 `MEDIUMTEXT` 变体，避免长父块受到普通短文本类型限制。

父块正文进入 MySQL 后，`parent_docstore.json` 不再是运行时父块存储。

### 4.3 `metric_events`：一行代表一次指标事件

对应模型是 [`models/metrics.py`](../models/metrics.py) 中的 `MetricEvent`。

```text
event_type
  tool_call、llm_call、conversation_timing

tool_name
  工具调用事件对应的工具名

success
  工具是否成功

duration_ms
  工具耗时或整轮对话耗时

user_id
  记录事件来源用户；当前展示仍按全局聚合

created_at
  事件发生时间
```

指标不再只保存最终数字，而是保存事件。这样可以在需要时重新聚合总调用次数、成功率、平均耗时和最近调用记录。

---

## 5. 数据库连接：为什么同时有异步和同步两套 Session

改造后的数据库配置在 [`config/db_conf.py`](../config/db_conf.py)。

```python
ASYNC_DATABASE_URL = URL.create("mysql+aiomysql", **_DATABASE_OPTIONS)
SYNC_DATABASE_URL = URL.create("mysql+pymysql", **_DATABASE_OPTIONS)
```

### 异步连接

`AsyncSessionLocal` 和 `get_db()` 服务于 FastAPI 请求：

```text
HTTP 请求
  -> Depends(get_db)
  -> AsyncSession
  -> crud/
  -> commit / rollback / close
```

用户、对话、ACL 等请求级 CRUD 保持异步，是因为它们运行在 FastAPI 的异步路由中。

### 同步连接

`SyncSessionLocal` 服务于 RAG、BM25 和指标这些同步调用链：

```text
RAG / Agent 工具 / 指标收集
  -> Repository
  -> SyncSessionLocal
  -> SQLAlchemy ORM
  -> MySQL
```

这样做不是重复建数据库，而是让不同执行模型使用匹配的 Session 生命周期：

- API 请求使用 `AsyncSession`。
- 同步 RAG 和同步指标代码使用 `Session`。
- 两者连接到同一个 MySQL 数据库和同一组表。

---

## 6. Repository 分层：数据库操作从业务代码中抽离

最终保留的 Repository 有三个：

```text
repositories/
  knowledge_repository.py
  parent_chunk_repository.py
  metrics_repository.py
```

统一依赖方向是：

```text
API / Agent / RAG
        -> Repository
        -> SQLAlchemy ORM Model
        -> Session
        -> MySQL
```

### 6.1 `KnowledgeRepository`

[`repositories/knowledge_repository.py`](../repositories/knowledge_repository.py) 负责 `knowledge_documents`。

它提供几类能力：

```python
get_by_doc_id(doc_id)
get_by_filename(filename)
get_by_hash(file_hash)
list_active()
```

这些方法统一了文档定位方式，不再让 `vector_store.py` 到处手写 SQL 或直接解析 JSON。

入库生命周期由这些方法表达：

```python
begin_ingestion(...)  # 新文档进入 processing
activate_document(...)  # 新版本完整生成后发布为 active
mark_failed(...)      # 失败记录错误
mark_deleting(...)    # 删除前标记 deleting
delete(doc_id)        # 删除文档元数据
```

它还提供两个给检索层使用的派生能力：

```python
active_chunk_ids()
fingerprint()
```

`active_chunk_ids()` 返回所有 active 文档当前版本的子块 ID，供 Chroma 和 BM25 过滤。

`fingerprint()` 根据 active 文档的身份、hash、版本位置、chunk 列表和计数等信息生成知识库指纹，供 BM25 缓存判断是否过期。

### 6.2 `ParentChunkRepository`

[`repositories/parent_chunk_repository.py`](../repositories/parent_chunk_repository.py) 负责 `parent_chunks`。

主要接口是：

```python
save(parent_id, page_content, metadata)
save_batch(records)
get(parent_id)
get_many(parent_ids)
delete_many(parent_ids)
delete_by_doc_id(doc_id)
count()
```

`save_batch()` 使用 MySQL upsert，同一个 `parent_id` 重复写入时更新已有记录。`get_many()` 一次查询多个父块，避免检索命中多个子块时产生 N+1 查询。

`ParentChildResolver` 只依赖父块存储接口，不关心底层是 JSON 还是 MySQL。因此最终运行时只需要把实现替换成 `ParentChunkRepository`，上层父子块逻辑不需要重写。

### 6.3 `MetricsRepository`

[`repositories/metrics_repository.py`](../repositories/metrics_repository.py) 负责 `metric_events`。

它把指标操作统一为：

```python
record_event(...)
aggregate(limit=50)
delete_all()
reset(memory_reset_callback)
```

`aggregate()` 在数据库层完成：

- 对话轮数统计。
- 工具总调用次数。
- 工具成功次数和成功率。
- 平均工具耗时。
- 平均响应时间。
- LLM 调用次数。
- 最近工具调用记录。

这样 API 不需要先把全部事件拉到 Python 内存，再自己做统计。

---

## 7. 知识库入库：从“直接覆盖”变成“生成后发布”

### 7.1 上传文件先进入版本目录

知识库 API 在 [`api/routers/knowledge.py`](../api/routers/knowledge.py) 中先把上传文件复制到：

```text
data/.knowledge_versions/<随机版本目录>/<filename>
```

数据库中的 `storage_key` 指向这个版本文件。这样新文件生成期间不会直接覆盖旧文件。

API 还使用知识库写锁，避免多个上传、删除或权限更新同时修改同一个知识库状态。

### 7.2 `VectorStoreService` 负责编排，不负责直接管理散落的 JSON

[`rag/vector_store.py`](../rag/vector_store.py) 仍然负责：

- 读取 PDF/TXT。
- 语义分块或固定分块。
- 父子块构建。
- 生成 embedding。
- 写入 Chroma。
- 调用 Repository 保存 MySQL 状态。

它不再把 manifest 和父块 JSON 当作主存储，而是初始化三个关键组件：

```python
self.knowledge_repository = KnowledgeRepository()
self.hybrid_engine = DynamicHybridRetriever(
    ...,
    knowledge_repository=self.knowledge_repository,
)
self.parent_docstore = ParentChunkRepository()
```

`get_vector_store_service()` 仍返回全局共享的 `VectorStoreService`，因此 API 上传、删除和 Agent 检索使用同一套 Chroma、BM25 和 Repository 依赖。

### 7.3 文件识别和去重

每个待入库文件都会先计算 hash，然后查询 MySQL：

```text
按 filename 找已有文档
按 file_hash 找相同内容
```

结果分为三类：

```text
同名且 hash 相同
  -> same，跳过 embedding

不同名但 hash 相同
  -> duplicate，跳过 embedding

同名但 hash 不同
  -> updated，保留原 doc_id，生成新版本

完全没有对应记录
  -> new，使用新的 doc_id 入库
```

这解决了“文件内容没有变化却重复 embedding”的问题，也保留了逻辑文档 ID 的稳定性。

### 7.4 生成新版本

每次新入库或更新都会创建一个带 generation 命名空间的 chunk ID。概念上类似：

```text
<doc_id>:gen:<hash片段>:parent:<index>:child:<index>
```

父块的 ID 再派生出子块 ID，子块 metadata 中保存：

```text
doc_id
parent_id
child_id
chunk_id
parent_index
child_index
filename
file_hash
source
page 等字段
```

这样旧版本和新版本的 chunk ID 不会互相覆盖。

### 7.5 父子块入库顺序

开启父子块模式时，入库过程是：

```text
原始文档
  -> 父块切分
  -> 父块进一步切成子块
  -> 父块正文和 metadata 批量写入 parent_chunks
  -> 子块和 embedding 写入 Chroma
  -> knowledge_documents 写入当前有效 chunk_ids、数量和状态
```

表格和公式等结构化块可以作为原子块保留，不再被普通 child splitter 拆散；普通文本块则按照当前配置进行父子切分。

### 7.6 发布和失败处理

新文档的状态变化是：

```text
不存在
  -> processing
  -> active
```

如果生成失败：

```text
processing
  -> failed
```

并清理本次已经写入的临时 Chroma 子块和父块。

更新已有文档时，旧版本不会先被删除：

```text
旧版本 active
  -> 生成新父块、新子块和新 embedding
  -> 新版本完整后更新 knowledge_documents 为 active
  -> 再清理旧子块和旧父块
```

因此，更新过程中即使 embedding 或解析失败，旧版本仍然可以继续被检索。旧版本清理失败时可能留下孤儿数据，但它们不属于 active chunk ID，不会被正常检索到。

这就是本次改造中最重要的可靠性变化：从“先删旧数据再写新数据”变成“先生成新数据，再切换 active 指针”。

---

## 8. 删除文件：从删除 JSON 记录变成跨存储清理

删除入口仍然可以按文件名兼容调用，但 API 主要按稳定的 `doc_id` 删除。

最终链路是：

```text
DELETE /api/knowledge/files/{doc_id}
  -> 获取 MySQL 文档记录
  -> KnowledgeRepository.mark_deleting()
  -> 按 chunk_ids 删除 Chroma 子块
  -> ParentChunkRepository.delete_by_doc_id()
  -> 可选删除原始版本文件
  -> KnowledgeRepository.delete()
  -> 重建 BM25
```

删除顺序的意义是：先让文档进入 `deleting`，再清理外部存储，最后删除 MySQL 主记录。

原来 `manifest.json` 里的 `chunk_ids` 由 `KnowledgeDocument.chunk_ids` 替代；原来 `parent_docstore.json` 中按 `doc_id` 删除父块的逻辑由 `ParentChunkRepository.delete_by_doc_id()` 替代。

---

## 9. 混合检索：MySQL 参与“有效性判断”，Chroma/BM25 负责“相关性判断”

### 9.1 启动时构建或加载 BM25

[`rag/hybrid_retriever.py`](../rag/hybrid_retriever.py) 不再根据 manifest 文件 hash 判断 BM25 是否新鲜，而是优先使用 `KnowledgeRepository.fingerprint()`。

```text
启动 DynamicHybridRetriever
  -> 查询 MySQL active 文档
  -> 生成知识库 fingerprint
  -> BM25 pkl 存在且 fingerprint 一致
       -> 直接加载
     fingerprint 不一致或加载失败
       -> 从 Chroma 重建 BM25
```

BM25 的 pkl 仍然存在，但它是可重建的运行时缓存，不是知识库事实来源。

### 9.2 active chunk 过滤

`KnowledgeRepository.active_chunk_ids()` 返回当前所有 active 文档的 chunk ID。

BM25 从 Chroma 构建时会过滤掉不在 active 集合中的子块：

```python
if metadata.get("chunk_id") in active_chunk_ids:
    docs.append(document)
```

向量检索则向 Chroma 传入 filter：

```python
{
    "chunk_id": {"$in": active_chunk_ids}
}
```

因此，旧版本、失败版本或删除后残留的 Chroma 向量不会正常进入召回结果。

### 9.3 ACL 过滤仍覆盖两条检索通道

如果当前用户有允许访问的文档 ID，向量检索会叠加：

```python
{"doc_id": {"$in": allowed_doc_ids}}
```

BM25 则使用 `FilteredBM25Retriever` 对 BM25 的候选文档再次过滤。也就是说，`EnsembleRetriever` 仍然可以使用，只是它接收的是已经具备 active/ACL 过滤能力的两个检索器。

### 9.4 RAG 处理顺序

[`rag/rag_service.py`](../rag/rag_service.py) 的最终流程是：

```text
query
  -> DynamicHybridRetriever
       -> Chroma 子块召回
       -> BM25 子块召回
       -> Ensemble 合并
  -> CrossEncoder 对子块 rerank
  -> ParentChildResolver 按 rerank 顺序回查父块
  -> 取前 final_k 个父块
  -> 拼接 context
  -> LLM 生成回答
  -> 追加参考来源
```

`ParentChildResolver` 使用 `ParentChunkRepository.get_many()` 批量回表，并且保留子块 rerank 后的父块顺序。回表后还会再次检查父块 metadata 中的 `doc_id`，避免无权限父块被带入最终上下文。

因此，MySQL 并没有替代向量召回，而是在召回链路中承担了三件事：

1. 判断哪些 chunk 属于当前 active 版本。
2. 判断哪些文档属于当前用户可见范围。
3. 根据子块命中结果取回完整父块正文。

---

## 10. 指标链路：从内存计数变成内存实时值加数据库历史值

[`agent/metrics_collector.py`](../agent/metrics_collector.py) 中的 `AgentMetrics` 没有被完全删除，而是变成了两层结构：

```text
AgentMetrics
  -> 内存计数器：用于当前进程内快速展示和兼容原有接口
  -> MetricsRepository：把每个事件写入 MySQL
```

工具中间件负责记录工具调用：

```text
@wrap_tool_call
  -> 工具开始时间
  -> 执行工具
  -> 记录成功/失败、工具名、耗时、user_id
```

模型调用中间件负责记录 LLM 调用：

```text
@before_model
  -> record_llm_call(user_id=...)
```

聊天 SSE 生成器负责记录整轮对话时间：

```text
start_conversation()
  -> agent 执行
  -> end_conversation(...)
  -> 写入 conversation_timing
```

如果数据库写指标失败，`AgentMetrics` 会记录 warning，但内存计数不会回滚，主问答流程也不会因为指标写入失败而失败。

### 指标 API 的变化

改造前：

```text
/api/metrics
  -> 读取 AgentMetrics 当前内存
```

改造后：

```text
/api/metrics
  -> MetricsRepository.aggregate()
  -> MySQL 聚合 metric_events
  -> 返回原有 API 结构
```

所以重启服务后，历史指标仍然存在。当前指标页面展示的是所有用户的全局聚合数据，还没有按用户隔离展示。

管理员执行 reset 时，会同时清理 MySQL 指标事件和内存计数器。

---

## 11. API 响应层的最后标准化

最后一个“标准化”提交没有改变数据库存储逻辑，而是规范了知识库 API 的输出。

以前 `api/routers/knowledge.py` 手动拼接字典：

```python
payload = {
    "name": document.filename,
    "doc_id": document.doc_id,
    ...
}
```

现在 [`schemas/knowledge.py`](../schemas/knowledge.py) 定义了：

```text
KnowledgeDocumentResponse
KnowledgeFilesResponse
UpdateDocumentPermissionsResponse
```

并通过：

```python
model_config = ConfigDict(from_attributes=True)
```

让 Pydantic 能从 SQLAlchemy ORM 对象生成响应。

`_to_document_response()` 只负责补充接口计算字段：

```text
can_manage
allowed_roles 是否向当前用户暴露
```

`can_manage` 不是数据库列，它是响应层字段；数据库只保存 `allowed_roles`。普通用户可以看到文件基本信息，但不会看到完整权限配置。

这个变化的意义是：数据库模型、接口响应模型和路由处理职责分开，后续修改 ORM 字段时不必继续手动维护大段字典拼装代码。

---

## 12. JSON 数据是怎么迁到 MySQL 的

这次迁移不是把 JSON 文件简单塞进一个大字段，而是按照最终模型拆成结构化记录。

### 12.1 `manifest.json` 到 `knowledge_documents`

原来的 manifest 每个文件条目映射为一行 `KnowledgeDocument`：

```text
manifest 的文件名 key
  -> filename

entry.doc_id
  -> doc_id

entry.file_hash
  -> file_hash

entry.chunk_ids
  -> chunk_ids JSON 列

entry.chunk_count
  -> chunk_count

entry.chunk_method
  -> chunk_method

entry.file_type / status / ingested_at
  -> 对应 MySQL 字段
```

父子块相关的父块数量、子块数量和当前权限也写入对应字段。

### 12.2 `parent_docstore.json` 到 `parent_chunks`

原来的 JSON 结构类似：

```json
{
  "parent_id": {
    "page_content": "父块正文",
    "metadata": {
      "doc_id": "...",
      "parent_index": 0
    }
  }
}
```

迁移后变成：

```text
parent_id
  -> parent_chunks.parent_id

metadata.doc_id
  -> parent_chunks.doc_id

metadata.parent_index
  -> parent_chunks.parent_index

page_content
  -> parent_chunks.page_content

metadata
  -> parent_chunks.metadata JSON 列
```

### 12.3 Chroma 不做同样的迁移

Chroma 中已经存在的子块向量不需要转换成 MySQL 行，也不需要因为这次数据层改造重新 embedding。

迁移只补齐 MySQL 的文档元数据和父块数据，然后通过 `chunk_ids` 把 MySQL 文档记录与现有 Chroma 子块关联起来。

后续新入库才使用版本化 generation ID，并由新的入库流程同时写 MySQL 和 Chroma。

### 12.4 旧 JSON 的最终角色

```text
manifest.json
parent_docstore.json
  -> 历史迁移来源和备份
  -> 不再作为运行时主写入目标
```

旧 JSON 父块实现已经移除，运行时父块查询只走 [`repositories/parent_chunk_repository.py`](../repositories/parent_chunk_repository.py)。

---

## 13. 文件职责总览

| 文件 | 改造后的职责 |
|---|---|
| [`config/db_conf.py`](../config/db_conf.py) | 创建异步和同步 MySQL Engine/Session |
| [`models/knowledge.py`](../models/knowledge.py) | 定义文档元数据和父块 ORM |
| [`models/metrics.py`](../models/metrics.py) | 定义指标事件 ORM |
| [`repositories/knowledge_repository.py`](../repositories/knowledge_repository.py) | 管理文档状态、active 版本、chunk ID、指纹 |
| [`repositories/parent_chunk_repository.py`](../repositories/parent_chunk_repository.py) | 保存、批量查询和删除父块 |
| [`repositories/metrics_repository.py`](../repositories/metrics_repository.py) | 写入和聚合指标事件 |
| [`rag/vector_store.py`](../rag/vector_store.py) | 编排解析、分块、embedding、Chroma 和入库生命周期 |
| [`rag/hybrid_retriever.py`](../rag/hybrid_retriever.py) | Chroma + BM25 混合召回，以及 active/ACL 过滤 |
| [`rag/parent_child_retriever.py`](../rag/parent_child_retriever.py) | 子块命中后批量回查父块并保持排序 |
| [`rag/rag_service.py`](../rag/rag_service.py) | rerank、父块组装、上下文拼接和 RAG 回答 |
| [`agent/metrics_collector.py`](../agent/metrics_collector.py) | 保留内存指标，同时持久化事件 |
| [`api/routers/knowledge.py`](../api/routers/knowledge.py) | 上传、列出、下载、修改权限和删除知识库文件 |
| [`api/routers/metrics.py`](../api/routers/metrics.py) | 从 Repository 获取历史指标并处理管理员重置 |
| [`schemas/knowledge.py`](../schemas/knowledge.py) | 定义知识库 API 请求和响应模型 |
中间方案中的 `KnowledgeStore`、`MetricsStore` 和 `MySQLParentDocstore` 已分别收敛为三个 Repository，不再保留两套长期并存的命名和入口。

---

## 14. 这次改造解决了什么，以及还保留什么边界

### 已解决

1. 知识库文件状态有了结构化数据库主记录。
2. 父块正文从 JSON 文件变成 MySQL 可查询数据。
3. 指标在进程重启后仍然可追溯。
4. RAG 不会召回已删除、失败或旧版本的 active 之外 chunk。
5. 更新文件采用“新版本生成完成后再发布”的方式，降低半成品数据被检索的风险。
6. 数据库操作集中到 Repository，API、RAG 和指标逻辑不再各自拼 SQL。
7. BM25 缓存可以根据数据库知识库指纹自动判断是否需要重建。
8. 父块回查支持批量查询，避免每个命中子块单独查一次数据库。

### 有意保留的边界

1. Chroma 仍然是本地持久化向量库，MySQL 不保存 embedding。
2. BM25 pkl 仍然是运行时缓存，不是主数据源。
3. MySQL、Chroma 和文件系统之间没有一个跨存储事务。
4. 旧数据清理属于维护动作，强制停止时可能短暂留下孤儿向量、父块或版本文件。
5. active chunk ID 过滤保证孤儿数据不会被正常召回，但不会自动抹除所有孤儿文件。
6. 指标虽然带有 `user_id`，当前展示接口仍然做全局聚合。
7. 根目录若仍有旧 JSON 文件，它们只是本地备份，不参与运行；兼容实现和一次性迁移代码已移除。

---

## 15. 一句话总结

这次改造把系统从：

```text
JSON 记录文件状态 + JSON 保存父块 + Chroma 保存向量 + 内存保存指标
```

变成了：

```text
MySQL 保存结构化业务状态
Chroma 保存子块向量
Repository 隔离数据库访问
VectorStoreService 编排入库生命周期
RAG 根据 MySQL 的 active 状态和权限过滤 Chroma/BM25
AgentMetrics 同时提供实时内存统计和可恢复的历史事件
```

核心变化不是“把 JSON 换成 SQL”这么简单，而是把**谁是当前有效版本、哪些 chunk 可以被检索、父块正文在哪里、指标如何恢复**这些运行规则正式变成了可查询、可验证的业务状态。
