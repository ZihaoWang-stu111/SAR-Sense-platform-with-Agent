# RAG 角色权限控制 + 多租户知识库隔离 — 实施方案

> 本方案基于对 SAR-Sense 全量源码扫描得出，所有事实结论均带 `file:line` 引用，可逐条核对。
> 调研方式：6 个子系统并行扫描（3 个由多智能体工作流完成：RAG 存储 / RAG 检索 / 认证数据库；3 个由人工补齐：知识库 API / Agent 身份传递 / 评测与部署）+ 三套候选方案对抗评审。
> 注：多智能体工作流因 API 账户余额不足（403）在中途中断，已完成的高质量扫描结果被保留并采纳，缺失部分由人工直接读源码补齐，最终方案由人工综合得出。

---

## 1. 背景与目标

### 1.1 现状一句话

SAR-Sense 当前是**单租户、全局共享知识库**：所有用户上传的文档进入同一个 Chroma collection（`agent_prjct`）、同一份 `parent_docstore.json`、同一份 `manifest.json`、同一个 BM25 内存索引；检索链路（Agent 的 `rag_summarize` 工具 → `RagSummarizeService` → `DynamicHybridRetriever`）**完全没有用户/租户概念**；RBAC 仅在路由层硬编码 `username == "admin"`。

### 1.2 目标

1. **多租户知识库隔离**：不同租户的文档互不可见（上传、检索、删除均按租户隔离）。
2. **角色权限控制（RBAC）**：把硬编码的 `username == "admin"` 升级为基于角色的权限模型（admin / editor / viewer），知识库写操作按角色 + 归属双重判定。
3. **Agent 检索链路也要隔离**：不仅 HTTP 接口要过滤，Agent 工具 `rag_summarize` 调用 RAG 时也要带当前用户身份，否则等于没隔离。
4. **向后兼容**：eval 评测管线（`eval/`、`eval_rag25/`）和存量测试不应被破坏；存量数据可迁移。
5. **工程正确性**：尊重现有不变量（BM25 共享 Document 并发契约、reranker 纯函数、单例架构、同步/异步边界）。

### 1.3 部署约束

- 单机演示 / 面试展示项目，MySQL 单库，单 worker（[docker-compose.yml:35-39](docker-compose.yml#L35)），用户量小。
- 但方案要体现工程正确性与可扩展性，便于面试讲述取舍。

---

## 2. 现状全量扫描结论

### 2.1 RAG 存储与入库链路

- 用 `langchain_chroma.Chroma` 封装（非原生 chromadb client），**单 collection** `agent_prjct`，`persist_directory=chroma_db`：[rag/vector_store.py:33-38](rag/vector_store.py#L33)
- 入库 chunk metadata 在 `_enrich_chunks`（非父子）与 `_build_parent_child_chunks`（父子）集中注入，字段有 `doc_id / chunk_id / file_hash / source / filename / file_type / chunk_index / page`，**无任何 tenant/owner 字段**：[rag/vector_store.py:215-302](rag/vector_store.py#L215)
- `doc_id = file_hash[:16]`（纯内容哈希）：[rag/vector_store.py:396](rag/vector_store.py#L396)；`parent_id = f'{doc_id}:parent:{parent_index:03d}'`（[:254](rag/vector_store.py#L254)）；`child_id` 同理（[:280](rag/vector_store.py#L280)）—— 全局内容去重前提下唯一，多租户放开后会**跨租户碰撞**。
- `manifest.json` 顶层以 `filename` 为 key（全局唯一），entry 字段 `doc_id / file_hash / chunk_count / chunk_ids / chunk_method / file_type / ingested_at / status`，父子模式追加 `parent_ids / parent_count / child_count`：[utils/file_handler.py:65-84](utils/file_handler.py#L65)
- 去重逻辑 `check_file_status`：filename 不在 manifest 但 file_hash 与已有条目相同 → `DUPLICATE` 跳过：[utils/file_handler.py:51-62](utils/file_handler.py#L51) —— **两个租户上传相同文件，后者会被静默跳过**。
- `ParentDocstore` 是单个 JSON 文件的扁平 dict，`key=parent_id`，`value={page_content, metadata}`，metadata 是自由 dict **可携带任意字段**：[rag/parent_docstore.py:33-50](rag/parent_docstore.py#L33)
- 入库入口 `load_document()` 是**无参全目录扫描** `listdir_with_allowed_type(data_path)`，没有 per-file 上传者上下文：[rag/vector_store.py:344-359](rag/vector_store.py#L344)
- `VectorStoreService` 是双检锁全局单例：[rag/vector_store.py:15-28](rag/vector_store.py#L15)

### 2.2 RAG 检索链路

- 检索入口：`RagSummarizeService.retriever_docs(query)` → `get_retriever(query).invoke(query)`：[rag/rag_service.py:37-50](rag/rag_service.py#L37)
- `DynamicHybridRetriever.get_retriever`：向量侧 `as_retriever(search_kwargs={'k': self.k})` **未用任何 filter**：[rag/hybrid_retriever.py:235](rag/hybrid_retriever.py#L235) —— langchain Chroma 原生支持 `search_kwargs={'filter': {...}}`，这是最干净的过滤插入点。
- BM25 是**全库内存索引**：`_rebuild_bm25_from_chroma` 调 `vector_store.get(include=['documents','metadatas'])` 拉全量构建 `BM25Retriever.from_documents`：[rag/hybrid_retriever.py:159-184](rag/hybrid_retriever.py#L159)；pkl 缓存指纹 = `manifest.json` 的 SHA-256：[:104-122](rag/hybrid_retriever.py#L104)；`BM25Retriever` **不支持 where 过滤**。
- **关键并发不变量**：BM25 长期持有同一批 Document 对象跨查询复用：[rag/hybrid_retriever.py:174-177](rag/hybrid_retriever.py#L174) —— 任何按租户过滤/打标的逻辑**禁止原地改 metadata**，必须像 reranker 一样复制新 Document。
- reranker 是纯函数，浅拷贝 metadata 写 `rerank_score` 到新 Document：[rag/reranker.py:43-50](rag/reranker.py#L43)
- 父子回表 `ParentChildResolver.resolve` 按 `parent_id` 直取共享 JSON docstore：[rag/parent_child_retriever.py:35-45](rag/parent_child_retriever.py#L35)

### 2.3 认证 / RBAC / 数据库层

- `User` 表仅 `id / username / password / nickname / created_at`，**无 role、无 tenant_id**：[models/users.py:18-22](models/users.py#L18)
- JWT payload 只含 `{user_id, username, exp}`，7 天过期：[utils/security.py:31-38](utils/security.py#L31)
- `get_current_user` 纯解码 token 返回 `{id, username}` dict，**不查库**（设计意图："token 即事实来源"）：[api/auth.py:24-33](api/auth.py#L24)
- `require_admin` 硬编码 `user["username"] != "admin"` → 403：[api/auth.py:36-40](api/auth.py#L36)
- 签发 token 的两处：register（[api/auth.py:49](api/auth.py#L49)）和 login（[api/auth.py:62](api/auth.py#L62)）
- `crud/conversations.py` 已有成熟的用户隔离模式：每函数带 `user_id` 参数 + `WHERE user_id=?`，越权返回空 dict/no-op（不抛 403 防探测）—— **这是知识库权限表可复用的模板**。
- 建表机制：`Base.metadata.create_all` 幂等建表，但**不会给已存在表加新列**（无 Alembic）：[api/app.py:43-53](api/app.py#L43) —— 给 `users` 加字段需手工 ALTER 或迁移脚本。
- 种子用户 `admin/admin123` 首启且 users 表为空时创建：[api/app.py:58-62](api/app.py#L58)

### 2.4 知识库 API

- `upload`：`Depends(require_admin)`，文件落共享 `data_dir` 后调 `load_document()` 处理**全目录**：[api/routers/knowledge.py:13-47](api/routers/knowledge.py#L13)
- `list`：任意登录用户可看，直接遍历全局 `manifest`：[:50-75](api/routers/knowledge.py#L50)
- `delete`：`Depends(require_admin)`，按 `doc_id` 遍历 manifest 定位，调 `delete_document_by_doc_id`：[:78-107](api/routers/knowledge.py#L78)
- 删除链路 `delete_document` 按 filename 取 `entry.chunk_ids` 删 Chroma + `parent_ids` 删 docstore + rebuild BM25：[rag/vector_store.py:304-334](rag/vector_store.py#L304)

### 2.5 Agent 身份传递链路（**改造核心难点**）

- chat 路由拿到 `user` 后调 `execute_stream(messages, conversation_id, on_step=on_step)` —— **只传 messages/conversation_id/on_step，没传用户身份**：[api/routers/chat.py:72](api/routers/chat.py#L72)
- `execute_stream` 调 `self.agent.stream(input_dict, stream_mode="values", context={"report": False})` —— 这个 `context` dict 就是 LangChain runtime context 通道：[agent/react_agent.py:43](agent/react_agent.py#L43)
- middleware 能读写 `request.runtime.context`（`report` 标志就是这么工作的）：[agent/tools/middleware.py:22-23](agent/tools/middleware.py#L22) —— **证明 runtime.context 可承载 tenant_id**
- `rag = RagSummarizeService()` 是模块级单例，`rag_summarize(query: str)` 工具签名只有 query：[agent/tools/agent_tools.py:19,27](agent/tools/agent_tools.py#L19) —— agent 是全局单例，**不能靠构造参数传租户**，必须走请求局部上下文。
- agent 在独立线程跑：[api/routers/chat.py:80](api/routers/chat.py#L80) `threading.Thread(target=run_agent)` —— ContextVar 在该线程内有效。

### 2.6 评测与部署影响面

- `eval_rag25/pipelines.py` 的 `_retrieve_children` 直接调 `vs.vector_store.as_retriever(search_kwargs={"k": k_pool})`（[:53](eval_rag25/pipelines.py#L53)）和 `engine.get_retriever(query).invoke(query)`（[:62](eval_rag25/pipelines.py#L62)），且会临时改 `engine.k` / `engine.bm25_retriever.k`（[:58-66](eval_rag25/pipelines.py#L58)）。
- **兼容性结论**：只要 `get_retriever(query, tenant_id=None)` 默认 None（None=不过滤=全库），eval 管线零改动。
- 部署：[docker-compose.yml:35-39](docker-compose.yml#L35) 单 `app` 容器单进程，`app_runtime` 卷存 BM25 pkl 缓存。单进程 → ContextVar 安全；若未来多 worker，per-tenant BM25 缓存需迁出本地卷。

---

## 3. 三套候选方案对比

| 维度 | A. 单集合 + metadata 过滤 | B. 知识空间 RBAC（KB 一等实体） | C. 物理隔离（per-tenant collection） |
|---|---|---|---|
| Chroma | 单 collection，chunk 加 `tenant_id`，查询期 `where` 过滤 | 同 A（chunk 加 `kb_id`），或 per-KB collection | 每租户独立 collection |
| BM25 | per-tenant 子索引（懒构建+缓存）或超采后过滤 | 同 A | 天然 per-tenant |
| manifest / docstore | 单文件，entry/record 加 tenant 字段 | 同 A + KB 字段 | per-tenant 独立文件 |
| doc_id | `sha256(tenant_id+file_hash)[:16]` | 同 A | 可保持纯 file_hash（物理隔离已无碰撞） |
| RBAC | role + 租户归属 | role + KB 成员关系（owner/editor/viewer） | role + 租户归属 |
| 隔离强度 | 逻辑隔离（靠 filter 正确性） | 逻辑隔离 | 物理隔离（最强） |
| 单例架构 | 兼容（参数化） | 兼容 | 破坏（需 registry） |
| eval 兼容 | ✅ 默认 None | ✅ | ❌ 需重构 |
| 改动面 | 中 | 大 | 大 |
| 面试展示价值 | 中 | 高（RBAC 设计亮点） | 中 |
| 风险 | filter 漏写路径则泄漏 | 表多、接口多 | 破坏单例与 eval |

### 对抗评审结论

- **正确性**：A 的 BM25 后过滤有"召回配额被挤占"问题 → 必须用 per-tenant BM25（在单集合内按 `where` 拉租户子集构建）解决，不能用纯后过滤。C 物理隔离无此问题但破坏 eval。
- **安全**：A/B 都是逻辑隔离，**强依赖每条检索路径都带 filter**——Agent 工具是最易漏的点（必须确保 ContextVar 有值）。C 物理边界最难泄漏但过度。
- **务实**：单机演示项目，C 过度工程；B 的完整 kb_members 代码量大但 RBAC 展示价值高；A 改动最小、eval 兼容、迁移可行。

---

## 4. 推荐方案：A 为主 + 轻量 B 的权限模型

**存储与检索采用方案 A**（单 collection + metadata 过滤 + per-tenant BM25），**权限模型引入方案 B 的轻量版**（User 加 role + 一张 `knowledge_documents` 归属表，可选 `knowledge_bases` 分组表），**不采用方案 C 的物理隔离**（破坏单例与 eval，过度工程）。

理由：方案 A 的 BM25 弱点可用"单集合内 per-tenant BM25 子索引"干净解决（不必走 C 的物理隔离）；方案 B 的完整 kb_members 对单机演示过重，但"role + 归属表"是 RBAC 的最小完备形态，面试讲得出彩又不过度。

### 4.1 数据模型变更

#### 4.1.1 `User` 表加字段（[models/users.py:18-22](models/users.py#L18)）

```python
# 新增字段
role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)        # admin / editor / viewer
tenant_id: Mapped[int] = mapped_column(Integer, ForeignKey("tenants.id"), nullable=False, default=1)
# 加索引
Index("idx_user_tenant", "tenant_id")
```

> 注意：`Base.metadata.create_all` 不会给已存在表加列（[api/app.py:52](api/app.py#L52)），需配套迁移脚本 `ALTER TABLE users ADD COLUMN ...`（见 §6）。

#### 4.1.2 新表 `tenants`（租户）

```python
class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)  # 租户名/组织名
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

#### 4.1.3 新表 `knowledge_documents`（文档归属 — 权限判定的真相源）

```python
class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (Index("idx_kd_tenant", "tenant_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)  # 对应 Chroma/manifest 的 doc_id
    tenant_id: Mapped[int] = mapped_column(Integer, ForeignKey("tenants.id"), nullable=False)
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="active")
    kb_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("knowledge_bases.id"), nullable=True)  # 可选分组
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

> 这张表替代"靠 manifest 做权限判定"——manifest 仍是 RAG 内部存储格式，但权限/归属判定走 MySQL（ACID、可查询、可审计），与 `crud/conversations.py` 的隔离模式一致。

#### 4.1.4（可选）新表 `knowledge_bases` + `kb_members`

```python
class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    id, name, tenant_id, owner_id, visibility(public/private/shared), created_at

class KBMember(Base):
    __tablename__ = "kb_members"
    id, kb_id, user_id, role(owner/editor/viewer)
```

> **P5 可选阶段**。v1 可只做 `tenant_id` + `role`，不引入 KB 分组，足够演示多租户 + RBAC。

### 4.2 存储隔离

#### 4.2.1 chunk metadata 注入 tenant_id（[rag/vector_store.py:215-302](rag/vector_store.py#L215)）

- `_enrich_chunks`（[:224-232](rag/vector_store.py#L224)）：meta dict 加 `'tenant_id': tenant_id`
- `_build_parent_child_chunks` 的 `parent_meta`（[:257-266](rag/vector_store.py#L257)）与 `child_meta`（[:281-294](rag/vector_store.py#L281)）同样加 `tenant_id`；parent_meta 会随 `save_batch` 原样落入 `parent_docstore.json`（[:301](rag/vector_store.py#L301)），父块租户信息免费获得。

> **Chroma metadata 值只能是标量**（str/int/float/bool），`tenant_id` 用 int 标量，符合约束。

#### 4.2.2 doc_id 掺入 tenant 防碰撞（[rag/vector_store.py:396](rag/vector_store.py#L396)）

```python
# 旧: doc_id = file_hash[:16]
# 新:
doc_id = hashlib.sha256(f"{tenant_id}:{file_hash}".encode()).hexdigest()[:16]
```

派生的 `parent_id` / `child_id`（[:254](rag/vector_store.py#L254), [:280](rag/vector_store.py#L280)）自动继承租户命名空间。

> **v1 建议**：先不掺 tenant 进 doc_id（见 §6 取舍），靠 `(tenant_id, file_hash)` 双键去重规避碰撞；v2 再命名空间化。

#### 4.2.3 manifest 命名空间化（[utils/file_handler.py:51-84](utils/file_handler.py#L51)）

- `update_manifest_entry` entry 加 `tenant_id` 字段。
- `check_file_status` 去重循环**限定同租户**：filename 冲突 + 同 tenant → 跳过/更新；同 filename 不同 tenant → 允许并存（或加 tenant 前缀重命名落盘）。
- 顶层 key 仍可用 `filename`，但去重判定必须 `tenant_id + file_hash` 双键。

#### 4.2.4 `load_document` 接受租户（[rag/vector_store.py:344-359](rag/vector_store.py#L344)）

```python
def load_document(self, tenant_id: int = None, owner_id: int = None, files: list[str] = None):
    # tenant_id/owner_id 透传给 _enrich_chunks / _build_parent_child_chunks
    # files 指定本次处理哪些文件（不再无参全目录扫描），落盘到 data/{tenant_id}/ 子目录
```

落盘路径建议按租户分子目录：`data/{tenant_id}/{filename}`（[api/routers/knowledge.py:31](api/routers/knowledge.py#L31) 的 `os.path.join(data_dir, filename)` 改为 `os.path.join(data_dir, str(tenant_id), filename)`），并同步改 `listdir_with_allowed_type` 与 `stale` 清理逻辑（[rag/vector_store.py:477-484](rag/vector_store.py#L477)）。

### 4.3 检索隔离

#### 4.3.1 向量侧：Chroma where 过滤（[rag/hybrid_retriever.py:235](rag/hybrid_retriever.py#L235)）

```python
# 旧
vector_retriever = self.vector_store.as_retriever(search_kwargs={'k': self.k})
# 新
search_kwargs = {'k': self.k}
if tenant_id is not None:
    search_kwargs['filter'] = {'tenant_id': tenant_id}
vector_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
```

#### 4.3.2 BM25 侧：per-tenant 子索引（推荐，解决配额侵蚀）

在单 collection 内按 `where` 拉租户子集构建 BM25，缓存按 `(tenant_id, manifest_hash)` 分键：

```python
# rag/hybrid_retriever.py 新增 per-tenant BM25 缓存
self._bm25_cache: dict[tuple[int, str], BM25Retriever] = {}
self._bm25_cache_lock = Lock()

def _get_bm25_for_tenant(self, tenant_id: int):
    # vector_store.get(where={'tenant_id': tenant_id}, include=['documents','metadatas'])
    # 缓存键 = (tenant_id, manifest_hash)
    # 复用现有 _bm25_lock 防并发双重构建（[:56,104](rag/hybrid_retriever.py#L56)）
```

调用点：[rag/hybrid_retriever.py:163](rag/hybrid_retriever.py#L163) `vector_store.get(include=[...])` 加 `where={'tenant_id': tenant_id}`。

> **备选（更简单但弱）**：BM25 仍全库，`get_retriever` 后对结果按 `doc.metadata['tenant_id']` **只读后过滤**（位于 [rag/rag_service.py:40-42](rag/rag_service.py#L40) `invoke` 之后、`rerank` 之前）。缺点：BM25 `k` 固定，后过滤会吞配额导致租户内召回不足，需放大 `k`（如 3×）。v1 可先用此法快速落地，v2 升级为 per-tenant 索引。

#### 4.3.3 签名透传

- `DynamicHybridRetriever.get_retriever(self, query, tenant_id=None)`（[:233](rag/hybrid_retriever.py#L233)）
- `VectorStoreService.get_retriever(self, query, tenant_id=None)`（[rag/vector_store.py:80-81](rag/vector_store.py#L80)）
- `RagSummarizeService.retriever_docs(self, query, tenant_id=None)` + `rag_summarize(self, query, tenant_id=None)`（[rag/rag_service.py:37,53](rag/rag_service.py#L37)）

#### 4.3.4 纵深防御：父子回表校验（[rag/parent_child_retriever.py:35-45](rag/parent_child_retriever.py#L35)）

`resolve` 回表后只读校验 `record['metadata'].get('tenant_id') == tenant_id`，不一致则跳过（正常情况下上游过滤已保证，此为兜底）。**不可改 docstore 记录**。

#### 4.3.5 不变量保持

- reranker 不动（纯函数，读 metadata 不写）：[rag/reranker.py:43-50](rag/reranker.py#L43)
- BM25 后过滤/per-tenant 索引返回的 Document 仍是共享对象，**禁止原地改 metadata** —— 任何 tenant 标记必须在入库时写入，检索期只读。

### 4.4 Agent 身份传递（**核心改造**）

通道：HTTP `user` → `execute_stream` → LangChain runtime context + ContextVar → `rag_summarize` 工具 → `RagSummarizeService`。

#### 4.4.1 chat 路由透传（[api/routers/chat.py:72](api/routers/chat.py#L72)）

```python
for chunk in agent.execute_stream(
    messages, conversation_id,
    user_id=user["id"], tenant_id=user["tenant_id"],
    on_step=on_step,
):
```

#### 4.4.2 execute_stream 注入 runtime context（[agent/react_agent.py:26,43](agent/react_agent.py#L26)）

```python
def execute_stream(self, chat_pack, conversation_id=None, user_id=None, tenant_id=None, on_step=None):
    ...
    ctx = {"report": False}
    if tenant_id is not None:
        ctx["tenant_id"] = tenant_id
        ctx["user_id"] = user_id
    # 同时设 ContextVar（工具函数体读取用）
    _request_ctx_var.set({"user_id": user_id, "tenant_id": tenant_id})
    for chunk in self.agent.stream(input_dict, stream_mode="values", context=ctx):
        ...
```

#### 4.4.3 工具读取身份（[agent/tools/agent_tools.py:27](agent/tools/agent_tools.py#L27)）

新增模块级 ContextVar：

```python
import contextvars
_request_ctx_var = contextvars.ContextVar("request_ctx", default={"user_id": None, "tenant_id": None})

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    ctx = _request_ctx_var.get()
    return rag.rag_summarize(query, tenant_id=ctx["tenant_id"])
```

> **线程安全说明**：agent 在 [chat.py:80](api/routers/chat.py#L80) 的独立线程内运行，`agent.stream` 同步执行，工具在该线程内调用，ContextVar 可见。若 langgraph 内部用线程池跑工具（极端情况），更稳的做法是用 middleware 桥接：在 `monitor_tool`（[agent/tools/middleware.py:9](agent/tools/middleware.py#L9)）里从 `request.runtime.context["tenant_id"]` 读出并 `set` ContextVar——middleware 与工具同上下文。**推荐两者都做**：execute_stream 设初值 + middleware 兜底同步。

#### 4.4.4 extract_file_content 工具同样改造

`extract_file_content(file_path)`（[agent_tools.py:516](agent/tools/agent_tools.py#L516)）若读取的是用户上传的附件，也需校验该文件属于当前租户（通过 `upload_store` 的归属记录），防止 LLM 被诱导读他人文件。P2 阶段处理。

### 4.5 API 权限改造

#### 4.5.1 JWT 加 claim（[utils/security.py:31-38](utils/security.py#L31), [api/auth.py:49,62](api/auth.py#L49)）

```python
def create_access_token(user_id, username, role, tenant_id):  # 签名扩展
    payload = {..., "role": role, "tenant_id": tenant_id, ...}
```

#### 4.5.2 `get_current_user` 返回 dict 加 role/tenant（[api/auth.py:24-33](api/auth.py#L24)）

```python
return {
    "id": payload["user_id"],
    "username": payload["username"],
    "role": payload.get("role", "user"),        # 旧 token 容错
    "tenant_id": payload.get("tenant_id", 1),   # 旧 token 默认租户
}
```

> 旧 token 用 `.get()` 带默认值，避免 KeyError 变 500（[utils/security.py:41-46](utils/security.py#L41) decode 不报缺失 claim）。

#### 4.5.3 `require_admin` 改基于 role + 新增 `require_role` 工厂（[api/auth.py:36-40](api/auth.py#L36)）

```python
def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "需要管理员权限")
    return user

def require_role(*roles):
    def dep(user: dict = Depends(get_current_user)):
        if user.get("role") not in roles:
            raise HTTPException(403, "权限不足")
        return user
    return dep
```

#### 4.5.4 知识库三端点（[api/routers/knowledge.py:13,50,78](api/routers/knowledge.py#L13)）

- `upload`：`Depends(require_role("admin", "editor"))`，从 `user["tenant_id"]` 取租户，落盘 `data/{tenant_id}/`，调 `load_document(tenant_id=..., owner_id=user["id"], files=[...])`，入库后写 `knowledge_documents` 归属记录。
- `list`：`Depends(get_current_user)`，按 `tenant_id` 过滤 manifest（或直接查 `knowledge_documents` 表）。
- `delete`：`Depends(require_role("admin", "editor"))`，**先校验 `doc_id` 的 `tenant_id == user["tenant_id"]`**（越权返回 404 而非 403，沿用 [crud/conversations.py:43](crud/conversations.py#L43) 防探测约定），再调 `delete_document_by_doc_id` + 删 `knowledge_documents` 记录。

### 4.6 前端改造（[js/auth.js:17](js/auth.js#L17)）

- `isAdmin()` 由 `getUsername() === "admin"` 改为读登录响应里的 `role === "admin"`（登录接口返回 user 对象加 `role`，存 `localStorage`）。
- 知识库上传/删除按钮按 `role` 显隐（[js/knowledge.js](js/knowledge.js)）。
- 可选：非 admin 用户隐藏知识库管理入口（仅 viewer）。

### 4.7 存量数据迁移

1. **建表**：启动时 `create_all` 自动建 `tenants / knowledge_documents / knowledge_bases`（新表，幂等）。
2. **加列**：`ALTER TABLE users ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user', ADD COLUMN tenant_id INT NOT NULL DEFAULT 1;`（迁移脚本 `scripts/migrate_multitenant.py`）。
3. **种子租户**：插入 `tenants` 默认租户 `id=1, name='default'`。
4. **种子 admin**：[api/app.py:58-62](api/app.py#L58) 种子用户赋 `role='admin', tenant_id=1`；历史 admin 用户 `UPDATE users SET role='admin', tenant_id=1 WHERE username='admin'`。
5. **RAG 存量回填**：脚本遍历 Chroma 全量 chunk + `parent_docstore.json` + `manifest.json`，给无 `tenant_id` 的记录回填 `tenant_id=1`（默认租户）。Chroma 更新 metadata 用 `collection.update(ids=..., metadatas=...)`。
6. **doc_id 迁移**（可选，若启用 §4.2.2 的 tenant 掺入）：旧 doc_id 不含 tenant，需重新入库或写映射表。**v1 建议先不掺 tenant 进 doc_id**（保持 `file_hash[:16]`），靠 `tenant_id` metadata 过滤隔离即可——doc_id 碰撞仅在"两租户上传同内容文件"时发生，v1 可在 `check_file_status` 用 `(tenant_id, file_hash)` 双键去重规避，碰撞时给后上传的加 tenant 前缀重命名。v2 再做 doc_id 命名空间化。

---

## 5. 分阶段实施

每阶段独立可验证，互不阻塞。

### P1 — 身份与 RBAC 基础（不动 RAG）
- [ ] `models/users.py` 加 `role / tenant_id`；新增 `models/tenant.py`
- [ ] `utils/security.py` JWT 加 claim；`api/auth.py` `get_current_user` 返回 role/tenant、`require_admin` 改基于 role、新增 `require_role`
- [ ] `schemas/users.py` `UserInfo` 加 role
- [ ] 迁移脚本：users 加列 + 种子租户 + 种子 admin role
- [ ] 前端 `js/auth.js` `isAdmin` 读 role
- **验证**：登录返回 role；非 admin 调 upload 返回 403；旧 token 仍可用（容错默认值）。

### P2 — 存储层加 tenant 元数据（检索仍兼容）
- [ ] `_enrich_chunks` / `_build_parent_child_chunks` 注入 `tenant_id`
- [ ] `load_document(tenant_id, owner_id, files)` 签名改造；落盘 `data/{tenant_id}/`
- [ ] `manifest` entry 加 tenant；`check_file_status` 同租户去重
- [ ] 新增 `knowledge_documents` 表 + `crud/knowledge.py`（仿 `crud/conversations.py`）
- [ ] 存量回填脚本：旧 chunk/parent/manifest 回填 `tenant_id=1`
- **验证**：上传带 tenant；list 按租户过滤；`tenant_id=None` 时 eval 仍全库跑通。

### P3 — 检索隔离 + Agent 身份通道（隔离生效）
- [ ] `hybrid_retriever.py:235` 向量侧加 `filter`
- [ ] BM25 per-tenant 子索引（或 v1 先用超采后过滤）
- [ ] `get_retriever / retriever_docs / rag_summarize` 签名透传 `tenant_id`（默认 None）
- [ ] `parent_child_retriever.py` resolve 后 tenant 纵深校验
- [ ] `react_agent.py` `execute_stream` 接 user_id/tenant_id，注入 runtime context + ContextVar
- [ ] `agent_tools.py` `rag_summarize` 读 ContextVar 传 tenant
- [ ] `middleware.py` `monitor_tool` 兜底同步 ContextVar
- [ ] `chat.py:72` 调 execute_stream 传 `user["tenant_id"]`
- **验证**：用 A 租户上传文档，B 租户提问检索不到（黑盒测试）；Agent 问答端到端隔离。

### P4 — 知识库 API RBAC + 前端
- [ ] `knowledge.py` 三端点按 role + tenant 归属判定
- [ ] `extract_file_content` 工具加租户归属校验
- [ ] 前端知识库页按 role 显隐上传/删除
- **验证**：editor 能上传不能删他人文档；viewer 不能上传；跨租户删除返回 404。

### P5（可选）— 知识空间 KB 分组
- [ ] `knowledge_bases` + `kb_members` 表
- [ ] KB 管理接口（创建/成员/可见性）
- [ ] 检索 filter 改 `{'kb_id': {'$in': accessible_kb_ids}}`（Chroma `$in` 支持）
- [ ] 前端 KB 选择器
- **验证**：公共库 + 私有库 + 共享库的可见性矩阵。

---

## 6. 风险与取舍

| 风险 | 影响 | 缓解 |
|---|---|---|
| **Agent 工具漏传 tenant** | 隔离失效（最严重） | ContextVar 默认 None=全库（eval 兼容），但**生产路径必须有值**；middleware 兜底同步；加单测断言"非 None tenant 必过滤" |
| BM25 per-tenant 首查重建慢 | 租户首查延迟数十秒 | 入库时预热该租户 BM25；或 v1 先用超采后过滤 |
| doc_id 不掺 tenant → 同内容跨租户碰撞 | Chroma 稳定 ID 互相覆盖 | `check_file_status` 用 `(tenant_id, file_hash)` 双键去重；碰撞时给后上传文件加 tenant 前缀重命名 |
| `Base.metadata.create_all` 不加列 | users 老库缺 role/tenant | 迁移脚本 `ALTER TABLE`；启动时检测列存在性 |
| 旧 JWT token 无 role/tenant claim | `payload['user_id']` 下标取值 → KeyError 500 | `get_current_user` 用 `.get('role','user')` `.get('tenant_id',1)` 容错 |
| 多 worker 部署 | ContextVar 失效；per-tenant BM25 pkl 各 worker 独立 | 当前单 worker（[docker-compose.yml](docker-compose.yml)）非问题；未来多 worker 需把 BM25 缓存迁 Redis 或共享卷 |
| 删除文档清理不完整 | 残留 chunk 被他租户检索到 | `delete_document` 已清理 Chroma+docstore+manifest（[rag/vector_store.py:304-334](rag/vector_store.py#L304)），加 tenant 后同步清 `knowledge_documents` 表 + 验证无孤儿 |
| `require_admin` 由 username 改 role | 历史非 admin 用户无 role | 迁移脚本 `UPDATE` 历史 admin；新注册默认 `role='user'` |
| 知识库内容公开→按租户隔离后，旧"全员可见"语义变化 | 历史用户预期被打破 | 默认租户 `tenant_id=1` 承接历史数据；新租户从 2 开始 |

### 取舍说明

- **不选物理隔离（C）**：破坏单例与 eval，单机演示过度工程。
- **不选完整 kb_members（B 全量）**：代码量大，v1 用 `role + tenant_id` 已满足"多租户 + RBAC"目标，KB 分组留 P5。
- **BM25 优先 per-tenant 子索引**：解决配额侵蚀；超采后过滤仅作 v1 临时方案。
- **doc_id v1 不掺 tenant**：降低迁移成本，靠双键去重规避碰撞；v2 再命名空间化。

---

## 7. 验证清单

### 7.1 单元/集成
- [ ] `tests/test_multitenant_retrieval.py`：A 租户上传 → B 租户检索命中数 = 0；A 租户检索命中。
- [ ] `tests/test_rbac.py`：viewer 上传 403；editor 删他人 404；admin 全权限。
- [ ] ContextVar 断言：`execute_stream` 调用后工具内 `_request_ctx_var.get()["tenant_id"]` 有值。
- [ ] 旧 token 容错：无 role claim 的 token 解码不 500。

### 7.2 回归
- [ ] `python eval_rag25/evaluate.py --validate-only` 通过（tenant=None 全库）。
- [ ] `python tests/test_parent_child_integration.py` 通过。
- [ ] 现有 admin 登录、上传、问答、检测全流程不破。

### 7.3 端到端隔离黑盒
- [ ] 两租户各上传一份不同文档，互问对方内容 → 均检索不到。
- [ ] 两租户上传同内容文件 → 各自成功，互不影响。
- [ ] Agent 问答（走 `rag_summarize` 工具）端到端隔离生效（不仅 HTTP 接口）。

---

## 8. 改动文件清单（按阶段）

| 阶段 | 文件 | 改动 |
|---|---|---|
| P1 | [models/users.py](models/users.py) | 加 role/tenant_id |
| P1 | models/tenant.py（新） | Tenant 表 |
| P1 | [utils/security.py](utils/security.py) | JWT 加 claim |
| P1 | [api/auth.py](api/auth.py) | get_current_user/require_admin/require_role |
| P1 | [schemas/users.py](schemas/users.py) | UserInfo 加 role |
| P1 | [crud/users.py](crud/users.py) | create_user 加 role/tenant |
| P1 | [api/app.py](api/app.py) | 种子用户赋 role；startup 注册新模型 |
| P1 | scripts/migrate_multitenant.py（新） | ALTER + 种子 |
| P1 | [js/auth.js](js/auth.js) | isAdmin 读 role |
| P2 | [rag/vector_store.py](rag/vector_store.py) | metadata 注入、load_document 签名、落盘分目录 |
| P2 | [utils/file_handler.py](utils/file_handler.py) | manifest entry + 去重双键 |
| P2 | models/knowledge.py（新） | KnowledgeDocument 表 |
| P2 | crud/knowledge.py（新） | 归属 CRUD |
| P2 | scripts/backfill_tenant.py（新） | 存量回填 |
| P3 | [rag/hybrid_retriever.py](rag/hybrid_retriever.py) | 向量 filter + per-tenant BM25 |
| P3 | [rag/rag_service.py](rag/rag_service.py) | retriever_docs/rag_summarize 加 tenant |
| P3 | [rag/vector_store.py](rag/vector_store.py) | get_retriever 透传 |
| P3 | [rag/parent_child_retriever.py](rag/parent_child_retriever.py) | resolve 后校验 |
| P3 | [agent/react_agent.py](agent/react_agent.py) | execute_stream 注入 context + ContextVar |
| P3 | [agent/tools/agent_tools.py](agent/tools/agent_tools.py) | rag_summarize 读 ContextVar |
| P3 | [agent/tools/middleware.py](agent/tools/middleware.py) | monitor_tool 兜底同步 ContextVar |
| P3 | [api/routers/chat.py](api/routers/chat.py) | 透传 tenant_id |
| P4 | [api/routers/knowledge.py](api/routers/knowledge.py) | 三端点 RBAC + 归属校验 |
| P4 | [agent/tools/agent_tools.py](agent/tools/agent_tools.py) | extract_file_content 归属校验 |
| P4 | [js/knowledge.js](js/knowledge.js) | 按 role 显隐 |
| P5 | models/knowledge_base.py（新） | KB + 成员表 |
| P5 | api/routers/knowledge_base.py（新） | KB 管理接口 |

---

## 附：方案选型一句话

> **单 Chroma collection + chunk metadata `tenant_id` + 查询期 `where` 过滤 + per-tenant BM25 子索引** 实现存储/检索隔离（方案 A）；**User 加 role + `knowledge_documents` 归属表** 实现权限模型（方案 B 轻量版）；**Agent 经 `execute_stream` → runtime context + ContextVar → `rag_summarize` 工具** 透传身份。不采用物理隔离（方案 C）以保单例与 eval 兼容。
