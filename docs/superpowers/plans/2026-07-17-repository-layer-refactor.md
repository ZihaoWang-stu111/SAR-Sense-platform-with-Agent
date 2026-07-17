# Repository 数据访问层重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把同步运行时数据库访问统一迁入顶层 `repositories/`，清除 `services/` 和 `rag/` 中的 MySQL/SQLAlchemy 实现，同时保持 API、RAG、指标和数据库行为不变。

**Architecture:** API 请求级异步查询继续使用 `crud/` 和 `AsyncSession`；RAG、Agent、指标等同步流程依赖 Repository 和 `SyncSessionLocal`。迁移脚本保留直接 SQLAlchemy 访问，因为它们负责 DDL、批量迁移和一致性检查，不属于运行时业务代码。

**Tech Stack:** Python 3.10、SQLAlchemy 2.0、MySQL、PyMySQL、aiomysql、`unittest`

---

## 文件结构

### 新增

- `repositories/__init__.py`：Repository 包标识。
- `repositories/knowledge_repository.py`：`knowledge_documents` 同步数据访问。
- `repositories/parent_chunk_repository.py`：`parent_chunks` 同步数据访问。
- `repositories/metrics_repository.py`：`metric_events` 同步写入、聚合和重置。

### 删除

- `services/knowledge_store.py`
- `services/metrics_store.py`
- `crud/metrics.py`

### 修改

- `rag/vector_store.py`：改用知识库和父块 Repository。
- `rag/hybrid_retriever.py`：把 `knowledge_store` 参数改为 `knowledge_repository`。
- `rag/parent_docstore.py`：只保留旧 JSON `ParentDocstore`。
- `agent/metrics_collector.py`：直接依赖 `MetricsRepository`，保留持久化失败不影响主流程的语义。
- `api/dependencies.py`：提供 `get_metrics_repository()` 懒加载单例。
- `api/routers/metrics.py`：改用 Repository 依赖，API 返回不变。
- 相关测试和 `AGENTS.md`：统一新模块路径和类名。

---

### Task 1：迁移知识库 Repository

**Files:**
- Create: `repositories/__init__.py`
- Move: `services/knowledge_store.py` → `repositories/knowledge_repository.py`
- Modify: `rag/vector_store.py`
- Modify: `rag/hybrid_retriever.py`
- Modify: `tests/test_mysql_knowledge_store.py`
- Modify: `tests/test_rag_mysql_runtime.py`
- Modify: `tests/test_hybrid_retriever.py`

- [ ] **Step 1：先修改测试使用期望的新接口**

把知识库测试导入改为：

```python
from repositories.knowledge_repository import KnowledgeRepository

self.repository = KnowledgeRepository(session_factory=self.session_factory)
```

把测试中的 `KnowledgeStore`、`FakeKnowledgeStore`、`knowledge_store` 分别改为 `KnowledgeRepository`、`FakeKnowledgeRepository`、`knowledge_repository`。

- [ ] **Step 2：运行测试并确认 RED**

```bash
conda run -n rag_env_backup python -m unittest tests.test_mysql_knowledge_store tests.test_rag_mysql_runtime tests.test_hybrid_retriever -v
```

预期：因 `repositories.knowledge_repository` 尚不存在而导入失败。

- [ ] **Step 3：移动实现并统一命名**

创建 `repositories/__init__.py`，将 `services/knowledge_store.py` 原实现移动到 `repositories/knowledge_repository.py`，类声明改为：

```python
class KnowledgeRepository:
    """知识库文档元数据的同步 SQLAlchemy Repository。"""

    def __init__(self, session_factory=SyncSessionLocal):
        self.session_factory = session_factory
```

保留现有方法及事务实现：

```text
get_by_doc_id, get_by_filename, get_by_hash, list_active,
begin_ingestion, mark_active, activate_document, active_chunk_ids,
mark_failed, mark_deleting, delete, as_manifest, fingerprint
```

删除原 `services/knowledge_store.py`，不保留 `KnowledgeStore` 别名。

- [ ] **Step 4：更新 RAG 调用和参数名**

`rag/vector_store.py` 使用：

```python
from repositories.knowledge_repository import KnowledgeRepository

self.knowledge_repository = KnowledgeRepository()
```

所有 `self.knowledge_store` 改为 `self.knowledge_repository`。

`DynamicHybridRetriever.__init__` 使用：

```python
def __init__(..., knowledge_repository=None, ...):
    if fingerprint_provider is None and knowledge_repository is not None:
        fingerprint_provider = knowledge_repository.fingerprint
    if active_chunk_ids_provider is None and knowledge_repository is not None:
        active_chunk_ids_provider = knowledge_repository.active_chunk_ids
```

- [ ] **Step 5：运行知识库相关测试并确认 GREEN**

```bash
conda run -n rag_env_backup python -m unittest tests.test_mysql_knowledge_store tests.test_rag_mysql_runtime tests.test_hybrid_retriever -v
```

预期：全部通过。

- [ ] **Step 6：提交**

```bash
git add repositories/__init__.py repositories/knowledge_repository.py rag/vector_store.py rag/hybrid_retriever.py tests/test_mysql_knowledge_store.py tests/test_rag_mysql_runtime.py tests/test_hybrid_retriever.py services/knowledge_store.py
git commit -m "refactor(知识库): 迁移同步数据访问到 Repository"
```

---

### Task 2：迁移父块 Repository

**Files:**
- Create: `repositories/parent_chunk_repository.py`
- Modify: `rag/parent_docstore.py`
- Modify: `rag/vector_store.py`
- Modify: `tests/test_mysql_knowledge_store.py`
- Modify: `tests/test_parent_child.py`
- Modify: `tests/test_rag_mysql_runtime.py`

- [ ] **Step 1：先修改测试使用期望的新类**

```python
from repositories.parent_chunk_repository import ParentChunkRepository

self.parent_repository = ParentChunkRepository(
    session_factory=self.session_factory,
)
```

测试补充以下结构断言：

```python
import rag.parent_docstore as legacy_parent_docstore

self.assertFalse(hasattr(legacy_parent_docstore, "MySQLParentDocstore"))
```

- [ ] **Step 2：运行测试并确认 RED**

```bash
conda run -n rag_env_backup python -m unittest tests.test_mysql_knowledge_store tests.test_parent_child tests.test_rag_mysql_runtime -v
```

预期：因 `repositories.parent_chunk_repository` 尚不存在而导入失败。

- [ ] **Step 3：创建父块 Repository**

把 `rag/parent_docstore.py` 中 `MySQLParentDocstore` 的 SQLAlchemy 实现移入新文件，并重命名：

```python
class ParentChunkRepository:
    """父块正文和元数据的同步 SQLAlchemy Repository。"""

    def __init__(self, session_factory=SyncSessionLocal):
        self.session_factory = session_factory
```

保留现有方法和 MySQL/SQLite upsert 行为：

```text
save, save_batch, get, get_many, delete_many,
delete_by_doc_id, count
```

`rag/parent_docstore.py` 最终只保留 JSON `ParentDocstore` 及其标准库依赖，不再导入 SQLAlchemy 或 `ParentChunk` 模型。

- [ ] **Step 4：更新 VectorStoreService**

```python
from repositories.parent_chunk_repository import ParentChunkRepository

if self.parent_child_enabled:
    self.parent_docstore = ParentChunkRepository()
```

保留 `parent_docstore` 属性名称，因为它表示 `ParentChildResolver` 使用的父块存储协议，而不是具体 MySQL 类名。

- [ ] **Step 5：运行父块相关测试并确认 GREEN**

```bash
conda run -n rag_env_backup python -m unittest tests.test_mysql_knowledge_store tests.test_parent_child tests.test_rag_mysql_runtime -v
```

预期：全部通过，并且旧模块不再暴露 `MySQLParentDocstore`。

- [ ] **Step 6：提交**

```bash
git add repositories/parent_chunk_repository.py rag/parent_docstore.py rag/vector_store.py tests/test_mysql_knowledge_store.py tests/test_parent_child.py tests/test_rag_mysql_runtime.py
git commit -m "refactor(父块): 独立父块 Repository"
```

---

### Task 3：迁移指标 Repository

**Files:**
- Move: `services/metrics_store.py` → `repositories/metrics_repository.py`
- Delete: `crud/metrics.py`
- Modify: `agent/metrics_collector.py`
- Modify: `api/dependencies.py`
- Modify: `api/routers/metrics.py`
- Modify: `tests/test_metrics_mysql.py`

- [ ] **Step 1：先修改测试使用新名称**

测试导入改为：

```python
from repositories.metrics_repository import MetricsRepository
```

API 依赖测试改为断言：

```python
first = dependencies.get_metrics_repository()
second = dependencies.get_metrics_repository()
self.assertIs(first, second)
```

再增加删除旧适配器的结构断言：

```python
from pathlib import Path

self.assertFalse(Path("crud/metrics.py").exists())
```

- [ ] **Step 2：运行指标测试并确认 RED**

```bash
conda run -n rag_env_backup python -m unittest tests.test_metrics_mysql -v
```

预期：因 `repositories.metrics_repository` 或 `get_metrics_repository` 尚不存在而失败。

- [ ] **Step 3：移动并重命名指标实现**

把 `services/metrics_store.py` 移到 `repositories/metrics_repository.py`，类名改为：

```python
class MetricsRepository:
    """指标事件的同步持久化、历史聚合与重置 Repository。"""

    _lock = threading.RLock()

    def __init__(self, session_factory=SyncSessionLocal):
        self.session_factory = session_factory
```

保留 `record_event()`、`aggregate()`、`delete_all()` 和 `reset()` 的现有行为。

- [ ] **Step 4：让 AgentMetrics 直接依赖 Repository**

删除 `crud.metrics.insert_metric_event` 依赖。单例初始化时创建 Repository：

```python
instance._repository = MetricsRepository()
```

增加唯一的容错入口：

```python
def _persist_event(self, **event) -> None:
    try:
        self._repository.record_event(**event)
    except Exception as exc:
        logger.warning(f"[metrics] failed to write MySQL event (memory unaffected): {exc}")
```

三个记录方法在 `MetricsRepository._lock` 和内存锁范围内调用 `_persist_event()`，保持“数据库失败不影响内存和主请求”的行为。删除 `crud/metrics.py`。

- [ ] **Step 5：更新 FastAPI 依赖与路由**

`api/dependencies.py` 使用：

```python
_metrics_repository = None

def get_metrics_repository():
    global _metrics_repository
    if _metrics_repository is None:
        from repositories.metrics_repository import MetricsRepository
        _metrics_repository = MetricsRepository()
    return _metrics_repository
```

`api/routers/metrics.py` 改为调用 `get_metrics_repository()`，接口路径和返回 JSON 不变。

- [ ] **Step 6：运行指标测试并确认 GREEN**

```bash
conda run -n rag_env_backup python -m unittest tests.test_metrics_mysql -v
```

预期：全部通过，包括持久化、聚合、并发、重置、用户 ID 和 API 契约测试。

- [ ] **Step 7：提交**

```bash
git add repositories/metrics_repository.py agent/metrics_collector.py api/dependencies.py api/routers/metrics.py tests/test_metrics_mysql.py services/metrics_store.py crud/metrics.py
git commit -m "refactor(指标): 统一指标 Repository 访问"
```

---

### Task 4：清理旧引用并完成总回归

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-07-16-repository-layer-design.md`（仅在实现与设计存在名称偏差时修正）
- Test: `tests/`

- [ ] **Step 1：更新架构文档**

`AGENTS.md` 增加 `repositories/` 层说明，并把旧名称统一改为：

```text
KnowledgeRepository
ParentChunkRepository
MetricsRepository
```

明确 `crud/` 只承担请求级异步 CRUD，Repository 承担非路由同步数据访问。

- [ ] **Step 2：扫描旧模块和类名**

```bash
rg -n "KnowledgeStore|MySQLParentDocstore|MetricsStore|services\.knowledge_store|services\.metrics_store|crud\.metrics|knowledge_store|get_metrics_store" --glob "*.py" --glob "*.md"
```

预期：除设计文档中的“旧名称 → 新名称”说明外无结果。

- [ ] **Step 3：扫描错误层级中的 SQLAlchemy 访问**

```bash
rg -n "from sqlalchemy|SyncSessionLocal|session_factory" services rag/parent_docstore.py --glob "*.py"
```

预期：无结果。迁移脚本、`crud/`、`models/`、`config/db_conf.py` 和 `repositories/` 不在该限制范围内。

- [ ] **Step 4：运行完整测试**

```bash
conda run -n rag_env_backup python -m unittest discover -s tests -p "test_*.py" -v
```

预期：当前 85 项及新增结构断言全部通过。

- [ ] **Step 5：运行编译和 diff 检查**

```bash
conda run -n rag_env_backup python -m compileall api agent rag crud models schemas services repositories utils tests
git diff --check
git status --short
```

预期：编译成功、无空白错误，工作树只包含本任务文档改动。

- [ ] **Step 6：提交文档与最终清理**

```bash
git add AGENTS.md docs/superpowers/plans/2026-07-17-repository-layer-refactor.md docs/superpowers/specs/2026-07-16-repository-layer-design.md
git commit -m "docs(架构): 更新 Repository 分层说明"
```

