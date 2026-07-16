# Repository 数据访问层重构设计

## 目标

把 `services/` 和 `rag/` 中的同步数据库访问代码统一迁移到顶层 `repositories/` 包，同时保持现有业务行为、数据库表结构、API 返回格式和事务语义不变。

本次重构只整理代码职责，不修改数据库数据，也不执行表结构迁移。

## 目标目录结构

```text
repositories/
├── __init__.py
├── knowledge_repository.py
├── metrics_repository.py
└── parent_chunk_repository.py
```

现有类统一重命名：

- `KnowledgeStore` → `KnowledgeRepository`
- `MetricsStore` → `MetricsRepository`
- `MySQLParentDocstore` → `ParentChunkRepository`

所有项目内部调用和测试迁移完成后，不保留旧类名兼容别名，避免新旧命名长期并存。

## 各层职责

### `KnowledgeRepository`

负责 `knowledge_documents` 表的同步 SQLAlchemy 数据访问，包括：

- 文档入库状态管理；
- 新旧文档版本激活；
- active chunk id 查询；
- 兼容旧 manifest 数据结构；
- 生成 BM25 缓存指纹；
- 文档查询和删除。

### `ParentChunkRepository`

负责 `parent_chunks` 表的同步 SQLAlchemy 数据访问，并保持 `ParentChildResolver` 依赖的父块存储接口不变，包括：

- 单条和批量保存父块；
- 单条和批量查询父块；
- 按父块 ID 或文档 ID 删除；
- 父块数量统计。

### `MetricsRepository`

负责 `metric_events` 表的同步 SQLAlchemy 数据访问，包括：

- 指标事件持久化；
- 历史指标聚合；
- 最近工具调用查询；
- 数据库指标与内存指标的单进程原子重置。

### 其他目录

- `services/` 只保留业务流程编排，不直接编写 SQLAlchemy 查询。
- `rag/` 只保留切块、向量入库、检索、重排和父块解析逻辑，不保存 MySQL 实现。
- `api/routers/` 只负责 HTTP 请求和响应。

## 依赖方向

```text
API / Agent / RAG / Services
            ↓
      Repositories
            ↓
 SQLAlchemy Models / Sessions
            ↓
           MySQL
```

RAG、Agent 和 Service 不通过 API 层访问数据库，避免形成 `RAG → API → 数据库` 的反向依赖。

## 明确保留不动的部分

### API 异步 CRUD

顶层 `crud/` 中面向 API 请求的异步函数继续保留。这些函数使用 FastAPI 注入的请求级 `AsyncSession`，负责用户、对话和文件 ACL 等请求数据操作。

本次不把异步 CRUD 与同步 Repository 合并，避免混淆两种 Session 生命周期。

### 数据库配置与模型

- `config/db_conf.py` 继续负责创建异步和同步 Engine、SessionFactory；
- `models/` 继续负责 SQLAlchemy 表映射；
- Repository 通过注入的同步 SessionFactory 操作模型。

### 一次性迁移脚本

`utils/` 下的数据库迁移脚本可以继续直接使用 SQLAlchemy，因为它们需要完成：

- 表结构检查；
- DDL 执行；
- 批量迁移；
- 重复数据检查；
- MySQL、JSON 和 Chroma 一致性报告。

这些属于运维迁移逻辑，不是运行时业务 Repository。

### 旧 JSON 父块存储

旧 `ParentDocstore` 保留在 `rag/parent_docstore.py`，仅用于迁移兼容。MySQL 实现迁移到 `repositories/parent_chunk_repository.py`。

## 兼容性与异常处理

- Repository 方法签名和返回值尽量保持现有行为；
- commit、rollback、锁和异常传播语义保持不变；
- 不修改数据库表结构；
- 不修改数据库现有数据；
- 不修改 API 请求和响应格式；
- 不修改 RAG 检索、父块回表和指标计算结果；
- 所有依赖注入、调用代码、测试和架构文档统一更新为新名称。

## 测试与验收

### 测试驱动

先增加结构测试，要求以下模块和类存在：

- `repositories.knowledge_repository.KnowledgeRepository`
- `repositories.metrics_repository.MetricsRepository`
- `repositories.parent_chunk_repository.ParentChunkRepository`

该测试在代码迁移前应失败，迁移完成后通过。

### 回归测试

- Repository 使用 SQLite 测试 SessionFactory 验证数据库行为；
- 知识库入库、更新、删除和 active generation 测试保持通过；
- 父子块回表测试保持通过；
- 指标写入、聚合、重置和并发测试保持通过；
- API、ACL、迁移和 RAG 检索测试保持通过。

### 静态验证

- `python -m compileall` 通过；
- `git diff --check` 通过；
- `services/` 不再包含运行时 SQLAlchemy 数据访问；
- `rag/parent_docstore.py` 不再包含 MySQL/SQLAlchemy 实现；
- 项目中不再引用旧类名和旧模块路径。
