# VectorStoreService Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把分块算法从 `rag/vector_store.py` 抽到一个独立组件，在不改变公开接口和数据行为的前提下缩短主服务。

**Architecture:** 新增一个 `DocumentChunker`，封装现有普通分块、语义分块、MinerU 结构化分块和父子块构建。`VectorStoreService` 保留 Chroma、Repository、文档生命周期与 BM25 协调，只通过两个入口调用分块器。

**Tech Stack:** Python 3.10、LangChain Document、RecursiveCharacterTextSplitter、SQLAlchemy Repository、Chroma

---

### Task 1: 建立重构基线

**Files:**
- Test: `tests/test_rag_mysql_runtime.py`
- Test: `tests/test_parent_child_integration.py`

- [ ] **Step 1: 运行现有运行时测试**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_rag_mysql_runtime -v
```

Expected: 当前测试全部通过；如果环境缺少依赖，记录具体缺失项，不修改业务代码规避环境问题。

- [ ] **Step 2: 记录当前公开接口调用者**

Run:

```powershell
rg -n "\.load_documents\(|\.load_document\(|\.delete_document_by_doc_id\(|\.retrieve\(" api rag tests -g "*.py"
```

Expected: API 仍通过 `load_documents()` 和 `delete_document_by_doc_id()`，RAG 仍通过 `retrieve()`。

### Task 2: 抽离 DocumentChunker

**Files:**
- Create: `rag/document_chunker.py`
- Modify: `rag/vector_store.py`
- Modify: `tests/test_rag_mysql_runtime.py`

- [ ] **Step 1: 调整初始化测试，要求 VectorStoreService 创建 DocumentChunker**

在 `test_initialization_uses_mysql_repositories_and_dynamic_manifest` 中 patch `DocumentChunker`，并断言 `service.chunker` 是返回的实例。运行该测试，确认因当前尚未创建 chunker 而失败。

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_rag_mysql_runtime.VectorStoreMySQLRuntimeTest.test_initialization_uses_mysql_repositories_and_dynamic_manifest -v
```

Expected: FAIL，原因是 `VectorStoreService` 尚未创建 `DocumentChunker`。

- [ ] **Step 2: 创建最小 DocumentChunker**

新增 `rag/document_chunker.py`：

```python
class DocumentChunker:
    def __init__(self, config, embedding_model):
        self.config = config
        self.embedding_model = embedding_model
        # 初始化 PDF、TXT、默认和 child splitter，以及语义分块参数

    def build_chunks(self, documents, *, file_path, doc_id, file_hash, file_type, id_namespace):
        # 语义分块失败时回退固定分块
        # 返回 chunks, chunk_ids, chunk_method

    def build_parent_child_chunks(self, documents, *, file_path, doc_id, file_hash, file_type, id_namespace):
        # MinerU 结构化、语义或固定父块，再生成父子块
        # 返回 child_chunks, child_ids, parent_ids, parent_records, chunk_method
```

把原有 `_build_spliter`、`_get_splitter`、`_cosine_sim`、`_cjk_ratio`、`_semantic_split`、`_enrich_chunks`、`_structured_split`、`_build_parent_child_chunks` 和 `_initial_chunk_method` 原样移动为该类的内部方法，只把 `embed_model` 和 `chroma_conf` 改为构造参数。

- [ ] **Step 3: 让 VectorStoreService 委托分块**

在 `VectorStoreService.__init__()` 中创建：

```python
self.chunker = DocumentChunker(chroma_conf, embed_model)
```

把 `load_document()` 中两处分块分支替换为：

```python
if self.parent_child_enabled:
    enriched_chunks, staged_child_ids, staged_parent_ids, parent_records, chunk_method = (
        self.chunker.build_parent_child_chunks(...)
    )
    self.parent_docstore.save_batch(parent_records)
else:
    enriched_chunks, staged_child_ids, chunk_method = self.chunker.build_chunks(...)
```

删除 `VectorStoreService` 中已经移动的分块属性和私有方法。

- [ ] **Step 4: 更新测试构造器**

`make_service()` 使用真实 `DocumentChunker` 的固定分块配置，并保留现有 FakeSplitter，从而继续覆盖入库生命周期，不新增第二套伪分块实现。

- [ ] **Step 5: 运行运行时测试**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_rag_mysql_runtime -v
```

Expected: 全部通过。

- [ ] **Step 6: 提交分块抽离**

```powershell
git add rag/document_chunker.py rag/vector_store.py tests/test_rag_mysql_runtime.py
git commit -m "refactor(RAG): 抽离文档分块组件"
```

### Task 3: 删除 VectorStoreService 冗余

**Files:**
- Modify: `rag/vector_store.py`
- Modify: `tests/test_rag_mysql_runtime.py`

- [ ] **Step 1: 精简旧版本快照测试**

保留现有“新版本激活后再清理旧版本”测试，并补充断言：父块 ID 能从形如 `doc:gen:hash:parent:000:child:000` 的旧 child ID 推导出来。

- [ ] **Step 2: 精简 `_snapshot_document()`**

实现只返回真实使用字段：

```python
@staticmethod
def _snapshot_document(record):
    chunk_ids = list(record.chunk_ids or [])
    parent_ids = list(dict.fromkeys(
        chunk_id.rsplit(":child:", 1)[0]
        for chunk_id in chunk_ids
        if ":child:" in chunk_id
    ))
    return {
        "chunk_ids": chunk_ids,
        "parent_ids": parent_ids,
        "storage_key": record.storage_key,
    }
```

调用方不再读取 `self.manifest.get(filename)`。

- [ ] **Step 3: 删除死接口**

删除项目内没有调用者的：

```python
def get_retriever(self, query: str):
    return self.hybrid_engine.get_retriyever(query)
```

保留 `retrieve()`。

- [ ] **Step 4: 运行目标测试与编译检查**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_rag_mysql_runtime tests.test_hybrid_retriever -v
conda run -n rag_env_backup python -m compileall -q rag api repositories tests
git diff --check
```

Expected: unittest 全部通过，compileall 和 diff check 退出码为 0。

- [ ] **Step 5: 提交精简结果**

```powershell
git add rag/vector_store.py tests/test_rag_mysql_runtime.py
git commit -m "refactor(RAG): 精简向量库生命周期代码"
```

### Task 4: 最终核对

**Files:**
- Verify: `rag/vector_store.py`
- Verify: `rag/document_chunker.py`
- Verify: `api/routers/knowledge.py`
- Verify: `rag/rag_service.py`

- [ ] **Step 1: 核对公开调用链和工作区**

Run:

```powershell
rg -n "\.load_documents\(|\.delete_document_by_doc_id\(|\.retrieve\(" api rag tests -g "*.py"
git status --short
git log -3 --oneline
```

Expected: 公开调用链未变化，工作区干净，最近提交包含设计、分块抽离和冗余精简。
