# VectorStoreService 最小化重构设计

## 目标

降低 `rag/vector_store.py` 的阅读复杂度，让它主要呈现检索、入库和删除三条业务流程，同时保持现有 API、数据格式和运行行为不变。

本次遵循最小改动原则：只抽离分块职责，不新增多层 Service、接口、工厂或配置。

## 当前问题

`VectorStoreService` 同时包含：

- Chroma、Repository 和混合检索器初始化；
- 中文、英文语义分块；
- MinerU 表格和公式的结构化分块；
- 父子块构建及 metadata 注入；
- 文档新增、更新、删除和失败补偿；
- BM25 重建协调；
- 旧版兼容接口。

其中分块算法约占文件三分之一，与文档生命周期混在一起，导致主入库流程难以阅读。

## 方案

新增 `rag/document_chunker.py`，提供一个 `DocumentChunker`，集中管理分块配置和分块算法。

移动以下职责：

- 文件类型对应的字符切分器；
- 中英文识别和语义断点分块；
- MinerU 文本、表格、公式的结构化分块；
- 普通 chunk metadata 和 ID 生成；
- 父块、子块及其 metadata 和 ID 生成；
- 分块方式名称判定。

`DocumentChunker` 对 `VectorStoreService` 只暴露两个业务入口：

- 构建普通子块；
- 构建父子块。

内部继续复用现有 `RecursiveCharacterTextSplitter`、`embed_model` 和 `chroma_conf`，不增加新依赖。

## VectorStoreService 保留职责

`rag/vector_store.py` 继续负责：

- 创建 Chroma、KnowledgeRepository、ParentChunkRepository 和 DynamicHybridRetriever；
- 执行带权限和 active chunk 过滤的检索；
- 文档去重与版本判断；
- 协调父块写入、子块向量写入和文档激活；
- 新版本成功后的旧版本清理；
- 入库失败后的补偿清理；
- 文档删除和 BM25 重建；
- 向旧调用方提供动态 `manifest` 兼容属性。

## 删除与精简

- 删除没有项目内调用者的 `VectorStoreService.get_retriever()`，统一使用 `retrieve()`。
- `_snapshot_document()` 只保留后续真实使用的 `chunk_ids`、`parent_ids` 和 `storage_key`。
- `parent_ids` 直接从旧 `chunk_ids` 推导，不再为了生成快照调用完整的 `manifest` 转换。
- 保留 `load_document()` 和 `load_documents()`：前者支持全目录扫描，后者继续保证批量上传中的单文件失败隔离。

## 不变边界

本次不修改：

- `load_document()`、`load_documents()`、`delete_document_by_doc_id()`、`retrieve()` 的参数和返回值；
- MySQL 表结构和已有数据；
- Chroma collection、metadata 和 chunk ID 格式；
- ACL 和 active chunk 过滤；
- MinerU 表格、公式原子块规则；
- 新 generation 先激活、旧 generation 后清理的顺序；
- 入库失败时保留旧 active generation 的行为；
- BM25 重建条件。

## 错误处理

分块失败仍由现有入库流程捕获。语义 embedding 失败继续回退固定分块。跨 MySQL 和 Chroma 的补偿清理逻辑保留在 `VectorStoreService`，避免把存储生命周期泄漏到分块器。

## 验证

- 编译 `rag`、`repositories`、`api` 和相关测试文件；
- 运行现有 MySQL 知识库运行时测试；
- 运行父子块和混合检索相关测试；
- 检查 API 仍只通过原公开方法上传、删除和检索；
- 检查 `git diff --check`。
