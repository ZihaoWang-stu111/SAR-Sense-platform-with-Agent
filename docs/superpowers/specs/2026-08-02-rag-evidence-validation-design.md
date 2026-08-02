# RAG 证据保护与高风险回答校验设计

## 背景

当前 RAG 链路为：

```text
ChromaDB + BM25 动态 RRF 融合
→ 截取 top-60 子块候选
→ BGE 子块重排
→ 父块回表并截取 final-k
→ LLM 生成回答
```

以“仅给出 CWW-Det 的对比实验表格”为例，目标表格已经进入粗召回：BM25 排名约第 13，但经 BGE 重排后约第 27，父块回表最终只保留 6 个，因此正确表格没有进入生成上下文。当前 `score_threshold=0.3` 只在最高分也低于阈值时清空整批结果，并不会逐条删除低于阈值的候选。

这类问题同时暴露两个缺口：

1. 结构化证据已经被召回，却在重排和 final-k 截断之间丢失。
2. 生成前没有验证证据是否足以满足“表格、具体数值、实验结论”等强约束，模型会尝试根据残缺材料补全答案。

## 目标

- 提高表格、指标和实验结论问题的最终证据命中率。
- 在证据不足时阻止 LLM 自行补全事实。
- 只对高风险回答增加一次校验模型调用。
- 校验失败时提供一次纠正机会，纠正后仍不可靠则拒答。
- 保持现有 ACL、active chunk、父子块、引用展示和工具返回协议不变。

## 非目标

- 不建设知识图谱、事实向量库或独立 NLI 服务。
- 不重写 `DynamicHybridRetriever`、RRF 或 BGE 重排算法。
- 不增加数据库表或字段。
- 不增加第三方依赖。
- 不为所有普通问答调用校验模型。

## 总体流程

```text
用户问题
  ↓
规则识别问题风险
  ↓
现有混合召回 top-60（ACL + active chunk 过滤保持不变）
  ↓
从候选中暂存强匹配表格
  ↓
BGE 子块重排
  ↓
将暂存证据放回前部并按 chunk_id 去重
  ↓
父块回表并截取 final-k
  ↓
生成前证据覆盖检查
  ├─ 不满足：直接拒答
  └─ 满足：生成回答
             ↓
        普通问题直接返回
        高风险问题调用一次校验模型
             ↓
        原回答 / 纠正答案 / 拒答
```

## 最小组件设计

### `rag/evidence_guard.py`

新增一个无状态模块，不创建 Service、Factory 或接口层。

#### `RiskProfile`

使用标准库只读 dataclass，字段为：

- `requires_table`
- `requires_numbers`
- `is_experimental_claim`
- `is_high_risk`
- `entities`

`entities` 只提取明显的英文型号或算法名，例如 `CWW-Det`、`MBE-Net`、`YOLO11n`，不尝试做通用命名实体识别。

#### `analyze_query(query)`

通过固定关键词和正则识别风险：

- 表格：`表格`、`对比表`、`实验表`、`消融表`、`结果表`。
- 数值：`mAP`、`Recall`、`FLOPs`、`参数量`、`准确率`、`召回率`、`多少`、`提升`。
- 实验结论：`实验结果`、`对比实验`、`消融实验`、`优于`、`证明`、`结论`。

任一类型命中即为高风险。该判断不调用 LLM。

#### `reserve_evidence(query, candidate_docs, risk)`

只在 `requires_table=True` 时工作：

1. 从已经完成 ACL 和 active chunk 过滤的 top-60 中寻找表格子块。
2. 表格判定优先使用 `mineru_type=table` 或 `table_id`，Markdown 表格特征仅作兼容兜底。
3. 优先选择正文包含查询核心实体的表格。
4. 多个表格同时匹配时，按实体命中数排序，再沿用原混合召回顺序。
5. 最多保留 1 个表格子块。

不从全库绕过权限重新检索，避免产生 ACL 旁路。

#### `merge_reserved(ranked_docs, reserved_docs)`

- 从 BGE 已打分结果中找到保留候选对应的副本，保留 `rerank_score`。
- 将强匹配表格放到结果前部。
- 按 `chunk_id` 去重，其余文档保持 BGE 顺序。
- 不修改共享 BM25 `Document`，继续遵守现有 Copy-on-Write 约束。

#### `validate_evidence(query, parent_docs, risk)`

返回 `(passed, reason)`：

- 表格问题必须存在表格父块，并包含查询核心实体。
- 指标问题必须存在数字和至少一个查询要求的指标词。
- 对比问题必须存在完整表格，或证据中至少覆盖两个明确比较对象。
- 普通问题直接通过，不增加额外约束。

## `RagSummarizeService` 改造

### `retriever_docs()`

保持公开签名不变，内部顺序调整为：

```python
risk = analyze_query(query)
candidates = vector_store.retrieve(...)
reserved = reserve_evidence(query, candidates, risk)
ranked = reranker.rerank(query, candidates)
ranked = merge_reserved(ranked, reserved)
parents = parent_resolver.resolve(ranked, ...)
passed, reason = validate_evidence(query, parents, risk)
```

为了让 `rag_summarize()` 获得风险和门控结果，内部使用一个私有结果结构或私有辅助方法；不改变 `agent_tools` 看到的最终字符串协议。

若显式表格问题的整批 BGE 最高分低于 `0.3`，该次重排使用 `score_threshold=0.0`，再由更严格的实体和表格证据门控决定是否可用。普通问题维持原阈值。

### 上下文构建

将现有上下文和来源拼装提取为 `_build_context(docs)`，返回：

```text
(context, sources, evidence_text)
```

`evidence_text` 只拼接 `Document.page_content`，不包含页码、chunk id 和重排分等来源元数据。生成模型和校验模型使用完全相同的 `context`，本地数字复核只对照 `evidence_text`，防止来源元数据中的数字误放行幻觉数值。

### 高风险回答校验

`_verify_high_risk_answer(query, context, answer)` 使用现有 `chat_model`，不创建第二个模型实例。通过 LangChain 已有的 Pydantic 输出解析能力得到：

```text
verdict: supported | corrected | unsupported
corrected_answer: string | null
unsupported_claims: string[]
```

校验提示词要求：

- 只能根据传入证据判断。
- 不允许使用模型自身知识补充数值。
- `supported` 返回原回答。
- 可在不引入新事实的情况下修复时返回 `corrected` 和完整纠正答案。
- 关键证据缺失时返回 `unsupported`。
- 纠正答案不得自行附加“参考来源”列表，来源仍由服务端统一生成。

一次校验调用同时完成判断和纠正，不再发起第二次重写调用。

### 纠正答案本地复核

校验模型返回 `corrected` 后执行轻量复核：

- 回答中的独立业务数值必须能在 `evidence_text` 中找到；来源编号、Markdown 序号以及 `YOLO11n` 这类型号内数字不参与比较。
- 数值比较统一去除千位分隔符和百分号两侧空格，但不计算或容许证据中不存在的派生数值。
- 用户要求表格时，纠正答案必须包含 Markdown 表格结构。
- 查询核心实体必须同时出现在证据和回答中。

复核通过才采用纠正答案，否则统一拒答。

## 返回与失败策略

| 情况 | 行为 |
|---|---|
| 普通问题、证据正常 | 维持现有生成和返回流程 |
| 高风险问题、必要证据缺失 | 生成前拒答，不调用 LLM |
| 高风险回答校验通过 | 返回原回答 |
| 校验模型给出可靠纠正答案 | 返回纠正答案 |
| 校验结果为 unsupported | 拒答 |
| 校验输出无法解析或调用异常 | 高风险问题保守拒答并记录日志 |
| `allowed_doc_ids=[]` | 维持现有空结果行为 |

拒答文案保持简短，并指出缺少的是表格、指标或可核验实验结论，不伪造来源。

## 指标

复用现有 `MetricEvent` 字段，不迁移数据库。新增事件类型：

- `rag_guard_rejected`
- `rag_verification_passed`
- `rag_answer_corrected`
- `rag_verification_rejected`
- `rag_verification_error`

`agent/tools/agent_tools.py` 将 `runtime.context["user_id"]` 传给 `RagSummarizeService.rag_summarize()`。没有用户上下文的本地直调只写日志，不写数据库。

`MetricsRepository.aggregate()` 增加一个简单的 `rag_quality` 计数字典，使现有 `/api/metrics` 可以看到这些结果；第一版不新增前端图表。

## 安全与并发

- 候选保留只处理现有 ACL 过滤后的候选，不扩大授权范围。
- 父块回表继续执行现有 `allowed_doc_ids` 二次校验。
- 所有证据处理函数只读 `Document`；需要调整顺序时创建新列表，不原地修改共享对象。
- 校验模型使用现有模块级模型实例，不增加模型加载和显存占用。

## 文件范围

新增：

- `rag/evidence_guard.py`
- `tests/test_rag_evidence_guard.py`

修改：

- `rag/rag_service.py`
- `agent/tools/agent_tools.py`
- `agent/metrics_collector.py`
- `repositories/metrics_repository.py`

明确不改：

- `rag/hybrid_retriever.py`
- `rag/reranker.py`
- `rag/parent_child_retriever.py`
- 数据库模型和表结构
- 前端 RAG 结果与参考来源协议
- 当前未提交的 `js/chat.js`、`templates/chat.html`

## 测试

只增加一个聚焦测试文件，并使用内存中的假 `Document` 和假校验结果，不加载真实 BGE、Chroma 或在线模型。覆盖：

1. CWW-Det 表格位于粗召回前 60、BGE 排名靠后时仍进入最终父块候选。
2. 非表格问题不改变原 BGE 排序。
3. 未授权表格不能通过候选保护进入结果。
4. 要求表格但没有匹配表格时，生成前拒答。
5. 原回答校验通过时保持不变。
6. 纠正答案含证据外数字时被本地复核拒绝。
7. 校验调用异常时高风险问题拒答。

回归验证：

```text
python -m compileall rag agent repositories tests
pytest tests/test_rag_evidence_guard.py -v
pytest tests/test_rag_acl_retrieval.py -v
```

## 验收标准

- “仅给我 CWW-Det 的对比实验表格”能够返回包含 CWW-Det 行的真实 Markdown 表格。
- 表格中的数值必须来自检索证据，不得由模型补写。
- 普通知识问答不增加第二次模型调用。
- 高风险问题最多增加一次模型调用。
- 无权限文件不会因表格候选保护被召回。
- 工具返回仍保持“回答正文 + 参考来源”格式，前端无需修改。

## 后续升级条件

只有评测证明正确表格经常无法进入 top-60 时，才新增独立表格召回通道；只有跨文档实体关系和多跳查询成为主要短板时，才评估知识图谱。当前版本不为这些假设提前建设基础设施。
