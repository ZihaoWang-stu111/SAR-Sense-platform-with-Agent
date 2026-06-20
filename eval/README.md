# RAG 评测：检索 + 生成（外部 LLM 评分）

为这个项目的 RAG 流水线（混合检索 + 父子块 + BGE Reranker）补一套独立评测，回答两个问题：

1. **每个组件单独贡献多少？**（5×3 检索消融）
2. **端到端答得对不对？**（生成评测，由外部更强的 judge LLM 评分）

> ⚠️ **judge LLM 用谁**：项目里 `chat_model`（DeepSeek-v3.2）既是答题模型也是潜在评委 → 同模型自评有偏差。**评分这一步交给外部更强的 LLM（推荐 Claude）做**，避免"自己评自己"。

---

## 文件结构

```
eval/
├── qa_dataset.json              # 50 道精选 QA（已写好，下面有 schema）
├── pipelines.py                 # 5 个 retrieval pipeline（向量 / 混合 / +PC / +RR / 全开）
├── metrics.py                   # is_hit / Recall@k / MRR
├── evaluate.py                  # Step 1：检索评测 (+ 可选 --with-answers 生成答案)
├── aggregate.py                 # Step 3：把 judge 评分聚合到最终报告
├── generate_qa.py               # （可选）LLM 自动出题脚本，本套已 deprecated 不用跑
└── results/
    ├── results_raw.csv          # ← evaluate.py 输出
    ├── ablation_table.md        # ← evaluate.py 出的检索表 → aggregate.py 重写为最终表
    ├── recall_at_k.png          # ← evaluate.py 输出
    ├── answers.jsonl            # ← evaluate.py --with-answers 输出
    ├── scores.jsonl             # ← 由外部 judge LLM 产出（用户保存）
    └── answer_correctness.png   # ← aggregate.py 输出
```

---

## 完整流程

### Step 1：检索评测 + 生成 RAG 答案

```bash
# 在 conda 环境 rag_env_backup 里跑
python eval/evaluate.py --with-answers
```

会发生：
- 检索评测：5 pipeline × 3 个 k 值 × 50 题 = 750 次检索（约 15-30 分钟）
- 答案生成：50 题 × 5 pipeline × k=5 = 250 次 LLM 调用（约 10-20 分钟）

输出：
- `eval/results/results_raw.csv` —— 每行 (pipeline, qa, k) 一条记录
- `eval/results/ablation_table.md` —— 检索消融表（这一步的中间产物，aggregate 后会被覆盖）
- `eval/results/recall_at_k.png` —— 5 条曲线
- `eval/results/answers.jsonl` —— 250 行 RAG 答案，**给外部 judge 用**

> 调试模式：`python eval/evaluate.py --limit 5 --with-answers` 先用 5 题跑通流程

### Step 2：交给外部 judge LLM 评分

把 `answers.jsonl` 给一个**比 chat_model 更强的 LLM**（推荐 Claude 4.x），让它打分。

**给 judge 的提示词模板**（直接复制到对话）：

```
你是 RAG 评测的评委。我会给你一个 JSONL，每行是一组 (问题, 标准答案,
RAG 答案)。请逐条评分，输出 JSONL 格式的评分文件。

评分规则：score ∈ {0.0, 0.5, 1.0}
  1.0 完全正确：RAG 答案与标准答案语义一致，关键标识词都答对，
                可以更详尽但不能错
  0.5 部分正确：关键标识词答对了主要的，但有缺失/表述不清/部分错误
  0.0 错误：答非所问 / 关键标识词答错 / 检索为空 / 编造内容

输出 schema（每行一条 JSON）：
  {"qa_id": "qa_001", "pipeline": "full", "score": 1.0, "reasoning": "..."}

reasoning 写一句话，说明为什么打这个分。

下面是 answers.jsonl 的内容：
[这里粘贴 answers.jsonl 的内容]
```

把 judge 的输出保存为 `eval/results/scores.jsonl`（与上面 schema 一致）。

> **如果是用本仓库的 Claude Code 评分**：
> 直接对 Claude 说 "请评判 eval/results/answers.jsonl，输出到 eval/results/scores.jsonl"。
> Claude 会用 Read 工具读 JSONL，按上面规则评分，再用 Write 工具写出 scores.jsonl。

### Step 3：聚合到最终报告

```bash
python eval/aggregate.py
```

会读 `results_raw.csv` + `scores.jsonl`，重写：
- `eval/results/ablation_table.md` —— 包含 Recall@k + MRR + Answer Correctness 三类指标
- `eval/results/answer_correctness.png` —— 各 pipeline 答对率柱状图

直接把 `ablation_table.md` 的内容贴到简历项目说明 + GitHub README 即可。

---

## 5 个 pipeline 都做什么

| pipeline | 检索 | 父子块 | Reranker | 角色 |
|---|---|---|---|---|
| `vector_only` | 纯向量 | ❌ | ❌ | baseline |
| `hybrid_no_pc_no_rr` | 混合 | ❌ | ❌ | 看 BM25 单独贡献 |
| `hybrid_pc_no_rr` | 混合 | ✅ | ❌ | 看父子块单独贡献 |
| `hybrid_no_pc_with_rr` | 混合 | ❌ | ✅ | 看 Reranker 单独贡献 |
| `full` | 混合 | ✅ | ✅ | 生产配置（线上跑这个） |

---

## 题集 schema（`qa_dataset.json`）

```json
{
  "id": "qa_001",
  "question": "MBE-Net 在哪两个公开数据集上做了验证？",
  "gold_answer": "HRSID 和 SSDD",
  "gold_filename": "MBE-Net算法详解.txt",
  "gold_snippet": "在HRSID和SSDD两个公开数据集上进行了充分验证",
  "qa_type": "fact"
}
```

- `gold_snippet` 必须是源文件中**精确出现的连续片段**（NFKC + 去空白后包含）
- 区分度高：含特定术语/数值，避免泛词导致跨文件假性命中
- 50 题分布：~50% fact / ~30% concept / ~20% comparison
- 覆盖 5 个 txt 语料文件（每个文件 ~10 题）

---

## 检索命中怎么判（面试官最爱问）

`is_hit(retrieved_docs, qa)` 在 [`metrics.py`](metrics.py)：

```python
gold_filename 匹配 + gold_snippet 包含
（NFKC 归一化 + 去全部空白 + lowercase）
```

**为什么不用 chunk_id 比对？**
父子块 pipeline 返回大父块（~240 字）、不开父子块返回小子块（~160 字），两套 ID 体系不一样。用"片段包含"判定让所有 pipeline 在同一标准——"**有没有检索到包含答案的段落**"——上比较，对父子块 / 纯子块同样公平。

---

## 为什么不评 faithfulness？

LLM 不变（都是 DeepSeek-v3.2），所以：
- `correctness` 的差异 = **检索贡献**（这是我们想测的）
- `faithfulness` = 检索贡献 + LLM 跟随上下文的能力（**LLM 自身能力混进来**）

为了让 hero number 干净，只评 **Recall@k（纯检索）+ Correctness（端到端）** 两根脊梁。

---

## 关键设计决策

1. **不修改任何生产代码** —— `eval/` 是独立模块，pipeline 全从 `rag/` 单例拼出来
2. **judge 用外部更强 LLM** —— 不让 DeepSeek 评 DeepSeek，避免自评偏差
3. **Reranker 评测时把 `score_threshold` 临时降到 0** —— 否则生产默认 0.3 的"门控"会清空低分结果，破坏消融对比公平性
4. **每个 pipeline 召回 `max(15, 3k)` 的宽候选池后再切到 top-k** —— 父子去重、Reranker 都需要候选池更宽
5. **`gold_snippet` 区分度优先** —— 题目都挑了含具体术语 / 数值 / 型号的片段，避免泛词导致 vector_only 也"假性命中"

---

## 常见坑

| 现象 | 原因 | 对策 |
|---|---|---|
| `parent_child_enabled` 报错 | 生产配置没开父子块 | `chroma.yml` 改成 true，删 `chroma_db/`，重跑入库 |
| `vector_only` Recall 高得反常 | gold_snippet 区分度不够 | 检查具体题，换更特定的片段 |
| Reranker 加载慢 | 第一次 `BGERerankerService()` 加载模型 | 单例已缓存，evaluate.py 里只加载一次 |
| 答案全是 `[空：...]` | 该 pipeline 检索为空（reranker 阈值？） | 已经设 score_threshold=0，应不会发生 |

---

## 简历 hero sentence 模板（aggregate 跑完替换数字）

> 自建 SAR 领域 **50** 题 RAG 评测集，对父子块 / BGE Reranker / 混合检索做 5×3 消融。**Recall@5 从 vector-only 基线的 XX% 提升到 full pipeline 的 YY%（+ZZ pp），MRR@5 从 0.AA 提升到 0.BB**；端到端答对率（外部 Claude 评分）从基线的 CC% 提升到 DD%。
