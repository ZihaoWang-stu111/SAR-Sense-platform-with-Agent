# RAG 40 题检索评估

该目录提供独立的检索消融评测，不修改生产 RAG 代码。目录名保留为
`eval_rag25` 以兼容已有命令，默认数据集已经升级为
`qa_dataset_40.json`。

## 评测口径

每道题只设置一份确定性的金标准证据：

- `gold_filename`：证据所属文件；
- `gold_snippet`：必须出现在召回文本中的原文片段；
- 命中条件：文件名匹配，并且召回块包含规范化后的原文片段。

这种单证据口径用于比较检索链路的 Recall、MRR 和延迟。真正的跨文档综合
问答应另做答案质量评测，不混入本组检索指标。

## 数据集覆盖

- 40 道题：10 道 easy、20 道 medium、10 道 hard；
- 题型包括事实、概念、机理、模块、比较、数值、表格、指标、应用场景、
  别名改写和多约束查询；
- 语料覆盖 5 个 TXT、poster，以及 MBE-Net、SFQ-Det、SADL、GL-DETR 论文 PDF；
- 其中 38 题的证据直接存在于子块；`q25` 和 `q38` 仅在父块中保留完整证据，
  用于验证父块回表对表格和多约束问题的实际价值；
- 不包含个人简历 PDF。

## 消融流水线

| name | meaning |
|---|---|
| `vector_only` | Chroma vector top-k |
| `hybrid_pc_no_rr` | vector + BM25 -> parent resolve |
| `hybrid_no_pc_with_rr` | vector + BM25 -> child rerank |
| `vector_pc_with_rr` | vector Top-60 -> child rerank -> parent resolve（BM25 公平对照） |
| `full` | vector + BM25 -> child rerank -> parent resolve |

`full` 与生产 RAG 共用同一个混合召回入口：保留向量 Top-40，BM25 从
Top-40 中补充最多 20 个不重复子块，再使用生产阈值完成子块重排和父块回表。
评测参数 `k` 只截取最终有序结果，便于计算 Recall@3/5/10。

## 运行命令

先校验 40 道题、源文件和金标准片段：

```powershell
conda activate rag_env_backup
python -m eval_rag25.evaluate --validate-only
```

运行完整检索消融：

```powershell
python -m eval_rag25.evaluate
```

需要额外生成各流水线答案时：

```powershell
python -m eval_rag25.evaluate --with-answers
```

调试时先做小规模 smoke：

```powershell
python -m eval_rag25.evaluate --limit 2 --ks 3 --only-pipelines vector_only full --output-suffix _smoke
```

## 输出

- `results/results_raw.csv`：每题、每条 pipeline、每个 k 的原始命中记录；
- `results/summary.csv`：Recall、MRR 和平均延迟；
- `results/ablation_table.md`：消融对比表；
- `results/answers.jsonl`：可选，供外部 judge LLM 评分。

外部 judge 的 `scores.jsonl` 格式：

```jsonl
{"qa_id":"q01","pipeline":"full","score":1.0,"reasoning":"answer is grounded and complete"}
{"qa_id":"q02","pipeline":"full","score":0.5,"reasoning":"partially correct"}
```

聚合 judge 结果：

```powershell
python -m eval_rag25.aggregate --scores eval_rag25/results/scores.jsonl
```
