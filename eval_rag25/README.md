# RAG 25题评估

这个目录是独立评估集，不改生产 RAG 代码。

## 覆盖范围

- 5 个 txt 知识库文件
- `MBE-Net A Multi-Branch Edge-Guided Feature Enhancement Algorithm for SAR Ship Detection(pdf).pdf`
- `SFQ-Det.pdf`
- `posterWZH.pdf`
- 不包含个人简历 PDF

## Pipeline

| name | meaning |
|---|---|
| `vector_only` | Chroma vector top-k |
| `hybrid_no_pc_no_rr` | vector + BM25 |
| `hybrid_pc_no_rr` | vector + BM25 -> parent resolve |
| `hybrid_no_pc_with_rr` | vector + BM25 -> child rerank |
| `full` | vector + BM25 -> child rerank -> parent resolve |

`full` 的顺序和生产 RAG 一致；评估里的 k 只控制返回 top-k，方便算 Recall@3/5/10。

## Commands

```bash
python eval_rag25/evaluate.py --validate-only
python eval_rag25/evaluate.py
python eval_rag25/evaluate.py --with-answers
```

调试时先小跑：

```bash
python eval_rag25/evaluate.py --limit 2 --ks 3 --only-pipelines vector_only full --output-suffix _smoke
```

## Outputs

- `results/results_raw.csv`: 每题、每 pipeline、每 k 的原始命中记录
- `results/summary.csv`: Recall、MRR、平均延迟
- `results/ablation_table.md`: 可直接放 README/简历项目描述的消融表
- `results/answers.jsonl`: 可选，给外部 judge LLM 打分

外部 judge 的 `scores.jsonl` 格式：

```jsonl
{"qa_id":"q01","pipeline":"full","score":1.0,"reasoning":"answer is grounded and complete"}
{"qa_id":"q02","pipeline":"full","score":0.5,"reasoning":"partially correct"}
```

聚合 judge 结果：

```bash
python eval_rag25/aggregate.py --scores eval_rag25/results/scores.jsonl
```

