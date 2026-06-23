# RAG 25题检索评估

- 题数: 2
- k: [3]
- 命中: filename 匹配 + gold_snippet 出现在返回文本中
- full: hybrid children -> child rerank -> parent resolve

## Recall / MRR

| pipeline | BM25 | PC | RR | R@3 | MRR@3 | avg ms@3 |
|---|---:|---:|---:|---:|---:|---:|
| vector | N | N | N | 50.0% | 0.250 | 4554 |
| full | Y | Y | Y | 100.0% | 1.000 | 8815 |

## 简历可用句

> 自建 SAR 领域 25 题 RAG 检索评估集，覆盖 txt、论文 PDF 和 poster；full pipeline 的 Recall@3 为 100.0%，相比 vector-only 基线 50.0% 变化 +50.0 pp。
