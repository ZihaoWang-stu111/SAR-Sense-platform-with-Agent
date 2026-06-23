# RAG 25题检索评估

- 题数: 25
- k: [3, 5, 10]
- 命中: filename 匹配 + gold_snippet 出现在返回文本中
- full: hybrid children -> child rerank -> parent resolve

## Recall / MRR

| pipeline | BM25 | PC | RR | R@3 | R@5 | R@10 | MRR@5 | avg ms@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vector | N | N | N | 52.0% | 60.0% | 72.0% | 0.433 | 252 |
| vector+BM25 | Y | N | N | 56.0% | 64.0% | 64.0% | 0.450 | 247 |
| vector+BM25+PC | Y | Y | N | 64.0% | 72.0% | 80.0% | 0.558 | 229 |
| vector+BM25+RR | Y | N | Y | 76.0% | 80.0% | 84.0% | 0.643 | 4302 |
| full | Y | Y | Y | 92.0% | 92.0% | 96.0% | 0.740 | 3458 |

## 简历可用句

> 自建 SAR 领域 25 题 RAG 检索评估集，覆盖 txt、论文 PDF 和 poster；full pipeline 的 Recall@5 为 92.0%，相比 vector-only 基线 60.0% 变化 +32.0 pp。
