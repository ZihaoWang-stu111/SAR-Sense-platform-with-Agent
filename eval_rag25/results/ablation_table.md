# RAG 40题检索评估

- 题数: 40
- k: [3, 5, 10]
- 命中: filename 匹配 + gold_snippet 出现在返回文本中
- full: hybrid children -> child rerank -> parent resolve

## Recall / MRR

| pipeline | BM25 | PC | RR | R@3 | R@5 | R@10 | MRR@5 | avg ms@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vector | N | N | N | 70.0% | 75.0% | 77.5% | 0.602 | 1678 |
| vector+BM25+PC | Y | Y | N | 72.5% | 85.0% | 97.5% | 0.657 | 1891 |
| vector+BM25+RR | Y | N | Y | 62.5% | 82.5% | 82.5% | 0.571 | 9665 |
| vector+RR+PC | N | Y | Y | 77.5% | 92.5% | 97.5% | 0.703 | 10836 |
| full | Y | Y | Y | 77.5% | 92.5% | 100.0% | 0.704 | 9628 |

## 简历可用句

> 自建 SAR 领域 40 题 RAG 检索评估集，覆盖 txt、论文 PDF 和 poster；full pipeline 的 Recall@5 为 92.5%，相比 vector-only 基线 75.0% 变化 +17.5 pp。
