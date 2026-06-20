# RAG 检索质量消融评测

- 题集大小：**50** 题
- k 值：[3, 5, 10]
- 命中判定：`gold_filename` 匹配 + `gold_snippet` 包含（NFKC + 去空格 + lowercase）

## 配置矩阵 × Recall@k

每行是一个 pipeline 配置，前 3 列直接显示三个开关的开/关状态。

| 配置             | 混合   | 父子块   | Reranker   | R@3   | R@5   | R@10   |   MRR@5 |
|:-----------------|:-------|:---------|:-----------|:------|:------|:-------|--------:|
| full（PC→RR）    | ✅     | ✅       | ✅         | 96.0% | 96.0% | 98.0%  |   0.82  |
| full_v2（RR→PC） | ✅     | ✅       | ✅         | 94.0% | 96.0% | 98.0%  |   0.842 |

## 单独贡献（基于 Recall@5）

每个模块单独开 vs 关在 R@5 上的差值，单位 pp（百分点）。



## 完整 Recall@k 矩阵（pipeline × k）

| pipeline             | k=3   | k=5   | k=10   |
|:---------------------|:------|:------|:-------|
| vector_only          | -     | -     | -      |
| hybrid_no_pc_no_rr   | -     | -     | -      |
| hybrid_pc_no_rr      | -     | -     | -      |
| hybrid_no_pc_with_rr | -     | -     | -      |
| full                 | 96.0% | 96.0% | 98.0%  |
| full_v2_rr_then_pc   | 94.0% | 96.0% | 98.0%  |

## MRR（首次命中倒数排名均值）

| pipeline             | k=3   | k=5   | k=10   |
|:---------------------|:------|:------|:-------|
| vector_only          | -     | -     | -      |
| hybrid_no_pc_no_rr   | -     | -     | -      |
| hybrid_pc_no_rr      | -     | -     | -      |
| hybrid_no_pc_with_rr | -     | -     | -      |
| full                 | 0.820 | 0.820 | 0.809  |
| full_v2_rr_then_pc   | 0.837 | 0.842 | 0.844  |

## 平均延迟 (ms)

| pipeline             | k=3   | k=5   | k=10   |
|:---------------------|:------|:------|:-------|
| vector_only          | -     | -     | -      |
| hybrid_no_pc_no_rr   | -     | -     | -      |
| hybrid_pc_no_rr      | -     | -     | -      |
| hybrid_no_pc_with_rr | -     | -     | -      |
| full                 | 5990  | 5774  | 10533  |
| full_v2_rr_then_pc   | 2182  | 2044  | 3473   |

## 简历 hero sentence

> 自建 SAR 领域 **50** 题 RAG 评测集，对父子块 / BGE Reranker / 混合检索做消融。
