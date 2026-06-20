"""
聚合检索评测 + 外部 judge 评分 → 最终消融报告。

输入：
    eval/results/results_raw.csv     —— evaluate.py 输出
    eval/results/scores.jsonl        —— 由外部 judge LLM 产出（schema 见 README）

输出：
    eval/results/ablation_table.md   —— 重写，加上 answer_correctness 列
    eval/results/answer_correctness.png  —— 各 pipeline 的 correctness 柱状图

scores.jsonl 一行一条记录：
    {"qa_id": "qa_001", "pipeline": "full", "score": 1.0, "reasoning": "..."}

score ∈ [0.0, 0.5, 1.0]：
    1.0 完全正确（答案与 gold_answer 语义一致或更详尽且不错）
    0.5 部分正确（关键标识词答对了，但有缺失或表述不清）
    0.0 错误 / 答非所问 / 检索为空 / 编造
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from eval.pipelines import PIPELINES
from eval.evaluate import (
    _PIPELINE_TOGGLES,
    _build_matrix_table,
    _build_contribution_table,
)
from utils.logger_handler import logger

RESULTS_DIR = ROOT / "eval" / "results"
CSV_PATH = RESULTS_DIR / "results_raw.csv"
SCORES_PATH = RESULTS_DIR / "scores.jsonl"
MD_PATH = RESULTS_DIR / "ablation_table.md"
CORR_PNG = RESULTS_DIR / "answer_correctness.png"


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}\n"
            "请先：\n"
            "  1) 跑 `python eval/evaluate.py --with-answers` 生成 answers.jsonl\n"
            "  2) 让外部 judge LLM 读 answers.jsonl 评分，保存为 scores.jsonl\n"
            "  3) 再回来跑 aggregate.py"
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} JSON 解析失败：{e}")
            for k in ("qa_id", "pipeline", "score"):
                if k not in rec:
                    raise ValueError(f"{path}:{i} 缺字段 {k}")
            try:
                rec["score"] = float(rec["score"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{i} score 不是数字：{rec.get('score')}")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path} 为空")
    return pd.DataFrame(rows)


def _retrieval_pivot(df: pd.DataFrame, metric: str, ks: list[int]) -> pd.DataFrame:
    grouped = df.groupby(["pipeline", "k"], sort=False).agg(
        recall=("hit", "mean"),
        mrr=("rr", "mean"),
    ).reset_index()
    pivot = grouped.pivot(index="pipeline", columns="k", values=metric)
    pivot = pivot.reindex(list(PIPELINES.keys())).reindex(columns=ks)
    if metric == "recall":
        pivot = pivot.map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-")
    else:
        pivot = pivot.map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
    pivot.columns = [f"k={k}" for k in pivot.columns]
    return pivot


def _correctness_table(scores: pd.DataFrame) -> pd.DataFrame:
    """每 pipeline 的平均 score（0-1） + 完全对/部分对/错的分布。"""
    grouped = scores.groupby("pipeline", sort=False).agg(
        n=("qa_id", "count"),
        correctness=("score", "mean"),
        n_correct=("score", lambda s: int((s == 1.0).sum())),
        n_partial=("score", lambda s: int((s == 0.5).sum())),
        n_wrong=("score", lambda s: int((s == 0.0).sum())),
    ).reindex(list(PIPELINES.keys()))
    return grouped


def _format_correctness_md(table: pd.DataFrame) -> pd.DataFrame:
    """配置矩阵风格的 correctness 表：含三个开关 + correctness 数值。"""
    rows = []
    for name, info in _PIPELINE_TOGGLES.items():
        if name not in table.index:
            continue
        r = table.loc[name]
        if pd.isnull(r["correctness"]):
            continue
        rows.append({
            "配置": info["label"],
            "混合": "✅" if info["hybrid"] else "❌",
            "父子块": "✅" if info["pc"] else "❌",
            "Reranker": "✅" if info["rr"] else "❌",
            "answer_correctness": f"{r['correctness']*100:.1f}%",
            "对/半对/错": f"{int(r['n_correct'])}/{int(r['n_partial'])}/{int(r['n_wrong'])}",
            "n": int(r["n"]),
        })
    return pd.DataFrame(rows)


def _plot_correctness(table: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        logger.warning(f"matplotlib 不可用，跳过画图：{e}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    pipelines = list(table.index)
    correctness = (table["correctness"].fillna(0) * 100).tolist()
    colors = ["#9aa0a6", "#4f87e6", "#5fb764", "#e5a445", "#c43c3c"][: len(pipelines)]
    bars = ax.bar(pipelines, correctness, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Answer Correctness (%)")
    ax.set_title("RAG 端到端答对率（外部 LLM judge 评分）")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar, v in zip(bars, correctness):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--scores", default=str(SCORES_PATH))
    parser.add_argument("--out-md", default=str(MD_PATH))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    scores_path = Path(args.scores)
    md_path = Path(args.out_md)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}，先跑 `python eval/evaluate.py`"
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    scores = _load_scores(scores_path)

    ks = sorted(df["k"].unique().tolist())
    n_qa = df["qa_id"].nunique()
    kmid = ks[len(ks) // 2]

    # 复用 evaluate.py 的检索矩阵 + 单独贡献逻辑
    agg_retrieval = df.groupby(["pipeline", "k"], sort=False).agg(
        recall=("hit", "mean"),
        mrr=("rr", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
    ).reset_index()
    matrix_md = _build_matrix_table(agg_retrieval, ks).to_markdown(index=False)
    contrib_df, gain = _build_contribution_table(agg_retrieval, ks)
    contrib_md = contrib_df.to_markdown(index=False)

    corr_table = _correctness_table(scores)
    corr_md = _format_correctness_md(corr_table).to_markdown(index=False)
    _plot_correctness(corr_table, CORR_PNG)

    # 简历友好的"hero number"提取
    full_corr = corr_table.loc["full", "correctness"] if "full" in corr_table.index else None
    base_corr = (
        corr_table.loc["vector_only", "correctness"]
        if "vector_only" in corr_table.index else None
    )
    full_recall_gain = gain.get("**全开 vs 基线**")

    hero_parts = [f"自建 SAR 领域 **{n_qa}** 题 RAG 评测集，对父子块 / BGE Reranker / 混合检索做消融。"]
    if full_recall_gain:
        hero_parts.append(
            f"**Recall@{kmid} 从 vector-only 的 {full_recall_gain['off']*100:.1f}% "
            f"提升到 full 的 {full_recall_gain['on']*100:.1f}%（"
            f"{'+' if full_recall_gain['delta_pp']>=0 else ''}{full_recall_gain['delta_pp']:.1f} pp）**"
        )
    if full_corr is not None and base_corr is not None:
        hero_parts.append(
            f"端到端答对率（Claude 评分）从 {base_corr*100:.1f}% 提升到 {full_corr*100:.1f}%。"
        )
    elif full_corr is not None:
        hero_parts.append(f"full pipeline 端到端答对率 {full_corr*100:.1f}%。")
    hero_line = "> " + " ".join(hero_parts)

    lines = [
        "# RAG 检索 + 生成消融评测",
        "",
        f"- 题集大小：**{n_qa}** 题",
        f"- k 值（检索）：{ks}",
        f"- judge 评分：{len(scores)} 条（外部 LLM 给的 0/0.5/1 三档）",
        "",
        "## 配置矩阵 × Recall@k",
        "",
        matrix_md,
        "",
        f"## 单独贡献（基于 Recall@{kmid}）",
        "",
        contrib_md,
        "",
        f"## Answer Correctness（k={kmid}，外部 judge 评分）",
        "",
        corr_md,
        "",
        "> `对/半对/错` = score=1.0 / 0.5 / 0.0 的题数分布。",
        "",
        "## 简历 hero sentence",
        "",
        hero_line,
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 70)
    print("✅ 聚合完成")
    print("=" * 70)
    print(f"  最终报告：{md_path}")
    print(f"  Correctness 图：{CORR_PNG}")
    print()
    print("Answer correctness 对比：")
    print(_format_correctness_md(corr_table).to_string())
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
