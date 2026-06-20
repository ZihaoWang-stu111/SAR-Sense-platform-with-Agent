"""
RAG 评测主入口。

模式 A（仅检索，~30 分钟）：
    python eval/evaluate.py
跑 5 pipeline × 3 个 k 值，输出 Recall@k / MRR / 延迟对比表 + 曲线图。

模式 B（检索 + 生成答案，多 ~10-20 分钟）：
    python eval/evaluate.py --with-answers
在模式 A 基础上，再为每个 (qa, pipeline) 在 k=5 跑一遍完整 RAG（用项目内的
chat_model 做答），把答案存到 eval/results/answers.jsonl。然后把 JSONL 交给
更强的 judge LLM（如 Claude）在外部评分，再用 eval/aggregate.py 合并到最终
报告里 —— 整套流程见 eval/README.md。

依赖：
    - eval/qa_dataset.json 存在
    - 生产配置 chroma.yml 中 parent_child_enabled: true 且已重建索引
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from eval.metrics import is_hit, first_hit_rank, reciprocal_rank
from eval.pipelines import PIPELINES
from utils.logger_handler import logger

DEFAULT_DATASET = ROOT / "eval" / "qa_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_KS = [3, 5, 10]


def _load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。\n"
            "确认 eval/qa_dataset.json 已存在；如需重建请打开 README.md 看流程。"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} 不是非空列表")
    required = {"id", "question", "gold_filename", "gold_snippet"}
    for qa in data:
        missing = required - qa.keys()
        if missing:
            raise ValueError(f"题 {qa.get('id', '?')} 缺字段：{missing}")
    return data


def _run_one(pipeline_name, pipeline, qa, k, max_retries: int = 3) -> dict:
    """跑一个 (pipeline, qa, k) 组合，返回一行结果。

    对 SSL / 连接 / 超时类的瞬时错误自动重试 max_retries 次（指数退避），
    避免 DashScope embedding API 偶发抖动污染评测数据。
    """
    transient_keywords = ("SSL", "Timeout", "Connection", "Max retries", "ReadTimeout")
    last_err: Exception | None = None
    t0 = time.perf_counter()

    for attempt in range(max_retries + 1):
        try:
            docs = pipeline(qa["question"], k=k)
            break
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_transient = any(kw in err_str for kw in transient_keywords)
            if attempt < max_retries and is_transient:
                wait = 2 ** attempt  # 1s, 2s, 4s 指数退避
                logger.warning(
                    f"[{pipeline_name}] qa={qa['id']} k={k} 瞬时错误，{wait}s 后重试 "
                    f"({attempt+1}/{max_retries})：{err_str[:120]}"
                )
                time.sleep(wait)
                continue
            # 非瞬时错误或重试用尽
            logger.warning(f"[{pipeline_name}] qa={qa['id']} k={k} 失败: {err_str[:200]}")
            return {
                "pipeline": pipeline_name,
                "qa_id": qa["id"],
                "k": k,
                "hit": False,
                "first_hit_rank": None,
                "rr": 0.0,
                "n_returned": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": err_str[:200],
            }
    else:
        # 理论上不会走到（break 或 return 提前退出），但兜底
        return {
            "pipeline": pipeline_name, "qa_id": qa["id"], "k": k,
            "hit": False, "first_hit_rank": None, "rr": 0.0,
            "n_returned": 0, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": str(last_err)[:200] if last_err else "unknown",
        }

    hit = is_hit(docs, qa)
    rank = first_hit_rank(docs, qa)
    rr = reciprocal_rank(docs, qa)
    return {
        "pipeline": pipeline_name,
        "qa_id": qa["id"],
        "k": k,
        "hit": hit,
        "first_hit_rank": rank,
        "rr": rr,
        "n_returned": len(docs),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "error": None,
    }


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """每 (pipeline, k) 算 recall@k / MRR / hit_rate。"""
    grouped = df.groupby(["pipeline", "k"], sort=False).agg(
        recall=("hit", "mean"),
        mrr=("rr", "mean"),
        hit_rate=("hit", "mean"),  # 与 recall 同义，留双列方便阅读
        n=("qa_id", "count"),
        avg_latency_ms=("latency_ms", "mean"),
    ).reset_index()
    return grouped


def _format_pivot(agg: pd.DataFrame, metric: str, ks: list[int]) -> pd.DataFrame:
    """
    pipeline × k 透视，cell 是百分比字符串 / MRR 保留 3 位小数。
    """
    pivot = agg.pivot(index="pipeline", columns="k", values=metric)
    pipeline_order = list(PIPELINES.keys())
    pivot = pivot.reindex(pipeline_order).reindex(columns=ks)
    if metric in ("recall", "hit_rate"):
        pivot = pivot.map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-")
    else:
        pivot = pivot.map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
    pivot.columns = [f"k={k}" for k in pivot.columns]
    return pivot


# 每个 pipeline 三个开关的开/关状态，用于"配置矩阵"行标
_PIPELINE_TOGGLES = {
    "vector_only":          {"label": "vector-only",         "hybrid": False, "pc": False, "rr": False},
    "hybrid_no_pc_no_rr":   {"label": "+BM25",               "hybrid": True,  "pc": False, "rr": False},
    "hybrid_pc_no_rr":      {"label": "+BM25 +PC",           "hybrid": True,  "pc": True,  "rr": False},
    "hybrid_no_pc_with_rr": {"label": "+BM25 +Reranker",     "hybrid": True,  "pc": False, "rr": True},
    "full":                 {"label": "full（PC→RR）",       "hybrid": True,  "pc": True,  "rr": True},
    "full_v2_rr_then_pc":   {"label": "full_v2（RR→PC）",    "hybrid": True,  "pc": True,  "rr": True},
}


def _build_matrix_table(agg: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    """
    构造"配置矩阵"主表：一行一个 pipeline，列里直接画 ✅/❌ 三个开关 + 各 k 的 Recall + MRR。
    """
    recall = agg.pivot(index="pipeline", columns="k", values="recall")
    mrr = agg.pivot(index="pipeline", columns="k", values="mrr")
    rows = []
    for name, info in _PIPELINE_TOGGLES.items():
        if name not in recall.index:
            continue
        row = {
            "配置": info["label"],
            "混合": "✅" if info["hybrid"] else "❌",
            "父子块": "✅" if info["pc"] else "❌",
            "Reranker": "✅" if info["rr"] else "❌",
        }
        for k in ks:
            v = recall.loc[name, k] if k in recall.columns else None
            row[f"R@{k}"] = f"{v*100:.1f}%" if pd.notnull(v) else "-"
        # 用中间那个 k 的 MRR 作主指标（最常被引用）
        kmid = ks[len(ks) // 2]
        v = mrr.loc[name, kmid] if kmid in mrr.columns else None
        row[f"MRR@{kmid}"] = f"{v:.3f}" if pd.notnull(v) else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def _build_contribution_table(agg: pd.DataFrame, ks: list[int]) -> tuple[pd.DataFrame, dict]:
    """
    "单独贡献"表：每个模块单独开 vs 关时，Recall@k_mid 的 Δ。
    这样可以直接讲"父子块独贡献 +X pp"这种结论。

    返回 (df, gain_dict) —— gain_dict 用于自动填充 hero sentence。
    """
    kmid = ks[len(ks) // 2]
    recall = agg.pivot(index="pipeline", columns="k", values="recall")[kmid]

    def _r(name):
        return recall[name] if name in recall.index else None

    pairs = [
        ("BM25 混合检索",  "vector_only",        "hybrid_no_pc_no_rr"),
        ("父子块",         "hybrid_no_pc_no_rr", "hybrid_pc_no_rr"),
        ("BGE Reranker",   "hybrid_no_pc_no_rr", "hybrid_no_pc_with_rr"),
        ("**全开 vs 基线**", "vector_only",        "full"),
    ]
    rows = []
    gain_dict = {}
    for label, off, on in pairs:
        v_off, v_on = _r(off), _r(on)
        if v_off is None or v_on is None:
            continue
        delta = (v_on - v_off) * 100
        rows.append({
            "对比项": label,
            "关": f"{v_off*100:.1f}%（{_PIPELINE_TOGGLES[off]['label']}）",
            "开": f"{v_on*100:.1f}%（{_PIPELINE_TOGGLES[on]['label']}）",
            f"ΔR@{kmid}": f"{'+' if delta >= 0 else ''}{delta:.1f} pp",
        })
        gain_dict[label] = {"off": v_off, "on": v_on, "delta_pp": delta}
    return pd.DataFrame(rows), gain_dict


def _write_markdown(agg: pd.DataFrame, ks: list[int], n_qa: int, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    kmid = ks[len(ks) // 2]
    matrix = _build_matrix_table(agg, ks)
    contrib, gain = _build_contribution_table(agg, ks)

    # 自动填 hero sentence 的数字
    full_gain = gain.get("**全开 vs 基线**")
    hero_line = (
        f"> 自建 SAR 领域 **{n_qa}** 题 RAG 评测集，对父子块 / BGE Reranker / 混合检索做消融。"
    )
    if full_gain:
        hero_line += (
            f"**Recall@{kmid} 从 vector-only 基线的 {full_gain['off']*100:.1f}% "
            f"提升到 full pipeline 的 {full_gain['on']*100:.1f}%（"
            f"{'+' if full_gain['delta_pp'] >= 0 else ''}{full_gain['delta_pp']:.1f} pp）**"
        )
        for k_label in ("BM25 混合检索", "父子块", "BGE Reranker"):
            g = gain.get(k_label)
            if g:
                hero_line += f"；{k_label}独贡献 {'+' if g['delta_pp'] >= 0 else ''}{g['delta_pp']:.1f} pp"
        hero_line += "。"

    lines = [
        "# RAG 检索质量消融评测",
        "",
        f"- 题集大小：**{n_qa}** 题",
        f"- k 值：{ks}",
        "- 命中判定：`gold_filename` 匹配 + `gold_snippet` 包含（NFKC + 去空格 + lowercase）",
        "",
        "## 配置矩阵 × Recall@k",
        "",
        "每行是一个 pipeline 配置，前 3 列直接显示三个开关的开/关状态。",
        "",
        matrix.to_markdown(index=False),
        "",
        f"## 单独贡献（基于 Recall@{kmid}）",
        "",
        f"每个模块单独开 vs 关在 R@{kmid} 上的差值，单位 pp（百分点）。",
        "",
        contrib.to_markdown(index=False),
        "",
        "## 完整 Recall@k 矩阵（pipeline × k）",
        "",
        _format_pivot(agg, "recall", ks).to_markdown(),
        "",
        "## MRR（首次命中倒数排名均值）",
        "",
        _format_pivot(agg, "mrr", ks).to_markdown(),
        "",
        "## 平均延迟 (ms)",
        "",
        agg.pivot(index="pipeline", columns="k", values="avg_latency_ms")
        .reindex(list(PIPELINES.keys()))
        .reindex(columns=ks)
        .map(lambda x: f"{x:.0f}" if pd.notnull(x) else "-")
        .rename(columns=lambda k: f"k={k}")
        .to_markdown(),
        "",
        "## 简历 hero sentence",
        "",
        hero_line,
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_chart(agg: pd.DataFrame, ks: list[int], out: Path) -> None:
    """5 条曲线，x=k，y=recall。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 中文字体（项目里已经踩过的坑：app.py 里这么配的）
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        logger.warning(f"matplotlib 不可用，跳过画图：{e}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for pipeline_name in PIPELINES.keys():
        sub = agg[agg["pipeline"] == pipeline_name].sort_values("k")
        if sub.empty:
            continue
        ax.plot(
            sub["k"], sub["recall"] * 100,
            marker="o", label=pipeline_name, linewidth=2,
        )
    ax.set_xticks(ks)
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Recall@k (%)")
    ax.set_title("RAG 检索质量消融：Recall@k")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ============== 生成答案（--with-answers 模式） ==============

def _build_rag_chain():
    """复用项目的 RAG prompt 和 chat_model，组装一个简单 chain。"""
    from utils.prompt_loader import load_rag_prompts
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from model.factory import chat_model
    prompt = PromptTemplate.from_template(load_rag_prompts())
    return prompt | chat_model | StrOutputParser()


def _render_answer(chain, question: str, docs) -> str:
    """给定问题和检索文档，调 chat_model 生成答案。"""
    if not docs:
        return "[空：检索结果为空]"
    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
    try:
        return chain.invoke({"input": question, "context": context}).strip()
    except Exception as e:
        return f"[错误: {str(e)[:200]}]"


def _generate_answers(qa_list: list[dict], answer_k: int, out: Path,
                      pipelines: dict | None = None) -> int:
    """
    对每个 (qa, pipeline) 在固定 k 跑完整 RAG，把答案写到 JSONL。
    返回写入的行数。
    """
    if pipelines is None:
        pipelines = PIPELINES
    chain = _build_rag_chain()
    total = len(qa_list) * len(pipelines)
    counter = 0
    t_start = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for pipeline_name, pipeline in pipelines.items():
            for qa in qa_list:
                counter += 1
                t0 = time.perf_counter()
                try:
                    docs = pipeline(qa["question"], k=answer_k)
                except Exception as e:
                    docs = []
                    logger.warning(
                        f"[{pipeline_name}] qa={qa['id']} 检索失败: {e}"
                    )
                rag_answer = _render_answer(chain, qa["question"], docs)
                row = {
                    "qa_id": qa["id"],
                    "pipeline": pipeline_name,
                    "k": answer_k,
                    "question": qa["question"],
                    "gold_answer": qa.get("gold_answer", ""),
                    "rag_answer": rag_answer,
                    "retrieved_filenames": [
                        d.metadata.get("filename", "?") for d in docs
                    ],
                    "n_retrieved": len(docs),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if counter % 10 == 0:
                    logger.info(
                        f"答案生成 {counter}/{total} "
                        f"（{counter/total*100:.0f}%，已耗 {time.perf_counter()-t_start:.0f}s）"
                    )
    return counter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--limit", type=int, default=None,
                        help="限制评测题数（调试用）")
    parser.add_argument("--with-answers", action="store_true",
                        help="额外为每个 (qa, pipeline) 生成 RAG 答案，输出 answers.jsonl，"
                             "供外部 LLM judge 评分")
    parser.add_argument("--answer-k", type=int, default=5,
                        help="生成答案时使用的 top-k（默认 5，仅 --with-answers 生效）")
    parser.add_argument("--skip-retrieval-eval", action="store_true",
                        help="跳过检索评测，仅生成答案（仅 --with-answers 生效；用于已经"
                             "跑过检索评测、想重新生成答案的场景）")
    parser.add_argument("--only-pipelines", nargs="+", default=None,
                        help="只跑指定的 pipeline 名（用于 A/B 对照实验）；"
                             "默认跑全部 PIPELINES")
    parser.add_argument("--output-suffix", default="",
                        help="输出文件名后缀，避免覆盖已有结果。"
                             "如 --output-suffix _ab 会写到 results_raw_ab.csv")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    qa_list = _load_dataset(dataset_path)
    if args.limit:
        qa_list = qa_list[: args.limit]

    logger.info(f"评测题集：{dataset_path}（{len(qa_list)} 题）")
    # 处理 --only-pipelines 过滤
    pipelines_to_run = dict(PIPELINES)
    if args.only_pipelines:
        pipelines_to_run = {n: p for n, p in PIPELINES.items() if n in args.only_pipelines}
        unknown = set(args.only_pipelines) - set(PIPELINES.keys())
        if unknown:
            raise ValueError(f"未知 pipeline: {unknown}。可选: {list(PIPELINES.keys())}")
        if not pipelines_to_run:
            raise ValueError("--only-pipelines 过滤后没有任何 pipeline")
    logger.info(f"Pipeline：{list(pipelines_to_run.keys())}")
    logger.info(f"k 值：{args.ks}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix
    csv_path = RESULTS_DIR / f"results_raw{suffix}.csv"
    md_path = RESULTS_DIR / f"ablation_table{suffix}.md"
    png_path = RESULTS_DIR / f"recall_at_k{suffix}.png"
    answers_path = RESULTS_DIR / f"answers{suffix}.jsonl"

    if not args.skip_retrieval_eval:
        rows = []
        total = len(pipelines_to_run) * len(args.ks) * len(qa_list)
        counter = 0
        t_start = time.perf_counter()
        for pipeline_name, pipeline in pipelines_to_run.items():
            for k in args.ks:
                for qa in qa_list:
                    counter += 1
                    row = _run_one(pipeline_name, pipeline, qa, k)
                    rows.append(row)
                    if counter % 25 == 0:
                        logger.info(
                            f"检索评测 {counter}/{total} "
                            f"（{counter/total*100:.0f}%，已耗 {time.perf_counter()-t_start:.0f}s）"
                        )

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        agg = _aggregate(df)
        _write_markdown(agg, args.ks, len(qa_list), md_path)
        _write_chart(agg, args.ks, png_path)
    else:
        logger.info("跳过检索评测（--skip-retrieval-eval）")
        agg = None

    if args.with_answers:
        logger.info(
            f"开始生成 RAG 答案：{len(qa_list)} 题 × {len(pipelines_to_run)} pipeline，k={args.answer_k}"
        )
        n = _generate_answers(qa_list, args.answer_k, answers_path, pipelines_to_run)
        logger.info(f"生成完成：{n} 行 → {answers_path}")

    print()
    print("=" * 70)
    print("✅ 评测完成")
    print("=" * 70)
    if agg is not None:
        print(f"  原始记录：{csv_path}")
        print(f"  消融表：{md_path}")
        print(f"  图：{png_path}")
        print()
        print("Recall@k 对比：")
        print(_format_pivot(agg, "recall", args.ks).to_string())
        print()
        print("MRR 对比：")
        print(_format_pivot(agg, "mrr", args.ks).to_string())
    if args.with_answers:
        print()
        print(f"  答案 JSONL：{answers_path}")
        print()
        print("⚠️  下一步：把 answers.jsonl 交给外部 judge LLM 评分（见 README.md）")
        print("    1. 让 Claude 读 answers.jsonl，输出 scores.jsonl")
        print("    2. 跑 `python eval/aggregate.py` 合并到最终报告")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
