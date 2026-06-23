from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_rag25.metrics import first_hit_rank, normalize, reciprocal_rank

DATA_DIR = ROOT / "data"
DEFAULT_DATASET = ROOT / "eval_rag25" / "qa_dataset_25.json"
RESULTS_DIR = ROOT / "eval_rag25" / "results"
DEFAULT_KS = [3, 5, 10]
PIPELINE_ORDER = [
    "vector_only",
    "hybrid_no_pc_no_rr",
    "hybrid_pc_no_rr",
    "hybrid_no_pc_with_rr",
    "full",
]
PIPELINE_TOGGLES = {
    "vector_only": {"label": "vector", "bm25": False, "pc": False, "rr": False},
    "hybrid_no_pc_no_rr": {"label": "vector+BM25", "bm25": True, "pc": False, "rr": False},
    "hybrid_pc_no_rr": {"label": "vector+BM25+PC", "bm25": True, "pc": True, "rr": False},
    "hybrid_no_pc_with_rr": {"label": "vector+BM25+RR", "bm25": True, "pc": False, "rr": True},
    "full": {"label": "full", "bm25": True, "pc": True, "rr": True},
}
REQUIRED_FIELDS = {
    "id",
    "question",
    "gold_answer",
    "gold_filename",
    "gold_snippet",
    "source_type",
    "qa_type",
    "difficulty",
}


def load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must be a non-empty JSON list")

    seen = set()
    for qa in data:
        missing = REQUIRED_FIELDS - qa.keys()
        if missing:
            raise ValueError(f"{qa.get('id', '?')} missing fields: {sorted(missing)}")
        if qa["id"] in seen:
            raise ValueError(f"duplicate id: {qa['id']}")
        seen.add(qa["id"])
    return data


def _read_pdf_text(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for --validate-only on PDFs") from exc

    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _read_source_text(filename: str, source_type: str) -> str:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    if source_type == "pdf":
        return _read_pdf_text(path)
    return path.read_text(encoding="utf-8")


def validate_dataset(qa_list: list[dict]) -> None:
    if len(qa_list) != 25:
        raise ValueError(f"expected 25 questions, got {len(qa_list)}")

    cache: dict[tuple[str, str], str] = {}
    for qa in qa_list:
        key = (qa["gold_filename"], qa["source_type"])
        if key not in cache:
            cache[key] = _read_source_text(*key)
        if normalize(qa["gold_snippet"]) not in normalize(cache[key]):
            raise ValueError(f"{qa['id']} gold_snippet not found in {qa['gold_filename']}")


def run_one(pipeline_name: str, pipeline, qa: dict, k: int) -> dict:
    t0 = time.perf_counter()
    try:
        docs = pipeline(qa["question"], k)
        err = None
    except Exception as exc:
        docs = []
        err = str(exc)[:240]

    rank = first_hit_rank(docs, qa)
    return {
        "pipeline": pipeline_name,
        "qa_id": qa["id"],
        "k": k,
        "hit": bool(rank),
        "first_hit_rank": rank or "",
        "rr": reciprocal_rank(docs, qa),
        "n_returned": len(docs),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "error": err or "",
    }


def aggregate(rows: list[dict], pipeline_order: list[str]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["pipeline"], int(row["k"]))].append(row)

    out = []
    for pipeline_name in pipeline_order:
        for k in sorted({int(row["k"]) for row in rows}):
            bucket = grouped.get((pipeline_name, k), [])
            if not bucket:
                continue
            out.append({
                "pipeline": pipeline_name,
                "k": k,
                "n": len(bucket),
                "recall": sum(1 for r in bucket if r["hit"]) / len(bucket),
                "mrr": sum(float(r["rr"]) for r in bucket) / len(bucket),
                "avg_latency_ms": sum(float(r["latency_ms"]) for r in bucket) / len(bucket),
            })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown(path: Path, agg_rows: list[dict], n_qa: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted({int(r["k"]) for r in agg_rows})
    by_name_k = {(r["pipeline"], int(r["k"])): r for r in agg_rows}
    kmid = ks[len(ks) // 2]

    lines = [
        "# RAG 25题检索评估",
        "",
        f"- 题数: {n_qa}",
        f"- k: {ks}",
        "- 命中: filename 匹配 + gold_snippet 出现在返回文本中",
        "- full: hybrid children -> child rerank -> parent resolve",
        "",
        "## Recall / MRR",
        "",
        "| pipeline | BM25 | PC | RR | " + " | ".join(f"R@{k}" for k in ks) + f" | MRR@{kmid} | avg ms@{kmid} |",
        "|---|---:|---:|---:|" + "---:|" * (len(ks) + 2),
    ]
    for name, toggles in PIPELINE_TOGGLES.items():
        row = by_name_k.get((name, kmid))
        if not row:
            continue
        recalls = []
        for k in ks:
            r = by_name_k.get((name, k))
            recalls.append(_fmt_pct(r["recall"]) if r else "-")
        lines.append(
            f"| {toggles['label']} | {'Y' if toggles['bm25'] else 'N'} | "
            f"{'Y' if toggles['pc'] else 'N'} | {'Y' if toggles['rr'] else 'N'} | "
            + " | ".join(recalls)
            + f" | {row['mrr']:.3f} | {row['avg_latency_ms']:.0f} |"
        )

    base = by_name_k.get(("vector_only", kmid))
    full = by_name_k.get(("full", kmid))
    if base and full:
        delta = (full["recall"] - base["recall"]) * 100
        lines += [
            "",
            "## 简历可用句",
            "",
            (
                f"> 自建 SAR 领域 25 题 RAG 检索评估集，覆盖 txt、论文 PDF 和 poster；"
                f"full pipeline 的 Recall@{kmid} 为 {_fmt_pct(full['recall'])}，"
                f"相比 vector-only 基线 {_fmt_pct(base['recall'])} 变化 {delta:+.1f} pp。"
            ),
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_rag_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    from model.factory import chat_model
    from utils.prompt_loader import load_rag_prompts

    return PromptTemplate.from_template(load_rag_prompts()) | chat_model | StrOutputParser()


def generate_answers(qa_list: list[dict], pipelines: dict, answer_k: int, out: Path) -> int:
    chain = _build_rag_chain()
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for pipeline_name, pipeline in pipelines.items():
            for qa in qa_list:
                try:
                    docs = pipeline(qa["question"], answer_k)
                    context = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))
                    answer = chain.invoke({"input": qa["question"], "context": context}).strip()
                except Exception as exc:
                    docs = []
                    answer = f"[ERROR] {str(exc)[:240]}"
                row = {
                    "qa_id": qa["id"],
                    "pipeline": pipeline_name,
                    "k": answer_k,
                    "question": qa["question"],
                    "gold_answer": qa["gold_answer"],
                    "rag_answer": answer,
                    "retrieved_filenames": [doc.metadata.get("filename", "") for doc in docs],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-pipelines", nargs="+")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--with-answers", action="store_true")
    parser.add_argument("--skip-retrieval-eval", action="store_true")
    parser.add_argument("--answer-k", type=int, default=5)
    args = parser.parse_args()

    qa_list = load_dataset(Path(args.dataset))
    validate_dataset(qa_list)
    if args.validate_only:
        print(f"OK: validated {len(qa_list)} questions")
        return 0
    if args.limit:
        qa_list = qa_list[:args.limit]

    from eval_rag25.pipelines import PIPELINES

    pipelines = dict(PIPELINES)
    if args.only_pipelines:
        unknown = set(args.only_pipelines) - set(PIPELINES)
        if unknown:
            raise ValueError(f"unknown pipelines: {sorted(unknown)}")
        pipelines = {name: PIPELINES[name] for name in args.only_pipelines}

    suffix = args.output_suffix
    rows = []
    if not args.skip_retrieval_eval:
        total = len(qa_list) * len(args.ks) * len(pipelines)
        done = 0
        for pipeline_name, pipeline in pipelines.items():
            for k in args.ks:
                for qa in qa_list:
                    done += 1
                    rows.append(run_one(pipeline_name, pipeline, qa, k))
                    if done % 25 == 0 or done == total:
                        print(f"retrieval {done}/{total}")

        write_csv(RESULTS_DIR / f"results_raw{suffix}.csv", rows)
        failures = [row for row in rows if row["error"]]
        write_csv(RESULTS_DIR / f"failures{suffix}.csv", failures)
        agg_rows = aggregate(rows, list(pipelines))
        write_csv(RESULTS_DIR / f"summary{suffix}.csv", agg_rows)
        write_markdown(RESULTS_DIR / f"ablation_table{suffix}.md", agg_rows, len(qa_list))

    if args.with_answers:
        n = generate_answers(qa_list, pipelines, args.answer_k, RESULTS_DIR / f"answers{suffix}.jsonl")
        print(f"answers {n}")

    print(f"done: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
