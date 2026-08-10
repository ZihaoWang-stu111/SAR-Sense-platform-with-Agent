from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval_rag25" / "results"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_scores(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("qa_id", "pipeline", "score"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing {key}")
            row["score"] = float(row["score"])
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default=str(RESULTS_DIR / "summary.csv"))
    parser.add_argument("--scores", default=str(RESULTS_DIR / "scores.jsonl"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "final_report.md"))
    args = parser.parse_args()

    retrieval = read_csv(Path(args.retrieval))
    scores = read_scores(Path(args.scores))
    question_count = len({row["qa_id"] for row in scores})

    by_pipeline = defaultdict(list)
    for row in scores:
        by_pipeline[row["pipeline"]].append(row["score"])

    lines = [
        f"# RAG {question_count}题最终评估",
        "",
        "## Retrieval Summary",
        "",
        "| pipeline | k | recall | mrr | avg latency ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in retrieval:
        lines.append(
            f"| {row['pipeline']} | {row['k']} | {float(row['recall']) * 100:.1f}% | "
            f"{float(row['mrr']):.3f} | {float(row['avg_latency_ms']):.0f} |"
        )

    lines += [
        "",
        "## Answer Correctness",
        "",
        "| pipeline | n | avg score | correct | partial | wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pipeline, vals in by_pipeline.items():
        lines.append(
            f"| {pipeline} | {len(vals)} | {sum(vals) / len(vals):.3f} | "
            f"{sum(1 for v in vals if v == 1.0)} | "
            f"{sum(1 for v in vals if v == 0.5)} | "
            f"{sum(1 for v in vals if v == 0.0)} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
