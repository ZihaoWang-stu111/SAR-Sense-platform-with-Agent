"""
LLM 自动生成 QA 草稿。

跑完后 **必须手工审核** → 改名为 eval/qa_dataset.json 才能给 evaluate.py 用。

用法：
    python eval/generate_qa.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# 让本脚本能从仓库根目录单独跑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import chat_model
from utils.config_handler import chroma_conf
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# ============== 配置 ==============

# 跳过的非领域文件（子串匹配，命中即跳过）
BLOCKLIST_SUBSTRINGS = ["王子豪", "posterWZH"]

# 每段 ~600 字给 LLM 出题：足够 LLM 抓上下文，又不会 token 爆炸
PASSAGE_CHUNK_SIZE = 600
PASSAGE_CHUNK_OVERLAP = 80

# 每段最多生成的 QA 数（LLM 可能少出，少出也 ok）
MAX_QA_PER_PASSAGE = 2

OUTPUT_PATH = ROOT / "eval" / "qa_dataset_raw.json"

PROMPT_TEMPLATE = """你是 SAR 舰船检测领域的资深研究员。下面给你一段技术资料原文，请基于它出 {n_qa} 道高质量检索评测题。

【硬要求】
1. 问题必须能从给定原文中找到明确答案；避免开放性、主观性、推断性问题。
2. gold_answer ≤ 50 字，必须含原文中可精确定位的关键标识词（型号名 / 数据集名 / 数值指标 / 专有术语）。
3. gold_snippet：从原文 verbatim 截取 30-80 字的连续片段（不要跨段、不要替换字符、不要总结），必须包含 gold_answer 的关键标识词，且区分度高（即只在该段出现，不易和其它段重复）。
4. qa_type ∈ ["fact", "concept", "comparison"]：
   - fact: 数值或标识查询（如 "mAP 是多少"）
   - concept: 概念解释（如 "什么是 EEWB"）
   - comparison: 对比 / 差异（如 "v1.0 和 v1.1 的区别"）
5. 严格输出 JSON 数组。不要 markdown 代码块、不要任何额外解释文字。

【原文】
\"\"\"
{passage}
\"\"\"

【输出格式】
[
  {{"question": "...", "gold_answer": "...", "gold_snippet": "...", "qa_type": "fact"}}
]
"""

_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """与 metrics.py 一致的归一化：用于校验 gold_snippet 是否真在原文里。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub("", s)
    return s.lower()


def _is_domain_file(filename: str) -> bool:
    return not any(s in filename for s in BLOCKLIST_SUBSTRINGS)


def _load_file_text(path: Path) -> str:
    """合并文件所有 page_content。txt 一般就 1 个 doc，PDF 是每页一个。"""
    p = str(path)
    if p.endswith(".txt"):
        docs = text_loader(p)
    elif p.endswith(".pdf"):
        docs = pdf_loader(p)
    else:
        return ""
    return "\n\n".join(d.page_content for d in docs if d.page_content)


def _split_passages(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PASSAGE_CHUNK_SIZE,
        chunk_overlap=PASSAGE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def _parse_llm_output(raw: str) -> list[dict]:
    """容错解析：剥 markdown 代码块、抽最外层 JSON 数组。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}")
        return []
    return items if isinstance(items, list) else []


def _validate_qa(item: dict, passage: str) -> dict | None:
    """字段齐全 + gold_snippet 真的在 passage 里 → 标准化 dict；否则 None。"""
    required = {"question", "gold_answer", "gold_snippet", "qa_type"}
    if not isinstance(item, dict) or not required.issubset(item.keys()):
        return None
    snippet = (item.get("gold_snippet") or "").strip()
    if not snippet or len(snippet) < 10:
        return None
    if _normalize(snippet) not in _normalize(passage):
        # LLM 编了，没在原文里 → 直接丢弃
        return None
    return {
        "question": str(item["question"]).strip(),
        "gold_answer": str(item["gold_answer"]).strip(),
        "gold_snippet": snippet,
        "qa_type": str(item.get("qa_type", "fact")).strip(),
    }


def _generate_qa_for_passage(passage: str) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(passage=passage, n_qa=MAX_QA_PER_PASSAGE)
    try:
        result = chat_model.invoke(prompt)
    except Exception as e:
        logger.warning(f"LLM 调用失败: {e}")
        return []
    raw = result.content if hasattr(result, "content") else str(result)
    items = _parse_llm_output(raw)
    return [v for item in items if (v := _validate_qa(item, passage)) is not None]


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def main() -> int:
    data_dir = get_abs_path(chroma_conf["data_path"])
    allowed = chroma_conf["allow_knowledge_file_type"]
    all_paths = listdir_with_allowed_type(data_dir, tuple(allowed))

    domain_paths = [p for p in all_paths if _is_domain_file(os.path.basename(p))]
    skipped = [p for p in all_paths if not _is_domain_file(os.path.basename(p))]
    logger.info(
        f"评测语料：{len(domain_paths)} 个领域文件，跳过 {len(skipped)} 个非领域文件"
    )
    for p in skipped:
        logger.info(f"  ↳ 跳过：{os.path.basename(p)}")

    all_qa: list[dict] = []
    qa_id = 0

    for path in domain_paths:
        filename = os.path.basename(path)
        try:
            text = _load_file_text(Path(path))
        except Exception as e:
            logger.warning(f"加载 {filename} 失败：{e}")
            continue
        if not text.strip():
            continue

        passages = _split_passages(text)
        logger.info(f"[{filename}] 切出 {len(passages)} 段，开始生成 QA …")

        for i, passage in enumerate(passages, 1):
            qa_items = _generate_qa_for_passage(passage)
            for q in qa_items:
                qa_id += 1
                q["id"] = f"qa_{qa_id:03d}"
                q["gold_filename"] = filename
                q["_source_chunk_text"] = passage  # 给人审核时对照
                all_qa.append(q)
            logger.info(f"  段 {i}/{len(passages)} → +{len(qa_items)} QA")

            # 每 10 段落盘一次，断了也不丢前面的
            if i % 10 == 0:
                _atomic_write_json(OUTPUT_PATH, all_qa)

    _atomic_write_json(OUTPUT_PATH, all_qa)

    print()
    print("=" * 70)
    print(f"✅ 生成完成：{len(all_qa)} 道 QA → {OUTPUT_PATH}")
    print("=" * 70)
    print("⚠️  下一步是手工审核：")
    print(f"   1. 打开 {OUTPUT_PATH}")
    print("   2. 删劣题、修错答案、必要时调短或替换 gold_snippet")
    print("   3. 留 50-60 道精品，另存为 eval/qa_dataset.json")
    print("   4. 跑 `python eval/evaluate.py`")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
