"""
RAG 检索质量评测指标。

is_hit 用 "文件名 + 文本片段包含" 的方式判定命中，
让父子块开/关的 pipeline 在同一标准下公平比较：
- 不开父子块：返回子块/小段，gold_snippet 可能就是该段一部分 → 命中
- 开父子块：返回父块大段，gold_snippet 也在父块里 → 命中
不依赖 chunk_id，避免不同 pipeline 的 ID 体系不一致。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from langchain_core.documents import Document


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """归一化：NFKC（半角化、统一全角符号）+ 折叠空白 + 小写。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub("", text)  # 中文场景下空白完全无意义，直接全删
    return text.lower()


def is_hit(retrieved_docs: Iterable[Document], qa: dict) -> bool:
    """
    判定：top-k 检索结果里是否存在"同文件 + 含 gold_snippet"的文档。
    """
    gold_filename = qa.get("gold_filename")
    snippet_norm = _normalize(qa.get("gold_snippet", ""))
    if not gold_filename or not snippet_norm:
        return False

    for doc in retrieved_docs:
        if doc.metadata.get("filename") != gold_filename:
            continue
        if snippet_norm in _normalize(doc.page_content):
            return True
    return False


def first_hit_rank(retrieved_docs: list[Document], qa: dict) -> int | None:
    """
    返回首次命中的 rank（1-based）；未命中返回 None。
    """
    gold_filename = qa.get("gold_filename")
    snippet_norm = _normalize(qa.get("gold_snippet", ""))
    if not gold_filename or not snippet_norm:
        return None

    for idx, doc in enumerate(retrieved_docs, start=1):
        if doc.metadata.get("filename") != gold_filename:
            continue
        if snippet_norm in _normalize(doc.page_content):
            return idx
    return None


def reciprocal_rank(retrieved_docs: list[Document], qa: dict) -> float:
    """MRR 的单题分量：1/rank，未命中返回 0。"""
    rank = first_hit_rank(retrieved_docs, qa)
    return 1.0 / rank if rank else 0.0
