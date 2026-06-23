from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return _WS_RE.sub("", text).lower()


def filename_match(actual: str | None, expected: str) -> bool:
    if not actual:
        return False
    actual_name = str(actual).replace("\\", "/").rsplit("/", 1)[-1]
    return actual_name == expected


def first_hit_rank(retrieved_docs: list[Any], qa: dict) -> int | None:
    gold_filename = qa.get("gold_filename", "")
    snippet = normalize(qa.get("gold_snippet", ""))
    if not gold_filename or not snippet:
        return None

    for rank, doc in enumerate(retrieved_docs, 1):
        if not filename_match(doc.metadata.get("filename"), gold_filename):
            continue
        if snippet in normalize(doc.page_content):
            return rank
    return None


def is_hit(retrieved_docs: Iterable[Any], qa: dict) -> bool:
    return first_hit_rank(list(retrieved_docs), qa) is not None


def reciprocal_rank(retrieved_docs: list[Any], qa: dict) -> float:
    rank = first_hit_rank(retrieved_docs, qa)
    return 1.0 / rank if rank else 0.0
