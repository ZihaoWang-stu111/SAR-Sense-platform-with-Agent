import pytest
from langchain_core.documents import Document

from rag import rag_service


class FakeChain:
    def __init__(self, answer):
        self.answer = answer
        self.invocations = []

    def invoke(self, payload):
        self.invocations.append(payload)
        return self.answer


def test_render_grounded_answer_converts_valid_citations():
    answer = "结论一[[EVIDENCE:1]]，结论二[[EVIDENCE:2]]，再次引用[[EVIDENCE:1]]。"

    assert rag_service.render_grounded_answer(
        answer, source_count=2
    ) == "结论一[1]，结论二[2]，再次引用[1]。"


@pytest.mark.parametrize(
    "answer",
    [
        "结论[[[EVIDENCE:1]]]",
        "结论[[EVIDENCE:1]]]",
    ],
)
def test_render_grounded_answer_rejects_citations_with_extra_brackets(answer):
    assert (
        rag_service.render_grounded_answer(answer, source_count=1)
        == rag_service.GROUNDING_FALLBACK
    )


def test_render_grounded_answer_rejects_answer_without_citation():
    assert (
        rag_service.render_grounded_answer("只有结论，没有证据引用。", source_count=2)
        == rag_service.GROUNDING_FALLBACK
    )


def test_render_grounded_answer_rejects_none_answer():
    assert (
        rag_service.render_grounded_answer(None, source_count=1)
        == rag_service.GROUNDING_FALLBACK
    )


@pytest.mark.parametrize(
    ("answer", "source_count"),
    [
        ("结论[[EVIDENCE:1]]", None),
        ("结论[[EVIDENCE:1]]", "1"),
        ("[[INSUFFICIENT]] 没有直接证据。", -1),
        ("结论[[EVIDENCE:1]]", True),
    ],
)
def test_render_grounded_answer_rejects_invalid_source_count(answer, source_count):
    assert (
        rag_service.render_grounded_answer(answer, source_count=source_count)
        == rag_service.GROUNDING_FALLBACK
    )


def test_insufficient_explanation_allows_zero_source_count_without_citation():
    assert (
        rag_service.render_grounded_answer(
            "[[INSUFFICIENT]] 没有直接证据。", source_count=0
        )
        == "没有直接证据。"
    )


@pytest.mark.parametrize(
    ("answer", "source_count"),
    [
        ("非法引用[[EVIDENCE:x]]", 2),
        ("零号引用[[EVIDENCE:0]]", 2),
        ("越界引用[[EVIDENCE:3]]", 2),
        ("未闭合引用[[EVIDENCE:1", 2),
        ("合法与非法混合[[EVIDENCE:1]][[EVIDENCE:nope]]", 2),
    ],
)
def test_render_grounded_answer_rejects_invalid_citations(answer, source_count):
    assert (
        rag_service.render_grounded_answer(answer, source_count)
        == rag_service.GROUNDING_FALLBACK
    )


def test_render_grounded_answer_keeps_insufficient_explanation_without_citation():
    answer = "  [[INSUFFICIENT]] 资料只描述了传感器，没有给出检测精度。  "

    assert rag_service.render_grounded_answer(
        answer, source_count=2
    ) == "资料只描述了传感器，没有给出检测精度。"


def test_render_grounded_answer_converts_valid_citation_after_insufficient_marker():
    answer = "[[INSUFFICIENT]] 现有资料仅支持局部结论[[EVIDENCE:1]]。"

    assert rag_service.render_grounded_answer(
        answer, source_count=1
    ) == "现有资料仅支持局部结论[1]。"


@pytest.mark.parametrize(
    "marker",
    [
        "[[EVIDENCE:0]]",
        "[[EVIDENCE:x]]",
        "[[EVIDENCE:2]]",
        "[[EVIDENCE:1",
    ],
)
def test_render_grounded_answer_rejects_invalid_citation_after_insufficient_marker(
    marker,
):
    answer = f"[[INSUFFICIENT]] 缺失说明{marker}"

    assert (
        rag_service.render_grounded_answer(answer, source_count=1)
        == rag_service.GROUNDING_FALLBACK
    )


def test_render_grounded_answer_uses_fallback_for_empty_insufficient_explanation():
    assert (
        rag_service.render_grounded_answer(" [[INSUFFICIENT]]  ", source_count=2)
        == rag_service.GROUNDING_FALLBACK
    )


@pytest.mark.parametrize(
    "residual_marker",
    [
        "[[evidence:x]]",
        "[EVIDENCE:x]]",
        "[[ EVIDENCE:2]]",
    ],
)
def test_render_grounded_answer_rejects_suspicious_residual_markers(residual_marker):
    answer = f"可靠结论[[EVIDENCE:1]]，残留标记{residual_marker}"

    assert (
        rag_service.render_grounded_answer(answer, source_count=1)
        == rag_service.GROUNDING_FALLBACK
    )


def test_render_grounded_answer_handles_overlong_numeric_index(monkeypatch):
    overlong_index = "9" * 5000

    def int_with_digit_limit(value):
        if len(value) > 4300:
            raise ValueError("Exceeds the limit for integer string conversion")
        return int(value)

    monkeypatch.setattr(rag_service, "int", int_with_digit_limit, raising=False)

    assert (
        rag_service.render_grounded_answer(
            f"结论[[EVIDENCE:{overlong_index}]]", source_count=1
        )
        == rag_service.GROUNDING_FALLBACK
    )


def test_rag_summarize_keeps_existing_refusal_when_no_docs_are_retrieved():
    chain = FakeChain("不应调用")
    service = object.__new__(rag_service.RagSummarizeService)
    service.chain = chain
    service.retriever_docs = lambda query, allowed_doc_ids=None: []

    result = service.rag_summarize("未知问题")

    assert result == "知识库中未检索到与该问题相关的可靠资料，无法基于知识库回答。"
    assert chain.invocations == []


def test_rag_summarize_appends_backend_sources_after_grounding_fallback():
    docs = [
        Document(
            page_content="资料正文",
            metadata={
                "filename": "source.txt",
                "chunk_id": "chunk-9",
                "rerank_score": 0.5,
            },
        )
    ]
    chain = FakeChain("没有证据标记的回答")
    service = object.__new__(rag_service.RagSummarizeService)
    service.chain = chain
    service.retriever_docs = lambda query, allowed_doc_ids=None: docs

    result = service.rag_summarize("问题")

    assert result == (
        f"{rag_service.GROUNDING_FALLBACK}\n\n"
        "参考来源：\n"
        "[1] source.txt | chunk_id=chunk-9 | score=0.5000"
    )


def test_rag_summarize_handles_none_answer_and_appends_backend_sources():
    docs = [
        Document(
            page_content="资料正文",
            metadata={
                "filename": "source.txt",
                "chunk_id": "chunk-9",
                "rerank_score": 0.5,
            },
        )
    ]
    chain = FakeChain(None)
    service = object.__new__(rag_service.RagSummarizeService)
    service.chain = chain
    service.retriever_docs = lambda query, allowed_doc_ids=None: docs

    result = service.rag_summarize("问题")

    assert result == (
        f"{rag_service.GROUNDING_FALLBACK}\n\n"
        "参考来源：\n"
        "[1] source.txt | chunk_id=chunk-9 | score=0.5000"
    )


def test_rag_summarize_builds_evidence_context_and_appends_backend_sources():
    docs = [
        Document(
            page_content="该方法在复杂海况下提升了检测精度。",
            metadata={
                "filename": "paper.pdf",
                "chunk_id": "chunk-1",
                "parent_index": 4,
                "match_child_id": "child-2",
                "page": 7,
                "rerank_score": 0.87654,
                "table_id": "table-3",
            },
        )
    ]
    chain = FakeChain(
        "结论得到内部证据支持[[EVIDENCE:1]]\n\n参考来源：\n伪造[[EVIDENCE:99]]"
    )
    service = object.__new__(rag_service.RagSummarizeService)
    service.chain = chain
    service.retriever_docs = lambda query, allowed_doc_ids=None: docs

    result = service.rag_summarize("效果如何？")

    assert len(chain.invocations) == 1
    context = chain.invocations[0]["context"]
    assert "[[EVIDENCE:n]]" in context
    assert "[[INSUFFICIENT]]" in context
    assert "[证据1]" in context
    assert "正文：\n该方法在复杂海况下提升了检测精度。" in context
    assert (
        "来源：paper.pdf | chunk_id=chunk-1 | parent_index=4 "
        "| match_child=child-2 | page=7 | score=0.8765 | table-3"
    ) in context
    assert result == (
        "结论得到内部证据支持[1]\n\n"
        "参考来源：\n"
        "[1] paper.pdf | chunk_id=chunk-1 | page=7 | score=0.8765"
    )
