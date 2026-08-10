from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from eval_rag25 import pipelines


def _docs(count: int) -> list[Document]:
    return [
        Document(
            page_content=f"child-{index}",
            metadata={"chunk_id": f"child-{index}"},
        )
        for index in range(count)
    ]


class _ProductionVectorStore:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.calls = []

    def retrieve(self, query: str, allowed_doc_ids=None):
        self.calls.append((query, allowed_doc_ids))
        return list(self.documents)


class _RecordingReranker:
    def __init__(self, events: list[str]):
        self.events = events
        self.thresholds = []

    def rerank(self, query, documents, score_threshold=0.3):
        self.events.append("rerank")
        self.thresholds.append(score_threshold)
        return list(documents)


class _RecordingResolver:
    def __init__(self, events: list[str]):
        self.events = events
        self.received_count = 0

    def resolve(self, documents):
        self.events.append("resolve")
        self.received_count = len(documents)
        return list(documents)


class EvalRagPipelineTests(unittest.TestCase):
    def test_vector_full_baseline_uses_same_rerank_and_parent_chain(self):
        self.assertIn("vector_pc_with_rr", pipelines.PIPELINES)

        children = _docs(60)
        events = []
        reranker = _RecordingReranker(events)
        resolver = _RecordingResolver(events)

        with (
            patch.object(pipelines, "_retrieve_children", return_value=children) as retrieve,
            patch.object(pipelines, "_get_reranker", return_value=reranker),
            patch.object(pipelines, "_get_resolver", return_value=resolver),
        ):
            result = pipelines.PIPELINES["vector_pc_with_rr"]("query", k=5)

        retrieve.assert_called_once_with("query", k_pool=60, hybrid=False)
        self.assertEqual(events, ["rerank", "resolve"])
        self.assertEqual(len(result), 5)

    def test_hybrid_candidates_use_production_retrieve_boundary(self):
        expected = _docs(60)
        vector_store = _ProductionVectorStore(expected)

        with patch.object(pipelines, "_get_vs", return_value=vector_store):
            actual = pipelines._retrieve_children("query", k_pool=40, hybrid=True)

        self.assertEqual(actual, expected)
        self.assertEqual(vector_store.calls, [("query", None)])

    def test_full_pipeline_matches_production_order_and_rerank_gate(self):
        children = _docs(60)
        events = []
        reranker = _RecordingReranker(events)
        resolver = _RecordingResolver(events)

        with (
            patch.object(pipelines, "_retrieve_children", return_value=children),
            patch.object(pipelines, "_get_reranker", return_value=reranker),
            patch.object(pipelines, "_get_resolver", return_value=resolver),
        ):
            result = pipelines.pipeline_full("query", k=5)

        self.assertEqual(events, ["rerank", "resolve"])
        self.assertEqual(reranker.thresholds, [0.3])
        self.assertEqual(resolver.received_count, 60)
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
