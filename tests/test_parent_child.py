import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from rag.parent_child_retriever import ParentChildResolver


class BatchOnlyParentStore:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def get_many(self, parent_ids):
        self.calls.append(list(parent_ids))
        return {
            parent_id: self.records[parent_id]
            for parent_id in parent_ids
            if parent_id in self.records
        }

    def get(self, parent_id):
        raise AssertionError(f"per-parent lookup is forbidden: {parent_id}")


def test_resolver_dedupe_and_fallback():
    store = BatchOnlyParentStore(
        {
            "doc1:parent:000": {
                "page_content": "full parent context about MBE-Net mAP 92.3%",
                "metadata": {
                    "parent_id": "doc1:parent:000",
                    "filename": "test.txt",
                },
            }
        }
    )
    resolver = ParentChildResolver(store, top_k_parents=2)
    children = [
        Document(
            page_content="mAP 92.3%",
            metadata={"parent_id": "doc1:parent:000", "child_id": "doc1:parent:000:child:000"},
        ),
        Document(
            page_content="duplicate parent child",
            metadata={"parent_id": "doc1:parent:000", "child_id": "doc1:parent:000:child:001"},
        ),
    ]
    parents = resolver.resolve(children)
    assert len(parents) == 1
    assert "MBE-Net" in parents[0].page_content
    assert parents[0].metadata["match_child_id"] == "doc1:parent:000:child:000"

    child_without_parent = [
        Document(page_content="standalone chunk", metadata={"chunk_id": "abc_0"})
    ]
    assert resolver.resolve(child_without_parent) == child_without_parent


def test_resolver_batches_parent_lookup_and_preserves_ranked_first_hits():
    parent_store = BatchOnlyParentStore(
        {
            "parent-1": {
                "page_content": "first parent",
                "metadata": {"doc_id": "doc-1", "parent_id": "parent-1"},
            },
            "parent-denied": {
                "page_content": "must be filtered by parent ACL",
                "metadata": {"doc_id": "doc-2", "parent_id": "parent-denied"},
            },
            "parent-3": {
                "page_content": "third parent",
                "metadata": {"doc_id": "doc-1", "parent_id": "parent-3"},
            },
        }
    )
    resolver = ParentChildResolver(parent_store, top_k_parents=4)
    children = [
        Document(
            page_content="best child",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-1",
                "child_id": "child-1",
                "rerank_score": 0.95,
            },
        ),
        Document(
            page_content="legacy fallback in ranked position",
            metadata={"doc_id": "doc-1", "chunk_id": "legacy-2", "rerank_score": 0.9},
        ),
        Document(
            page_content="missing parent child",
            metadata={"doc_id": "doc-1", "parent_id": "missing", "child_id": "child-3"},
        ),
        Document(
            page_content="spoofed child metadata",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-denied",
                "child_id": "child-4",
            },
        ),
        Document(
            page_content="duplicate lower-ranked child",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-1",
                "child_id": "child-5",
                "rerank_score": 0.5,
            },
        ),
        Document(
            page_content="third parent child",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-3",
                "child_id": "child-6",
                "rerank_score": 0.4,
            },
        ),
    ]

    resolved = resolver.resolve(children, allowed_doc_ids={"doc-1"})

    assert parent_store.calls == [["parent-1", "missing", "parent-denied", "parent-3"]]
    assert [doc.page_content for doc in resolved] == [
        "first parent",
        "legacy fallback in ranked position",
        "third parent",
    ]
    assert resolved[0].metadata["match_child_id"] == "child-1"
    assert resolved[0].metadata["rerank_score"] == 0.95


class ParentChildResolverBatchTest(unittest.TestCase):
    def test_batch_lookup_contract(self):
        test_resolver_batches_parent_lookup_and_preserves_ranked_first_hits()


def test_config_enabled():
    from utils.config_handler import chroma_conf

    assert chroma_conf.get("parent_child_enabled") is True


if __name__ == "__main__":
    test_resolver_dedupe_and_fallback()
    test_config_enabled()
    print("ALL UNIT TESTS PASSED")
