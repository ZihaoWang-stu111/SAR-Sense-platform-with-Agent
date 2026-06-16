import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from rag.parent_docstore import ParentDocstore
from rag.parent_child_retriever import ParentChildResolver


def test_parent_docstore_crud():
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, "test_docstore.json")
    ds = ParentDocstore(store_path)
    ds.save(
        "doc1:parent:000",
        "parent text here",
        {"parent_id": "doc1:parent:000", "doc_id": "doc1", "chunk_type": "parent"},
    )
    assert ds.get("doc1:parent:000")["page_content"] == "parent text here"
    assert ds.count() == 1
    ds.delete_many(["doc1:parent:000"])
    assert ds.count() == 0


def test_resolver_dedupe_and_fallback():
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, "test_docstore.json")
    ds = ParentDocstore(store_path)
    ds.save(
        "doc1:parent:000",
        "full parent context about MBE-Net mAP 92.3%",
        {"parent_id": "doc1:parent:000", "filename": "test.txt"},
    )
    resolver = ParentChildResolver(ds, top_k_parents=2)
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

    legacy = [Document(page_content="old chunk", metadata={"chunk_id": "abc_0"})]
    assert resolver.resolve(legacy) == legacy


def test_config_enabled():
    from utils.config_handler import chroma_conf

    assert chroma_conf.get("parent_child_enabled") is True


if __name__ == "__main__":
    test_parent_docstore_crud()
    test_resolver_dedupe_and_fallback()
    test_config_enabled()
    print("ALL UNIT TESTS PASSED")
