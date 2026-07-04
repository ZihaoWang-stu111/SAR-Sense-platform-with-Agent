import unittest

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.hybrid_retriever import FilteredBM25Retriever
from rag.parent_child_retriever import ParentChildResolver


class DictParentDocstore:
    def __init__(self, records):
        self.records = records

    def get(self, parent_id):
        return self.records.get(parent_id)


class RagAclRetrievalTest(unittest.TestCase):
    def test_filtered_bm25_scores_all_docs_but_only_returns_allowed_docs(self):
        docs = [
            Document(page_content="secret exact mbe-net sfq-det", metadata={"doc_id": "secret", "chunk_id": "s1"}),
            Document(page_content="public weak match", metadata={"doc_id": "public", "chunk_id": "p1"}),
        ]
        bm25 = BM25Retriever.from_documents(docs)

        retriever = FilteredBM25Retriever(source=bm25, allowed_doc_ids={"public"}, k=2)
        results = retriever.invoke("secret exact mbe-net")

        self.assertEqual([doc.metadata["doc_id"] for doc in results], ["public"])

    def test_filtered_bm25_empty_acl_returns_no_docs(self):
        bm25 = BM25Retriever.from_documents([
            Document(page_content="public", metadata={"doc_id": "public"}),
        ])

        retriever = FilteredBM25Retriever(source=bm25, allowed_doc_ids=set(), k=2)

        self.assertEqual(retriever.invoke("public"), [])

    def test_parent_child_resolver_filters_parent_doc_id_again(self):
        child_docs = [
            Document(
                page_content="child secret",
                metadata={"doc_id": "secret", "parent_id": "parent-secret", "child_id": "child-secret"},
            ),
            Document(
                page_content="child public",
                metadata={"doc_id": "public", "parent_id": "parent-public", "child_id": "child-public"},
            ),
        ]
        docstore = DictParentDocstore({
            "parent-secret": {
                "page_content": "secret parent",
                "metadata": {"doc_id": "secret", "filename": "secret.txt"},
            },
            "parent-public": {
                "page_content": "public parent",
                "metadata": {"doc_id": "public", "filename": "public.txt"},
            },
        })

        resolver = ParentChildResolver(parent_docstore=docstore, top_k_parents=2)
        results = resolver.resolve(child_docs, allowed_doc_ids={"public"})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["doc_id"], "public")
        self.assertEqual(results[0].page_content, "public parent")


if __name__ == "__main__":
    unittest.main()
