import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from rag.hybrid_retriever import DynamicHybridRetriever, FilteredBM25Retriever


class DynamicHybridRetrieverTest(unittest.TestCase):
    def test_bm25_preprocess_keeps_mixed_domain_terms(self):
        text = "SAR舰船检测中 MBE-Net 的 mAP50-95、召回率和 SFQ-Det 在 HH/VV 极化下表现如何？"

        tokens = DynamicHybridRetriever._preprocess_for_bm25(text)

        self.assertIn("sar", tokens)
        self.assertIn("舰船检测", tokens)
        self.assertIn("mbe-net", tokens)
        self.assertIn("map50-95", tokens)
        self.assertIn("召回率", tokens)
        self.assertIn("sfq-det", tokens)
        self.assertIn("hh", tokens)
        self.assertIn("vv", tokens)
        self.assertIn("极化", tokens)

    def test_dynamic_weights_favor_bm25_for_exact_metrics(self):
        retriever = object.__new__(DynamicHybridRetriever)

        weights = retriever._get_dynamic_weights("MBE-Net 的 mAP50 和召回率是多少？")

        self.assertLess(weights[0], weights[1])
        self.assertGreaterEqual(weights[1], 0.6)

    def test_fingerprint_provider_drives_cache_without_manifest_access(self):
        cached_bm25 = object()
        provider = Mock(return_value="db-fingerprint")

        with (
            patch.object(
                DynamicHybridRetriever,
                "_load_bm25_cache",
                return_value=cached_bm25,
            ) as load_cache,
            patch.object(
                DynamicHybridRetriever,
                "_rebuild_bm25_from_chroma",
                side_effect=AssertionError("cache should be reused"),
            ),
            patch(
                "rag.hybrid_retriever.get_file_hash",
                side_effect=AssertionError("manifest must not be read"),
            ),
        ):
            retriever = DynamicHybridRetriever(
                vector_store=object(),
                bm25_cache_path="unused-cache.pkl",
                manifest_path="legacy-manifest.json",
                fingerprint_provider=provider,
            )

        self.assertIs(retriever.bm25_retriever, cached_bm25)
        provider.assert_called_once_with()
        load_cache.assert_called_once_with("db-fingerprint")

    def test_knowledge_store_fingerprint_drives_cache_persistence(self):
        rebuilt_bm25 = object()
        knowledge_store = Mock()
        knowledge_store.fingerprint.return_value = "knowledge-v2"

        with (
            patch.object(DynamicHybridRetriever, "_load_bm25_cache", return_value=None),
            patch.object(
                DynamicHybridRetriever,
                "_rebuild_bm25_from_chroma",
                return_value=rebuilt_bm25,
            ),
            patch.object(DynamicHybridRetriever, "_persist_bm25_cache") as persist_cache,
            patch(
                "rag.hybrid_retriever.get_file_hash",
                side_effect=AssertionError("manifest must not be read"),
            ),
        ):
            retriever = DynamicHybridRetriever(
                vector_store=object(),
                bm25_cache_path="unused-cache.pkl",
                manifest_path="legacy-manifest.json",
                knowledge_store=knowledge_store,
            )

        self.assertIs(retriever.bm25_retriever, rebuilt_bm25)
        knowledge_store.fingerprint.assert_called_once_with()
        persist_cache.assert_called_once_with(rebuilt_bm25, "knowledge-v2")

    def test_empty_active_chunk_ids_short_circuits_both_retrieval_routes(self):
        vector_store = Mock()
        retriever = object.__new__(DynamicHybridRetriever)
        retriever.vector_store = vector_store
        retriever.k = 3
        retriever.bm25_retriever = Mock()
        retriever.active_chunk_ids_provider = Mock(return_value=set())

        self.assertEqual(retriever.retrieve("ship"), [])
        vector_store.as_retriever.assert_not_called()
        retriever.active_chunk_ids_provider.assert_called_once_with()

    def test_vector_filter_combines_active_chunks_with_allowed_documents(self):
        vector_retriever = object()
        vector_store = Mock()
        vector_store.as_retriever.return_value = vector_retriever
        retriever = object.__new__(DynamicHybridRetriever)
        retriever.vector_store = vector_store
        retriever.k = 3
        retriever.bm25_retriever = None
        retriever.active_chunk_ids_provider = Mock(
            return_value={"active-child-1", "active-child-2"}
        )

        result = retriever.get_retriever("ship", allowed_doc_ids={"doc-visible"})

        self.assertIs(result, vector_retriever)
        search_filter = vector_store.as_retriever.call_args.kwargs["search_kwargs"]["filter"]
        self.assertEqual(
            search_filter,
            {
                "$and": [
                    {"chunk_id": {"$in": ["active-child-1", "active-child-2"]}},
                    {"doc_id": {"$in": ["doc-visible"]}},
                ]
            },
        )

    def test_bm25_filters_orphans_before_top_k(self):
        source = SimpleNamespace(
            preprocess_func=lambda query: [query],
            vectorizer=SimpleNamespace(get_scores=lambda _tokens: [100.0, 2.0, 1.0]),
            docs=[
                Document(
                    page_content="orphan exact match",
                    metadata={"doc_id": "doc-1", "chunk_id": "old-orphan"},
                ),
                Document(
                    page_content="active weaker match",
                    metadata={"doc_id": "doc-1", "chunk_id": "active-child"},
                ),
                Document(
                    page_content="other document",
                    metadata={"doc_id": "doc-2", "chunk_id": "other-active"},
                ),
            ],
        )
        retriever = FilteredBM25Retriever.model_construct(
            source=source,
            allowed_doc_ids={"doc-1"},
            active_chunk_ids={"active-child", "other-active"},
            k=1,
        )

        results = retriever.invoke("needle")

        self.assertEqual([doc.metadata["chunk_id"] for doc in results], ["active-child"])
        self.assertEqual(results[0].metadata["bm25_score"], 2.0)


if __name__ == "__main__":
    unittest.main()
