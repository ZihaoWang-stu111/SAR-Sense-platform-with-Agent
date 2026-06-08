import unittest

from rag.hybrid_retriever import DynamicHybridRetriever


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


if __name__ == "__main__":
    unittest.main()
