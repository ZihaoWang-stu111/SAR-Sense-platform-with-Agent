from __future__ import annotations

import unittest
from collections import Counter

from eval_rag25.evaluate import DEFAULT_DATASET, load_dataset


class EvalRagDatasetTests(unittest.TestCase):
    def test_default_dataset_is_balanced_40_question_set(self):
        qa_list = load_dataset(DEFAULT_DATASET)

        self.assertEqual(DEFAULT_DATASET.name, "qa_dataset_40.json")
        self.assertEqual(len(qa_list), 40)
        self.assertEqual(len({qa["id"] for qa in qa_list}), 40)
        self.assertEqual(len({qa["question"] for qa in qa_list}), 40)
        self.assertGreaterEqual(len({qa["qa_type"] for qa in qa_list}), 8)
        self.assertEqual(
            Counter(qa["difficulty"] for qa in qa_list),
            {"easy": 10, "medium": 20, "hard": 10},
        )

        filenames = {qa["gold_filename"] for qa in qa_list}
        self.assertIn(
            "A_Self-Attention_Dictionary_Learning-Based_Method_for_Ship_Detection_in_SAR_Images.pdf",
            filenames,
        )
        self.assertIn(
            "GL-DETR_Global-to-Local_Transformers_for_Small_Ship_Detection_in_SAR_Images.pdf",
            filenames,
        )

if __name__ == "__main__":
    unittest.main()
