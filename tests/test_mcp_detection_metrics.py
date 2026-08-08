import unittest

from agent.tools.mcp_tools import calculate_detection_metrics_mcp
from mcp_server.detection_metrics_server import calculate_detection_metrics


class DetectionMetricsToolTest(unittest.TestCase):
    def test_calculates_precision_recall_and_f1(self):
        result = calculate_detection_metrics(tp=80, fp=10, fn=20)

        self.assertIn("Precision=88.89%", result)
        self.assertIn("Recall=80.00%", result)
        self.assertIn("F1=84.21%", result)

    def test_zero_denominators_return_zero(self):
        result = calculate_detection_metrics(tp=0, fp=0, fn=0)

        self.assertIn("Precision=0.00%", result)
        self.assertIn("Recall=0.00%", result)
        self.assertIn("F1=0.00%", result)

    def test_rejects_negative_counts(self):
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            calculate_detection_metrics(tp=-1, fp=0, fn=0)


class MCPBridgeTest(unittest.TestCase):
    def test_langchain_tool_calls_real_mcp_server(self):
        result = calculate_detection_metrics_mcp.invoke(
            {"tp": 80, "fp": 10, "fn": 20}
        )

        self.assertIn("Precision=88.89%", result)
        self.assertIn("Recall=80.00%", result)
        self.assertIn("F1=84.21%", result)


if __name__ == "__main__":
    unittest.main()
