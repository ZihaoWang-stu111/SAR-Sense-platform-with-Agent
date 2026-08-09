import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agent.tools.mcp_tools import (
    _call_detection_metrics,
    calculate_detection_metrics_mcp,
)
from agent.react_agent import ReactAgent
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
    def test_bridge_uses_v2_high_level_client(self):
        transport = object()
        result = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="probe-ok")],
            is_error=False,
        )
        client = AsyncMock()
        client.call_tool.return_value = result
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=client)
        client_context.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "agent.tools.mcp_tools.stdio_client", return_value=transport
        ):
            with patch(
                "agent.tools.mcp_tools.Client", return_value=client_context
            ) as client_cls:
                actual = asyncio.run(_call_detection_metrics(1, 2, 3))

        self.assertEqual(actual, "probe-ok")
        client_cls.assert_called_once_with(transport)
        client.call_tool.assert_awaited_once_with(
            "calculate_detection_metrics",
            {"tp": 1, "fp": 2, "fn": 3},
        )

    def test_langchain_tool_calls_real_mcp_server(self):
        result = calculate_detection_metrics_mcp.invoke(
            {"tp": 80, "fp": 10, "fn": 20}
        )

        self.assertIn("Precision=88.89%", result)
        self.assertIn("Recall=80.00%", result)
        self.assertIn("F1=84.21%", result)


class MCPAgentRegistrationTest(unittest.TestCase):
    def test_react_agent_registers_mcp_tool(self):
        with patch("agent.react_agent.create_agent") as create_agent:
            ReactAgent()

        tools = create_agent.call_args.kwargs["tools"]
        self.assertIn("calculate_detection_metrics_mcp", [item.name for item in tools])


if __name__ == "__main__":
    unittest.main()
