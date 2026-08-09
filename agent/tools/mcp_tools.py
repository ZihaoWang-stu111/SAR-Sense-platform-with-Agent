import asyncio
import sys
from pathlib import Path

from langchain_core.tools import tool
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def _call_detection_metrics(tp: int, fp: int, fn: int) -> str:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.detection_metrics_server"],
        cwd=PROJECT_ROOT,
    )
    async with Client(stdio_client(server)) as client:
        result = await client.call_tool(
            "calculate_detection_metrics",
            {"tp": tp, "fp": fp, "fn": fn},
        )

    text = "\n".join(
        block.text
        for block in result.content
        if getattr(block, "type", None) == "text"
    )
    if result.is_error:
        raise RuntimeError(text or "MCP 工具调用失败")
    if not text:
        raise RuntimeError("MCP 工具未返回文本结果")
    return text


@tool
def calculate_detection_metrics_mcp(tp: int, fp: int, fn: int) -> str:
    """当用户给出 TP、FP、FN 时，通过 MCP 计算 Precision、Recall 和 F1。"""
    return asyncio.run(_call_detection_metrics(tp, fp, fn))
