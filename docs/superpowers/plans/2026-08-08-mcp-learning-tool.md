# MCP 2.0 Learning Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让现有同步 `ReactAgent` 通过 MCP 2.0 的 stdio 协议调用目标检测指标计算工具。

**Architecture:** 独立 MCP Server 负责计算 Precision、Recall 和 F1；同步 LangChain Tool 使用官方 MCP Client 2.0 启动短生命周期 stdio 子进程，显式执行 `server/discover` 和 `tools/call`。现有 Agent 只增加一个工具，不改变线程池、流式输出和中间件。

**Tech Stack:** Python 3.10、MCP Python SDK 2.0、LangChain 1.x、pytest/unittest

---

## 文件结构

- Create: `mcp_server/__init__.py`：声明 MCP 示例服务包。
- Create: `mcp_server/detection_metrics_server.py`：MCP Server 与指标计算工具。
- Create: `agent/tools/mcp_tools.py`：MCP Client 到同步 LangChain Tool 的薄桥接。
- Create: `tests/test_mcp_detection_metrics.py`：计算、stdio 协议和 Agent 注册测试。
- Modify: `agent/react_agent.py`：注册 MCP 工具。
- Modify: `requirements.txt`：固定 `mcp==2.0.0`。

### Task 1: MCP Server 与指标计算

**Files:**
- Create: `mcp_server/__init__.py`
- Create: `mcp_server/detection_metrics_server.py`
- Modify: `requirements.txt`
- Test: `tests/test_mcp_detection_metrics.py`

- [ ] **Step 1: 安装并记录 MCP 2.0 依赖**

在 `requirements.txt` 增加：

```text
mcp==2.0.0
```

运行：

```powershell
conda run -n rag_env_backup python -m pip install "mcp==2.0.0"
```

预期：安装成功，`python -c "import mcp"` 退出码为 0。

- [ ] **Step 2: 先写指标行为测试**

创建 `tests/test_mcp_detection_metrics.py`：

```python
import unittest

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
```

- [ ] **Step 3: 运行测试并确认按预期失败**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_mcp_detection_metrics -v
```

Expected: FAIL/ERROR，原因是 `mcp_server.detection_metrics_server` 尚不存在。

- [ ] **Step 4: 实现最小 MCP Server**

创建空的 `mcp_server/__init__.py`，并创建 `mcp_server/detection_metrics_server.py`：

```python
from mcp.server import MCPServer


mcp = MCPServer(
    "sar-sense-detection-metrics",
    description="SAR-Sense 目标检测指标计算示例",
)


@mcp.tool()
def calculate_detection_metrics(tp: int, fp: int, fn: int) -> str:
    """根据 TP、FP、FN 计算 Precision、Recall 和 F1。"""
    if min(tp, fp, fn) < 0:
        raise ValueError("TP、FP、FN 不能为负数")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        f"Precision={precision:.2%}, "
        f"Recall={recall:.2%}, "
        f"F1={f1:.2%}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- [ ] **Step 5: 运行测试并确认通过**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_mcp_detection_metrics -v
```

Expected: 3 tests PASS。

- [ ] **Step 6: 提交 MCP Server**

```powershell
git add requirements.txt mcp_server tests/test_mcp_detection_metrics.py
git commit -m "feat: 添加 MCP 检测指标服务"
```

### Task 2: stdio MCP Client 与同步 LangChain 桥接

**Files:**
- Create: `agent/tools/mcp_tools.py`
- Modify: `tests/test_mcp_detection_metrics.py`

- [ ] **Step 1: 先写真实 stdio 调用测试**

在 `tests/test_mcp_detection_metrics.py` 增加：

```python
from agent.tools.mcp_tools import calculate_detection_metrics_mcp


class MCPBridgeTest(unittest.TestCase):
    def test_langchain_tool_calls_real_mcp_server(self):
        result = calculate_detection_metrics_mcp.invoke(
            {"tp": 80, "fp": 10, "fn": 20}
        )

        self.assertIn("Precision=88.89%", result)
        self.assertIn("Recall=80.00%", result)
        self.assertIn("F1=84.21%", result)
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_mcp_detection_metrics.MCPBridgeTest -v
```

Expected: FAIL/ERROR，原因是 `agent.tools.mcp_tools` 尚不存在。

- [ ] **Step 3: 实现同步桥接**

创建 `agent/tools/mcp_tools.py`：

```python
import asyncio
import sys
from pathlib import Path

from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def _call_detection_metrics(tp: int, fp: int, fn: int) -> str:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.detection_metrics_server"],
        cwd=PROJECT_ROOT,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.discover()
            result = await session.call_tool(
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
    """当用户给出 TP、FP、FN 并要求计算检测指标时，调用 MCP 计算 Precision、Recall 和 F1。"""
    return asyncio.run(_call_detection_metrics(tp, fp, fn))
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_mcp_detection_metrics.MCPBridgeTest -v
```

Expected: PASS；测试过程中会启动并正常退出一个 MCP stdio 子进程。

- [ ] **Step 5: 提交桥接代码**

```powershell
git add agent/tools/mcp_tools.py tests/test_mcp_detection_metrics.py
git commit -m "feat: 接入 MCP stdio 工具调用"
```

### Task 3: 注册进 ReactAgent 并回归验证

**Files:**
- Modify: `agent/react_agent.py`
- Modify: `tests/test_mcp_detection_metrics.py`

- [ ] **Step 1: 先写 Agent 注册测试**

在 `tests/test_mcp_detection_metrics.py` 增加：

```python
from unittest.mock import patch

from agent.react_agent import ReactAgent


class MCPAgentRegistrationTest(unittest.TestCase):
    def test_react_agent_registers_mcp_tool(self):
        with patch("agent.react_agent.create_agent") as create_agent:
            ReactAgent()

        tools = create_agent.call_args.kwargs["tools"]
        self.assertIn("calculate_detection_metrics_mcp", [item.name for item in tools])
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_mcp_detection_metrics.MCPAgentRegistrationTest -v
```

Expected: FAIL，工具列表中不存在 `calculate_detection_metrics_mcp`。

- [ ] **Step 3: 在现有工具列表注册 MCP 工具**

在 `agent/react_agent.py` 增加导入：

```python
from agent.tools.mcp_tools import calculate_detection_metrics_mcp
```

并在 `create_agent(..., tools=[...])` 的末尾加入：

```python
calculate_detection_metrics_mcp,
```

- [ ] **Step 4: 运行局部测试**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_mcp_detection_metrics -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 运行基础回归验证**

Run:

```powershell
conda run -n rag_env_backup python -m compileall agent mcp_server tests
conda run -n rag_env_backup python -m unittest tests.test_agent_executor tests.test_mcp_detection_metrics -v
git diff --check
```

Expected: 新增 MCP 测试通过、无语法错误、无空白错误。若基线已有与本改动无关的失败，记录具体测试与原因，不修改无关模块。

- [ ] **Step 6: 提交 Agent 注册**

```powershell
git add agent/react_agent.py tests/test_mcp_detection_metrics.py
git commit -m "feat: 将 MCP 工具注册到智能体"
```

### Task 4: 手工调用验证

**Files:**
- None

- [ ] **Step 1: 启动应用**

```powershell
conda run -n rag_env_backup python api_server_fastapi.py
```

- [ ] **Step 2: 在聊天页提问**

```text
TP=80、FP=10、FN=20，请调用检测指标工具计算精确率、召回率和 F1。
```

Expected:

- 思考步骤显示调用 `calculate_detection_metrics_mcp`。
- 最终答案包含 Precision `88.89%`、Recall `80.00%`、F1 `84.21%`。
- 服务端日志由现有 `monitor_tool` 记录该工具的成功调用。

- [ ] **Step 3: 确认工作区状态**

```powershell
git status --short --branch
```

Expected: 工作区干净；提交历史依次包含设计、MCP Server、stdio 桥接和 Agent 注册。
