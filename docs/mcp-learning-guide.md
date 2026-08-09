# SAR-Sense MCP 源码学习大纲

这份文档只围绕本项目已经实现的 MCP 调用链展开。目标不是背诵 MCP SDK，
而是能够对照源码回答：**工具在哪里定义、Agent 为什么能看见它、调用如何跨进程、
结果如何回到模型。**

## 一、学完应达到什么程度

学完后，你应该能够独立说明：

1. MCP Server、MCP Client、LangChain Tool 和 Agent 分别负责什么。
2. `@mcp.tool()` 与 LangChain 的 `@tool` 为什么同时存在。
3. 用户问题如何变成一次真实的 stdio MCP 调用。
4. `server/discover` 和 `tools/call` 在链路中的作用。
5. 为什么当前代码使用 `asyncio.run()`，而没有把整个 Agent 改成异步。
6. 当前方案为什么适合学习，以及生产化时应怎样演进。

## 二、先记住完整调用链

先不要钻进函数，先把下面这条链记住：

```text
用户输入 TP、FP、FN
        |
        v
ReactAgent 看到 calculate_detection_metrics_mcp 的工具描述
        |
        v
模型生成 tool_call
        |
        v
LangChain 调用同步工具 calculate_detection_metrics_mcp
        |
        v
asyncio.run(_call_detection_metrics(...))
        |
        v
stdio_client 启动独立 Python 子进程
        |
        v
MCP ClientSession -> server/discover
        |
        v
MCP ClientSession -> tools/call
        |
        v
MCP Server 执行 calculate_detection_metrics
        |
        v
结果通过 stdout 返回 Client，再成为 ToolMessage
        |
        v
模型根据工具结果生成最终回答
```

这里存在两个容易混淆的“工具”：

- MCP Server 暴露的工具名：`calculate_detection_metrics`。
- Agent 直接注册的 LangChain 工具名：`calculate_detection_metrics_mcp`。

后者是一个薄桥接器，内部通过 MCP 调用前者。

## 三、推荐阅读顺序

### 第一遍：只理解职责，约 20 分钟

按下面顺序浏览，不要研究 SDK 内部实现：

1. [`requirements.txt`](../requirements.txt) 第 71 行：确认项目使用 `mcp==2.0.0`。
2. [`mcp_server/detection_metrics_server.py`](../mcp_server/detection_metrics_server.py)：看 MCP 工具如何发布。
3. [`agent/tools/mcp_tools.py`](../agent/tools/mcp_tools.py)：看客户端如何调用 MCP Server。
4. [`agent/react_agent.py`](../agent/react_agent.py) 第 9、21-24 行：看工具如何进入 Agent。
5. [`agent/tools/middleware.py`](../agent/tools/middleware.py) 第 19-41 行：看调用如何被统一监控。
6. [`tests/test_mcp_detection_metrics.py`](../tests/test_mcp_detection_metrics.py)：看三层行为如何验证。

第一遍结束时，只要能画出上一节的调用链即可。

### 第二遍：逐个理解核心函数，约 40 分钟

重点阅读后文第四至第七节。每读完一个函数，都回答三个问题：

1. 它收到什么？
2. 它返回什么？
3. 它属于 MCP Server、MCP Client，还是 Agent？

### 第三遍：运行验证，约 20 分钟

执行测试和直接调用，观察一次完整的跨进程过程。不要只看代码猜结果。

### 第四遍：准备面试表达，约 20 分钟

用第十节的问题进行口述。能用自己的话讲清楚取舍，比背 SDK API 更重要。

## 四、MCP Server：工具提供方

源码：[`mcp_server/detection_metrics_server.py`](../mcp_server/detection_metrics_server.py)

### 4.1 创建 MCP Server

```python
mcp = MCPServer(
    "sar-sense-detection-metrics",
    description="SAR-Sense 目标检测指标计算示例",
)
```

`MCPServer` 是工具提供方。它维护工具注册表，并负责处理 MCP 协议消息。

这里的名字是服务名称，不是工具名称。一个 MCP Server 可以暴露多个工具，
但当前学习示例只保留一个。

### 4.2 `@mcp.tool()` 注册工具

```python
@mcp.tool()
def calculate_detection_metrics(tp: int, fp: int, fn: int) -> str:
```

这一层是 **MCP 工具注册**。SDK 会利用：

- 函数名生成工具名。
- 类型注解生成输入 Schema。
- docstring 生成工具描述。
- 返回值类型描述输出。

所以我们不需要手写 JSON Schema。

必须理解：装饰器不会主动执行函数，只是把函数登记到 MCP Server。只有客户端发送
`tools/call` 后，服务端才执行它。

### 4.3 指标计算

```python
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
```

这部分只是业务逻辑，不属于 MCP 协议。将来即使不用 MCP，公式本身也不变。

边界处理：

- TP、FP、FN 不能为负数。
- 分母为 0 时返回 `0.0`，避免 `ZeroDivisionError`。
- 结果格式化为百分比文本，模型可以直接阅读。

### 4.4 启动 stdio 服务

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

`stdio` 表示客户端和服务端通过标准输入、标准输出传输 MCP 消息：

```text
Client 写入 Server stdin
Server 从 stdin 读取请求
Server 将响应写入 stdout
Client 从 stdout 读取响应
```

因此直接运行这个文件后，它会等待输入，看起来像“卡住”，这是正常现象。真正的
启动者是 `stdio_client`。

## 五、MCP Client：工具使用方

源码：[`agent/tools/mcp_tools.py`](../agent/tools/mcp_tools.py)

### 5.1 `PROJECT_ROOT`

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

作用是获得项目根目录 `E:\sar-sense`。子进程只有在正确工作目录下，才能通过：

```text
python -m mcp_server.detection_metrics_server
```

找到项目中的 `mcp_server` 包。

### 5.2 `_call_detection_metrics()`

```python
async def _call_detection_metrics(tp: int, fp: int, fn: int) -> str:
```

这是整个 MCP 客户端链路的核心函数。它接收三个整数，最终返回 MCP Server 的文本结果。

#### 第一步：描述如何启动服务端

```python
server = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_server.detection_metrics_server"],
    cwd=PROJECT_ROOT,
)
```

- `command=sys.executable`：使用当前 FastAPI/Agent 所在环境的 Python。
- `args`：以模块方式启动 MCP Server。
- `cwd`：把子进程工作目录固定在项目根目录。

使用 `sys.executable` 比硬编码 `python` 更可靠，否则系统可能启动另一个没有安装 MCP
依赖的 Python。

#### 第二步：建立 stdio 连接

```python
async with stdio_client(server) as (read_stream, write_stream):
```

`stdio_client` 会：

1. 启动 MCP Server 子进程。
2. 将子进程 stdin/stdout 包装成异步读写流。
3. 退出 `async with` 时关闭管道并回收子进程。

因此不需要手工调用 `subprocess.Popen()`，也不会留下常驻端口。

#### 第三步：创建协议会话

```python
async with ClientSession(read_stream, write_stream) as session:
```

`ClientSession` 在读写流之上处理 MCP 消息、请求 ID、响应匹配和数据模型校验。

你需要理解它是“协议客户端”，但不需要深挖内部 Dispatcher、JSON-RPC 编解码和
AnyIO task group。

#### 第四步：协议发现

```python
await session.discover()
```

这是本示例专门体现 MCP `2026-07-28` 版本的地方。客户端通过
`server/discover` 获取服务端支持的现代协议版本，并采用双方都支持的版本。

本项目实测结果：

```text
supported_versions=['2026-07-28']
```

旧版 MCP 主要通过 `initialize` 握手；当前代码没有调用 `initialize()`。

#### 第五步：调用工具

```python
result = await session.call_tool(
    "calculate_detection_metrics",
    {"tp": tp, "fp": fp, "fn": fn},
)
```

它在协议层对应 `tools/call`：

- 第一个参数是 MCP Server 注册的工具名。
- 第二个参数是符合工具输入 Schema 的参数字典。
- 返回值是 MCP 的 `CallToolResult`，不是普通字符串。

#### 第六步：解析结果与错误

```python
text = "\n".join(
    block.text
    for block in result.content
    if getattr(block, "type", None) == "text"
)
```

MCP 结果的 `content` 是内容块列表，因为协议还允许图片、音频等类型。本工具只需要
文本，所以只收集 `type == "text"` 的块。

```python
if result.is_error:
    raise RuntimeError(text or "MCP 工具调用失败")
if not text:
    raise RuntimeError("MCP 工具未返回文本结果")
```

两种错误被明确区分：

- Server 明确返回工具错误。
- 调用没有报错，但没有返回预期文本。

异常继续向上抛出，由现有 Agent 中间件统一记录，不在桥接层重复写日志。

## 六、LangChain 同步桥接

源码：[`agent/tools/mcp_tools.py`](../agent/tools/mcp_tools.py) 第 39-42 行。

```python
@tool
def calculate_detection_metrics_mcp(tp: int, fp: int, fn: int) -> str:
    """当用户给出 TP、FP、FN 时，通过 MCP 计算 Precision、Recall 和 F1。"""
    return asyncio.run(_call_detection_metrics(tp, fp, fn))
```

这一层使用的是 **LangChain 的 `@tool`**，不是 MCP 的 `@mcp.tool()`。

它解决的是接口不一致：

```text
现有 ReactAgent 需要同步 LangChain Tool
MCP Python Client 提供异步调用
```

`asyncio.run()` 创建临时事件循环，执行 MCP 异步调用，拿到结果后关闭事件循环。

为什么当前能这样用：

- 你的 Agent 使用同步 `agent.stream()`。
- Agent 任务运行在共享工作线程中。
- 工作线程内没有正在运行的 asyncio 事件循环。

注意：如果未来把 Agent 改成 `agent.astream()` 并直接运行在事件循环中，就不应继续在
同一线程调用 `asyncio.run()`；届时应改为异步 LangChain Tool 或官方 MCP 适配器。

## 七、Agent 如何发现和展示 MCP 工具

### 7.1 注册到工具列表

源码：[`agent/react_agent.py`](../agent/react_agent.py) 第 9、21-24 行。

```python
from agent.tools.mcp_tools import calculate_detection_metrics_mcp
```

然后传给：

```python
create_agent(..., tools=[..., calculate_detection_metrics_mcp])
```

`create_agent` 会把工具名称、docstring 和参数 Schema 提供给模型。模型根据用户问题
决定是否生成针对该工具的 `tool_call`。

MCP Server 不直接连接模型。模型只看见 LangChain 桥接工具；桥接工具被执行后，才会
启动 MCP Client。

### 7.2 为什么现有监控自动生效

源码：[`agent/tools/middleware.py`](../agent/tools/middleware.py) 第 19-41 行。

`monitor_tool` 包裹 Agent 的所有工具调用。MCP 桥接工具进入统一工具列表后，也会自动获得：

- 工具名和参数日志。
- 成功/失败统计。
- 调用耗时统计。
- `user_id` 关联。

MCP Server 不需要再写一套指标采集逻辑。

### 7.3 前端为什么能看到调用过程

源码：[`agent/react_agent.py`](../agent/react_agent.py) 第 65-120 行。

`execute_stream()` 处理 LangGraph 产生的消息：

- AIMessage 含 `tool_calls` 时，发出 `tool_call` 思考步骤。
- ToolMessage 返回时，发出 `tool_result` 思考步骤。
- 最终 AIMessage 才进入回答正文。

所以 MCP 工具无需单独修改 SSE 或前端，仍沿用现有工具展示链路。

## 八、测试应该怎么看

源码：[`tests/test_mcp_detection_metrics.py`](../tests/test_mcp_detection_metrics.py)

### 8.1 `DetectionMetricsToolTest`

只测试 MCP Server 内部业务函数：

- 正常计算结果。
- 零分母处理。
- 负数参数校验。

这组测试快，但它不能证明 MCP 协议真正工作。

### 8.2 `MCPBridgeTest`

```python
calculate_detection_metrics_mcp.invoke({"tp": 80, "fp": 10, "fn": 20})
```

这组测试没有 mock MCP Client，会真实启动 stdio 子进程。因此它能够同时验证：

- 子进程能启动。
- `server/discover` 成功。
- `tools/call` 成功。
- 响应能被解析。
- LangChain 同步桥接可用。

这是本功能价值最高的一项测试。

### 8.3 `MCPAgentRegistrationTest`

它只 mock `create_agent()`，然后检查传入的工具列表。这样可以证明注册行为，而不真正
调用大模型。

## 九、动手验证顺序

### 9.1 先运行全部 MCP 测试

```powershell
conda activate rag_env_backup
python -m unittest tests.test_mcp_detection_metrics -v
```

预期：5 项测试通过。

### 9.2 绕过大模型，直接验证 MCP 桥接

```powershell
python -c "from agent.tools.mcp_tools import calculate_detection_metrics_mcp as t; print(t.invoke({'tp': 80, 'fp': 10, 'fn': 20}))"
```

预期：

```text
Precision=88.89%, Recall=80.00%, F1=84.21%
```

### 9.3 在真实 Agent 中验证

启动项目：

```powershell
python api_server_fastapi.py
```

聊天页输入：

```text
TP=80、FP=10、FN=20，请调用检测指标工具计算精确率、召回率和 F1。
```

观察：

1. 思考过程出现 `calculate_detection_metrics_mcp`。
2. 工具参数为 `tp=80, fp=10, fn=20`。
3. 最终回答包含 `88.89% / 80.00% / 84.21%`。
4. 服务日志记录工具调用耗时与成功状态。

## 十、面试常见问题

### 1. MCP 和普通 Tool Calling 有什么区别？

Tool Calling 是模型表达“我要调用哪个函数、传什么参数”的能力。MCP 是应用与外部
工具服务之间的标准协议，定义工具发现、参数 Schema、调用和结果格式。两者位于不同层。

### 2. 为什么不直接把计算函数写成 LangChain Tool？

直接写当然更短。本功能使用 MCP 的目的不是解决计算难题，而是演示工具服务与 Agent
解耦：同一个 MCP Server 可以被其他 MCP Host 使用，而不绑定 LangChain。

### 3. 为什么有两个装饰器？

- `@mcp.tool()` 把业务函数发布给 MCP Client。
- `@tool` 把 MCP 调用入口发布给当前 LangChain Agent。

中间的桥接层负责协议转换。

### 4. 为什么选择 stdio，不选择 HTTP？

当前只有一个本地学习工具，stdio 不需要端口、服务发现和独立部署。工具增多、调用频繁
或需要跨机器访问时，再改为长生命周期 Streamable HTTP Server。

### 5. 这真的是跨进程 MCP 调用吗？

是。`stdio_client` 使用当前 Python 启动独立子进程，客户端先执行
`server/discover`，再执行 `tools/call`。集成测试没有绕过协议调用服务端函数。

### 6. 为什么自己写桥接，不用 `langchain-mcp-adapters`？

实现时最新版 `langchain-mcp-adapters==0.3.2` 依赖 `mcp>=1.24,<2`，而项目希望学习
支持 `2026-07-28` 协议的 `mcp==2.0.0`。因此使用一个很薄的桥接保持版本目标，未来
官方适配后可以替换。

### 7. 当前方案有什么性能问题？

每次调用都会启动一个 Python 子进程，实测约为秒级，适合学习和低频工具。生产高频
调用应使用长生命周期 MCP 服务，避免反复启动进程。

### 8. MCP 调用失败后如何处理？

桥接层把 MCP 错误转换为异常；现有 `monitor_tool` 统一记录失败、耗时和用户，LangChain
再把工具失败反馈给 Agent。没有在每一层重复吞异常或重复记指标。

### 9. 为什么用 `sys.executable`？

确保 MCP 子进程使用与主应用相同的 Conda Python 和依赖环境，避免系统 Python 找不到
`mcp` 或项目模块。

### 10. 7 月 28 日版本在源码中体现在哪里？

体现在 `session.discover()`。实测服务返回 `supported_versions=['2026-07-28']`，客户端
没有调用旧式 `initialize()`。

## 十一、哪些必须看懂，哪些先不要深挖

### 必须看懂

- `MCPServer` 的职责。
- `@mcp.tool()` 如何发布工具。
- stdio 客户端为什么会启动子进程。
- `ClientSession.discover()` 和 `call_tool()` 的先后关系。
- MCP 内容块如何转换成 LangChain 工具字符串。
- LangChain `@tool` 为什么是同步桥接。
- 工具如何注册进 `ReactAgent`。
- 单元测试与真实协议集成测试的区别。

### 知道存在即可

- MCP SDK 内部 JSON-RPC Dispatcher。
- AnyIO 的 task group 和流实现。
- `mcp_types` 自动生成的数据模型细节。
- stdio 在 Windows 上如何终止整个子进程树。
- MCP 扩展、资源、Prompt、订阅和 OAuth。
- Streamable HTTP 的底层 ASGI 实现。

目前项目没有使用后面这些能力，提前深挖只会增加记忆负担。

## 十二、建议练习

完成一个练习后再做下一个：

1. 把 TP/FP/FN 换成另一组数，先手算，再运行 MCP 工具核对。
2. 暂时把 MCP 工具从 `ReactAgent` 工具列表移除，观察注册测试如何失败，然后恢复。
3. 给服务端传入负数，观察 MCP 返回的工具错误如何被桥接成 `RuntimeError`。
4. 新增一个同服务器工具 `calculate_iou(intersection, union)`，走完注册、桥接和测试。
5. 当本地工具超过 3 个且启动延迟明显时，再研究 Streamable HTTP，不要现在提前改造。

## 十三、官方资料阅读顺序

只建议读以下三处：

1. [MCP Python SDK v2 首页](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/index.md)
2. [MCP Python SDK v2 更新说明](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/whats-new.md)
3. [MCP 2026-07-28 Streamable HTTP 规范](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/basic/transports/streamable-http.mdx)

先用本项目理解工具链，再读协议细节，会比从规范第一页硬啃更容易。
