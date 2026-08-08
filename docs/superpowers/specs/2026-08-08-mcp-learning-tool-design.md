# MCP 2.0 学习工具设计

## 目标

在不改变现有同步 Agent、线程池和 SSE 链路的前提下，让 SAR-Sense 的
`ReactAgent` 真正通过 MCP 调用一个独立工具，帮助理解 MCP Server、MCP Client、
工具协议和 LangChain Tool 之间的关系。

示例工具为 `calculate_detection_metrics(tp, fp, fn)`，根据目标检测混淆统计计算
Precision、Recall 和 F1。它与项目领域一致、结果确定、无需数据库或外部 API。

## 方案选择

采用 **MCP 2.0 + stdio + 薄 LangChain 同步桥接**。

- 不采用独立 HTTP 服务：学习示例不值得增加端口和启动顺序。
- 不采用 `langchain-mcp-adapters`：当前 `0.3.2` 要求 `mcp<2`，无法直接使用
  支持 `2026-07-28` 协议的 `mcp==2.0.0`。
- 不把 Agent 全面改成异步：现有同步 `agent.stream()` 在线程池内运行稳定，MCP
  示例不应扩大改动范围。

## 组件

### MCP Server

新增 `mcp_server/detection_metrics_server.py`：

- 使用官方 `MCPServer` 注册 `calculate_detection_metrics`。
- 参数必须为非负整数。
- 分母为零时对应指标返回 `0.0`。
- 通过 `mcp.run(transport="stdio")` 运行，由客户端按需启动子进程。

### 同步桥接

新增 `agent/tools/mcp_tools.py`：

- 对 LangChain 暴露同步工具 `calculate_detection_metrics_mcp`。
- 工具内部使用 `asyncio.run()` 执行一次 MCP 客户端调用；这与当前 Agent
  工作线程的同步执行方式一致。
- 使用 `sys.executable` 启动 MCP Server，保证服务端与 FastAPI 使用同一 Python
  环境。
- MCP Client 先调用 `server/discover` 协商 `2026-07-28` 协议，再通过
  `tools/call` 调用工具。
- MCP 或子进程失败时抛出清晰异常，交给现有 `monitor_tool` 中间件统一记录。

这个桥接只负责协议转换，不包含业务计算。待 LangChain 官方适配器支持 MCP 2.0
后，可直接删除该文件并换成官方工具加载器。

### Agent 注册

修改 `agent/react_agent.py`，把 `calculate_detection_metrics_mcp` 加入现有
`create_agent(..., tools=[...])` 列表。Agent 的流式处理、思考步骤、指标采集和 SSE
输出均保持不变。

### 依赖

在 `requirements.txt` 增加 `mcp==2.0.0`。不安装
`langchain-mcp-adapters`，不增加 MCP CLI、数据库表、配置文件或前端代码。

## 调用链

```text
用户问题
  -> ReactAgent 选择 calculate_detection_metrics_mcp
  -> 同步 LangChain Tool
  -> asyncio.run(MCP Client)
  -> stdio 启动 MCP Server 子进程
  -> server/discover
  -> tools/call: calculate_detection_metrics
  -> MCP 文本结果
  -> Agent 生成最终回答
```

每次调用都会启动并关闭一个短生命周期 MCP 子进程。该取舍适合学习和低频演示；
如果以后 MCP 工具变多或调用频繁，再改为长生命周期 Streamable HTTP 服务。

## 验证

- 单元测试验证 Precision、Recall、F1 的正常值和零分母行为。
- 集成测试通过 stdio 启动真实 MCP Server，完成 `discover` 和 `tools/call`。
- Agent 工具列表测试确认 MCP 工具已注册。
- 运行现有相关 Agent 测试和 `compileall`，确认未影响原工具及流式链路。
- 手工提问：`TP=80、FP=10、FN=20，请计算精确率、召回率和 F1。`
  思考链中应显示调用 `calculate_detection_metrics_mcp`，最终回答应给出约
  `88.89% / 80.00% / 84.21%`。

## 明确不做

- 不增加 MCP 管理页面。
- 不提供多 MCP Server 配置中心。
- 不实现连接池、重试、鉴权或远程部署。
- 不把已有本地工具批量迁移为 MCP。
- 不改变现有 Agent 的同步执行模型。
