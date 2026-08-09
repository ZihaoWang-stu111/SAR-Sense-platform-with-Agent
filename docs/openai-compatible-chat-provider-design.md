# OpenAI 兼容聊天模型接入设计

## 目标

在不影响现有 Ollama 和 DashScope 能力的前提下，为聊天模型工厂增加 OpenAI 兼容协议支持，并在本机临时切换到 `mimo-v2.5-pro`。

## 设计

- 在 `ChatModelFactory` 中增加 `openai` Provider 分支，使用 `langchain_openai.ChatOpenAI`。
- 优先读取项目专用的 `OPENAI_COMPATIBLE_API_KEY`、`OPENAI_COMPATIBLE_BASE_URL`、`OPENAI_COMPATIBLE_TIMEOUT_S`，避免开发机已有的标准 OpenAI 环境变量覆盖本项目配置。专用 Key 与地址必须成对配置，缺少任一项会立即报错，避免把一套凭据发送到另一套服务；二者均未设置时才整体回退到标准 `OPENAI_API_KEY` 与 `OPENAI_BASE_URL`。超时可独立回退到 `OPENAI_TIMEOUT_S`。
- 继续通过已有 `CHAT_PROVIDER` 和 `CHAT_MODEL_NAME` 选择聊天 Provider 与模型。
- 本机 `.env` 设置为 OpenAI Provider；真实密钥只保存在 `.env`，不写入受 Git 跟踪的文件。
- `config/rag.yml` 保持 Ollama 默认配置，避免把临时模型服务变成仓库默认依赖。
- DashScope Embedding 保持现状，不随聊天模型切换。
- 在 `.env.example` 中补充空配置项，说明可选配置但不暴露密钥。
- 在 `requirements.txt` 中显式声明 `langchain-openai`，避免只依赖当前环境中的传递安装。

## 调用链

```text
CHAT_PROVIDER=openai
        |
resolve_chat_provider_model()
        |
ChatModelFactory.generator()
        |
ChatOpenAI(base_url, api_key, model)
        |
现有 LangChain Agent / Tool Calling / 流式输出链路
```

## 错误处理

- SDK 内部重试保持最小，由项目现有 `call_governance` 统一处理外层重试。
- 认证错误直接失败，不重复重试。
- 超时优先通过 `OPENAI_COMPATIBLE_TIMEOUT_S` 配置，未设置时兼容 `OPENAI_TIMEOUT_S`，不复用 DashScope 的环境变量。

## 验证

1. 模型工厂能生成 `ChatOpenAI` 实例，且模型名和接口地址正确。
2. 发起最小文本调用，确认兼容接口、密钥和模型可用。
3. 验证一次带工具的 Agent 调用，确认 OpenAI Tool Calling 兼容性。
4. 运行相关单元测试与 `compileall`。

## 不在本次范围

- 不修改 Embedding Provider。
- 不删除 Ollama 或 DashScope 支持。
- 不把真实密钥、临时接口地址或临时模型设置提交到 Git。
