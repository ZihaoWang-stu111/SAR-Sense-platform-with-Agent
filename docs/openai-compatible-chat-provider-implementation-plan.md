# OpenAI Compatible Chat Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SAR-Sense 增加通用 OpenAI 兼容聊天 Provider，并在本机临时切换到 `mimo-v2.5-pro`。

**Architecture:** 模型选择继续由 `CHAT_PROVIDER` 与 `CHAT_MODEL_NAME` 控制；`ChatModelFactory` 在现有 Ollama、DashScope 分支旁新增 `ChatOpenAI` 分支。通用能力进入代码仓库，服务地址、模型切换和真实密钥只进入被忽略的本机 `.env`。

**Tech Stack:** Python 3.10、LangChain 1.3、`langchain-openai` 1.2.2、OpenAI 兼容 API、unittest

---

## 文件结构

- `model/factory.py`：根据 Provider 创建 `ChatOpenAI`，继续复用现有调用治理包装。
- `utils/call_governance.py`：提供独立的 OpenAI 单次调用默认超时。
- `tests/test_model_factory.py`：锁定 OpenAI Provider 参数映射及未知 Provider 行为。
- `requirements.txt`：显式声明项目直接使用的 `langchain-openai`。
- `.env.example`：公开可选配置名，不包含真实密钥和临时服务地址。
- `.env`：本机临时模型配置；已被 Git 忽略。

### Task 1: 用测试锁定 OpenAI Provider 行为

**Files:**
- Create: `tests/test_model_factory.py`
- Modify: `utils/call_governance.py`
- Modify: `model/factory.py`

- [ ] **Step 1: 编写失败测试**

创建 `tests/test_model_factory.py`：

```python
import os
import unittest
from unittest.mock import patch

from model.factory import ChatModelFactory


class ChatModelFactoryTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "CHAT_PROVIDER": "openai",
            "CHAT_MODEL_NAME": "mimo-v2.5-pro",
            "OPENAI_API_KEY": "system-key",
            "OPENAI_BASE_URL": "https://standard.invalid/v1",
            "OPENAI_TIMEOUT_S": "15",
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENAI_COMPATIBLE_BASE_URL": "https://example.invalid/v1",
            "OPENAI_COMPATIBLE_TIMEOUT_S": "45",
        },
        clear=False,
    )
    @patch("model.factory.ChatOpenAI")
    def test_builds_openai_compatible_model(self, chat_openai):
        expected = object()
        chat_openai.return_value = expected

        actual = ChatModelFactory().generator()

        self.assertIs(actual, expected)
        chat_openai.assert_called_once_with(
            model="mimo-v2.5-pro",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            timeout=45.0,
            max_retries=0,
            use_responses_api=False,
        )

    @patch.dict(
        os.environ,
        {"CHAT_PROVIDER": "unsupported", "CHAT_MODEL_NAME": "model"},
        clear=False,
    )
    def test_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "Unsupported chat provider"):
            ChatModelFactory().generator()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_model_factory -v
```

Expected: FAIL，因为 `model.factory.ChatOpenAI` 尚不存在，且未知 Provider 仍会落入 DashScope 分支。

- [ ] **Step 3: 增加 OpenAI 超时常量**

在 `utils/call_governance.py` 的 Provider 默认超时区域加入：

```python
DEFAULT_OPENAI_TIMEOUT_S = 120.0
```

- [ ] **Step 4: 实现最小 OpenAI Provider 分支**

在 `model/factory.py` 导入：

```python
from langchain_openai import ChatOpenAI

from utils.call_governance import DEFAULT_OPENAI_TIMEOUT_S
```

将 `ChatModelFactory.generator()` 的 Provider 分支改为：

```python
if provider == "ollama":
    return ChatOllama(
        model=model_name,
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            rag_conf.get("ollama_base_url", "http://localhost:11434"),
        ),
        reasoning=False,
        num_ctx=8192,
        client_kwargs={
            "trust_env": False,
            "timeout": get_timeout_s(
                "OLLAMA_TIMEOUT_S", DEFAULT_OLLAMA_TIMEOUT_S
            ),
        },
    )
if provider == "openai":
    compatible_api_key = (
        os.getenv("OPENAI_COMPATIBLE_API_KEY") or ""
    ).strip() or None
    compatible_base_url = (
        os.getenv("OPENAI_COMPATIBLE_BASE_URL") or ""
    ).strip() or None
    if bool(compatible_api_key) != bool(compatible_base_url):
        raise ValueError(
            "OPENAI_COMPATIBLE_API_KEY and "
            "OPENAI_COMPATIBLE_BASE_URL must be configured together"
        )

    standard_timeout = get_timeout_s(
        "OPENAI_TIMEOUT_S", DEFAULT_OPENAI_TIMEOUT_S
    )
    return ChatOpenAI(
        model=model_name,
        api_key=compatible_api_key or os.getenv("OPENAI_API_KEY"),
        base_url=compatible_base_url or os.getenv("OPENAI_BASE_URL"),
        timeout=get_timeout_s(
            "OPENAI_COMPATIBLE_TIMEOUT_S", standard_timeout
        ),
        max_retries=0,
        use_responses_api=False,
    )
if provider == "dashscope":
    return ChatTongyi(
        model=model_name,
        max_retries=1,
        model_kwargs={
            "request_timeout": get_timeout_s(
                "DASHSCOPE_TIMEOUT_S", DEFAULT_DASHSCOPE_TIMEOUT_S
            ),
        },
    )
raise ValueError(f"Unsupported chat provider: {provider}")
```

- [ ] **Step 5: 运行测试并确认通过**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_model_factory tests.test_call_governance -v
```

Expected: 所有测试 PASS。

### Task 2: 声明依赖和公开配置契约

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] **Step 1: 显式声明 LangChain OpenAI 依赖**

在 `requirements.txt` 的 LangChain 区域加入：

```text
langchain-openai==1.2.2
```

- [ ] **Step 2: 补充不含凭据的示例配置**

在 `.env.example` 的聊天模型覆盖配置下加入：

```dotenv
# OpenAI 兼容聊天服务；CHAT_PROVIDER=openai 时填写。
# OPENAI_COMPATIBLE_API_KEY=""
# OPENAI_COMPATIBLE_BASE_URL="https://example.com/v1"
# OPENAI_COMPATIBLE_TIMEOUT_S=120
# 未设置专用变量时兼容 OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_TIMEOUT_S。
```

- [ ] **Step 3: 运行静态验证**

Run:

```bash
conda run -n rag_env_backup python -m compileall model utils tests
conda run -n rag_env_backup python -m unittest tests.test_model_factory tests.test_call_governance -v
git diff --check
```

Expected: 编译成功、测试全部 PASS、`git diff --check` 无输出。

- [ ] **Step 4: 提交通用 Provider 能力**

```bash
git add model/factory.py utils/call_governance.py tests/test_model_factory.py requirements.txt .env.example
git commit -m "feat: 接入 OpenAI 兼容聊天模型"
```

### Task 3: 本机切换与真实接口验证

**Files:**
- Modify locally only: `.env`

- [ ] **Step 1: 写入本机临时配置**

在已被 Git 忽略的 `.env` 中设置 `CHAT_PROVIDER=openai`、`CHAT_MODEL_NAME=mimo-v2.5-pro`、`OPENAI_COMPATIBLE_BASE_URL=<兼容服务地址>`、`OPENAI_COMPATIBLE_TIMEOUT_S=120`，并将用户提供的真实凭据写入 `OPENAI_COMPATIBLE_API_KEY`。密钥和实际临时服务地址不得出现在计划、日志、测试或 Git diff 中。

- [ ] **Step 2: 验证模型初始化和最小文本调用**

Run:

```bash
conda run -n rag_env_backup python -c "from model.factory import chat_model; r=chat_model.invoke('只回复 OK'); print(type(chat_model).__name__); print(r.content)"
```

Expected: 第一行是 `ChatOpenAI`，第二行包含 `OK`。

- [ ] **Step 3: 验证 Tool Calling**

Run:

```bash
conda run -n rag_env_backup python -c "from langchain_core.tools import tool; from model.factory import chat_model; exec('@tool\ndef probe():\n    \"\"\"返回固定探针值。\"\"\"\n    return \"probe-ok\"', globals()); r=chat_model.bind_tools([probe], tool_choice='required').invoke('调用 probe 工具'); print(bool(r.tool_calls)); print(r.tool_calls[0]['name'] if r.tool_calls else '')"
```

Expected: 输出 `True` 和 `probe`。若文本调用成功但这里失败，则该兼容服务不能作为当前 Agent 的聊天模型，恢复本机 `.env` 中的 Ollama 配置，不改通用 Provider 代码。

- [ ] **Step 4: 确认密钥未被 Git 跟踪**

Run:

```bash
git status --short
git diff -- .env
```

Expected: `.env` 不出现在状态或 diff 中。
