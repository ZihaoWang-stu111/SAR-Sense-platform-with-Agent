# 聊天实时加载状态 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在聊天 assistant 回答区域增加行内旋转指示器和实时阶段文字，并在第一段最终回答出现时立即隐藏。

**Architecture:** 后端只新增一个表示 Agent 工作线程已经开始的 `status` SSE 事件；检索、工具和生成阶段继续复用现有 `thought_step`。前端把 `loadingStatus` 保存在当前流式 assistant 消息对象中，前台与后台 SSE 解析路径共同维护它，渲染层只在正文尚未出现时输出加载行。

**Tech Stack:** FastAPI SSE、Vanilla JavaScript、CSS、Python `unittest`

---

## 文件结构

- `api/routers/chat.py`：发送 Agent 已开始执行的 `status` SSE 事件。
- `js/chat.js`：维护临时加载状态、映射思考步骤并渲染 assistant 加载行。
- `css/style_v2.css`：复用现有 spinner，补充加载行和减少动画偏好。
- `tests/test_agent_executor.py`：验证 `status` 事件先于正文事件。
- `tests/test_chat_live_status.py`：验证前端两条 SSE 路径、首字符隐藏和 CSS 规则。

### Task 0: 提交已验证的有界 Agent 执行器基线

**Files:**
- Create: `services/agent_executor.py`
- Modify: `api/dependencies.py`
- Modify: `api/routers/chat.py`
- Modify: `api/app.py`
- Modify: `AGENTS.md`
- Test: `tests/test_agent_executor.py`

- [ ] **Step 1: 重新运行基线测试**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_agent_executor -v
```

Expected: 4 tests pass.

- [ ] **Step 2: 检查基线 diff**

Run:

```bash
git diff --check
git status --short
```

Expected: 无 whitespace error；只有上述执行器文件处于未提交状态。

- [ ] **Step 3: 单独提交执行器改造**

```bash
git add AGENTS.md api/app.py api/dependencies.py api/routers/chat.py services/agent_executor.py tests/test_agent_executor.py
git commit -m "refactor: 限制智能体并发线程数"
```

### Task 1: 增加 Agent started SSE 事件

**Files:**
- Modify: `tests/test_agent_executor.py`
- Modify: `api/routers/chat.py`

- [ ] **Step 1: 写失败测试**

在 `ChatAgentDispatchTest.test_chat_submits_agent_work_to_shared_executor` 的末尾增加：

```python
self.assertIn("event: status", body)
self.assertIn("正在思考...", body)
self.assertLess(body.index("event: status"), body.index("event: chunk"))
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_agent_executor.ChatAgentDispatchTest -v
```

Expected: FAIL，因为响应中还没有 `event: status`。

- [ ] **Step 3: 在 Agent 工作任务开始时写入状态**

在 `run_agent()` 最前面、开始指标计时之前加入：

```python
loop.call_soon_threadsafe(
    sse_queue.put_nowait,
    ("status", "正在思考..."),
)
```

在 SSE 事件分支中加入：

```python
elif event_type == "status":
    data = json.dumps({"content": event_data}, ensure_ascii=False)
    yield f"event: status\ndata: {data}\n\n"
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_agent_executor.ChatAgentDispatchTest -v
```

Expected: PASS。

- [ ] **Step 5: 提交后端状态事件**

```bash
git add api/routers/chat.py tests/test_agent_executor.py
git commit -m "feat: 推送智能体执行状态"
```

### Task 2: 增加前端状态模型和两条 SSE 解析路径

**Files:**
- Create: `tests/test_chat_live_status.py`
- Modify: `js/chat.js`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_chat_live_status.py`：

```python
import unittest
from pathlib import Path


class ChatLiveStatusFrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("js/chat.js").read_text(encoding="utf-8")

    def test_foreground_and_background_streams_handle_status(self):
        self.assertEqual(self.source.count("eventType === 'status'"), 2)
        self.assertIn("loadingStatus: '正在等待处理...'", self.source)

    def test_thought_steps_map_to_friendly_status(self):
        self.assertIn("function getLoadingStatusForStep(step)", self.source)
        self.assertIn("rag_summarize: '正在检索知识库...'", self.source)
        self.assertIn("web_search: '正在搜索网络...'", self.source)
        self.assertIn("detect_ships: '正在检测图像...'", self.source)
        self.assertIn("return '正在整理结果...'", self.source)
        self.assertIn("return '正在生成回答...'", self.source)

    def test_first_typewriter_character_hides_status(self):
        typewriter_start = self.source.index("async function processTypewriterQueue()")
        typewriter_end = self.source.index("async function appendDetectImages", typewriter_start)
        typewriter = self.source[typewriter_start:typewriter_end]
        self.assertIn("msg.loadingStatus = null", typewriter)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_chat_live_status -v
```

Expected: FAIL，因为 `loadingStatus` 和状态映射尚不存在。

- [ ] **Step 3: 增加状态映射函数**

在 `sendMessageStreaming()` 之前增加：

```javascript
const TOOL_LOADING_STATUS = Object.freeze({
  rag_summarize: '正在检索知识库...',
  web_search: '正在搜索网络...',
  detect_ships: '正在检测图像...',
  extract_file_content: '正在解析文件...'
});

function getLoadingStatusForStep(step) {
  if (!step) return null;
  if (step.step_type === 'tool_call') {
    return TOOL_LOADING_STATUS[step.tool_name] || '正在调用工具...';
  }
  if (step.step_type === 'tool_result') {
    return '正在整理结果...';
  }
  if (step.step_type === 'final_answer') {
    return '正在生成回答...';
  }
  if (step.step_type === 'thinking') {
    return '正在思考...';
  }
  return null;
}
```

- [ ] **Step 4: 初始化前台和后台 assistant 状态**

两个 assistant 消息对象都增加：

```javascript
loadingStatus: '正在等待处理...'
```

- [ ] **Step 5: 处理前台 SSE 状态**

前台 SSE 分支增加：

```javascript
if (eventType === 'status') {
  assistantMessage.loadingStatus = data.content || '正在思考...';
  updateLastMessage(false);
} else if (eventType === 'chunk') {
  assistantMessage.pendingChunks.push(data.content);
} else if (eventType === 'rag_result') {
  assistantMessage.loadingStatus = '正在生成回答...';
  if (!assistantMessage.rag_results) assistantMessage.rag_results = [];
  assistantMessage.rag_results.push(data.content);
  updateLastMessage(false);
} else if (eventType === 'thought_step') {
  assistantMessage.thoughtSteps.push(data.step);
  const nextStatus = getLoadingStatusForStep(data.step);
  if (nextStatus) assistantMessage.loadingStatus = nextStatus;
  updateLastMessage(false);
  updateThoughtChainRealtime(assistantMessage.thoughtSteps);
}
```

`done`、`error` 和请求异常分支中设置：

```javascript
assistantMessage.loadingStatus = null;
```

- [ ] **Step 6: 处理后台 SSE 状态**

`processStreamInBackground()` 使用同一映射：

```javascript
if (eventType === 'status') {
  assistantMessage.loadingStatus = data.content || '正在思考...';
} else if (eventType === 'chunk') {
  assistantMessage.loadingStatus = null;
  assistantMessage.content += data.content;
} else if (eventType === 'rag_result') {
  assistantMessage.loadingStatus = '正在生成回答...';
  if (!assistantMessage.rag_results) assistantMessage.rag_results = [];
  assistantMessage.rag_results.push(data.content);
} else if (eventType === 'thought_step') {
  assistantMessage.thoughtSteps.push(data.step);
  const nextStatus = getLoadingStatusForStep(data.step);
  if (nextStatus) assistantMessage.loadingStatus = nextStatus;
}
```

后台 `done`、`error` 和连接异常同样清空 `loadingStatus`。

- [ ] **Step 7: 在第一段正文实际出现时隐藏状态**

`processTypewriterQueue()` 取出非空 chunk 后，在逐字循环前加入：

```javascript
if (chunk && msg.loadingStatus) {
  msg.loadingStatus = null;
}
```

这样 spinner 会与第一个字符的首次 `updateLastMessage(false)` 同时消失。

- [ ] **Step 8: 运行测试和 JS 语法检查**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_chat_live_status -v
node --check js/chat.js
```

Expected: 3 tests pass，Node 退出码为 0。

- [ ] **Step 9: 提交状态模型**

```bash
git add js/chat.js tests/test_chat_live_status.py
git commit -m "feat: 跟踪聊天实时处理状态"
```

### Task 3: 渲染行内 spinner

**Files:**
- Modify: `tests/test_chat_live_status.py`
- Modify: `js/chat.js`
- Modify: `css/style_v2.css`

- [ ] **Step 1: 写失败测试**

向 `tests/test_chat_live_status.py` 增加：

```python
    def test_loading_status_has_accessible_markup_and_reduced_motion(self):
        css = Path("css/style_v2.css").read_text(encoding="utf-8")
        self.assertIn("function renderAssistantLoadingStatus(status)", self.source)
        self.assertIn('class="message-status assistant-loading-status"', self.source)
        self.assertIn('role="status"', self.source)
        self.assertIn('aria-live="polite"', self.source)
        self.assertIn(".assistant-loading-status", css)
        self.assertIn("@media (prefers-reduced-motion: reduce)", css)
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_chat_live_status.ChatLiveStatusFrontendTest.test_loading_status_has_accessible_markup_and_reduced_motion -v
```

Expected: FAIL，因为加载状态 HTML 和减少动画规则尚不存在。

- [ ] **Step 3: 增加统一渲染函数**

在 `buildAssistantDisplayContent()` 后增加：

```javascript
function renderAssistantLoadingStatus(status) {
  if (!status) return '';
  return `
    <div class="message-status assistant-loading-status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span>
      <span>${escapeHtml(status)}</span>
    </div>
  `;
}

function renderAssistantDisplayHtml(message, isStreaming) {
  const hasAnswer = Boolean((message?.content || '').trim());
  const showLoading = Boolean(isStreaming && message?.loadingStatus && !hasAnswer);
  let html = renderMarkdown(buildAssistantDisplayContent(message));
  html = renderWithCitations(html, isStreaming && !showLoading);
  return `${showLoading ? renderAssistantLoadingStatus(message.loadingStatus) : ''}${html}`;
}
```

将 `updateLastMessage()` 中的：

```javascript
let html = renderMarkdown(buildAssistantDisplayContent(assistantMessage));
html = renderWithCitations(html, isStreaming);
```

替换为：

```javascript
let html = renderAssistantDisplayHtml(assistantMessage, isStreaming);
```

将 `renderMessages()` 中 assistant 的 Markdown 和引用渲染替换为同一个函数：

```javascript
if (msg.role === 'assistant') {
  html = renderAssistantDisplayHtml(msg, isStreamingMsg);
}
```

- [ ] **Step 4: 收紧现有状态样式**

在 `css/style_v2.css` 现有 `.message-status` 附近增加：

```css
.assistant-loading-status {
  min-height: 1.5rem;
  padding: 0.125rem 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.assistant-loading-status .spinner {
  width: 14px;
  height: 14px;
  flex: 0 0 14px;
  border-color: var(--border-color);
  border-top-color: var(--accent-primary);
}

@media (prefers-reduced-motion: reduce) {
  .assistant-loading-status .spinner {
    animation: none;
  }
}
```

- [ ] **Step 5: 运行前端测试和语法检查**

Run:

```bash
conda run -n rag_env_backup python -m unittest tests.test_chat_live_status -v
node --check js/chat.js
```

Expected: 4 tests pass，Node 退出码为 0。

- [ ] **Step 6: 提交行内加载 UI**

```bash
git add js/chat.js css/style_v2.css tests/test_chat_live_status.py
git commit -m "feat: 展示聊天实时加载状态"
```

### Task 4: 回归和视觉验证

**Files:**
- Verify: `api/routers/chat.py`
- Verify: `js/chat.js`
- Verify: `css/style_v2.css`

- [ ] **Step 1: 运行完整相关回归**

```bash
conda run -n rag_env_backup python -m unittest tests.test_agent_executor tests.test_chat_live_status tests.test_chat_image_ocr tests.test_rag_acl_retrieval -v
conda run -n rag_env_backup python -m compileall api services tests
node --check js/chat.js
git diff --check
```

Expected: 所有测试通过，无 Python/JavaScript 语法错误，无 whitespace error。

- [ ] **Step 2: 启动服务并验证状态顺序**

```bash
conda run -n rag_env_backup python api_server_fastapi.py
```

在浏览器中发送普通问题和 RAG 问题，确认：

```text
正在等待处理...
→ 正在思考...
→ 正在检索知识库...（RAG 问题）
→ 正在整理结果...
→ 正在生成回答...
→ 第一段正文出现时 spinner 消失
```

- [ ] **Step 3: 验证会话切换**

回答生成期间切到其他会话再切回，确认：

- 状态不会串到其他会话。
- 后台回答继续生成。
- 切回时要么显示当前状态，要么已经显示正文。
- 完成后历史会话中不保留 spinner。

- [ ] **Step 4: 检查最终工作区**

```bash
git status --short
git log -5 --oneline
```

Expected: 只有用户原有的未提交改动；本计划产生的提交按任务分开。
