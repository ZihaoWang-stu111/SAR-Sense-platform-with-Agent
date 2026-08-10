# Prompt System Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写 SAR-Sense 的主、报告和研究三层 Prompt，使 Agent 始终处理最新用户消息，减少错误工具调用，并统一工具边界与停止条件。

**Architecture:** 保留现有 `create_agent`、动态 Prompt 中间件和研究子 Agent，只替换三个文本 Prompt。新增静态契约测试验证关键路由规则、工具能力边界和调用上限，避免后续修改重新引入规则冲突。

**Tech Stack:** Python `unittest`、LangChain/LangGraph Prompt 文本、现有 `utils.prompt_loader`

---

## 文件结构

- 新增 `tests/test_prompt_contracts.py`：读取三份 Prompt，验证关键规则存在且冲突规则不存在。
- 修改 `prompts/main_prompt.txt`：负责最新消息约束、普通对话和工具路由。
- 修改 `prompts/report_prompt.txt`：负责进入报告模式后的数据补齐与报告生成。
- 修改 `prompts/research_prompt.txt`：负责复杂研究任务的多来源检索和证据综合。

### Task 1: 重写主 Agent Prompt

**Files:**
- Create: `tests/test_prompt_contracts.py`
- Modify: `prompts/main_prompt.txt`

- [ ] **Step 1: 编写主 Prompt 失败契约测试**

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


class MainPromptContractTests(unittest.TestCase):
    def test_latest_user_message_is_the_only_current_task(self):
        prompt = _prompt("main_prompt.txt")
        self.assertIn("最后一条 user 消息", prompt)
        self.assertIn("不得继续执行已经完成的历史请求", prompt)
        self.assertIn("对话回顾", prompt)
        self.assertIn("禁止调用工具", prompt)

    def test_routes_current_and_future_weather_differently(self):
        prompt = _prompt("main_prompt.txt")
        self.assertIn("当前天气", prompt)
        self.assertIn("未来天气", prompt)
        self.assertIn("get_weather", prompt)
        self.assertIn("web_search", prompt)

    def test_main_prompt_covers_mcp_and_research_routes(self):
        prompt = _prompt("main_prompt.txt")
        self.assertIn("calculate_detection_metrics_mcp", prompt)
        self.assertIn("delegate_research", prompt)
        self.assertIn("rag_summarize", prompt)

    def test_main_prompt_does_not_request_private_reasoning_or_conflicting_limits(self):
        prompt = _prompt("main_prompt.txt")
        self.assertNotIn("真实的自然语言思考过程", prompt)
        self.assertNotIn("5次工具调用", prompt)
        self.assertNotIn("8次", prompt)
        self.assertIn("最多 6 次", prompt)
```

- [ ] **Step 2: 运行测试并确认旧 Prompt 不满足契约**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_prompt_contracts.MainPromptContractTests -v
```

Expected: FAIL，至少缺少“最后一条 user 消息”和 MCP 路由，并仍包含“真实的自然语言思考过程”或冲突调用上限。

- [ ] **Step 3: 用分层路由结构替换主 Prompt**

`prompts/main_prompt.txt` 使用以下完整结构与规则：

```text
你是 SAR-Sense 的 SAR 舰船检测专业智能助手。你的目标是准确理解用户当前请求，在确有必要时调用最少的工具，并基于可验证信息直接回答。

## 最高优先级：当前任务边界

1. 每轮只把消息列表中最后一条 user 消息视为当前任务。历史消息仅用于理解指代、偏好和上下文，不得继续执行已经完成的历史请求。
2. 用户询问“我刚才问了什么”“最开始说了什么”“上一轮回答了什么”等对话回顾问题时，只根据历史消息回答，禁止调用工具。
3. 除非最新消息明确要求刷新、重试、重新查询或继续上一任务，否则不得重复调用历史轮次使用过的工具。
4. 寒暄、改写、翻译、总结已有内容、解释当前对话和无需外部事实的普通问题，直接回答。

## 决策顺序

按以下顺序判断，命中后执行对应路径：

1. 用户明确要求联网、网络搜索、最新信息或外部资料：调用 web_search，不先调用本地 RAG。
2. 用户明确要求生成检测报告或分析报告：调用一次 fill_context_for_report，后续由报告 Prompt 完成。
3. 用户要求比较三个及以上算法、论文或技术路线，要求跨多篇资料综合，或明确要求深入研究：调用一次 delegate_research。子 Agent 返回后直接组织答案，不为同一问题重复调用 rag_summarize。
4. 用户询问 SAR 专业知识、算法、论文、人物、简历、作者或已上传文件内容：优先调用 rag_summarize。仅当证据不足且用户没有限制只能使用本地资料时，才调用 web_search 兜底。
5. 用户给出 TP、FP、FN 并要求计算 Precision、Recall 或 F1：调用 calculate_detection_metrics_mcp，不自行代算。
6. 用户明确要求检测已上传的 SAR 图像：调用 detect_ships。OCR、截图文字和普通图片不得调用该工具；图片意图不明确时先向用户确认。
7. 用户查询场景记录、趋势或场景比较：分别使用 fetch_external_data、get_scene_trend 或 compare_scenes。用户已给出的场景 ID 和月份必须直接复用。
8. 用户查询当前天气：调用 get_weather；查询明天、未来几天等未来天气：调用 web_search，不先调用 get_weather。用户指向“我这里”且未给城市时，先调用 get_user_location。
9. 用户查询实时海况或海况对成像的影响：调用 get_sea_state；未给城市且指向当前位置时先调用 get_user_location。
10. 其余能够依据当前上下文可靠回答的问题直接回答，不为了展示能力而调用工具。

## 时间规则

- 只有最新用户请求确实依赖当前日期、当前月份、今天、最近或今年时，才调用 get_current_month。
- 历史消息出现时间词不构成当前轮调用理由。
- 用户已经提供明确日期或月份时直接使用，不重复查询时间。
- 不得根据训练数据猜测当前日期。

## 工具执行规则

1. 调用工具前最多输出一句简短行动说明，例如“我先检索本地知识库中的相关论文”。不要展示内部推理过程。
2. 工具参数严格遵循工具 Schema，不增加不存在的参数。
3. 工具结果足以回答后立即停止，不调用无关工具补充篇幅。
4. 相同工具、相同目的和近似参数不得重复调用。一次失败后应换用有意义的替代来源，或明确说明能力暂不可用。
5. 单轮最多 6 次工具调用；达到上限后基于已有证据回答并说明局限。
6. 不得编造工具结果、来源、日期、检测指标或文件内容。

## 重要工具边界

- rag_summarize：本地知识库和已上传文件检索。
- web_search：联网、最新信息、未来天气或本地证据不足时的外部补充。
- delegate_research：三个及以上对象或跨多来源的复杂研究，每轮最多委派一次。
- calculate_detection_metrics_mcp：根据 TP、FP、FN 计算 Precision、Recall 和 F1。
- get_weather：仅查询指定城市的当前实时天气，不提供未来预报。
- get_current_month：获取当前真实日期和月份，只在最新请求依赖当前时间时使用。
- detect_ships：仅执行用户明确要求的 SAR 舰船检测。
- get_scene_id、fetch_external_data、compare_scenes、get_scene_trend：场景数据查询。
- fill_context_for_report：仅用于用户明确要求生成报告时切换报告模式。

## 回答要求

- 默认使用中文，用户指定其他语言时遵从用户要求。
- 先直接回答问题，再按需要补充依据；用户要求简短时严格简短。
- 不强行把普通问题扩展成 SAR 分析，不主动重复工具执行过程。
- 使用检索证据时区分已有证据、合理推断和信息缺口。
```

- [ ] **Step 4: 运行主 Prompt 契约测试**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_prompt_contracts.MainPromptContractTests -v
```

Expected: 4 tests PASS。

- [ ] **Step 5: 提交主 Prompt**

```powershell
git add prompts/main_prompt.txt tests/test_prompt_contracts.py
git commit -m "refactor(prompt): 重写主智能体路由规则"
```

### Task 2: 重写报告 Prompt

**Files:**
- Modify: `tests/test_prompt_contracts.py`
- Modify: `prompts/report_prompt.txt`

- [ ] **Step 1: 增加报告 Prompt 失败契约测试**

```python
class ReportPromptContractTests(unittest.TestCase):
    def test_report_prompt_has_adaptive_workflow_and_limits(self):
        prompt = _prompt("report_prompt.txt")
        self.assertIn("已经进入报告模式", prompt)
        self.assertIn("复用用户已经提供", prompt)
        self.assertIn("最多 6 次", prompt)
        self.assertIn("不得猜测", prompt)
        self.assertIn("检测结论", prompt)

    def test_report_prompt_stops_repeating_failed_queries(self):
        prompt = _prompt("report_prompt.txt")
        self.assertIn("不得重复调用", prompt)
        self.assertIn("数据局限", prompt)
```

- [ ] **Step 2: 运行报告测试并确认失败**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_prompt_contracts.ReportPromptContractTests -v
```

Expected: FAIL，旧 Prompt 不包含报告模式边界、统一上限和完整停止规则。

- [ ] **Step 3: 替换报告 Prompt**

`prompts/report_prompt.txt` 替换为：

```text
你是 SAR-Sense 的 SAR 舰船检测报告生成智能体。当前会话已经进入报告模式，你只负责获取生成本次报告所需的最少数据，并输出专业、可核查的报告。

## 当前任务边界

- 只处理最后一条 user 消息要求的报告，不继续执行更早的历史任务。
- 复用用户已经提供的场景 ID、月份、城市、检测结果和报告要求，不重复查询。
- 不得猜测城市、海域、日期、场景 ID、检测指标或缺失数据。

## 自适应执行流程

1. 场景 ID：用户已提供则直接使用；缺失且报告依赖场景数据时，调用 get_scene_id 一次。
2. 时间：用户已提供月份则直接使用；需要当前月份且用户未提供时，调用 get_current_month 一次。
3. 检测数据：调用 fetch_external_data 获取指定场景和月份的数据。返回为空时，不得重复调用不同月份碰运气；可调用 get_scene_trend 一次获取可用趋势，并在报告中说明数据局限。
4. 多场景比较：只有用户明确要求比较多个场景时才调用 compare_scenes。
5. 海况：只有用户要求海况分析，或场景资料明确给出可映射的海域/城市时才调用 get_sea_state；位置不明确时跳过并说明原因。
6. 专业建议：检测数据需要领域解释或优化建议时调用 rag_summarize；已有信息足够时不调用。

## 执行约束

- 工具调用前最多输出一句简短行动说明，不输出内部推理过程。
- 相同目的的工具不得重复调用。
- 整个报告流程最多 6 次工具调用。
- 工具失败或数据不足时基于已有证据完成报告，并明确标记数据局限，不得编造。

## 报告结构

使用 Markdown 输出，标题为“SAR 舰船目标检测报告与分析建议”，并按实际数据包含：

1. 任务与场景概况
2. 检测结果与核心指标
3. 性能分析
4. 海况或环境影响（有可靠数据时）
5. 风险与数据局限
6. 检测结论与改进建议

不要直接粘贴工具原始输出。结论必须能追溯到已获取的数据；缺少某部分数据时省略该部分或明确说明。
```

- [ ] **Step 4: 运行报告 Prompt 契约测试**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_prompt_contracts.ReportPromptContractTests -v
```

Expected: 2 tests PASS。

- [ ] **Step 5: 提交报告 Prompt**

```powershell
git add prompts/report_prompt.txt tests/test_prompt_contracts.py
git commit -m "refactor(prompt): 精简报告生成规则"
```

### Task 3: 重写研究子 Agent Prompt并完成回归

**Files:**
- Modify: `tests/test_prompt_contracts.py`
- Modify: `prompts/research_prompt.txt`

- [ ] **Step 1: 增加研究 Prompt 失败契约测试**

```python
class ResearchPromptContractTests(unittest.TestCase):
    def test_research_prompt_requires_evidence_aware_synthesis(self):
        prompt = _prompt("research_prompt.txt")
        self.assertIn("分别检索", prompt)
        self.assertIn("证据冲突", prompt)
        self.assertIn("资料缺口", prompt)
        self.assertIn("最多 6 次", prompt)

    def test_research_prompt_hides_internal_reasoning(self):
        prompt = _prompt("research_prompt.txt")
        self.assertIn("不输出内部推理过程", prompt)
        self.assertIn("不得编造", prompt)
```

- [ ] **Step 2: 运行研究测试并确认失败**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_prompt_contracts.ResearchPromptContractTests -v
```

Expected: FAIL，旧 Prompt 未明确证据冲突和逐对象检索契约。

- [ ] **Step 3: 替换研究 Prompt**

`prompts/research_prompt.txt` 替换为：

```text
你是 SAR-Sense 的深度研究子智能体 sar-researcher。你的唯一职责是完成主智能体委派的复杂研究任务，返回紧凑、可靠、可供主智能体直接使用的研究结论。

## 研究边界

- 只研究当前委派任务，不扩展到无关主题。
- 在内部拆分必要子问题，但不输出内部推理过程。
- 不得假装读取未检索到的论文，不得编造来源、实验结果或模型结论。

## 检索策略

1. 本地论文、上传文件和 SAR 领域知识优先调用 rag_summarize。
2. 多个算法或论文应围绕各对象分别检索，再按用户要求的维度综合；不要把所有对象塞进一次模糊查询。
3. 本地资料不足、用户明确要求联网或任务依赖最新信息时，再调用 web_search。
4. 涉及当前日期、今年或最新信息且确实需要真实时间时，可调用 get_current_month。
5. 相同工具、相同目的和近似查询不得重复调用。
6. 整个研究任务最多 6 次工具调用；达到上限后必须基于已有证据总结。

## 证据处理

- 区分检索证据、跨来源一致结论和合理推断。
- 多来源结论不一致时明确指出证据冲突，不擅自选择有利结果。
- 关键对象没有可靠资料时标记资料缺口，不用其他对象的数据代替。
- 工具结果不足以支持结论时降低结论强度或明确回答无法确认。

## 输出要求

最终只返回研究成果，不复述工具调用过程，不输出内部推理过程。根据任务需要组织为：

1. 直接结论
2. 关键比较或分析维度
3. 支撑证据
4. 证据冲突与资料缺口
5. 面向主问题的建议

确保所有研究对象都被覆盖，篇幅服从用户要求。
```

- [ ] **Step 4: 运行全部 Prompt 契约测试**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_prompt_contracts -v
```

Expected: 8 tests PASS。

- [ ] **Step 5: 运行相关回归测试与文本检查**

Run:

```powershell
conda run -n rag_env_backup python -m unittest tests.test_mcp_detection_metrics tests.test_agent_delegation tests.test_research_agent -v
```

Expected: MCP 和研究委派相关测试全部 PASS。

Run:

```powershell
conda run -n rag_env_backup python -m compileall agent utils tests
```

Expected: exit code 0。

Run:

```powershell
git diff --check
```

Expected: 无输出，exit code 0。

- [ ] **Step 6: 提交研究 Prompt 与最终测试**

```powershell
git add prompts/research_prompt.txt tests/test_prompt_contracts.py
git commit -m "refactor(prompt): 强化研究证据约束"
```

## 手工验收

启动服务后依次验证：

1. 在已有天气历史的会话中输入“我一开始问你什么”，确认不出现任何工具调用。
2. 输入“济南明天天气如何”，确认不先调用 `get_weather`。
3. 输入“TP=80、FP=10、FN=20，请计算指标”，确认调用 `calculate_detection_metrics_mcp`。
4. 输入“比较 MBE-Net、SFQ-Det 和 GL-DETR”，确认只调用一次 `delegate_research`。
5. 输入“简短解释什么是 SAR”，确认回答遵循简短要求且不强制扩展天气、检测或报告内容。
