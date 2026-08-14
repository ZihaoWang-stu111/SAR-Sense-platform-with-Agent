"""Research Agent 单元测试：工具白名单 + _extract_final_ai_text 纯逻辑。

不触发真实 create_agent 构建（避免连 Ollama/MySQL/Chroma），只测纯函数与常量。
"""
import unittest

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class ResearchToolsWhitelistTest(unittest.TestCase):
    """工具白名单：Research Agent 只许有 RAG/Web/时间，禁止检测/报告/委派。"""

    def test_research_tools_exactly_three(self):
        from agent.research_agent import RESEARCH_TOOLS
        names = {t.name for t in RESEARCH_TOOLS}
        self.assertEqual(names, {"rag_summarize", "web_search", "get_current_month"})

    def test_no_delegate_research_to_prevent_recursion(self):
        """防递归：delegate_research 不得进入子 Agent 工具列表。"""
        from agent.research_agent import RESEARCH_TOOLS
        names = {t.name for t in RESEARCH_TOOLS}
        self.assertNotIn("delegate_research", names)

    def test_no_business_tools(self):
        from agent.research_agent import RESEARCH_TOOLS
        names = {t.name for t in RESEARCH_TOOLS}
        for forbidden in (
            "detect_ships", "fill_context_for_report", "fetch_external_data",
            "get_weather", "get_scene_trend", "compare_scenes", "get_sea_state",
        ):
            self.assertNotIn(forbidden, names, f"{forbidden} 不应出现在子 Agent 工具集")


class ResearchBudgetTest(unittest.TestCase):
    def test_research_budget_leaves_room_for_final_summary(self):
        from agent.research_agent import (
            _RESEARCH_MAX_TOOL_CALLS,
            _RESEARCH_RECURSION_LIMIT,
        )

        self.assertEqual(_RESEARCH_MAX_TOOL_CALLS, 5)
        self.assertEqual(_RESEARCH_RECURSION_LIMIT, 30)


class ExtractFinalAiTextTest(unittest.TestCase):
    def test_returns_last_nonempty_ai(self):
        from agent.research_agent import _extract_final_ai_text
        messages = [
            HumanMessage(content="比较 A B C"),
            AIMessage(content="", tool_calls=[{"name": "rag_summarize", "args": {}, "id": "1"}]),
            ToolMessage(content="结果", tool_call_id="1"),
            AIMessage(content="最终研究结论"),
        ]
        self.assertEqual(_extract_final_ai_text(messages), "最终研究结论")

    def test_returns_fallback_when_no_ai(self):
        from agent.research_agent import _extract_final_ai_text
        messages = [HumanMessage(content="x"), ToolMessage(content="y", tool_call_id="1")]
        self.assertEqual(_extract_final_ai_text(messages), "研究子智能体未返回有效结论。")

    def test_returns_fallback_when_empty(self):
        from agent.research_agent import _extract_final_ai_text
        self.assertEqual(_extract_final_ai_text([]), "研究子智能体未返回有效结论。")

    def test_returns_fallback_when_none(self):
        from agent.research_agent import _extract_final_ai_text
        self.assertEqual(_extract_final_ai_text(None), "研究子智能体未返回有效结论。")

    def test_skips_empty_ai_content(self):
        from agent.research_agent import _extract_final_ai_text
        messages = [AIMessage(content="   "), AIMessage(content="有效结论")]
        self.assertEqual(_extract_final_ai_text(messages), "有效结论")

    def test_does_not_raise_on_non_string_content(self):
        from agent.research_agent import _extract_final_ai_text
        # content 为 list（部分模型返回结构化 content）时不应抛异常，跳过找下一个
        messages = [AIMessage(content=[]), AIMessage(content="文本结论")]
        self.assertEqual(_extract_final_ai_text(messages), "文本结论")


if __name__ == "__main__":
    unittest.main()
