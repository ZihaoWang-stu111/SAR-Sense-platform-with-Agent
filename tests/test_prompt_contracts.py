from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


class MainPromptContractTests(unittest.TestCase):
    def test_latest_user_message_is_the_only_current_task(self):
        prompt = _prompt("main_prompt.txt")
        self.assertRegex(
            prompt,
            r"每轮[^\n]*最后一条 user 消息[^\n]*当前任务",
        )
        self.assertRegex(
            prompt,
            r"对话回顾[^\n]*只根据历史消息回答[^\n]*禁止调用工具",
        )
        self.assertRegex(
            prompt,
            r"不得仅因为历史轮次调用过某工具[^\n]*当前轮自动重放",
        )
        self.assertRegex(
            prompt,
            r"最新消息[^\n]*新的、独立的工具需求[^\n]*正常调用[^\n]*同一工具",
        )
        self.assertRegex(
            prompt,
            r"上海天气[^\n]*北京天气[^\n]*get_weather",
        )
        self.assertNotIn(
            "除非最新消息明确要求刷新、重试、重新查询或继续上一任务，"
            "否则不得重复调用历史轮次使用过的工具",
            prompt,
        )

    def test_routes_current_and_future_weather_differently(self):
        prompt = _prompt("main_prompt.txt")
        self.assertRegex(
            prompt,
            r"当前天气[^\n]*调用 get_weather[^\n]*未来天气[^\n]*调用 web_search",
        )

    def test_decision_priority_and_specialized_routes_are_unambiguous(self):
        prompt = _prompt("main_prompt.txt")
        decision_start = prompt.index("## 决策顺序")
        route_labels = (
            "1. 检测结果/场景数据类报告",
            "2. 复杂研究",
            "3. 普通显式联网查询",
        )
        for label in route_labels:
            self.assertIn(label, prompt)
        route_positions = [prompt.index(label, decision_start) for label in route_labels]
        self.assertEqual(route_positions, sorted(route_positions))
        self.assertLess(decision_start, route_positions[0])

        self.assertRegex(
            prompt,
            r"优先级固定为[^\n]*检测结果/场景数据类报告"
            r"[^\n]*复杂研究[^\n]*普通显式联网查询",
        )
        self.assertRegex(
            prompt,
            r"1\. 检测结果/场景数据类报告[^\n]*fill_context_for_report"
            r"[^\n]*(?:算法|论文)[^\n]*不属于报告模式",
        )
        self.assertRegex(
            prompt,
            r"2\. 复杂研究[^\n]*三个及以上算法、论文、模型或技术路线"
            r"[^\n]*跨多篇算法/论文/技术资料综合"
            r"[^\n]*针对算法/论文/模型/技术路线开展深入研究"
            r"[^\n]*delegate_research"
            r"[^\n]*同时明确要求联网[^\n]*研究子 Agent[^\n]*web_search",
        )
        self.assertNotIn("三个及以上对象", prompt)
        self.assertNotIn("跨多来源的复杂研究", prompt)
        self.assertRegex(
            prompt,
            r"3\. 普通显式联网查询[^\n]*不属于报告模式和复杂研究"
            r"[^\n]*web_search",
        )
        self.assertRegex(
            prompt,
            r"TP、FP、FN[^\n]*Precision[^\n]*调用 calculate_detection_metrics_mcp",
        )
        self.assertRegex(
            prompt,
            r"SAR 专业知识[^\n]*优先调用 rag_summarize",
        )
        self.assertRegex(
            prompt,
            r"场景记录、趋势和比较[^\n]*不属于研究委派"
            r"[^\n]*始终按场景工具路由[^\n]*fetch_external_data"
            r"[^\n]*get_scene_trend[^\n]*compare_scenes"
            r"[^\n]*不得调用 delegate_research",
        )

    def test_main_prompt_does_not_request_private_reasoning_or_conflicting_limits(self):
        prompt = _prompt("main_prompt.txt")
        self.assertNotIn("真实的自然语言思考过程", prompt)
        limit_pattern = re.compile(
            r"(?:"
            r"(?:最多|不超过|不得超过)\s*(\d+)\s*次[^\n。；]{0,12}工具调用"
            r"|工具调用[^\n。；]{0,12}(?:最多|不超过|不得超过|上限\s*(?:为|是|[:：])?)"
            r"\s*(\d+)\s*次"
            r"|(\d+)\s*次\s*工具调用后"
            r")"
        )
        limits = [
            next(value for value in match.groups() if value is not None)
            for match in limit_pattern.finditer(prompt)
        ]
        self.assertEqual(limits, ["6"])
