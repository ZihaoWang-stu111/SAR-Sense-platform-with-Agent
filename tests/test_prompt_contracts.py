from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def _normalized_prompt(name: str) -> str:
    return re.sub(r"\s+", " ", _prompt(name)).strip()


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

    def test_rag_evidence_ids_and_supported_facts_survive_final_answer(self):
        prompt = _normalized_prompt("main_prompt.txt")
        self.assertRegex(
            prompt,
            r"rag_summarize.*?已经过.*?证据校验",
        )
        self.assertRegex(
            prompt,
            r"最终回答.*?保留.*?\[n\].*?引用编号.*?对应事实",
        )
        self.assertRegex(
            prompt,
            r"(?:不得|禁止).*?删除.*?(?:改号|更改编号).*?(?:伪造|新增).*?正文引用",
        )

    def test_rag_body_citations_are_preserved_without_copying_source_list(self):
        prompt = _normalized_prompt("main_prompt.txt")
        self.assertRegex(
            prompt,
            r"(?:保留|不得删除).*?rag_summarize 回答正文.*?\[n\].*?对应事实",
        )
        self.assertRegex(
            prompt,
            r"(?:不得|禁止).*?删除.*?改号.*?伪造.*?新增.*?正文引用",
        )
        self.assertRegex(
            prompt,
            r"后端.*?追加.*?参考来源：.*?列表.*?仅.*?展示.*?追溯",
        )
        self.assertRegex(
            prompt,
            r"参考来源：.*?列表.*?(?:不得|禁止).*?最终回答正文.*?(?:重复|照抄)",
        )

    def test_rag_partial_answer_boundary_is_not_filled_from_model_knowledge(self):
        prompt = _normalized_prompt("main_prompt.txt")
        self.assertRegex(
            prompt,
            r"RAG.*?仅支持部分.*?保留.*?已确认.*?缺失.*?边界",
        )
        self.assertRegex(
            prompt,
            r"(?:不得|禁止).*?模型自身知识.*?(?:补齐|补全).*?缺失事实",
        )

    def test_rag_insufficiency_respects_local_only_and_existing_web_fallback(self):
        prompt = _normalized_prompt("main_prompt.txt")
        self.assertRegex(
            prompt,
            r"RAG.*?(?:资料|证据)不足.*?只能.*?本地知识库.*?不要联网"
            r".*?直接说明.*?缺失.*?(?:不得调用|不调用) web_search",
        )
        self.assertRegex(
            prompt,
            r"用户.*?未.*?限制.*?确有必要.*?(?:允许|可).*?web_search.*?兜底",
        )


class RagSummarizePromptContractTests(unittest.TestCase):
    def setUp(self):
        self.prompt = _normalized_prompt("rag_summarize.txt")

    def test_prohibits_illegal_infringing_or_abusive_content(self):
        self.assertRegex(
            self.prompt,
            r"(?:禁止|不得).*?输出.*?违法.*?侵权.*?攻击性.*?内容",
        )

    def test_uses_only_input_evidence_and_defines_evidence_roles(self):
        self.assertRegex(
            self.prompt,
            r"只.*?依据.*?输入.*?\[证据n\].*?回答",
        )
        self.assertRegex(
            self.prompt,
            r"(?:不得|禁止).*?模型自身知识.*?(?:补全|添加)",
        )
        self.assertRegex(
            self.prompt,
            r"直接证据.*?正文.*?直接包含.*?回答.*?事实.*?支撑.*?结论",
        )
        self.assertRegex(
            self.prompt,
            r"背景信息.*?(?:主题相关|定义|上下文).*?不能.*?单独支撑.*?具体结论",
        )
        self.assertRegex(
            self.prompt,
            r"信息缺口.*?(?:没有|未).*?明确.*?条件.*?数字.*?比较.*?因果",
        )

    def test_requires_real_internal_evidence_ids_for_each_key_claim(self):
        self.assertRegex(
            self.prompt,
            r"每个.*?关键事实.*?结论.*?紧跟.*?\[\[EVIDENCE:n\]\]",
        )
        self.assertRegex(
            self.prompt,
            r"n.*?来自输入.*?真实存在.*?证据编号",
        )
        self.assertRegex(
            self.prompt,
            r"(?:不得|禁止).*?编造.*?越界.*?文件名.*?替代",
        )
        self.assertNotRegex(
            self.prompt,
            r"必须用《文件名》|根据《[^》]+》",
        )

    def test_partial_and_total_insufficiency_are_distinct(self):
        self.assertRegex(
            self.prompt,
            r"部分.*?直接证据.*?回答.*?已支持.*?引用.*?明确.*?缺失",
        )
        self.assertRegex(
            self.prompt,
            r"不要.*?完整.*?(?:补全|编造)",
        )
        self.assertRegex(
            self.prompt,
            r"只有.*?所有请求内容.*?没有.*?直接证据.*?答案.*?"
            r"\[\[INSUFFICIENT\]\].*?开头.*?具体说明.*?缺",
        )
        self.assertRegex(
            self.prompt,
            r"(?:主题相关|背景信息).*?(?:不等于|不得视为).*?(?:有答案|直接证据)",
        )

    def test_outputs_only_answer_body_without_sources_or_reasoning(self):
        self.assertRegex(
            self.prompt,
            r"只输出.*?回答正文",
        )
        self.assertRegex(
            self.prompt,
            r"(?:绝不|不得|不要).*?参考来源.*?列表.*?(?:后端|系统).*?追加",
        )
        self.assertRegex(
            self.prompt,
            r"(?:不输出|禁止输出).*?分类过程.*?内部推理.*?JSON",
        )

    def test_keeps_template_variables(self):
        raw_prompt = _prompt("rag_summarize.txt")
        self.assertEqual(raw_prompt.count("{input}"), 1)
        self.assertEqual(raw_prompt.count("{context}"), 1)

    def test_preserves_complete_tables_when_requested(self):
        self.assertRegex(
            self.prompt,
            r"用户.*?完整表格.*?(?:原样|完整).*?输出",
        )
        self.assertRegex(
            self.prompt,
            r"每一行.*?每一列.*?所有数值",
        )
        self.assertRegex(
            self.prompt,
            r"(?:禁止|不得).*?省略.*?(?:压缩|概括).*?表格",
        )


class ReportPromptContractTests(unittest.TestCase):
    def setUp(self):
        self.prompt = _normalized_prompt("report_prompt.txt")

    def _branch(self, start: str, end: str) -> str:
        self.assertIn(start, self.prompt)
        self.assertIn(end, self.prompt)
        start_index = self.prompt.index(start)
        end_index = self.prompt.index(end, start_index)
        return self.prompt[start_index:end_index]

    def test_report_mode_reuses_all_user_context_in_one_rule(self):
        self.assertRegex(
            self.prompt,
            r"当前会话.*?已经进入报告模式",
        )
        self.assertRegex(
            self.prompt,
            r"复用用户已经提供的.*?场景 ID.*?月份"
            r".*?城市.*?检测结果.*?不重复查询",
        )
        self.assertRegex(
            self.prompt,
            r"不得猜测.*?城市.*?海域.*?日期"
            r".*?场景 ID.*?检测指标.*?缺失数据",
        )
        self.assertRegex(
            self.prompt,
            r"整个报告流程.*?最多\s*6\s*次工具调用",
        )

    def test_report_mode_has_a_global_duplicate_call_rule(self):
        self.assertRegex(
            self.prompt,
            r"相同目的的工具.*?不得重复调用",
        )

    def test_report_mode_completes_from_evidence_when_tools_or_data_fail(self):
        self.assertRegex(
            self.prompt,
            r"工具失败或数据不足时.*?基于已有证据完成报告"
            r".*?标记数据局限.*?不得编造",
        )

    def test_detection_data_routes_are_mutually_exclusive(self):
        self.assertRegex(
            self.prompt,
            r"检测数据.*?场景数量.*?时间范围"
            r".*?互斥.*?只(?:执行|选择).*?一个分支",
        )

    def test_single_scene_multi_month_trend_uses_only_get_scene_trend(self):
        branch = self._branch(
            "单场景、多月份趋势",
            "多场景、同一月份比较",
        )
        self.assertRegex(
            branch,
            r"先复用.*?趋势数据.*?数据足够时.*?不查询"
            r".*?仅当用户要求单场景多月份趋势"
            r".*?相应数据缺失或不足时.*?才调用 get_scene_trend",
        )
        self.assertIn("不得调用 compare_scenes", branch)

    def test_multi_scene_same_month_comparison_uses_only_compare_scenes(self):
        branch = self._branch(
            "多场景、同一月份比较",
            "多场景、多月份趋势比较",
        )
        self.assertRegex(
            branch,
            r"先复用.*?同月多场景比较数据.*?数据足够时.*?不查询"
            r".*?仅当用户要求多场景同一月份比较"
            r".*?相应数据缺失或不足时.*?才调用 compare_scenes",
        )
        self.assertIn("不得调用 get_scene_trend", branch)

    def test_multi_scene_multi_month_comparison_queries_each_scene_trend(self):
        branch = self._branch(
            "多场景、多月份趋势比较",
            "普通单场景报告",
        )
        self.assertRegex(
            branch,
            r"先复用.*?各场景趋势数据.*?数据足够时.*?不查询"
            r".*?仅当用户要求多场景多月份趋势比较"
            r".*?相应数据缺失或不足时"
            r".*?按场景分别调用 get_scene_trend.*?再综合",
        )
        self.assertIn("不得调用 compare_scenes", branch)
        self.assertRegex(
            branch,
            r"单轮.*?最多\s*6\s*次工具调用.*?总上限",
        )

    def test_single_scene_report_fetches_only_when_data_is_insufficient(self):
        self.assertRegex(
            self.prompt,
            r"普通单场景报告.*?用户(?:已)?提供足够检测结果时直接复用"
            r".*?只有普通单场景检测报告.*?已有检测数据缺失或不足时"
            r".*?才调用 fetch_external_data",
        )
        self.assertNotRegex(
            self.prompt,
            r"检测数据\s*[:：]\s*调用 fetch_external_data",
        )
        self.assertRegex(
            self.prompt,
            r"fetch_external_data.*?返回为空时"
            r".*?不得(?:更换月份|重复调用不同月份)"
            r".*?get_scene_trend 一次.*?(?:披露|说明)数据局限",
        )

    def test_sea_state_is_only_for_explicit_realtime_reports(self):
        self.assertRegex(
            self.prompt,
            r"get_sea_state.*?仅用于.*?用户明确要求"
            r".*?当前时点/实时海况.*?报告",
        )
        self.assertRegex(
            self.prompt,
            r"只有用户明确要求.*?当前时点或实时海况"
            r".*?位置可映射.*?才调用 get_sea_state",
        )
        self.assertRegex(
            self.prompt,
            r"仅写当前月份.*?历史月份.*?整月报告"
            r".*?均不得调用 get_sea_state"
            r".*?不得把查询时的实时海况当作同期证据"
            r".*?跳过海况分析.*?说明数据局限",
        )

    def test_report_template_is_complete_and_evidence_based(self):
        self.assertRegex(
            self.prompt,
            r"标题(?:固定)?为\s*[“\"]?SAR 舰船目标检测报告与分析建议[”\"]?",
        )
        self.assertRegex(
            self.prompt,
            r"以下六个章节.*?完整报告模板",
        )

        template_start = self.prompt.index("## 报告结构")
        chapters = (
            "任务与场景概况",
            "检测结果与核心指标",
            "性能分析",
            "海况或环境影响（有可靠数据时）",
            "风险与数据局限",
            "检测结论与改进建议",
        )
        chapter_positions = [
            self.prompt.index(chapter, template_start) for chapter in chapters
        ]
        self.assertEqual(chapter_positions, sorted(chapter_positions))
        self.assertRegex(
            self.prompt,
            r"生成(?:具体)?报告时.{0,80}可按实际数据省略"
            r".{0,40}海况.{0,40}可选章节",
        )
        self.assertRegex(
            self.prompt,
            r"(?:禁止|不得|不要).*?直接粘贴工具原始输出",
        )
        self.assertRegex(
            self.prompt,
            r"结论.*?必须.*?追溯.*?已获取的数据",
        )

    def test_detection_conclusion_is_always_retained(self):
        self.assertRegex(
            self.prompt,
            r"检测结论与改进建议.*?必须始终保留",
        )
        self.assertRegex(
            self.prompt,
            r"缺少可靠数据时.*?检测结论.*?明确"
            r".*?证据不足/无法确认.*?不得编造",
        )


class ResearchPromptContractTests(unittest.TestCase):
    def setUp(self):
        self.prompt = _prompt("research_prompt.txt")

    def test_research_prompt_defines_retrieval_and_evidence_limits(self):
        for requirement in ("分别检索", "证据冲突", "资料缺口", "最多 5 次"):
            self.assertIn(requirement, self.prompt)

    def test_research_prompt_prohibits_private_reasoning_and_fabrication(self):
        for requirement in ("不输出内部推理过程", "不得编造"):
            self.assertIn(requirement, self.prompt)
