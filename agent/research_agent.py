from __future__ import annotations

import threading
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

from model.factory import chat_model
from agent.tools.agent_tools import rag_summarize, web_search, get_current_month
from agent.tools.middleware import monitor_tool, log_before_model
from utils.prompt_loader import load_research_prompts
from utils.config_handler import agent_conf


# 子 Agent 运行限制（从 config/agent.yml 读取，提供硬约束兜底）：
# - max_tool_calls：单次研究工具调用上限（ToolCallLimitMiddleware run_limit）
# - recursion_limit：LangGraph 图步数上限（防 4b 模型循环不收敛）
_research_conf = (agent_conf.get("research_agent") or {})
_RESEARCH_MAX_TOOL_CALLS = _research_conf.get("max_tool_calls", 6)
_RESEARCH_RECURSION_LIMIT = _research_conf.get("recursion_limit", 20)


# Research Agent 工具白名单：仅检索 / 联网 / 时间。
# 严禁 detect_ships / fill_context_for_report / 数据查询类工具（与研究职责无关）；
# 严禁 delegate_research（防递归委派与 Agent loop 扩散）。
RESEARCH_TOOLS = [rag_summarize, web_search, get_current_month]


# 线程安全 lazy singleton：复用 chat_model / RAG service singleton，
# 不保存任何请求级状态（user_id / allowed_doc_ids / conversation_id 等全部走 invoke input + runtime context）。
_research_agent = None
_research_agent_lock = threading.Lock()


def get_research_agent():
    """返回 sar-researcher 单例。第一次调用时构建 create_agent graph。"""
    global _research_agent

    if _research_agent is None:
        with _research_agent_lock:
            if _research_agent is None:
                _research_agent = create_agent(
                    model=chat_model,
                    system_prompt=load_research_prompts(),
                    tools=RESEARCH_TOOLS,
                    middleware=[
                        # 硬约束：单次研究最多 N 次工具调用，超过则阻止该工具、
                        # 让模型基于已有证据总结（exit_behavior=continue 不抛异常）
                        ToolCallLimitMiddleware(
                            run_limit=_RESEARCH_MAX_TOOL_CALLS,
                            exit_behavior="continue",
                        ),
                        monitor_tool,
                        log_before_model,
                    ],
                    name="sar-researcher",
                )

    return _research_agent


def _extract_final_ai_text(messages) -> str:
    """从 Research Agent 的最终 messages 中提取最后一条非空 AIMessage 文本。

    没有 AIMessage 或均为空时返回安全 fallback，不抛异常。
    """
    for message in reversed(messages or []):
        if getattr(message, "type", "") == "ai":
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return "研究子智能体未返回有效结论。"


def execute_research(
    task: str,
    *,
    user_context: dict,
    rag_results: Optional[list] = None,
) -> str:
    """运行 Research Agent。

    子 Agent 内部步骤不展示给前端（市面 Deep Research 产品惯例）：
    - tool_call/tool_result 不 emit，thought_steps 不含子 Agent 步骤；
      用户只看到主 Agent 的 delegate_research（"正在进行深度研究..." -> "深度研究完成"）。
    - RAG 来源（rag_summarize 返回含"参考来源"）走 rag_results collector，
      经主 Agent execute_stream 轮询进父 rag_results（参考来源面板）。
    - 研究总结作为返回值（父 ToolMessage，content 仅总结，保持上下文隔离）。

    ACL：user_context（含 allowed_doc_ids）透传给子 Agent runtime context，
    子 Agent 调 RAG 时仍执行文件级 ACL，不绕过。
    """
    research_agent = get_research_agent()
    input_dict = {"messages": [{"role": "user", "content": task}]}
    processed_message_count = 0
    final_messages: list = []

    for chunk in research_agent.stream(
        input_dict,
        stream_mode="values",
        context=user_context,
        config={"recursion_limit": _RESEARCH_RECURSION_LIMIT},
    ):
        messages = chunk.get("messages", [])
        if messages:
            final_messages = messages

        for i in range(processed_message_count, len(messages)):
            message = messages[i]
            # 只收集 RAG 来源到 collector；子 Agent 内部 tool_call/tool_result
            # 不 emit（不进 thought_steps），保持前端思维链干净。
            if getattr(message, "type", "") == "tool":
                tool_name = getattr(message, "name", "")
                content = message.content.strip() if message.content else ""
                if (
                    tool_name == "rag_summarize"
                    and "参考来源" in content
                    and rag_results is not None
                ):
                    rag_results.append(content)

        processed_message_count = len(messages)

    return _extract_final_ai_text(final_messages)
