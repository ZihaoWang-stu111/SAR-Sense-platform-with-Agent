from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from agent.research_agent import execute_research
from utils.logger_handler import logger


@tool(
    description=(
        "委派复杂研究任务给 sar-researcher 子智能体进行多轮检索。"
        "必须使用的场景（满足任一即用本工具，不得用 rag_summarize 代替）："
        "(1)比较 3 个及以上算法/论文/模型/技术路线；"
        "(2)跨多篇资料综合归纳；(3)多来源交叉分析；"
        "(4)用户要求深入研究/系统比较/多来源分析。"
        "例：比较 SFQ-Det、MBE-Net、YOLOv8 的特征提取与融合思路。"
        "不得用于：单个概念问答、舰船检测、天气海况、场景数据、报告生成。"
    )
)
def delegate_research(task: str, runtime: ToolRuntime) -> str:
    """委派研究任务给 sar-researcher 子智能体。

    ACL / runtime context 从父 runtime 自动透传，LLM 无法通过 task 参数控制：
    - user_id / role / allowed_doc_ids / client_ip 只从 runtime.context 取，不暴露为 tool 入参；
    - 子 Agent 调 RAG 时仍拿到同一份 allowed_doc_ids，不绕过 ACL。

    两条数据通道（V3）：
    - 子 Agent RAG 来源 -> _subagent_rag_results collector -> 父 rag_results（参考来源面板）；
    - 返回值只含研究总结 -> 父 ToolMessage（content 不含来源材料，保持上下文隔离）。

    子 Agent 内部执行步骤不展示给前端（市面产品惯例），用户只看到本工具的
    tool_call（"正在进行深度研究..."）与 tool_result（"深度研究完成"）。

    调用 delegate_research 时，task 必须是自包含研究任务（研究对象 + 比较维度 + 用户要求），
    不得只传"帮我研究一下"之类模糊内容。
    """
    context = runtime.context

    # 单轮防重复委派（工程约束，不只靠 Prompt）：
    # context 是请求级（每请求新建 runtime_context），同一 turn 内第二次调用直接返回，
    # 下次用户请求重新生成 context，flag 自动重置。
    if context.get("_research_delegated"):
        return "本轮已完成一次深度研究，请基于已有研究结果继续回答，不得再次委派。"
    context["_research_delegated"] = True

    rag_results = context.get("_subagent_rag_results")

    try:
        return execute_research(
            task=task,
            user_context={
                "user_id": context.get("user_id"),
                "role": context.get("role"),
                "allowed_doc_ids": context.get("allowed_doc_ids"),
                "client_ip": context.get("client_ip"),
            },
            rag_results=rag_results,
        )
    except Exception:
        logger.error("delegate_research 执行失败", exc_info=True)
        # 不泄露 traceback，主 Agent 据 prompt fallback 或说明能力暂不可用。
        return "深度研究子智能体暂时执行失败，请基于现有信息继续回答。"
