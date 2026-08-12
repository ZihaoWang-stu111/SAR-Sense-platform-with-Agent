import time

from langchain.agents.middleware import before_model, dynamic_prompt, wrap_tool_call
from langchain_core.messages.utils import count_tokens_approximately

from agent.metrics_collector import AgentMetrics
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


def _runtime_user_id(runtime) -> int:
    context = getattr(runtime, "context", None)
    if context is None:
        return 1
    if hasattr(context, "get"):
        return context.get("user_id", 1)
    return getattr(context, "user_id", 1)


@wrap_tool_call
def monitor_tool(request, handler):
    metrics = AgentMetrics()
    tool_name = request.tool_call["name"]
    user_id = _runtime_user_id(request.runtime)
    started_at = time.time()

    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        duration_ms = (time.time() - started_at) * 1000
        logger.info(f"{request.tool_call['name']}工具调用成功")
        metrics.record_tool_call(tool_name, True, duration_ms, user_id=user_id)
        if tool_name == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as exc:
        duration_ms = (time.time() - started_at) * 1000
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(exc)}")
        metrics.record_tool_call(tool_name, False, duration_ms, user_id=user_id)
        raise


@before_model
def log_before_model(state, runtime):
    metrics = AgentMetrics()
    metrics.record_llm_call(user_id=_runtime_user_id(runtime))
    messages = state["messages"]
    approximate_tokens = count_tokens_approximately(messages)
    logger.info(
        f"[log_before_model]包含：{len(messages)}条消息，约{approximate_tokens} tokens"
    )
    logger.debug(
        f"[log_before_model]{type(state['messages'][-1]).__name__}| "
        f"{state['messages'][-1].content.strip()}"
    )
    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request):
    if request.runtime.context.get("report", False):
        return load_report_prompts()
    return load_system_prompts()
