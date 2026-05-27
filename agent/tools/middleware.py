import time
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts,load_system_prompts
from agent.metrics_collector import AgentMetrics


@wrap_tool_call
def monitor_tool(request, handler):
    metrics = AgentMetrics()
    tool_name = request.tool_call["name"]
    t_start = time.time()

    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        duration_ms = (time.time() - t_start) * 1000
        logger.info(f"{request.tool_call['name']}工具调用成功")
        metrics.record_tool_call(tool_name, True, duration_ms)
        if request.tool_call['name'] == "fetch_external_data":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        duration_ms = (time.time() - t_start) * 1000
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        metrics.record_tool_call(tool_name, False, duration_ms)
        raise e

@before_model
def log_before_model(state,runtime):
    metrics = AgentMetrics()
    metrics.record_llm_call()
    logger.info(f"[log_before_model]包含：{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__}| {state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request):

    if request.runtime.context.get("report",False):
        return load_report_prompts()
    return load_system_prompts()

