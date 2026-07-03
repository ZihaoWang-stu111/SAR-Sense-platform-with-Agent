from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import get_weather,get_scene_id,\
    get_current_month,get_user_location,rag_summarize,fetch_external_data,fill_context_for_report,\
    get_sea_state,compare_scenes,get_scene_trend,web_search,detect_ships
from agent.tools.middleware import monitor_tool, report_prompt_switch,log_before_model
from datetime import datetime


class ReactAgent:

    def __init__(self):

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[get_weather, get_scene_id, get_current_month,
                   get_user_location, rag_summarize, fetch_external_data, fill_context_for_report,
                   get_sea_state, compare_scenes, get_scene_trend, web_search, detect_ships],
            middleware=[monitor_tool, report_prompt_switch,log_before_model]

        )

    def execute_stream(self, chat_pack, conversation_id=None, on_step=None):
        """流式产出 agent 回答。on_step(step_dict) 回调在每次产生思维链步骤时被调用，
        调用方自行收集（请求局部变量），agent 不再持有任何全局状态。"""
        # 把"判断 on_step + 调用"封装一层，避免每处 append 都写 if
        def emit_step(step):
            if on_step:
                on_step(step)

        if chat_pack is None:
            chat_pack = []

        input_dict = {
            "messages": chat_pack
        }

        processed_message_count = len(chat_pack)  # 跟踪已处理的消息数量

        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            messages = chunk["messages"]

            # 处理所有新消息（从上次处理位置到当前）
            for i in range(processed_message_count, len(messages)):
                message = messages[i]
                msg_type = getattr(message, 'type', '')

                if msg_type == "ai":
                    tool_calls = getattr(message, 'tool_calls', None)
                    content = message.content.strip() if message.content else ""

                    if tool_calls:
                        if content:
                            emit_step({
                                "step_type": "thinking",
                                "content": content[:300],
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        for tc in tool_calls:
                            emit_step({
                                "step_type": "tool_call",
                                "content": f"调用 {tc.get('name', '')}",
                                "tool_name": tc.get('name', ''),
                                "tool_args": tc.get('args', {}),
                                "tool_call_id": tc.get('id', ''),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                    elif content:
                        emit_step({
                            "step_type": "final_answer",
                            "content": "生成最终回答",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })

                elif msg_type == "tool":
                    result_content = message.content.strip() if message.content else ""
                    truncated = result_content[:200] + "..." if len(result_content) > 200 else result_content
                    emit_step({
                        "step_type": "tool_result",
                        "content": truncated,
                        "tool_name": getattr(message, 'name', ''),
                        "tool_call_id": getattr(message, 'tool_call_id', ''),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

            # 输出所有新的 AI 消息和 ToolMessage
            for i in range(processed_message_count, len(messages)):
                message = messages[i]
                msg_type = getattr(message, 'type', '')

                if msg_type == 'ai':
                    tool_calls = getattr(message, 'tool_calls', None)
                    content = message.content.strip() if message.content else ""
                    # 带 tool_calls 的是"思考+调用工具"，走思维链（上面已 emit_step），不进正文
                    # 不带 tool_calls 的才是最终回答，yield 到正文
                    if content and not tool_calls:
                        yield content + "\n"
                elif msg_type == 'tool':
                    content = message.content.strip() if message.content else ""
                    # 只把 RAG 检索结果（含参考来源）输出到正文，前端要解析参考来源按钮
                    # 其他工具结果（天气、位置等）走思维链，不进正文
                    if content and "参考来源" in content:
                        yield content + "\n"

            processed_message_count = len(messages)

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("mbe-net和sfq-det哪个好，简短回答"):
        print(chunk, end="", flush=True)
