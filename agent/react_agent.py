from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import get_weather,get_scene_id,\
    get_current_month,get_user_location,rag_summarize,fetch_external_data,fill_context_for_report,\
    get_sea_state,compare_scenes,get_scene_trend,web_search,detect_ships,extract_file_content
from agent.tools.middleware import monitor_tool, report_prompt_switch,log_before_model
from agent.metrics_collector import AgentMetrics
from datetime import datetime

_thought_chains = {}  # conversation_id -> {"steps": [...]}

class ReactAgent:

    def __init__(self):

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[get_weather, get_scene_id, get_current_month,
                   get_user_location, rag_summarize, fetch_external_data, fill_context_for_report,
                   get_sea_state, compare_scenes, get_scene_trend, web_search, detect_ships, extract_file_content],
            middleware=[monitor_tool, report_prompt_switch,log_before_model]

        )

    def execute_stream(self, chat_pack, conversation_id=None):
        # 为这个会话创建独立的思考链
        if conversation_id:
            _thought_chains[conversation_id] = {"steps": []}
            current_chain = _thought_chains[conversation_id]
        else:
            # 兼容没有 ID 的调用（如命令行测试）
            current_chain = {"steps": []}

        if chat_pack is None:
            chat_pack = []

        metrics = AgentMetrics()
        metrics.start_conversation()

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
                            current_chain["steps"].append({
                                "step_type": "thinking",
                                "content": content[:300],
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        for tc in tool_calls:
                            current_chain["steps"].append({
                                "step_type": "tool_call",
                                "content": f"调用 {tc.get('name', '')}",
                                "tool_name": tc.get('name', ''),
                                "tool_args": tc.get('args', {}),
                                "tool_call_id": tc.get('id', ''),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                    elif content:
                        current_chain["steps"].append({
                            "step_type": "final_answer",
                            "content": "生成最终回答",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })

                elif msg_type == "tool":
                    result_content = message.content.strip() if message.content else ""
                    truncated = result_content[:200] + "..." if len(result_content) > 200 else result_content
                    current_chain["steps"].append({
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
                    content = message.content.strip()
                    if content:
                        yield content + "\n"
                elif msg_type == 'tool':
                    # 输出工具结果（包含参考来源），供前端溯源使用
                    content = message.content.strip()
                    if content:
                        yield content + "\n"

            processed_message_count = len(messages)

        metrics.end_conversation()

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("mbe-net和sfq-det哪个好，简短回答"):
        print(chunk, end="", flush=True)
