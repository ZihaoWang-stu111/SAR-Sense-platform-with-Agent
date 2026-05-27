from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import get_weather,get_scene_id,\
    get_current_month,get_user_location,rag_summarize,fetch_external_data,fill_context_for_report,\
    get_sea_state,compare_scenes,get_scene_trend,web_search,detect_ships,extract_file_content
from agent.tools.middleware import monitor_tool, report_prompt_switch,log_before_model
from agent.metrics_collector import AgentMetrics
from datetime import datetime

_thought_chain = {"steps": []}

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

    def execute_stream(self, chat_pack):
        _thought_chain["steps"] = []

        if chat_pack is None:
            chat_pack = []

        metrics = AgentMetrics()
        metrics.start_conversation()

        input_dict = {
            "messages": chat_pack
        }

        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message:
                msg_type = getattr(latest_message, 'type', '')

                if msg_type == "ai":
                    tool_calls = getattr(latest_message, 'tool_calls', None)
                    content = latest_message.content.strip() if latest_message.content else ""

                    if tool_calls:
                        if content:
                            _thought_chain["steps"].append({
                                "step_type": "thinking",
                                "content": content[:300],
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        for tc in tool_calls:
                            _thought_chain["steps"].append({
                                "step_type": "tool_call",
                                "content": f"调用 {tc.get('name', '')}",
                                "tool_name": tc.get('name', ''),
                                "tool_args": tc.get('args', {}),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                    elif content:
                        _thought_chain["steps"].append({
                            "step_type": "final_answer",
                            "content": "生成最终回答",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })

                elif msg_type == "tool":
                    result_content = latest_message.content.strip() if latest_message.content else ""
                    truncated = result_content[:200] + "..." if len(result_content) > 200 else result_content
                    _thought_chain["steps"].append({
                        "step_type": "tool_result",
                        "content": truncated,
                        "tool_name": getattr(latest_message, 'name', ''),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

                yield latest_message.content.strip() + "\n"

        metrics.end_conversation()

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("檀香山今天天气如何，海况适不适合检测任务"):
        print(chunk, end="", flush=True)
