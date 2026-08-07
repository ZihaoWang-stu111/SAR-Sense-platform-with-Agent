from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import get_weather,get_scene_id,\
    get_current_month,get_user_location,rag_summarize,fetch_external_data,fill_context_for_report,\
    get_sea_state,compare_scenes,get_scene_trend,web_search,detect_ships
from agent.tools.middleware import monitor_tool, report_prompt_switch,log_before_model
from agent.tools.delegation_tools import delegate_research
from datetime import datetime
import re


class ReactAgent:

    def __init__(self):

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[get_weather, get_scene_id, get_current_month,
                   get_user_location, rag_summarize, fetch_external_data, fill_context_for_report,
                   get_sea_state, compare_scenes, get_scene_trend, web_search, detect_ships,
                   delegate_research],
            middleware=[monitor_tool, report_prompt_switch,log_before_model]

        )

    def execute_stream(self, chat_pack, conversation_id=None, user_context=None, on_step=None):
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

        runtime_context = {"report": False}
        if user_context:
            runtime_context.update(user_context)

        # 子 Agent 桥接（V3）：请求级 RAG 来源 collector。
        # subagent_rag_results 是当前请求级 list 引用，delegate_research 内部 append，
        # 本循环在 chunk 之间轮询并 yield，复用现有 rag_result 通道（chat.py 零改动）。
        subagent_rag_results = []
        runtime_context["_subagent_rag_results"] = subagent_rag_results
        emitted_rag_count = 0

        for chunk in self.agent.stream(input_dict, stream_mode="values", context=runtime_context):
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
                    # detect_ships 工具画完检测图存 upload_store 后，在返回字符串末尾附 [viz:<upload_id>] marker。
                    # 这里抽出来发专门的 detect_image step，由前端拉图渲染成回答下方卡片（不进思维链）；
                    # 同时从 tool_result 展示内容里剥掉 marker，保持思维链干净。LLM 仍会在 observation
                    # 里看到原始 marker，但那只是个短 id（非 base64），不影响推理。
                    viz_match = re.search(r'\[viz:(img_\w+)\]', result_content)
                    if viz_match:
                        viz_upload_id = viz_match.group(1)
                        result_content = (result_content[:viz_match.start()] + result_content[viz_match.end():]).rstrip()
                        emit_step({
                            "step_type": "detect_image",
                            "upload_id": viz_upload_id,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                    tool_name = getattr(message, 'name', '')
                    if tool_name == "delegate_research":
                        # 研究总结不进思考链，避免与最终回答重复、thought_steps 膨胀；
                        # LLM 仍在 ToolMessage 里看到完整研究结论，这里只改前端展示文本。
                        display_content = "深度研究完成，正在整理研究结论"
                    else:
                        display_content = result_content[:200] + "..." if len(result_content) > 200 else result_content
                    emit_step({
                        "step_type": "tool_result",
                        "content": display_content,
                        "tool_name": tool_name,
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
                    # RAG 检索结果单独作为结构化事件输出，前端用于渲染参考来源面板，不混进最终回答正文。
                    # 其他工具结果（天气、位置等）走思维链，不进正文。
                    # 加 tool_name == "rag_summarize" 守卫：避免 delegate_research 研究总结里
                    # 出现"参考来源"字样被误判（子 Agent 来源走 collector，不进这里）。
                    tool_name = getattr(message, 'name', '')
                    if (
                        tool_name == "rag_summarize"
                        and content
                        and "参考来源" in content
                    ):
                        yield {"type": "rag_result", "content": content}

            processed_message_count = len(messages)

            # V3: 轮询子 Agent RAG 来源 collector，复用现有 rag_result 通道。
            # delegate_research 执行期间子 Agent 的 RAG 来源 append 进 collector，
            # 在 tool 完成后的 chunk 一次性 yield 给 chat.py 的 rag_results。
            while emitted_rag_count < len(subagent_rag_results):
                yield {
                    "type": "rag_result",
                    "content": subagent_rag_results[emitted_rag_count],
                }
                emitted_rag_count += 1

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("mbe-net和sfq-det哪个好，简短回答"):
        print(chunk, end="", flush=True)
