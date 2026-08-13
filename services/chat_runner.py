"""聊天业务编排：组上下文、跑 agent、落库。

传输层（SSE）只负责把 event_queue 里的事件推给客户端；
断线不影响本模块已提交的执行与 persist。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_agent, get_agent_executor, get_metrics
from config.db_conf import AsyncSessionLocal
from crud import conversations as conv_crud
from crud.knowledge_acl import get_allowed_doc_ids
from utils.conversation_builder import build_chat_pack, fit_messages_to_budget

logger = logging.getLogger(__name__)


def _try_load_memory_context(user_id: int, query: str) -> str:
    try:
        from services.memory_service import load_memory_context

        return load_memory_context(user_id, query) or ""
    except Exception as e:
        logger.warning(f"[memory] 读取失败，本轮不注入: {e}")
        return ""


def _try_process_memory_turn(**kwargs) -> None:
    try:
        from services.memory_service import process_memory_turn

        process_memory_turn(**kwargs)
    except Exception as e:
        logger.warning(f"[memory] 更新失败: {e}")


async def prepare_chat(
    *,
    db: AsyncSession,
    user: dict,
    message: str,
    display_message: str,
    conversation_id: str | None,
    history_messages: list,
    client_ip: str | None,
) -> tuple[list, dict]:
    """写 user（若有会话）、组 chat_pack / 历史，返回 (messages, user_context)。"""
    allowed_doc_ids = await get_allowed_doc_ids(db, user.get("role", "guest"))
    user_context = {
        "user_id": user["id"],
        "role": user.get("role", "guest"),
        "allowed_doc_ids": None if allowed_doc_ids is None else list(allowed_doc_ids),
        "client_ip": client_ip,
    }

    if conversation_id:
        await conv_crud.append_message(db, conversation_id, user["id"], "user", display_message)
        messages = await build_chat_pack(db, conversation_id, user["id"])
        # 先 commit：user + summary 落库；agent 在线程内跑，不再占用请求 db
        await db.commit()
        if messages and messages[-1].get("role") == "user":
            messages[-1] = {"role": "user", "content": message}
    else:
        history = [m.model_dump() if hasattr(m, "model_dump") else m for m in history_messages]
        messages = history + [{"role": "user", "content": message}]

    messages = fit_messages_to_budget(messages)
    return messages, user_context


def submit_chat_run(
    *,
    messages: list,
    memory_user_message: str,
    conversation_id: str | None,
    user_context: dict,
    user_id: int,
    event_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """提交到 AgentExecutor：跑 agent、往 queue 推事件、结束后独立 session 落库。"""

    executor = get_agent_executor()
    full_content = ""
    rag_results: list = []
    thought_steps_list: list = []
    run_success = False

    def emit(event_type: str, event_data: Any = None) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, (event_type, event_data))

    def on_step(step):
        thought_steps_list.append(step)
        emit("thought_step", step)

    async def persist() -> None:
        persisted = False
        if conversation_id and (full_content.strip() or rag_results):
            try:
                async with AsyncSessionLocal() as session:
                    await conv_crud.append_message(
                        session,
                        conversation_id,
                        user_id,
                        "assistant",
                        full_content,
                        thought_steps=thought_steps_list or None,
                        rag_results=rag_results or None,
                    )
                    await session.commit()
                persisted = True
                logger.info(f"assistant 已落库（conv={conversation_id}, len={len(full_content)}）")
            except Exception as e:
                logger.warning(f"存 assistant 消息失败: {e}")

        if persisted and run_success and memory_user_message.strip() and full_content.strip():
            executor.submit(
                _try_process_memory_turn,
                user_id=user_id,
                user_message=memory_user_message,
                assistant_content=full_content,
                conversation_id=conversation_id,
            )

    def run_agent() -> None:
        nonlocal full_content, run_success
        emit("status", "正在思考...")
        metrics = get_metrics()
        started_at = metrics.start_conversation()
        agent = get_agent()
        try:
            memory_context = _try_load_memory_context(user_id, memory_user_message)
            if memory_context:
                user_context["memory_context"] = memory_context
            for chunk in agent.execute_stream(
                messages, conversation_id, user_context=user_context, on_step=on_step
            ):
                if isinstance(chunk, dict) and chunk.get("type") == "rag_result":
                    content = chunk.get("content", "")
                    if content:
                        rag_results.append(content)
                    emit("rag_result", content)
                elif isinstance(chunk, str) and chunk.strip():
                    full_content += chunk
                    emit("chunk", chunk)
            run_success = True
        except Exception as e:
            emit("error", str(e))
        finally:
            emit("done", None)
            metrics.end_conversation(started_at, user_id=user_id)
            asyncio.run_coroutine_threadsafe(persist(), loop)

    executor.submit(run_agent)
