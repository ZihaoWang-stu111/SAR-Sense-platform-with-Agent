import json
import logging
import traceback

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_agent, get_metrics
from api.auth import get_current_user
from config.db_conf import get_db
from crud import conversations as conv_crud
from crud.knowledge_acl import get_allowed_doc_ids
from utils.conversation_builder import build_chat_pack
from utils.traffic_control import rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _client_ip(request: Request) -> str | None:
    """取真实客户端 IP：反代后从 X-Forwarded-For 最左取，直连取 request.client.host。

    cpolar/nginx 转发时 XFF=「用户真实IP, 节点IP」，最左是用户 IP。
    本地直连无 XFF，request.client.host 是 127.0.0.1/内网 IP（工具会 fallback 查服务器 IP）。
    """
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        ip = xff.split(",")[0].strip()
        if ip and ip.lower() != "unknown":
            return ip
    return request.client.host if request.client else None


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Streaming chat endpoint using SSE"""
    await rate_limit(f"user:{user['id']}:chat", 6, 60)
    logger.info("Streaming chat request received")
    data = await request.json()
    message = data.get('message', '')
    display_message = data.get('display_message', message)
    messages_history = data.get('messages', [])
    conversation_id = data.get('conversation_id')

    if not message:
        raise HTTPException(status_code=400, detail='No message provided')

    logger.info(f"Processing streaming message: {message[:50]}...")

    logger.info("Loading agent for streaming...")
    agent = get_agent()
    logger.info("Agent loaded for streaming")
    allowed_doc_ids = await get_allowed_doc_ids(db, user.get("role", "guest"))
    user_context = {
        "user_id": user["id"],
        "role": user.get("role", "guest"),
        "allowed_doc_ids": None if allowed_doc_ids is None else list(allowed_doc_ids),
        # 客户端 IP（get_user_location 工具用）：经反代从 X-Forwarded-For 最左取真实用户 IP，
        # 直连取 request.client.host；localhost/内网 IP 时工具会 fallback 查服务器出口 IP
        "client_ip": _client_ip(request),
    }

    if conversation_id:
        await conv_crud.append_message(db, conversation_id, user["id"], "user", display_message)
        messages = await build_chat_pack(db, conversation_id, user["id"])
        # 路由级别先 commit 一次，让 user 消息 + chat_pack 用到的 summary 更新立刻落库；
        # 后续 agent 在线程内跑、不再用 db，避免长时间持有连接。
        await db.commit()
        if messages and messages[-1].get("role") == "user":
            messages[-1] = {"role": "user", "content": message}
    else:
        messages = messages_history[-10:] + [{"role": "user", "content": message}]

    async def generate():
        """SSE generator for streaming response"""
        import asyncio
        import threading
        from queue import Queue

        event_queue = Queue()

        metrics = get_metrics()
        metrics.start_conversation()

        # 累积 assistant 回答 + 思维链，finally 存库
        # （后端存，不依赖前端 streamDone；前端断开/切页面也存，回来不空）
        full_content = ""
        rag_results = []
        thought_steps_list = []

        # on_step 回调：agent 每产生一个思维链步骤就推到 event_queue + 累积，
        # 由主循环 yield 给前端。状态走事件队列，不再用全局 dict。
        def on_step(step):
            thought_steps_list.append(step)
            event_queue.put(('thought_step', step))

        def run_agent():
            """Run agent in a separate thread"""
            try:
                for chunk in agent.execute_stream(messages, conversation_id, user_context=user_context, on_step=on_step):
                    if isinstance(chunk, dict) and chunk.get("type") == "rag_result":
                        event_queue.put(('rag_result', chunk.get("content", "")))
                    elif isinstance(chunk, str) and chunk.strip():
                        event_queue.put(('chunk', chunk))
            except Exception as e:
                event_queue.put(('error', str(e)))
            finally:
                event_queue.put(('done', None))

        agent_thread = threading.Thread(target=run_agent)
        agent_thread.start()

        try:
            logger.info("Starting agent stream...")

            while True:
                try:
                    event_type, event_data = event_queue.get_nowait()

                    if event_type == 'thought_step':
                        data = json.dumps({
                            'type': 'thought_step',
                            'step': event_data
                        }, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    elif event_type == 'chunk':
                        full_content += event_data   # 累积，finally 存库
                        data = json.dumps({
                            'type': 'chunk',
                            'content': event_data
                        }, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    elif event_type == 'rag_result':
                        if event_data:
                            rag_results.append(event_data)
                        data = json.dumps({
                            'type': 'rag_result',
                            'content': event_data
                        }, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    elif event_type == 'error':
                        data = json.dumps({
                            'type': 'error',
                            'message': event_data
                        }, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    elif event_type == 'done':
                        break
                except Exception:
                    pass

                await asyncio.sleep(0.05)

            data = json.dumps({'type': 'done'})
            yield f"data: {data}\n\n"

        except Exception as e:
            traceback.print_exc()
            data = json.dumps({
                'type': 'error',
                'message': str(e)
            }, ensure_ascii=False)
            yield f"data: {data}\n\n"

        finally:
            metrics.end_conversation()
            # 后端存 assistant（前端断开/切页面也存，回来不空）
            if conversation_id and (full_content.strip() or rag_results):
                try:
                    await conv_crud.append_message(
                        db, conversation_id, user["id"], "assistant", full_content,
                        thought_steps=thought_steps_list or None,
                        rag_results=rag_results or None,
                    )
                    await db.commit()
                except Exception as e:
                    logger.warning(f"存 assistant 消息失败: {e}")
            logger.info("Streaming response completed")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )
