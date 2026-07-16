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
        """SSE generator：只把 agent 事件推给前端（消费视图）。

        业务任务（跑 agent + 持久化）在独立子线程里，不绑 SSE client 生命周期——
        client 断开（切页面/刷新/断网）时 generate 退出，子线程继续跑完并存完整回答。
        SSE 退化为"可选的进度订阅视图"，符合"业务任务与传输层解耦"。
        """
        import asyncio
        import threading

        from config.db_conf import AsyncSessionLocal

        loop = asyncio.get_running_loop()
        # asyncio.Queue：SSE 侧 await get()，agent 子线程用 call_soon_threadsafe 跨线程 put。
        sse_queue: asyncio.Queue = asyncio.Queue()

        # 累积器在子线程写、persist 读；SSE 只读 sse_queue，不碰这些。
        full_content = ""
        rag_results: list = []
        thought_steps_list: list = []

        def on_step(step):
            thought_steps_list.append(step)
            loop.call_soon_threadsafe(sse_queue.put_nowait, ('thought_step', step))

        async def persist():
            """独立 session 存库——不依赖 request 的 db（client 断后 request db 已关）。

            由子线程 finally 经 run_coroutine_threadsafe 调度到 event loop 执行；
            event loop 长期运行，与 client 是否在线无关。
            """
            if not conversation_id or not (full_content.strip() or rag_results):
                return
            try:
                async with AsyncSessionLocal() as session:
                    await conv_crud.append_message(
                        session, conversation_id, user["id"], "assistant", full_content,
                        thought_steps=thought_steps_list or None,
                        rag_results=rag_results or None,
                    )
                    await session.commit()
                logger.info(f"assistant 已落库（conv={conversation_id}, len={len(full_content)}）")
            except Exception as e:
                logger.warning(f"存 assistant 消息失败: {e}")

        def run_agent():
            """子线程：跑 agent + 累积 + 推 SSE + 调度存库。client 断了也跑完。"""
            nonlocal full_content
            metrics = get_metrics()
            started_at = metrics.start_conversation()
            try:
                for chunk in agent.execute_stream(
                    messages, conversation_id, user_context=user_context, on_step=on_step
                ):
                    if isinstance(chunk, dict) and chunk.get("type") == "rag_result":
                        content = chunk.get("content", "")
                        if content:
                            rag_results.append(content)
                        loop.call_soon_threadsafe(sse_queue.put_nowait, ('rag_result', content))
                    elif isinstance(chunk, str) and chunk.strip():
                        full_content += chunk
                        loop.call_soon_threadsafe(sse_queue.put_nowait, ('chunk', chunk))
            except Exception as e:
                loop.call_soon_threadsafe(sse_queue.put_nowait, ('error', str(e)))
            finally:
                loop.call_soon_threadsafe(sse_queue.put_nowait, ('done', None))
                metrics.end_conversation(started_at, user_id=user["id"])
                # 调度存库到 event loop——即使 SSE client 已断，agent 结果仍完整落库
                asyncio.run_coroutine_threadsafe(persist(), loop)

        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()

        try:
            logger.info("Starting agent stream...")
            while True:
                event_type, event_data = await sse_queue.get()
                if event_type == 'done':
                    break
                if event_type == 'thought_step':
                    data = json.dumps({'step': event_data}, ensure_ascii=False)
                    yield f"event: thought_step\ndata: {data}\n\n"
                elif event_type == 'chunk':
                    data = json.dumps({'content': event_data}, ensure_ascii=False)
                    yield f"event: chunk\ndata: {data}\n\n"
                elif event_type == 'rag_result':
                    data = json.dumps({'content': event_data}, ensure_ascii=False)
                    yield f"event: rag_result\ndata: {data}\n\n"
                elif event_type == 'error':
                    data = json.dumps({'message': event_data}, ensure_ascii=False)
                    yield f"event: error\ndata: {data}\n\n"
            yield f"event: done\ndata: {json.dumps({})}\n\n"
        except asyncio.CancelledError:
            # client 断开：generate 退出，agent 子线程继续跑完存库，回来 loadConversation 能看到完整回答
            logger.info(f"SSE client 断开（conv={conversation_id}），agent 后台继续")
            raise  # 传播取消（规范）；子线程独立，不受影响
        except Exception as e:
            traceback.print_exc()
            data = json.dumps({'message': str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {data}\n\n"
        # 不在 generate 里存库——存库由子线程 persist 调度，不绑 client 生命周期

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )
