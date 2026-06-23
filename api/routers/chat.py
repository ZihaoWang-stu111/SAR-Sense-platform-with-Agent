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
from utils.conversation_builder import build_chat_pack

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Streaming chat endpoint using SSE"""
    try:
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

            # on_step 回调：agent 每产生一个思维链步骤就推到 event_queue，
            # 由主循环 yield 给前端。状态走事件队列，不再用全局 dict。
            def on_step(step):
                event_queue.put(('thought_step', step))

            def run_agent():
                """Run agent in a separate thread"""
                try:
                    for chunk in agent.execute_stream(messages, conversation_id, on_step=on_step):
                        if chunk and chunk.strip():
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
                            data = json.dumps({
                                'type': 'chunk',
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

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
