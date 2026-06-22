import json
import logging
import traceback

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_agent, get_conv_manager, get_metrics
from api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat/stream")
async def chat_stream(request: Request, user: dict = Depends(get_current_user)):
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

        from agent.react_agent import _thought_chains

        logger.info("Loading agent for streaming...")
        agent = get_agent()
        logger.info("Agent loaded for streaming")

        conv_mgr = get_conv_manager()
        if conversation_id:
            conv_mgr.append_message(conversation_id, "user", display_message, user_id=user["id"])
            messages = conv_mgr.build_chat_pack(conversation_id, user_id=user["id"])
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

            def run_agent():
                """Run agent in a separate thread"""
                try:
                    for chunk in agent.execute_stream(messages, conversation_id):
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
                last_step_count = 0

                while True:
                    if conversation_id and conversation_id in _thought_chains:
                        current_steps = _thought_chains[conversation_id]["steps"]
                    else:
                        current_steps = []

                    if len(current_steps) > last_step_count:
                        new_steps = current_steps[last_step_count:]
                        for step in new_steps:
                            data = json.dumps({
                                'type': 'thought_step',
                                'step': step
                            }, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                        last_step_count = len(current_steps)

                    try:
                        event_type, event_data = event_queue.get_nowait()

                        if event_type == 'chunk':
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

                if conversation_id and conversation_id in _thought_chains:
                    current_steps = _thought_chains[conversation_id]["steps"]
                    if len(current_steps) > last_step_count:
                        new_steps = current_steps[last_step_count:]
                        for step in new_steps:
                            data = json.dumps({
                                'type': 'thought_step',
                                'step': step
                            }, ensure_ascii=False)
                            yield f"data: {data}\n\n"

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

                if conversation_id and conversation_id in _thought_chains:
                    del _thought_chains[conversation_id]

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
