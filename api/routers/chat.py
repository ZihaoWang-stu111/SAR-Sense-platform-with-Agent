"""聊天 SSE 传输适配：鉴权、限流、把 ChatRunner 事件编成 SSE 帧。"""

import asyncio
import json
import logging
import traceback

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from config.db_conf import get_db
from schemas.conversations import ChatStreamRequest
from services.chat_runner import prepare_chat, submit_chat_run
from utils.traffic_control import rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _client_ip(request: Request) -> str | None:
    """取真实客户端 IP：反代后从 X-Forwarded-For 最左取，直连取 request.client.host。"""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        ip = xff.split(",")[0].strip()
        if ip and ip.lower() != "unknown":
            return ip
    return request.client.host if request.client else None


def _sse_line(event_type: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Streaming chat endpoint using SSE。"""
    await rate_limit(f"user:{user['id']}:chat", 6, 60)
    logger.info("Streaming chat request received")

    message = payload.message
    display_message = payload.display_message or payload.message
    conversation_id = payload.conversation_id
    logger.info(f"Processing streaming message: {message[:50]}...")

    messages, user_context = await prepare_chat(
        db=db,
        user=user,
        message=message,
        display_message=display_message,
        conversation_id=conversation_id,
        history_messages=payload.messages,
        client_ip=_client_ip(request),
    )

    async def generate():
        """只订阅事件并写成 SSE；agent + 落库在 ChatRunner，不绑 client 生命周期。"""
        loop = asyncio.get_running_loop()
        sse_queue: asyncio.Queue = asyncio.Queue()
        submit_chat_run(
            messages=messages,
            conversation_id=conversation_id,
            user_context=user_context,
            user_id=user["id"],
            event_queue=sse_queue,
            loop=loop,
        )

        try:
            logger.info("Starting agent stream...")
            while True:
                event_type, event_data = await sse_queue.get()
                if event_type == "done":
                    break
                if event_type == "status":
                    yield _sse_line("status", {"content": event_data})
                elif event_type == "thought_step":
                    yield _sse_line("thought_step", {"step": event_data})
                elif event_type == "chunk":
                    yield _sse_line("chunk", {"content": event_data})
                elif event_type == "rag_result":
                    yield _sse_line("rag_result", {"content": event_data})
                elif event_type == "error":
                    yield _sse_line("error", {"message": event_data})
            yield _sse_line("done", {})
        except asyncio.CancelledError:
            logger.info(f"SSE client 断开（conv={conversation_id}），agent 后台继续")
            raise
        except Exception as e:
            traceback.print_exc()
            yield _sse_line("error", {"message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
