"""对话路由：5 端点，全部 async + Depends(get_db, get_current_user)。"""
import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.dependencies import get_conv_manager
from config.db_conf import get_db
from schemas.conversations import AppendMessageRequest, CreateConversationRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["conversations"])


@router.get("")
async def list_conversations(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        convs = await get_conv_manager().list_conversations(db, user["id"])
        return {"success": True, "conversations": convs}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_conversation(
    req: CreateConversationRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        conv_id = await get_conv_manager().create_conversation(db, user["id"], req.message or "")
        return {"success": True, "conversation_id": conv_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conv_id}")
async def load_conversation(
    conv_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        conv = await get_conv_manager().load_conversation(db, conv_id, user["id"])
        return {"success": True, "conversation": conv}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await get_conv_manager().delete_conversation(db, conv_id, user["id"])
        return {"success": True}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conv_id}/messages")
async def append_message(
    conv_id: str,
    req: AppendMessageRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await get_conv_manager().append_message(
            db, conv_id, user["id"], req.role, req.content,
            thought_steps=req.thought_steps,
        )
        return {"success": True}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
