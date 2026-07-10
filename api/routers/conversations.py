"""对话路由：4 端点，全部 async + Depends(get_db, get_current_user)。直接调 crud。
异常由全局处理器兜底，路由不再写 try/except。"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from config.db_conf import get_db
from crud import conversations as conv_crud
from schemas.conversations import CreateConversationRequest

router = APIRouter(tags=["conversations"])


@router.get("")
async def list_conversations(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    convs = await conv_crud.list_conversations(db, user["id"])
    return {"success": True, "conversations": convs}


@router.post("")
async def create_conversation(
    req: CreateConversationRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conv_id = await conv_crud.create_conversation(db, user["id"], req.message or "")
    return {"success": True, "conversation_id": conv_id}


@router.get("/{conv_id}")
async def load_conversation(
    conv_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conv = await conv_crud.load_conversation(db, conv_id, user["id"])
    return {"success": True, "conversation": conv}


@router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await conv_crud.delete_conversation(db, conv_id, user["id"])
    return {"success": True}
