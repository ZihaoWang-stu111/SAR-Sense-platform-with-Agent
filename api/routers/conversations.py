import logging
import traceback

from fastapi import APIRouter, Request, HTTPException, Depends

from api.dependencies import get_conv_manager
from api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["conversations"])


@router.get("")
async def list_conversations(user: dict = Depends(get_current_user)):
    """List all conversations of the current user"""
    try:
        conv_manager = get_conv_manager()
        conversations = conv_manager.list_conversations(user_id=user["id"])
        return {
            'success': True,
            'conversations': conversations
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_conversation(request: Request, user: dict = Depends(get_current_user)):
    """Create a new conversation"""
    try:
        data = await request.json()
        first_message = data.get('message', '新对话')
        conv_manager = get_conv_manager()
        conv_id = conv_manager.create_conversation(first_message, user_id=user["id"])
        return {
            'success': True,
            'conversation_id': conv_id
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conv_id}")
async def load_conversation(conv_id: str, user: dict = Depends(get_current_user)):
    """Load a conversation. 越权或不存在的对话返回空，不报 404/403。"""
    try:
        conv_manager = get_conv_manager()
        conversation = conv_manager.load_conversation(conv_id, user_id=user["id"])
        # load_conversation 永远返回 truthy dict（不存在/越权都返回空默认）
        return {
            'success': True,
            'conversation': conversation
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: str, user: dict = Depends(get_current_user)):
    """Delete a conversation"""
    try:
        conv_manager = get_conv_manager()
        conv_manager.delete_conversation(conv_id, user_id=user["id"])
        return {'success': True}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conv_id}/messages")
async def append_message(conv_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Append a message to a conversation"""
    try:
        data = await request.json()
        role = data.get('role', 'user')
        content = data.get('content', '')
        thought_steps = data.get('thought_steps')

        conv_manager = get_conv_manager()
        conv_manager.append_message(
            conv_id, role, content,
            thought_steps=thought_steps, user_id=user["id"],
        )

        return {'success': True}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
