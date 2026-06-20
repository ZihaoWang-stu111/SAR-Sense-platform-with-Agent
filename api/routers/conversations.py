import logging
import traceback

from fastapi import APIRouter, Request, HTTPException

from api.dependencies import get_conv_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["conversations"])


@router.get("")
async def list_conversations():
    """List all conversations"""
    try:
        conv_manager = get_conv_manager()
        conversations = conv_manager.list_conversations()
        return {
            'success': True,
            'conversations': conversations
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_conversation(request: Request):
    """Create a new conversation"""
    try:
        data = await request.json()
        first_message = data.get('message', '新对话')
        conv_manager = get_conv_manager()
        conv_id = conv_manager.create_conversation(first_message)
        return {
            'success': True,
            'conversation_id': conv_id
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conv_id}")
async def load_conversation(conv_id: str):
    """Load a conversation"""
    try:
        conv_manager = get_conv_manager()
        conversation = conv_manager.load_conversation(conv_id)
        if conversation:
            return {
                'success': True,
                'conversation': conversation
            }
        else:
            raise HTTPException(status_code=404, detail='Conversation not found')
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation"""
    try:
        conv_manager = get_conv_manager()
        conv_manager.delete_conversation(conv_id)
        return {'success': True}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conv_id}/messages")
async def append_message(conv_id: str, request: Request):
    """Append a message to a conversation"""
    try:
        data = await request.json()
        role = data.get('role', 'user')
        content = data.get('content', '')
        thought_steps = data.get('thought_steps')

        conv_manager = get_conv_manager()
        conv_manager.append_message(conv_id, role, content, thought_steps=thought_steps)

        return {'success': True}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
