"""对话相关请求 Pydantic schemas。"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    message: Optional[str] = ""


class ChatHistoryMessage(BaseModel):
    """无 conversation_id 时客户端自带的历史消息（仅允许合法角色）。"""
    role: Literal["user", "assistant", "system"]
    content: str = ""


class ChatStreamRequest(BaseModel):
    """POST /api/chat/stream 请求体。"""
    message: str = Field(min_length=1)
    # 附件场景下 message 含附件全文，display_message 是存库/展示用的原始输入
    display_message: Optional[str] = None
    # 仅在无 conversation_id 时使用；后端统一按首尾保留和 Token 预算裁剪
    messages: list[ChatHistoryMessage] = Field(default_factory=list)
    conversation_id: Optional[str] = None
