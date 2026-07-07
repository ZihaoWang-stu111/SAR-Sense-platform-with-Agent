"""对话相关请求 Pydantic schemas。"""
from typing import Optional
from pydantic import BaseModel


class CreateConversationRequest(BaseModel):
    message: Optional[str] = ""


class AppendMessageRequest(BaseModel):
    role: str = "user"
    content: str = ""
    thought_steps: Optional[list] = None
    rag_results: Optional[list] = None
