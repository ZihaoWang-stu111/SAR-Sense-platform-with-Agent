"""认证请求/响应 Pydantic schemas（对齐参考方式）。"""
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=4, max_length=128)


class UserInfo(BaseModel):
    id: int
    username: str
    role: str


class UpdateUserRoleRequest(BaseModel):
    role: str = Field(min_length=1, max_length=32)
