"""认证路由 + FastAPI 依赖。

JWT + Authorization 头。PyJWT + bcrypt，无 passlib/jose。
端点：POST /api/auth/register、POST /api/auth/login、GET /api/auth/me。
"""
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from utils.db import get_conn, init_db
from utils.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)

# auth 路由依赖 users 表 + 种子用户，import 时确保建表（幂等）。
# ConversationManager 构造也会调 init_db，但 auth 路由可能先被访问。
init_db()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    username: str
    password: str


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """解析 Authorization: Bearer <token>，返回 {id, username}。失败 401。"""
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="无效或过期的 token")
    return {"id": payload["user_id"], "username": payload["username"]}


@router.post("/register")
async def register(creds: Credentials):
    """注册：用户名重复 409，成功直接返回 token（注册后自动登录）。"""
    with get_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM users WHERE username=?", (creds.username,)
        ).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="用户名已存在")
        user_id = uuid.uuid4().hex
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, creds.username, hash_password(creds.password), now),
        )
    token = create_access_token(user_id, creds.username)
    return {"success": True, "user": {"id": user_id, "username": creds.username}, "token": token}


@router.post("/login")
async def login(creds: Credentials):
    """登录：用户名或密码错误 401，成功返回 token。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username=?",
            (creds.username,),
        ).fetchone()
    if row is None or not verify_password(creds.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(row["id"], row["username"])
    return {"success": True, "user": {"id": row["id"], "username": row["username"]}, "token": token}


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    """校验 token 有效性，返回当前用户。前端加载时调用。"""
    return {"success": True, "user": user}
