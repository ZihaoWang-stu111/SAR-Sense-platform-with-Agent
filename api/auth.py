"""认证路由 + FastAPI 依赖（基于 MySQL + AsyncSession）。

POST /api/auth/register、POST /api/auth/login、GET /api/auth/me
JWT 走 Authorization: Bearer <token> 头。
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_conf import get_db
from crud.users import create_user, get_user_by_username
from schemas.users import LoginRequest, RegisterRequest
from utils.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
router = APIRouter(prefix="/api/auth", tags=["auth"])


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """解析 Authorization: Bearer <token>，返回 {id, username}。失败 401。

    注意：返回的是 dict（不是 SQLAlchemy User 对象），因为 token 已携带必需信息，
    避免每次请求都查一次 users 表。需要新鲜数据时调 /api/auth/me。
    """
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="无效或过期的 token")
    return {"id": payload["user_id"], "username": payload["username"]}


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """知识库管理需要 admin 权限。"""
    if user["username"] != "admin":
        raise HTTPException(status_code=403, detail="只有管理员才能操作知识库")
    return user


@router.post("/register")
async def register(creds: RegisterRequest, db: AsyncSession = Depends(get_db)):
    exists = await get_user_by_username(db, creds.username)
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")
    user = await create_user(db, creds.username, hash_password(creds.password))
    token = create_access_token(user.id, user.username)
    return {
        "success": True,
        "user": {"id": user.id, "username": user.username},
        "token": token,
    }


@router.post("/login")
async def login(creds: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_username(db, creds.username)
    if user is None or not verify_password(creds.password, user.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user.id, user.username)
    return {
        "success": True,
        "user": {"id": user.id, "username": user.username},
        "token": token,
    }


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return {"success": True, "user": user}
