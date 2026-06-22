"""认证安全工具（纯函数，无 FastAPI 依赖）。

bcrypt 做密码哈希，PyJWT 签发/校验 token。
求职项目最简实现：SECRET_KEY 硬编码常量（生产环境改为环境变量）。
"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

# 求职项目：硬编码 secret。生产环境应改为 os.getenv("JWT_SECRET") 或从配置读。
SECRET_KEY = os.getenv("JWT_SECRET", "sar-sense-dev-secret-change-in-production-2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7


def hash_password(plain: str) -> str:
    """bcrypt 哈希密码，返回 str（约 100ms）。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """校验密码。任何异常返回 False（不抛）。"""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: str, username: str) -> str:
    """签发 JWT，载荷 {user_id, username, exp}，7 天过期。"""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    """解码 JWT。过期/伪造/格式错统一返回 None。"""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
