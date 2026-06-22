"""用户表 crud（异步）。"""
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.users import User


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    return await db.get(User, user_id)


async def create_user(db: AsyncSession, username: str, password_hash: str) -> User:
    user = User(
        username=username,
        password=password_hash,
        created_at=datetime.now(),
    )
    db.add(user)
    await db.flush()    # 拿到自增 id
    await db.refresh(user)
    return user


async def count_users(db: AsyncSession) -> int:
    from sqlalchemy import func
    result = await db.execute(select(func.count(User.id)))
    return result.scalar_one()
