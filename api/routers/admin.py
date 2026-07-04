from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import require_admin
from config.db_conf import get_db
from crud.users import (
    count_admin_users,
    get_user_by_id,
    list_users,
    update_user_role,
)
from schemas.users import UpdateUserRoleRequest
from utils.rbac import ROLE_ADMIN, validate_role

router = APIRouter(tags=["admin"])


def _user_payload(user) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "nickname": user.nickname,
        "role": user.role,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.get("/users")
async def admin_list_users(
    _admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    users = await list_users(db)
    return {"success": True, "users": [_user_payload(user) for user in users]}


@router.patch("/users/{user_id}/role")
async def admin_update_user_role(
    user_id: int,
    req: UpdateUserRoleRequest,
    _admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    try:
        role = validate_role(req.role)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效角色")

    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")

    if user.role == ROLE_ADMIN and role != ROLE_ADMIN and await count_admin_users(db) <= 1:
        raise HTTPException(status_code=400, detail="不能降级最后一个管理员")

    user = await update_user_role(db, user, role)
    return {"success": True, "user": _user_payload(user)}
