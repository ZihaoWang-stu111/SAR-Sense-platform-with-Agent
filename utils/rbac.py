ROLES = ("admin", "researcher", "business", "guest")
VISIBLE_ROLES = ("researcher", "business", "guest")
ROLE_ADMIN = "admin"
ROLE_RESEARCHER = "researcher"
ROLE_BUSINESS = "business"
ROLE_GUEST = "guest"


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"invalid role: {role}")
    return role


def validate_allowed_roles(roles: list[str] | None) -> list[str]:
    roles = roles or []
    invalid = [role for role in roles if role not in VISIBLE_ROLES]
    if invalid:
        raise ValueError(f"invalid visible roles: {invalid}")
    return sorted(set(roles))


def is_admin(user: dict | None) -> bool:
    return bool(user and user.get("role") == ROLE_ADMIN)
