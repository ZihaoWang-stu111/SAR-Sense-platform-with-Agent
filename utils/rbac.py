ROLES = ("admin", "researcher", "business", "guest")
ROLE_ADMIN = "admin"
ROLE_RESEARCHER = "researcher"
ROLE_BUSINESS = "business"
ROLE_GUEST = "guest"


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"invalid role: {role}")
    return role


def is_admin(user: dict | None) -> bool:
    return bool(user and user.get("role") == ROLE_ADMIN)
