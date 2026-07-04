from pydantic import BaseModel, Field


class UpdateDocumentPermissionsRequest(BaseModel):
    visibility_mode: str = Field(pattern="^(admin_only|roles)$")
    allowed_roles: list[str] = []


class UploadVisibilityRequest(BaseModel):
    visibility_mode: str = Field(pattern="^(admin_only|roles)$")
    allowed_roles: list[str] = []
