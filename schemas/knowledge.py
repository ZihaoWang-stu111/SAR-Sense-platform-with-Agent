from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UpdateDocumentPermissionsRequest(BaseModel):
    visibility_mode: str = Field(pattern="^(admin_only|roles)$")
    allowed_roles: list[str] = []


class UploadVisibilityRequest(BaseModel):
    visibility_mode: str = Field(pattern="^(admin_only|roles)$")
    allowed_roles: list[str] = []


class KnowledgeDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str = Field(validation_alias="filename")
    doc_id: str
    file_type: str | None = None
    chunk_count: int
    parent_count: int | None = None
    child_count: int | None = None
    chunk_method: str | None = None
    status: str
    ingested_at: datetime | None = None
    file_hash: str | None = None
    can_manage: bool = False
    allowed_roles: list[str] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class KnowledgeFilesResponse(BaseModel):
    success: bool = True
    files: list[KnowledgeDocumentResponse]
    total_files: int
    total_chunks: int


class KnowledgeEvidenceResponse(BaseModel):
    filename: str
    page: int | None = None
    content: str
    doc_id: str
    download_url: str | None = None


class UpdateDocumentPermissionsResponse(BaseModel):
    success: bool = True
    doc_id: str
    allowed_roles: list[str]
