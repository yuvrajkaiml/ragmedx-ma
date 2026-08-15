from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRole(str, Enum):
    HEALTHCARE_PROFESSIONAL = "healthcare_professional"
    MEDICAL_STUDENT = "medical_student"
    RESEARCHER = "researcher"
    SYSTEM_ADMIN = "system_admin"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: EmailStr
    username: str
    full_name: str | None
    role: UserRole
    specialty: str | None
    institution: str | None
    is_active: bool
    is_verified: bool
    created_at: datetime


class UserProfileUpdate(BaseModel):
    full_name: str | None = Field(default=None, max_length=255)
    specialty: str | None = Field(default=None, max_length=100)
    institution: str | None = Field(default=None, max_length=255)
