from enum import Enum

from pydantic import BaseModel, EmailStr, Field

from app.schemas.user import UserResponse


class RegistrationRole(str, Enum):
    HEALTHCARE_PROFESSIONAL = "healthcare_professional"
    MEDICAL_STUDENT = "medical_student"
    RESEARCHER = "researcher"


class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)
    role: RegistrationRole = RegistrationRole.HEALTHCARE_PROFESSIONAL
    specialty: str | None = Field(default=None, max_length=100)
    institution: str | None = Field(default=None, max_length=255)


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class EmailActionRequest(BaseModel):
    email: EmailStr


class EmailTokenRequest(BaseModel):
    token: str = Field(min_length=1)


class PasswordResetRequest(EmailTokenRequest):
    new_password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse