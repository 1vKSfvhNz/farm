# backend/app/schemas/auth.py
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    userlogin: str = Field(..., min_length=8, max_length=32)
    password: str = Field(..., min_length=6, max_length=16)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int
    user_id: int
    email: str
    username: str
    phone: str
    roles: list[str]


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int

class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=4, max_length=72)
    
class ResetPasswordConfirmRequest(BaseModel):
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=4, max_length=72)

class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=4, max_length=72)
    new_password: str = Field(..., min_length=4, max_length=72)