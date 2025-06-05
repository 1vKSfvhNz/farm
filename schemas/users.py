from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional
from schemas import Pagination

class UserBase(BaseModel):
    username: str
    email: EmailStr
    phone: str

class UserCreate(UserBase):
    password: str
    code: Optional[str] = None

    class Config:
        from_attributes = True


class UserResponse(UserBase):
    id: int
    username: str
    email: str
    phone: str
    role: str
    created_at: datetime
    last_login: Optional[datetime]  # Peut être null si l'utilisateur ne s'est jamais connecté

    class Config:
        from_attributes = True  # Active la compatibilité avec les ORM (SQLAlchemy)

class UsersResponse(BaseModel):
    users: list[UserResponse]
    pagination: Pagination
