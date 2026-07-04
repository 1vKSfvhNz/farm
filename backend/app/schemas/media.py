# backend/app/schemas/media.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class CameraBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    url: str = Field(..., min_length=1, max_length=500)
    enclos_id: Optional[int] = None
    is_active: bool = True
    roi_config: Optional[str] = None
    fps: int = Field(15, ge=1, le=60)


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    url: Optional[str] = Field(None, min_length=1, max_length=500)
    enclos_id: Optional[int] = None
    is_active: Optional[bool] = None
    roi_config: Optional[str] = None
    fps: Optional[int] = Field(None, ge=1, le=60)


class CameraResponse(CameraBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class VideoRecordBase(BaseModel):
    camera_id: int
    animal_id: Optional[int] = None
    enclos_id: Optional[int] = None
    debut: datetime
    fin: Optional[datetime] = None
    duree_secondes: Optional[float] = Field(None, ge=0)
    file_path: str
    file_size_mb: Optional[float] = Field(None, ge=0)
    action_type: Optional[str] = None
    action_id: Optional[int] = None
    utilisateur_id: Optional[int] = None


class VideoRecordCreate(VideoRecordBase):
    pass


class VideoRecordResponse(VideoRecordBase):
    id: int
    ia_anomalie_detectee: bool
    ia_anomalie_type: Optional[str]
    ia_confiance: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True