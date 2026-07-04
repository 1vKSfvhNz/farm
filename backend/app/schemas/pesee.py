# backend/app/schemas/pesee.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime


class PeseeBase(BaseModel):
    animal_id: Optional[int] = None
    lot_entomo_id: Optional[int] = None
    lot_avicole_id: Optional[int] = None
    date_pesee: date
    poids: float = Field(..., gt=0)
    methode: Optional[str] = None
    video_record_id: Optional[int] = None
    notes: Optional[str] = None


class PeseeCreate(PeseeBase):
    pass


class PeseeUpdate(BaseModel):
    date_pesee: Optional[date] = None
    poids: Optional[float] = Field(None, gt=0)
    methode: Optional[str] = None
    notes: Optional[str] = None


class PeseeResponse(PeseeBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True