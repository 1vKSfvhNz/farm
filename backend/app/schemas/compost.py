# backend/app/schemas/compost.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
from enum import Enum


class CompostTypeEnum(str, Enum):
    DECRETS_VERTS = "déchets verts"
    FUMIER = "fumier"
    MIXTE = "mixte"


class CompostBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: CompostTypeEnum
    date_demarrage: date
    volume_initial: float = Field(..., gt=0)
    volume_final: Optional[float] = Field(None, gt=0)
    date_maturite_estimee: Optional[date] = None
    date_maturite_reelle: Optional[date] = None
    utilisation_finale: Optional[str] = None
    notes: Optional[str] = None


class CompostCreate(CompostBase):
    pass


class CompostUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    type: Optional[CompostTypeEnum] = None
    date_demarrage: Optional[date] = None
    volume_initial: Optional[float] = Field(None, gt=0)
    volume_final: Optional[float] = Field(None, gt=0)
    date_maturite_estimee: Optional[date] = None
    date_maturite_reelle: Optional[date] = None
    utilisation_finale: Optional[str] = None
    notes: Optional[str] = None


class CompostResponse(CompostBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RetournementCompostBase(BaseModel):
    compost_id: int
    date_retournement: date
    responsable: Optional[str] = None
    temperature_avant: Optional[float] = None
    temperature_apres: Optional[float] = None
    humidite_avant: Optional[float] = Field(None, ge=0, le=100)
    humidite_apres: Optional[float] = Field(None, ge=0, le=100)
    notes: Optional[str] = None


class RetournementCompostCreate(RetournementCompostBase):
    pass


class RetournementCompostResponse(RetournementCompostBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True