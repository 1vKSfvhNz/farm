# backend/app/schemas/water_quality.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class WaterQualityBase(BaseModel):
    enclos_id: int
    timestamp: datetime
    ph: Optional[float] = Field(None, ge=0, le=14)
    temperature: Optional[float] = Field(None, ge=-10, le=50)
    oxygene_dissous: Optional[float] = Field(None, ge=0)
    ammoniac: Optional[float] = Field(None, ge=0)
    nitrites: Optional[float] = Field(None, ge=0)
    nitrates: Optional[float] = Field(None, ge=0)
    conductivite: Optional[float] = Field(None, ge=0)
    turbidite: Optional[float] = Field(None, ge=0)
    source: Optional[str] = None


class WaterQualityCreate(WaterQualityBase):
    pass


class WaterQualityResponse(WaterQualityBase):
    id: int
    alerte_generee: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class WaterQualityAlerteResponse(BaseModel):
    id: int
    water_quality_id: int
    parametre: str
    valeur: float
    seuil: float
    niveau: str
    message: Optional[str]
    traitee: bool
    created_at: datetime
    
    class Config:
        from_attributes = True