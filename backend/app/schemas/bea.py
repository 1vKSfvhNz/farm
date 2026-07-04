# backend/app/schemas/bea.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime


class BienEtreIndiceBase(BaseModel):
    enclos_id: int
    date: date
    indice_global: float = Field(..., ge=0, le=100)
    indice_proprete: Optional[float] = Field(None, ge=0, le=100)
    indice_acces_eau: Optional[float] = Field(None, ge=0, le=100)
    indice_densite: Optional[float] = Field(None, ge=0, le=100)
    indice_comportement: Optional[float] = Field(None, ge=0, le=100)
    notes: Optional[str] = None


class BienEtreIndiceCreate(BienEtreIndiceBase):
    pass


class BienEtreIndiceResponse(BienEtreIndiceBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class BienEtreCritereResponse(BaseModel):
    id: int
    nom: str
    description: Optional[str]
    ponderation: float
    seuil_alerte: Optional[float]
    
    class Config:
        from_attributes = True