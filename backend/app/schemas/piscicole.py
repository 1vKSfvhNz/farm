# backend/app/schemas/piscicole.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from .animal import AnimalBase


class PiscicoleBase(AnimalBase):
    production_viande: bool = False
    production_reproduction: bool = False
    taille_moyenne: Optional[float] = Field(None, ge=0)


class PiscicoleCreate(PiscicoleBase):
    pass


class PiscicoleUpdate(BaseModel):
    production_viande: Optional[bool] = None
    production_reproduction: Optional[bool] = None
    taille_moyenne: Optional[float] = Field(None, ge=0)


class PiscicoleResponse(PiscicoleBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True