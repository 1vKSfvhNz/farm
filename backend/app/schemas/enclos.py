# backend/app/schemas/enclos.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class EnclosTypeEnum(str, Enum):
    ENCLOS = "enclos"
    BASSIN = "bassin"
    PATURAGE = "pâturage"
    CAGE = "cage"
    BAC = "bac"


class EnclosBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: EnclosTypeEnum
    longueur: float = Field(..., gt=0)
    largeur: float = Field(..., gt=0)
    hauteur: Optional[float] = Field(None, gt=0)
    localisation_gps: Optional[str] = None
    zone: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None


class EnclosCreate(EnclosBase):
    pass


class EnclosUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    type: Optional[EnclosTypeEnum] = None
    longueur: Optional[float] = Field(None, gt=0)
    largeur: Optional[float] = Field(None, gt=0)
    hauteur: Optional[float] = Field(None, gt=0)
    # capacite_maximale: Optional[int] = Field(None, gt=0)
    localisation_gps: Optional[str] = None
    zone: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None


class EnclosResponse(EnclosBase):
    id: int
    surface: float
    volume: Optional[float]

    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class EnclosLength(BaseModel):
    length: int
    
class EnclosStats(BaseModel):
    total_enclos: int
    stats_by_type: Dict[str, int]
    capacite_totale: int
    animaux_actifs_total: int
    surface_totale_m2: float
    volume_total_m3: float
    taux_occupation_moyen: float
    enclos_saturation_elevee: int
    enclos_saturation_faible: int
    enclos_vides: int
    enclos_pleins: int


class EnclosDetailStats(BaseModel):
    id: int
    name: str
    type: EnclosTypeEnum
    surface_m2: float
    volume_m3: Optional[float] = None
    capacite_maximale: int
    occupation_actuelle: int
    taux_occupation: float
    animaux_par_type: Dict[str, int]
    places_restantes: int
    zone: Optional[str] = None
    localisation_gps: Optional[str] = None
    class Config:
        from_attributes = True