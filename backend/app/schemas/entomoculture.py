# backend/app/schemas/entomoculture.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
from enum import Enum


class StadeInsecteEnum(str, Enum):
    OEUF = "oeuf"
    LARVE = "larve"
    PUPE = "pupe"
    ADULTE = "adulte"


class TypeProductionInsecteEnum(str, Enum):
    LARVES = "larves"
    REPRODUCTION = "reproduction"
    OEUFS = "oeufs"


class EntomocultureLotBase(BaseModel):
    identification: str = Field(..., min_length=1, max_length=100)
    espece: str = Field(..., min_length=1, max_length=100)
    stade_actuel: StadeInsecteEnum
    date_arrivee: date
    provenance: Optional[str] = None
    prix_achat: Optional[float] = Field(None, ge=0)
    poids_initial: Optional[float] = Field(None, ge=0)
    quantite_estimative: Optional[int] = Field(None, ge=0)
    enclos_id: Optional[int] = None
    type_production: TypeProductionInsecteEnum
    notes: Optional[str] = None


class EntomocultureLotCreate(EntomocultureLotBase):
    pass


class EntomocultureLotUpdate(BaseModel):
    espece: Optional[str] = Field(None, min_length=1, max_length=100)
    stade_actuel: Optional[StadeInsecteEnum] = None
    provenance: Optional[str] = None
    poids_initial: Optional[float] = Field(None, ge=0)
    quantite_estimative: Optional[int] = Field(None, ge=0)
    enclos_id: Optional[int] = None
    type_production: Optional[TypeProductionInsecteEnum] = None
    notes: Optional[str] = None


class EntomocultureLotResponse(EntomocultureLotBase):
    id: int
    taux_mortalite: Optional[float]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class EntomocultureCycleBase(BaseModel):
    lot_id: int
    date_debut: date
    date_fin: Optional[date] = None
    stade_debut: StadeInsecteEnum
    stade_fin: Optional[StadeInsecteEnum] = None
    production_grammes: Optional[float] = Field(None, ge=0)
    taux_mortalite: Optional[float] = Field(None, ge=0, le=100)
    substrat_utilise: Optional[str] = None


class EntomocultureCycleCreate(EntomocultureCycleBase):
    pass


class EntomocultureCycleResponse(EntomocultureCycleBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True