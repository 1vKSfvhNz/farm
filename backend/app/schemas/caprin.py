# backend/app/schemas/caprin.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime

from .animal import AnimalBase, AnimalVenteStats
from app.core.constants import SexeEnum, StatutAnimalEnum


class CaprinBase(AnimalBase):
    production_viande: bool = False
    production_reproduction: bool = False
    poids_initial: Optional[float] = Field(None, ge=0, description="Poids initial à la création")
    photo_base64: Optional[str] = Field(None, description="Photo encodée en base64")


class CaprinCreate(CaprinBase):
    pass


class CaprinUpdate(BaseModel):
    """Tous les champs sont optionnels pour la mise à jour"""
    race: Optional[str] = Field(None, min_length=1, max_length=100)
    sexe: Optional[SexeEnum] = None
    date_naissance: Optional[date] = None
    date_arrivee: Optional[date] = None
    provenance: Optional[str] = None
    prix_achat: Optional[float] = None
    enclos_id: Optional[int] = None
    statut: Optional[StatutAnimalEnum] = None
    type_production: Optional[str] = None
    photo_url: Optional[str] = None
    notes: Optional[str] = None
    production_viande: Optional[bool] = None
    production_reproduction: Optional[bool] = None
    poids_initial: Optional[float] = Field(None, ge=0)
    photo_base64: Optional[str] = None

    class Config:
        from_attributes = True


class CaprinResponse(CaprinBase):
    id: int
    identification: str
    enclos_name: str
    photo_url: Optional[str] = None
    age_mois: Optional[int] = None
    dernier_poids: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CaprinStats(AnimalVenteStats):
    total: int
    production_viande: int
    production_reproduction: int
    poids_moyen_kg: float

    class Config:
        from_attributes = True