# backend/app/schemas/bovin.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime

from app.core.constants import SexeEnum, StatutAnimalEnum
from .animal import AnimalBase, AnimalVenteStats


class BovinBase(AnimalBase):
    production_laitiere: bool = False
    production_viande: bool = False
    production_reproduction: bool = False
    lactation_en_cours: bool = False
    production_lait_quotidienne: Optional[float] = Field(None, ge=0)
    poids_initial: Optional[float] = Field(None, ge=0, description="Poids initial à la création")
    photo_base64: Optional[str] = Field(None, description="Photo encodée en base64")


class BovinCreate(BovinBase):
    pass


class BovinUpdate(BaseModel):
    # Tous les champs sont optionnels pour la mise à jour
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
    production_laitiere: Optional[bool] = None
    production_viande: Optional[bool] = None
    production_reproduction: Optional[bool] = None
    lactation_en_cours: Optional[bool] = None
    production_lait_quotidienne: Optional[float] = Field(None, ge=0)
    poids_initial: Optional[float] = Field(None, ge=0)
    photo_base64: Optional[str] = None

    # === NOUVEAUX CHAMPS ===
    prix_vente: Optional[int] = Field(None, ge=0)
    date_vente: Optional[date] = None
    client_acheteur: Optional[str] = Field(None, max_length=255)
    note_vente: Optional[str] = None

    class Config:
        from_attributes = True


class BovinResponse(BovinBase):
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

class BovinStats(AnimalVenteStats):
    total: int
    production_laitiere: int
    production_viande: int
    production_reproduction: int
    lactation_en_cours: int
    production_lait_moyenne: float
