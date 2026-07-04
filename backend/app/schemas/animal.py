# backend/app/schemas/animal.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from enum import Enum


class SexeEnum(str, Enum):
    MALE = "male"
    FEMELLE = "femelle"
    HERMAPHRODITE = "hermaphrodite"


class StatutAnimalEnum(str, Enum):
    VIVANT = "vivant"
    VENDU = "vendu"
    DECEDE = "decede"
    TRANSFERE = "transfere"


class AnimalBase(BaseModel):
    type_espece: str
    race: str = Field(..., min_length=1, max_length=100)
    sexe: SexeEnum
    date_naissance: Optional[date] = None
    date_arrivee: date
    provenance: Optional[str]
    prix_achat: Optional[float] = None
    enclos_id: int
    statut: StatutAnimalEnum = StatutAnimalEnum.VIVANT
    type_production: Optional[str] = None
    photo_url: Optional[str] = None
    notes: Optional[str] = None

    # === NOUVEAUX CHAMPS ===
    prix_vente: Optional[int] = Field(None, ge=0, description="Prix de vente en €")
    date_vente: Optional[date] = Field(None, description="Date de vente")
    client_acheteur: Optional[str] = Field(None, max_length=255, description="Nom du client acheteur")
    note_vente: Optional[str] = Field(None, description="Notes sur la vente")

class AnimalVenteStats(BaseModel):
    # === NOUVEAU ===
    total_ventes: int
    montant_total_ventes: int
    prix_vente_moyen: int

# === NOUVEAU SCHÉMA POUR LA VENTE ===
class AnimalVenteCreate(BaseModel):
    """Schéma pour enregistrer une vente de bovin, ovin, caprin"""
    prix_vente: int = Field(..., ge=0, description="Prix de vente en FCFA")
    date_vente: Optional[date] = Field(None, description="Date de vente (par défaut aujourd'hui)")
    client_acheteur: Optional[str] = Field(None, max_length=255, description="Nom du client acheteur")
    note_vente: Optional[str] = Field(None, description="Notes sur la vente")
    statut: Optional[StatutAnimalEnum] = Field(StatutAnimalEnum.VENDU, description="Statut après vente")