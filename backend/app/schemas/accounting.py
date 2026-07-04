# backend/app/schemas/accounting.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
from enum import Enum


class CategorieDepenseEnum(str, Enum):
    ACHAT_ANIMAUX = "achat_animaux"
    ACHAT_OEUFS = "achat_oeufs"
    ALIMENTATION = "alimentation"
    VACCINS_SOINS = "vaccins_soins"
    EQUIPEMENT = "equipement"
    PERSONNEL = "personnel"
    EAU_ELECTRICITE = "eau_electricite"
    ENTRETIEN = "entretien"
    COMPOSTAGE = "compostage"
    TRANSPORT = "transport"
    FRAIS_DIVERS = "frais_divers"


class CategorieRecetteEnum(str, Enum):
    VENTE_ANIMAUX_VIVANTS = "vente_animaux_vivants"
    VENTE_VIANDE = "vente_viande"
    VENTE_LAIT = "vente_lait"
    VENTE_LAINE = "vente_laine"
    VENTE_OEUFS = "vente_oeufs"
    VENTE_LARVES = "vente_larves"
    VENTE_COMPOST = "vente_compost"
    VENTE_FUMIER = "vente_fumier"
    SUBVENTIONS = "subventions"
    AUTRES = "autres"


class DepenseBase(BaseModel):
    categorie: CategorieDepenseEnum
    montant: float = Field(..., gt=0)
    date: date
    description: Optional[str] = None
    fournisseur: Optional[str] = None
    quantite: Optional[float] = Field(None, gt=0)
    prix_unitaire: Optional[float] = Field(None, gt=0)
    animal_id: Optional[int] = None
    lot_entomo_id: Optional[int] = None
    piece_jointe_url: Optional[str] = None


class DepenseCreate(DepenseBase):
    pass


class DepenseUpdate(BaseModel):
    categorie: Optional[CategorieDepenseEnum] = None
    montant: Optional[float] = Field(None, gt=0)
    date: Optional[date] = None # type: ignore
    description: Optional[str] = None
    fournisseur: Optional[str] = None
    quantite: Optional[float] = Field(None, gt=0)
    prix_unitaire: Optional[float] = Field(None, gt=0)


class DepenseResponse(DepenseBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RecetteBase(BaseModel):
    categorie: CategorieRecetteEnum
    montant: float = Field(..., gt=0)
    date: date
    description: Optional[str] = None
    client: Optional[str] = None
    quantite: Optional[float] = Field(None, gt=0)
    prix_unitaire: Optional[float] = Field(None, gt=0)
    animal_id: Optional[int] = None
    lot_entomo_id: Optional[int] = None


class RecetteCreate(RecetteBase):
    pass


class RecetteUpdate(BaseModel):
    categorie: Optional[CategorieRecetteEnum] = None
    montant: Optional[float] = Field(None, gt=0)
    date: Optional[date] = None # type: ignore
    description: Optional[str] = None
    client: Optional[str] = None
    quantite: Optional[float] = Field(None, gt=0)
    prix_unitaire: Optional[float] = Field(None, gt=0)


class RecetteResponse(RecetteBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AccountSummary(BaseModel):
    total_depenses: float
    total_recettes: float
    benefice: float
    marge_brute_pourcent: float
    depenses_par_categorie: dict
    recettes_par_categorie: dict
    tresorerie_previsionnelle: dict