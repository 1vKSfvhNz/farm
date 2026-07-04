# backend/app/schemas/apiculture.py
"""
Schémas Pydantic pour la gestion apicole
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
from enum import Enum


# ============ ENUMS ============

class StatutRucheEnum(str, Enum):
    """Statut d'une ruche"""
    ACTIVE = "active"
    ORPHELINE = "orpheline"
    EN_ESSIMAGE = "en_essaimage"
    MORTE = "morte"


# ============ RUCHES ============

class RucheBase(BaseModel):
    """Base pour les ruches"""
    identification: str = Field(..., min_length=1, max_length=50, description="Identifiant unique de la ruche")
    emplacement: Optional[str] = Field(None, max_length=255, description="Emplacement de la ruche")
    date_installation: date = Field(..., description="Date d'installation")
    race: Optional[str] = Field(None, max_length=100, description="Race des abeilles")
    statut: StatutRucheEnum = Field(default=StatutRucheEnum.ACTIVE, description="Statut de la ruche")
    nombre_cadres: Optional[int] = Field(None, ge=0, description="Nombre de cadres")
    notes: Optional[str] = Field(None, description="Notes supplémentaires")


class RucheCreate(RucheBase):
    """Schéma pour la création d'une ruche"""
    pass


class RucheUpdate(BaseModel):
    """Schéma pour la mise à jour d'une ruche"""
    identification: Optional[str] = Field(None, min_length=1, max_length=50)
    emplacement: Optional[str] = Field(None, max_length=255)
    date_installation: Optional[date] = None
    race: Optional[str] = Field(None, max_length=100)
    statut: Optional[StatutRucheEnum] = None
    nombre_cadres: Optional[int] = Field(None, ge=0)
    notes: Optional[str] = None


class RucheResponse(RucheBase):
    """Schéma pour la réponse d'une ruche"""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# ============ RÉCOLTES DE MIEL ============

class RecolteMielBase(BaseModel):
    """Base pour les récoltes de miel"""
    ruche_id: int = Field(..., description="ID de la ruche")
    date_recolte: date = Field(..., description="Date de récolte")
    poids_kg: float = Field(..., gt=0, description="Poids de miel récolté (kg)")
    qualite: Optional[str] = Field(None, max_length=50, description="Qualité du miel (toutes fleurs, miellat, etc.)")
    taux_eau: Optional[float] = Field(None, ge=0, le=30, description="Taux d'eau (%)")
    notes: Optional[str] = Field(None, description="Notes supplémentaires")


class RecolteMielCreate(RecolteMielBase):
    """Schéma pour la création d'une récolte"""
    pass


class RecolteMielUpdate(BaseModel):
    """Schéma pour la mise à jour d'une récolte"""
    date_recolte: Optional[date] = None
    poids_kg: Optional[float] = Field(None, gt=0)
    qualite: Optional[str] = Field(None, max_length=50)
    taux_eau: Optional[float] = Field(None, ge=0, le=30)
    notes: Optional[str] = None


class RecolteMielResponse(RecolteMielBase):
    """Schéma pour la réponse d'une récolte"""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============ STATISTIQUES ============

class ProductionStatsResponse(BaseModel):
    """Statistiques de production de miel"""
    year: int
    total_honey_kg: float
    monthly_production: dict  # {1: 10.5, 2: 15.2, ...}
    average_per_ruche: float


class RucheStatsResponse(BaseModel):
    """Statistiques sur les ruches"""
    total: int
    active: int
    orphelines: int
    en_essaimage: int
    mortes: int
    taux_activite: float
    age_moyen_jours: float
    age_moyen_annees: float


class AlerteApicoleResponse(BaseModel):
    """Alerte apicole"""
    severity: str  # warning, critical
    message: str
    ruche_id: Optional[int]
    ruche_identification: Optional[str]
    reine_id: Optional[int]
    date: datetime


class DashboardApicoleResponse(BaseModel):
    """Tableau de bord apicole"""
    ruches: RucheStatsResponse
    production: ProductionStatsResponse
    dernieres_recoltes: List[RecolteMielResponse]
    alertes: List[AlerteApicoleResponse]
    derniere_mise_a_jour: str


# ============ INSPECTIONS ============

class InspectionRucheBase(BaseModel):
    """Base pour les inspections de ruche"""
    ruche_id: int = Field(..., description="ID de la ruche")
    date_inspection: date = Field(..., description="Date de l'inspection")
    etat_cadres: Optional[str] = Field(None, max_length=255, description="État des cadres")
    presence_couvain: Optional[bool] = Field(None, description="Présence de couvain")
    presence_miel: Optional[bool] = Field(None, description="Présence de miel")
    presence_pollen: Optional[bool] = Field(None, description="Présence de pollen")
    signes_maladie: Optional[str] = Field(None, description="Signes de maladie")
    notes: Optional[str] = Field(None, description="Notes supplémentaires")
    technicien: Optional[str] = Field(None, max_length=100, description="Nom du technicien")


class InspectionRucheCreate(InspectionRucheBase):
    """Schéma pour la création d'une inspection"""
    pass


class InspectionRucheResponse(InspectionRucheBase):
    """Schéma pour la réponse d'une inspection"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ ESSAIMAGE ============

class EssaimageBase(BaseModel):
    """Base pour les essaimages"""
    ruche_id: int = Field(..., description="ID de la ruche source")
    date_essaimage: date = Field(..., description="Date de l'essaimage")
    nouvelle_ruche_id: Optional[int] = Field(None, description="ID de la nouvelle ruche créée")
    nombre_abeilles: Optional[int] = Field(None, ge=0, description="Nombre estimé d'abeilles")
    reine_age: Optional[int] = Field(None, ge=0, description="Âge de la reine (mois)")
    recoltement: bool = Field(default=False, description="Essaim récolté")
    notes: Optional[str] = Field(None, description="Notes supplémentaires")


class EssaimageCreate(EssaimageBase):
    """Schéma pour la création d'un essaimage"""
    pass


class EssaimageResponse(EssaimageBase):
    """Schéma pour la réponse d'un essaimage"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

