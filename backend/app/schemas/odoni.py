# backend/app/schemas/odoni.py
"""
Schémas Pydantic pour la gestion des nuisibles (odoni)
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date


# ============ PIÈGES ============

class PiegeOdoniBase(BaseModel):
    """Base pour les pièges à nuisibles"""
    name: str = Field(..., min_length=1, max_length=100, description="Nom du piège")
    emplacement: Optional[str] = Field(None, max_length=255, description="Emplacement")
    enclos_id: Optional[int] = Field(None, description="ID de l'enclos associé")
    compost_id: Optional[int] = Field(None, description="ID du compost associé")
    date_installation: date = Field(..., description="Date d'installation")
    type_piege: Optional[str] = Field(None, max_length=50, description="Type de piège (collant, lumineux, etc.)")
    is_active: bool = Field(default=True, description="Piège actif")


class PiegeOdoniCreate(PiegeOdoniBase):
    """Schéma pour la création d'un piège"""
    pass


class PiegeOdoniUpdate(BaseModel):
    """Schéma pour la mise à jour d'un piège"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    emplacement: Optional[str] = Field(None, max_length=255)
    enclos_id: Optional[int] = None
    compost_id: Optional[int] = None
    type_piege: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class PiegeOdoniResponse(PiegeOdoniBase):
    """Schéma pour la réponse d'un piège"""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============ COMPTAGES ============

class ComptageOdoniBase(BaseModel):
    """Base pour les comptages de nuisibles"""
    piege_id: int = Field(..., description="ID du piège")
    nombre: int = Field(..., ge=0, description="Nombre de nuisibles comptés")
    espece: Optional[str] = Field(None, max_length=100, description="Espèce identifiée")
    methode: Optional[str] = Field(default="manuel", max_length=50, description="Méthode de comptage")


class ComptageOdoniCreate(ComptageOdoniBase):
    """Schéma pour la création d'un comptage"""
    pass


class ComptageOdoniUpdate(BaseModel):
    """Schéma pour la mise à jour d'un comptage"""
    nombre: Optional[int] = Field(None, ge=0)
    espece: Optional[str] = Field(None, max_length=100)
    methode: Optional[str] = Field(None, max_length=50)


class ComptageOdoniResponse(ComptageOdoniBase):
    """Schéma pour la réponse d'un comptage"""
    id: int
    timestamp: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ STATISTIQUES ============

class InfestationLevelResponse(BaseModel):
    """Niveau d'infestation"""
    level: str  # low, medium, high, critical
    total_count: int
    pieges_actifs: int
    alertes: List[dict] = []


class PiegeStatistiquesResponse(BaseModel):
    """Statistiques d'un piège"""
    piege_id: int
    piege_name: str
    total_comptages: int
    moyenne_par_jour: float
    max_par_jour: int
    tendance: str  # increasing, decreasing, stable
    dernier_comptage: Optional[ComptageOdoniResponse]
    evolution: List[dict]  # [{date: "2024-01-01", count: 10}]


class AlerteOdoniResponse(BaseModel):
    """Alerte pour les nuisibles"""
    severity: str  # warning, critical
    message: str
    piege_id: int
    piege_name: str
    current_count: int
    threshold: int
    date: datetime