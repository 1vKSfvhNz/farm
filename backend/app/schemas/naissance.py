# backend/app/schemas/naissance.py
"""
Schémas Pydantic pour les naissances
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime


class NaissanceBase(BaseModel):
    """Base pour les naissances"""
    mere_id: int = Field(..., description="ID de la mère")
    pere_bovin_id: Optional[int] = Field(None, description="ID du père (bovin)")
    pere_ovin_id: Optional[int] = Field(None, description="ID du père (ovin)")
    pere_caprin_id: Optional[int] = Field(None, description="ID du père (caprin)")
    animal_ne_id: int = Field(..., description="ID du nouveau-né")
    date_naissance: date = Field(..., description="Date de naissance")
    poids_naissance: Optional[float] = Field(None, ge=0, description="Poids à la naissance (kg)")
    sexe: str = Field(..., description="Sexe du nouveau-né")
    complications: bool = Field(default=False, description="Complications lors de la naissance")
    notes: Optional[str] = Field(None, description="Notes supplémentaires")


class NaissanceCreate(NaissanceBase):
    """Schéma pour la création d'une naissance"""
    pass


class NaissanceUpdate(BaseModel):
    """Schéma pour la mise à jour d'une naissance"""
    poids_naissance: Optional[float] = Field(None, ge=0)
    complications: Optional[bool] = None
    notes: Optional[str] = None


class NaissanceResponse(NaissanceBase):
    """Schéma pour la réponse d'une naissance"""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class NaissanceListResponse(BaseModel):
    """Réponse pour une liste de naissances"""
    items: list[NaissanceResponse]
    total: int
    skip: int
    limit: int