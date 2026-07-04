# backend/app/schemas/alerts.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class AlertBase(BaseModel):
    type: str  # vaccination, pesee, nettoyage, mortalite, water_quality, bea, comptable
    niveau: str  # info, warning, critical
    message: str
    espece: Optional[str] = None
    animal_id: Optional[int] = None
    enclos_id: Optional[int] = None
    entite_type: Optional[str] = None
    entite_id: Optional[int] = None
    date_alerte: datetime
    date_limite: Optional[datetime] = None


class AlertCreate(AlertBase):
    pass


class AlertResponse(AlertBase):
    id: int
    est_lue: bool
    est_traitee: bool
    utilisateur_traitement_id: Optional[int]
    date_traitement: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True