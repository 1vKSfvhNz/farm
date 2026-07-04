# backend/app/schemas/experimental.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import date, datetime


class ReferenceHypothesisBase(BaseModel):
    espece: str = Field(..., min_length=1, max_length=50)
    race: Optional[str] = Field(None, max_length=100)
    parametre: str = Field(..., min_length=1, max_length=100)
    valeur_estimee: float
    unite: Optional[str] = None


class ReferenceHypothesisCreate(ReferenceHypothesisBase):
    pass


class ReferenceHypothesisResponse(ReferenceHypothesisBase):
    id: int
    utilisateur_id: Optional[int]
    date_creation: date
    validee: bool
    date_validation: Optional[date]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ExperimentalModeResponse(BaseModel):
    mode: str  # complet, hybride, experimental
    jours_collecte: int
    nombre_donnees_par_espece: Dict[str, int]
    confiance_moyenne: float
    seuils_atteints: List[str]
    recommandations: List[str]


class ConfidenceResponse(BaseModel):
    prediction_type: str
    confidence: float
    confidence_label: str  # Faible, Moyenne, Élevée
    facteurs: List[str]
    donnees_manquantes: List[str]
    recommandations: List[str]


class ReferenceGenerationRequest(BaseModel):
    espece: str
    force_regenerate: bool = False


class ReferenceGenerationResponse(BaseModel):
    espece: str
    nouvelles_references: List[Dict]
    nombre_donnees_utilisees: int
    confiance_atteinte: float
    message: str