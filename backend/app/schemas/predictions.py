# backend/app/schemas/predictions.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import date, datetime


class PredictionRequest(BaseModel):
    espece: str
    race: Optional[str] = None
    animal_id: Optional[int] = None
    lot_id: Optional[int] = None
    prediction_type: str  # growth, production, health, cashflow, compost, water_quality
    horizon_jours: int = Field(30, ge=1, le=365)


class PredictionResponse(BaseModel):
    prediction_id: str
    espece: str
    prediction_type: str
    horizon_jours: int
    predictions: Dict
    confidence: float = Field(..., ge=0, le=100)
    confidence_level: str  # faible, moyenne, élevée
    warnings: List[str] = []
    generated_at: datetime


class GrowthPredictionResponse(BaseModel):
    animal_id: int
    age_actuel_jours: int
    poids_actuel_kg: float
    poids_prevu_jours: List[Dict]  # [{"jour": 30, "poids_min": 10, "poids_max": 15, "poids_moyen": 12.5}]
    date_atteinte_poids_vente: Optional[date]
    retard_croissance_detecte: bool
    recommandations: List[str]


class ProductionPredictionResponse(BaseModel):
    espece: str
    type_production: str  # lait, oeufs, larves
    production_quotidienne_actuelle: float
    production_prevue_15j: float
    production_prevue_30j: float
    production_prevue_90j: float
    saisonnalite_impact: Optional[float]
    recommandations: List[str]


class CashflowPredictionResponse(BaseModel):
    tresorerie_actuelle: float
    entrees_prevues_30j: float
    sorties_prevues_30j: float
    tresorerie_prevue_30j: float
    tresorerie_prevue_60j: float
    tresorerie_prevue_90j: float
    seuil_alerte_atteint: bool
    recommandations: List[str]