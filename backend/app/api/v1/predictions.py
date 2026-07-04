# backend/app/api/v1/predictions.py
"""
Routes des prédictions
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import date

from ...database import get_db
from ...schemas.predictions import PredictionRequest, PredictionResponse, GrowthPredictionResponse, ProductionPredictionResponse, CashflowPredictionResponse
from ...services.prediction_service import prediction_service
from ...api.dependencies.auth import get_current_user
from ...models.user import User

router = APIRouter(prefix="/predictions", tags=["Prédictions"])


@router.post("/", response_model=PredictionResponse)
async def make_prediction(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Générer une prédiction basée sur le type demandé
    """
    prediction = await prediction_service.make_prediction(db, request)
    return prediction


@router.get("/growth/{animal_id}", response_model=GrowthPredictionResponse)
async def predict_growth(
    animal_id: int,
    horizon_jours: int = Query(90, description="Horizon de prédiction en jours"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Prédire la croissance d'un animal
    """
    prediction = await prediction_service.predict_growth(db, animal_id, horizon_jours)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Animal non trouvé"
        )
    return prediction


@router.get("/production/{espece}", response_model=ProductionPredictionResponse)
async def predict_production(
    espece: str,
    race: Optional[str] = Query(None, description="Race"),
    enclos_id: Optional[int] = Query(None, description="Enclos"),
    horizon_jours: int = Query(30, description="Horizon en jours"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Prédire la production (lait, œufs, larves)
    """
    prediction = await prediction_service.predict_production(
        db, espece, race, enclos_id, horizon_jours
    )
    return prediction


@router.get("/cashflow", response_model=CashflowPredictionResponse)
async def predict_cashflow(
    horizon_jours: int = Query(90, description="Horizon en jours"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Prédire l'évolution de la trésorerie
    """
    prediction = await prediction_service.predict_cashflow(db, horizon_jours)
    return prediction


@router.get("/health/{espece}")
async def predict_health_risk(
    espece: str,
    enclos_id: Optional[int] = Query(None, description="Enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Prédire les risques sanitaires
    """
    risk = await prediction_service.predict_health_risk(db, espece, enclos_id)
    return risk


@router.get("/compost/{compost_id}")
async def predict_compost_maturity(
    compost_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Prédire la date de maturité d'un compost
    """
    prediction = await prediction_service.predict_compost_maturity(db, compost_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Compost non trouvé"
        )
    return prediction


@router.get("/overcrowding/{enclos_id}")
async def predict_overcrowding(
    enclos_id: int,
    horizon_jours: int = Query(90, description="Horizon en jours"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Prédire la date de surpopulation d'un enclos
    """
    prediction = await prediction_service.predict_overcrowding(db, enclos_id, horizon_jours)
    return prediction


@router.get("/water-quality/{enclos_id}")
async def predict_water_quality(
    enclos_id: int,
    hours_ahead: int = Query(24, description="Heures à prédire"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Prédire l'évolution de la qualité de l'eau
    """
    prediction = await prediction_service.predict_water_quality(db, enclos_id, hours_ahead)
    return prediction