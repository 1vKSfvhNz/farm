# backend/app/api/v1/experimental.py
"""
Routes du mode expérimental
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ...database import get_db
from ...schemas.experimental import (
    ReferenceHypothesisCreate, ReferenceHypothesisResponse,
    ExperimentalModeResponse, ConfidenceResponse,
    ReferenceGenerationRequest, ReferenceGenerationResponse
)
from ...services.experimental_service import experimental_service
from ...api.dependencies.auth import get_current_user, get_current_admin_user
from ...models.user import User

router = APIRouter(prefix="/experimental", tags=["Mode Expérimental"])


@router.get("/status", response_model=ExperimentalModeResponse)
async def get_experimental_status(
    espece: Optional[str] = Query(None, description="Filtrer par espèce"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le statut du mode expérimental
    """
    status = await experimental_service.get_experimental_mode_status(db, espece)
    return status


@router.get("/confidence/{espece}/{prediction_type}", response_model=ConfidenceResponse)
async def get_confidence(
    espece: str,
    prediction_type: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le niveau de confiance pour un type de prédiction
    """
    confidence = await experimental_service.get_confidence(db, espece, prediction_type)
    return confidence


@router.post("/references/generate", response_model=ReferenceGenerationResponse)
async def generate_reference(
    request: ReferenceGenerationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Générer une référence à partir des données collectées (admin uniquement)
    """
    result = await experimental_service.generate_reference(
        db, request.espece, request.force_regenerate
    )
    return result


@router.post("/hypotheses", response_model=ReferenceHypothesisResponse)
async def create_hypothesis(
    hypothesis: ReferenceHypothesisCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Créer une hypothèse de référence (mode expérimental)
    """
    result = await experimental_service.create_hypothesis(db, hypothesis, current_user.id)
    return result


@router.get("/hypotheses", response_model=list[ReferenceHypothesisResponse])
async def get_hypotheses(
    espece: Optional[str] = Query(None, description="Filtrer par espèce"),
    validee: Optional[bool] = Query(None, description="Filtrer par validation"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les hypothèses de référence
    """
    hypotheses = await experimental_service.get_hypotheses(db, espece, validee)
    return hypotheses


@router.put("/hypotheses/{hypothesis_id}/validate")
async def validate_hypothesis(
    hypothesis_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Valider une hypothèse de référence (admin uniquement)
    """
    success = await experimental_service.validate_hypothesis(db, hypothesis_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hypothèse non trouvée"
        )
    return {"message": "Hypothèse validée avec succès"}


@router.get("/data/collect-stats")
async def get_collection_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les statistiques de collecte de données
    """
    stats = await experimental_service.get_collection_stats(db)
    return stats