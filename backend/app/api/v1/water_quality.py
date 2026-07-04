# backend/app/api/v1/water_quality.py
"""
Routes de gestion de la qualité de l'eau
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import datetime

from ...database import get_db
from ...schemas.water_quality import WaterQualityCreate, WaterQualityResponse, WaterQualityAlerteResponse
from ...services.water_quality_service import water_quality_service
from ...api.dependencies.auth import get_current_user, get_current_technicien_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/water-quality", tags=["Qualité de l'eau"])


@router.post("/measurements", response_model=WaterQualityResponse, status_code=status.HTTP_201_CREATED)
async def create_measurement(
    measurement: WaterQualityCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Ajouter une mesure de qualité d'eau
    """
    result = await water_quality_service.create_measurement(db, measurement)
    return result


@router.get("/measurements/{enclos_id}/latest", response_model=WaterQualityResponse)
async def get_latest_measurement(
    enclos_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la dernière mesure pour un enclos
    """
    measurement = await water_quality_service.get_last_measurement(db, enclos_id)
    if not measurement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucune mesure trouvée pour cet enclos"
        )
    return measurement


@router.get("/measurements/{enclos_id}", response_model=PaginatedResponse[WaterQualityResponse])
async def get_measurements(
    enclos_id: int,
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir l'historique des mesures pour un enclos
    """
    measurements = await water_quality_service.get_measurements(
        db,
        enclos_id,
        skip=pagination.skip,
        limit=pagination.limit,
        start_date=start_date,
        end_date=end_date
    )
    
    return PaginatedResponse.create(
        items=measurements,
        total=len(measurements),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/alerts", response_model=PaginatedResponse[WaterQualityAlerteResponse])
async def get_alerts(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    traitee: Optional[bool] = Query(None, description="Filtrer par statut"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les alertes de qualité d'eau
    """
    alerts = await water_quality_service.get_alerts(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        enclos_id=enclos_id,
        traitee=traitee
    )
    
    return PaginatedResponse.create(
        items=alerts,
        total=len(alerts),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Marquer une alerte comme traitée
    """
    success = await water_quality_service.resolve_alert(db, alert_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alerte non trouvée"
        )
    return {"message": "Alerte marquée comme traitée"}