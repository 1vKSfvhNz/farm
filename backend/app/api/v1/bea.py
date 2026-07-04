# backend/app/api/v1/bea.py
"""
Routes du bien-être animal (BEA)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import date

from ...database import get_db
from ...schemas.bea import BienEtreIndiceCreate, BienEtreIndiceResponse, BienEtreCritereResponse
from ...services.bea_service import bea_service
from ...api.dependencies.auth import get_current_user, get_current_technicien_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/bea", tags=["Bien-être Animal"])


@router.get("/indices/{enclos_id}/latest", response_model=BienEtreIndiceResponse)
async def get_latest_index(
    enclos_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le dernier indice de bien-être pour un enclos
    """
    index = await bea_service.get_latest_index(db, enclos_id)
    if not index:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucun indice trouvé pour cet enclos"
        )
    return index


@router.get("/indices/{enclos_id}", response_model=PaginatedResponse[BienEtreIndiceResponse])
async def get_indices(
    enclos_id: int,
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir l'historique des indices de bien-être
    """
    indices = await bea_service.get_indices(
        db,
        enclos_id,
        skip=pagination.skip,
        limit=pagination.limit,
        start_date=start_date,
        end_date=end_date
    )
    
    return PaginatedResponse.create(
        items=indices,
        total=len(indices),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.post("/indices", response_model=BienEtreIndiceResponse, status_code=status.HTTP_201_CREATED)
async def create_index(
    index_data: BienEtreIndiceCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Créer un indice de bien-être
    """
    index = await bea_service.create_index(db, index_data, current_user.id)
    return index


@router.get("/criteres", response_model=List[BienEtreCritereResponse])
async def get_criteres(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des critères de bien-être
    """
    criteres = await bea_service.get_criteres(db)
    return criteres


@router.get("/dashboard/{enclos_id}")
async def get_bea_dashboard(
    enclos_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le tableau de bord BEA pour un enclos
    """
    dashboard = await bea_service.get_dashboard(db, enclos_id)
    return dashboard