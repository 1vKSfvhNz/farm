# backend/app/api/v1/piscicoles.py
"""
Routes de gestion des piscicoles (poissons)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...schemas.piscicole import PiscicoleCreate, PiscicoleUpdate, PiscicoleResponse
from ...schemas.pesee import PeseeCreate, PeseeResponse
from ...services.piscicole_service import piscicole_service
from ...services.pesee_service import pesee_service
from ...api.dependencies.auth import can_read_piscicoles, can_write_piscicoles
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User
from ...models.animal import SexeEnum, StatutAnimalEnum

router = APIRouter(prefix="/piscicoles", tags=["Piscicoles"])


@router.get("/", response_model=PaginatedResponse[PiscicoleResponse])
async def get_piscicoles(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    race: Optional[str] = Query(None, description="Filtrer par race"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par bassin"),
    statut: Optional[StatutAnimalEnum] = Query(None, description="Filtrer par statut"),
    current_user: User = Depends(can_read_piscicoles),
):
    """
    Obtenir la liste des piscicoles
    """
    piscicoles = await piscicole_service.get_piscicoles(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        race=race,
        enclos_id=enclos_id,
        statut=statut
    )
    
    all_piscicoles = await piscicole_service.get_piscicoles(db, limit=10000)
    
    return PaginatedResponse.create(
        items=piscicoles,
        total=len(all_piscicoles),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{piscicole_id}", response_model=PiscicoleResponse)
async def get_piscicole(
    piscicole_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_piscicoles),
):
    """
    Obtenir un poisson par son ID
    """
    piscicole = await piscicole_service.get_piscicole(db, piscicole_id)
    if not piscicole:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Piscicole non trouvé"
        )
    return piscicole


@router.post("/", response_model=PiscicoleResponse, status_code=status.HTTP_201_CREATED)
async def create_piscicole(
    piscicole_data: PiscicoleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_piscicoles),
):
    """
    Créer un nouveau poisson
    """
    piscicole, error = await piscicole_service.create_piscicole(db, piscicole_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return piscicole


@router.put("/{piscicole_id}", response_model=PiscicoleResponse)
async def update_piscicole(
    piscicole_id: int,
    piscicole_data: PiscicoleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_piscicoles),
):
    """
    Mettre à jour un poisson
    """
    piscicole, error = await piscicole_service.update_piscicole(db, piscicole_id, piscicole_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return piscicole



@router.get("/bassin/{enclos_id}/biomasse")
async def get_biomass(
    enclos_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_piscicoles),
):
    """
    Calculer la biomasse totale dans un bassin
    """
    biomass = await piscicole_service.get_biomass(db, enclos_id)
    return {"enclos_id": enclos_id, "biomass_kg": round(biomass, 2)}


@router.get("/stats/global")
async def get_piscicole_stats(
    enclos_id: Optional[int] = Query(None, description="Filtrer par bassin"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_piscicoles),
):
    """
    Obtenir des statistiques globales sur les piscicoles
    """
    stats = await piscicole_service.get_piscicole_stats(db, enclos_id)
    return stats