# backend/app/api/v1/compost.py
"""
Routes de gestion du compostage
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...schemas.compost import CompostCreate, CompostUpdate, CompostResponse, RetournementCompostCreate, RetournementCompostResponse
from ...services.compost_service import compost_service
from ...api.dependencies.auth import get_current_user, get_current_technicien_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User
from ...models.compost import CompostType

router = APIRouter(prefix="/compost", tags=["Compostage"])


@router.get("/", response_model=PaginatedResponse[CompostResponse])
async def get_composts(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    compost_type: Optional[CompostType] = Query(None, description="Filtrer par type"),
    is_mature: Optional[bool] = Query(None, description="Filtrer par maturité"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des composts
    """
    composts = await compost_service.get_composts(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        compost_type=compost_type,
        is_mature=is_mature
    )
    
    all_composts = await compost_service.get_composts(db, limit=10000)
    
    return PaginatedResponse.create(
        items=composts,
        total=len(all_composts),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{compost_id}", response_model=CompostResponse)
async def get_compost(
    compost_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir un compost par son ID
    """
    compost = await compost_service.get_compost(db, compost_id)
    if not compost:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Compost non trouvé"
        )
    return compost


@router.get("/{compost_id}/status")
async def get_compost_status(
    compost_id: int,
    temperature: Optional[float] = Query(None, description="Température actuelle"),
    humidity: Optional[float] = Query(None, description="Humidité actuelle"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le statut avancé d'un compost
    """
    status = await compost_service.get_compost_status(db, compost_id, temperature, humidity)
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Compost non trouvé"
        )
    return status


@router.post("/", response_model=CompostResponse, status_code=status.HTTP_201_CREATED)
async def create_compost(
    compost_data: CompostCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Créer un nouveau tas de compost
    """
    compost, error = await compost_service.create_compost(db, compost_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return compost


@router.put("/{compost_id}", response_model=CompostResponse)
async def update_compost(
    compost_id: int,
    compost_data: CompostUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Mettre à jour un compost
    """
    compost, error = await compost_service.update_compost(db, compost_id, compost_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return compost


@router.delete("/{compost_id}")
async def delete_compost(
    compost_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Supprimer un compost
    """
    success, message = await compost_service.delete_compost(db, compost_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    return {"message": message}


@router.get("/{compost_id}/retournements", response_model=PaginatedResponse[RetournementCompostResponse])
async def get_turnings(
    compost_id: int,
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir l'historique des retournements
    """
    turnings = await compost_service.get_turnings(
        db, compost_id, skip=pagination.skip, limit=pagination.limit
    )
    
    all_turnings = await compost_service.get_turnings(db, compost_id, limit=10000)
    
    return PaginatedResponse.create(
        items=turnings,
        total=len(all_turnings),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.post("/{compost_id}/retournements", response_model=RetournementCompostResponse)
async def add_turning(
    compost_id: int,
    turning_data: RetournementCompostCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Ajouter un retournement de compost
    """
    turning_data.compost_id = compost_id
    turning, error = await compost_service.add_turning(db, turning_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return turning


@router.post("/{compost_id}/mature")
async def mark_as_mature(
    compost_id: int,
    volume_final: Optional[float] = Query(None, description="Volume final en m³"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Marquer un compost comme mature
    """
    success, message = await compost_service.mark_as_mature(db, compost_id, current_user.id, volume_final)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    return {"message": message, "compost_id": compost_id}


@router.get("/stats/global")
async def get_compost_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir des statistiques globales sur les composts
    """
    stats = await compost_service.get_compost_stats(db)
    return stats