# backend/app/api/v1/entomoculture.py
"""
Routes de gestion de l'entomoculture (insectes)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...schemas.entomoculture import (
    EntomocultureLotCreate, EntomocultureLotUpdate, EntomocultureLotResponse,
    EntomocultureCycleCreate, EntomocultureCycleResponse
)
from ...services.entomoculture_service import entomoculture_service
from ...api.dependencies.auth import get_current_user, get_current_technicien_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/entomoculture", tags=["Entomoculture"])


@router.get("/lots", response_model=PaginatedResponse[EntomocultureLotResponse])
async def get_lots(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    espece: Optional[str] = Query(None, description="Filtrer par espèce"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des lots d'insectes
    """
    lots = await entomoculture_service.get_lots(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        espece=espece,
        enclos_id=enclos_id
    )
    
    all_lots = await entomoculture_service.get_lots(db, limit=10000)
    
    return PaginatedResponse.create(
        items=lots,
        total=len(all_lots),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/lots/{lot_id}", response_model=EntomocultureLotResponse)
async def get_lot(
    lot_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir un lot par son ID
    """
    lot = await entomoculture_service.get_lot(db, lot_id)
    if not lot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lot non trouvé"
        )
    return lot


@router.post("/lots", response_model=EntomocultureLotResponse, status_code=status.HTTP_201_CREATED)
async def create_lot(
    lot_data: EntomocultureLotCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Créer un nouveau lot d'insectes
    """
    lot, error = await entomoculture_service.create_lot(db, lot_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return lot


@router.put("/lots/{lot_id}", response_model=EntomocultureLotResponse)
async def update_lot(
    lot_id: int,
    lot_data: EntomocultureLotUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Mettre à jour un lot
    """
    lot, error = await entomoculture_service.update_lot(db, lot_id, lot_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return lot


@router.delete("/lots/{lot_id}")
async def delete_lot(
    lot_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
    soft_delete: bool = True,
):
    """
    Supprimer un lot
    """
    success = await entomoculture_service.delete_lot(db, lot_id, current_user.id, soft_delete)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lot non trouvé"
        )
    return {"message": "Lot supprimé avec succès"}


@router.get("/lots/{lot_id}/cycles", response_model=List[EntomocultureCycleResponse])
async def get_cycles(
    lot_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les cycles d'un lot
    """
    lot = await entomoculture_service.get_lot(db, lot_id)
    if not lot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lot non trouvé"
        )
    return lot.cycles


@router.post("/cycles", response_model=EntomocultureCycleResponse, status_code=status.HTTP_201_CREATED)
async def add_cycle(
    cycle_data: EntomocultureCycleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Ajouter un cycle pour un lot
    """
    cycle, error = await entomoculture_service.add_cycle(db, cycle_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return cycle


@router.get("/stats/global")
async def get_entomoculture_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir des statistiques globales sur l'entomoculture
    """
    stats = await entomoculture_service.get_stats(db)
    return stats