# backend/app/api/v1/endpoints/pesees.py
"""
Routes de gestion des pesées
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...schemas.pesee import PeseeCreate, PeseeUpdate, PeseeResponse
from ...services.pesee_service import pesee_service
from ...api.dependencies.auth import get_current_user, get_current_technicien_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/pesees", tags=["Pesées"])


@router.get("/", response_model=PaginatedResponse[PeseeResponse])
async def get_pesees(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    animal_id: Optional[int] = Query(None, description="Filtrer par ID de l'animal"),
    lot_entomo_id: Optional[int] = Query(None, description="Filtrer par ID du lot entomoculture"),
    lot_avicole_id: Optional[int] = Query(None, description="Filtrer par ID du lot avicole"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des pesées avec filtres
    """
    pesees = await pesee_service.get_pesees(
        db,
        animal_id=animal_id,
        lot_entomo_id=lot_entomo_id,
        lot_avicole_id=lot_avicole_id,
        skip=pagination.skip,
        limit=pagination.limit
    )
    
    # Compter le total
    total = await pesee_service.count_pesees(
        db,
        animal_id=animal_id,
        lot_entomo_id=lot_entomo_id,
        lot_avicole_id=lot_avicole_id
    )
    
    return PaginatedResponse.create(
        items=pesees,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{pesee_id}", response_model=PeseeResponse)
async def get_pesee(
    pesee_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir une pesée par son ID
    """
    pesee = await pesee_service.get_pesee(db, pesee_id)
    if not pesee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pesée non trouvée"
        )
    return pesee


@router.post("/create", response_model=PeseeResponse, status_code=status.HTTP_201_CREATED)
async def create_pesee(
    pesee_data: PeseeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Créer une nouvelle pesée (technicien requis)
    """
    # Vérifier qu'au moins un ID est fourni
    if not any([pesee_data.animal_id, pesee_data.lot_entomo_id, pesee_data.lot_avicole_id]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Au moins un ID (animal, lot entomoculture ou lot avicole) est requis"
        )
    
    pesee = await pesee_service.create_pesee(db, pesee_data, current_user.id)
    return pesee


@router.put("/update/{pesee_id}", response_model=PeseeResponse)
async def update_pesee(
    pesee_id: int,
    pesee_data: PeseeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Mettre à jour une pesée (technicien requis)
    """
    pesee, error = await pesee_service.update_pesee(db, pesee_id, pesee_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error
        )
    return pesee


@router.get("/animal/{animal_id}/growth-curve", response_model=List[dict])
async def get_growth_curve(
    animal_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la courbe de croissance d'un animal
    """
    curve = await pesee_service.get_growth_curve(db, animal_id)
    return curve


@router.get("/animal/{animal_id}/last", response_model=PeseeResponse)
async def get_last_pesee(
    animal_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la dernière pesée d'un animal
    """
    pesee = await pesee_service.get_last_pesee(db, animal_id)
    if not pesee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucune pesée trouvée pour cet animal"
        )
    return pesee