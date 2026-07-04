# backend/app/api/v1/vaccination.py
"""
Routes de gestion des vaccinations
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import date

from ...database import get_db
from ...schemas.vaccination import (
    VaccinationCreate, VaccinationUpdate, VaccinationResponse,
    MaladieCreate, MaladieResponse, VaccinCreate, VaccinResponse
)
from ...services.vaccination_service import vaccination_service
from ...api.dependencies.auth import get_current_user, get_current_veterinaire_user, get_current_admin_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/vaccination", tags=["Vaccinations"])


@router.get("/", response_model=PaginatedResponse[VaccinationResponse])
async def get_vaccinations(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    animal_id: Optional[int] = Query(None, description="Filtrer par animal"),
    maladie_id: Optional[int] = Query(None, description="Filtrer par maladie"),
    realisee: Optional[bool] = Query(None, description="Filtrer par statut de réalisation"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des vaccinations
    """
    vaccinations = await vaccination_service.get_vaccinations(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        animal_id=animal_id,
        maladie_id=maladie_id,
        realisee=realisee
    )
    
    return PaginatedResponse.create(
        items=vaccinations,
        total=len(vaccinations),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/upcoming")
async def get_upcoming_vaccinations(
    days_ahead: int = Query(7, description="Nombre de jours à venir"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les vaccinations à venir
    """
    vaccinations = await vaccination_service.get_upcoming_vaccinations(db, days_ahead)
    return {"count": len(vaccinations), "vaccinations": vaccinations}


@router.get("/overdue")
async def get_overdue_vaccinations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les vaccinations en retard
    """
    vaccinations = await vaccination_service.get_overdue_vaccinations(db)
    return {"count": len(vaccinations), "vaccinations": vaccinations}


@router.get("/{vaccination_id}", response_model=VaccinationResponse)
async def get_vaccination(
    vaccination_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir une vaccination par son ID
    """
    vaccination = await vaccination_service.get_vaccination(db, vaccination_id)
    if not vaccination:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vaccination non trouvée"
        )
    return vaccination


@router.post("/", response_model=VaccinationResponse, status_code=status.HTTP_201_CREATED)
async def create_vaccination(
    vaccination_data: VaccinationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_veterinaire_user),
):
    """
    Créer une vaccination
    """
    vaccination, error = await vaccination_service.create_vaccination(db, vaccination_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return vaccination


@router.put("/{vaccination_id}", response_model=VaccinationResponse)
async def update_vaccination(
    vaccination_id: int,
    vaccination_data: VaccinationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_veterinaire_user),
):
    """
    Mettre à jour une vaccination
    """
    vaccination, error = await vaccination_service.update_vaccination(db, vaccination_id, vaccination_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return vaccination


@router.post("/{vaccination_id}/realize")
async def realize_vaccination(
    vaccination_id: int,
    date_realisee: Optional[date] = Query(None, description="Date de réalisation"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_veterinaire_user),
):
    """
    Marquer une vaccination comme réalisée
    """
    vaccination, error = await vaccination_service.realize_vaccination(db, vaccination_id, date_realisee, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return {"message": "Vaccination marquée comme réalisée"}


@router.delete("/{vaccination_id}")
async def delete_vaccination(
    vaccination_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Supprimer une vaccination (admin uniquement)
    """
    success = await vaccination_service.delete_vaccination(db, vaccination_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vaccination non trouvée"
        )
    return {"message": "Vaccination supprimée avec succès"}


# Maladies
@router.get("/maladies", response_model=List[MaladieResponse])
async def get_maladies(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des maladies
    """
    maladies = await vaccination_service.get_maladies(db)
    return maladies


@router.post("/maladies", response_model=MaladieResponse, status_code=status.HTTP_201_CREATED)
async def create_maladie(
    maladie_data: MaladieCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_veterinaire_user),
):
    """
    Créer une maladie
    """
    maladie, error = await vaccination_service.create_maladie(db, maladie_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return maladie


# Vaccins
@router.get("/vaccins", response_model=List[VaccinResponse])
async def get_vaccins(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des vaccins
    """
    vaccins = await vaccination_service.get_vaccins(db)
    return vaccins


@router.post("/vaccins", response_model=VaccinResponse, status_code=status.HTTP_201_CREATED)
async def create_vaccin(
    vaccin_data: VaccinCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_veterinaire_user),
):
    """
    Créer un vaccin
    """
    vaccin, error = await vaccination_service.create_vaccin(db, vaccin_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return vaccin