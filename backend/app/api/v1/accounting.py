# backend/app/api/v1/accounting.py
"""
Routes de gestion comptable
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import date

from ...database import get_db
from ...schemas.accounting import DepenseCreate, DepenseUpdate, DepenseResponse, RecetteCreate, RecetteUpdate, RecetteResponse, AccountSummary
from ...services.accounting_service import accounting_service
from ...api.dependencies.auth import get_current_user, get_current_admin_user, get_current_responsable_enclos_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/accounting", tags=["Comptabilité"])


@router.get("/summary", response_model=AccountSummary)
async def get_summary(
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le résumé comptable
    """
    summary = await accounting_service.get_summary(db, start_date, end_date)
    return summary


@router.get("/depenses", response_model=PaginatedResponse[DepenseResponse])
async def get_depenses(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    categorie: Optional[str] = Query(None, description="Filtrer par catégorie"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des dépenses
    """
    depenses = await accounting_service.get_depenses(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        start_date=start_date,
        end_date=end_date,
        categorie=categorie
    )
    
    return PaginatedResponse.create(
        items=depenses,
        total=len(depenses),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/recettes", response_model=PaginatedResponse[RecetteResponse])
async def get_recettes(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    categorie: Optional[str] = Query(None, description="Filtrer par catégorie"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des recettes
    """
    recettes = await accounting_service.get_recettes(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        start_date=start_date,
        end_date=end_date,
        categorie=categorie
    )
    
    return PaginatedResponse.create(
        items=recettes,
        total=len(recettes),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.post("/depenses", response_model=DepenseResponse, status_code=status.HTTP_201_CREATED)
async def create_depense(
    depense_data: DepenseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Créer une dépense
    """
    depense = await accounting_service.create_depense(db, depense_data, current_user.id)
    return depense


@router.post("/recettes", response_model=RecetteResponse, status_code=status.HTTP_201_CREATED)
async def create_recette(
    recette_data: RecetteCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Créer une recette
    """
    recette = await accounting_service.create_recette(db, recette_data, current_user.id)
    return recette


@router.put("/depenses/{depense_id}", response_model=DepenseResponse)
async def update_depense(
    depense_id: int,
    depense_data: DepenseUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Mettre à jour une dépense (admin uniquement)
    """
    depense, error = await accounting_service.update_depense(db, depense_id, depense_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return depense


@router.put("/recettes/{recette_id}", response_model=RecetteResponse)
async def update_recette(
    recette_id: int,
    recette_data: RecetteUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Mettre à jour une recette (admin uniquement)
    """
    recette, error = await accounting_service.update_recette(db, recette_id, recette_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return recette

@router.get("/profitability/species")
async def get_profitability_by_species(
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Obtenir la rentabilité par espèce
    """
    profitability = await accounting_service.get_profitability_by_species(db, start_date, end_date)
    return profitability