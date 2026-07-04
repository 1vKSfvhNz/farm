# backend/app/api/v1/enclos.py
"""
Routes de gestion des enclos
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ...database import get_db
from ...schemas.enclos import EnclosCreate, EnclosUpdate, EnclosResponse, EnclosLength
from ...services.enclos_service import enclos_service
from ...api.dependencies.auth import get_current_responsable_enclos_user, get_current_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User
from ...models.enclos import Enclos
from ...core import logger

router = APIRouter(prefix="/enclos", tags=["Enclos"])

@router.get("", response_model=PaginatedResponse[EnclosResponse])
async def get_enclos_list(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    enclos_type: Optional[list[str]] = Query(None, description="Filtrer par type"),
    zone: Optional[str] = Query(None, description="Filtrer par zone"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des enclos
    """
    # Construire la requête de base
    query = select(Enclos)
    
    # Appliquer les filtres
    if enclos_type:
        query = query.where(Enclos.type.in_(enclos_type))
    if zone:
        query = query.where(Enclos.zone == zone)
    
    # Récupérer le total
    count_query = select(func.count()).select_from(Enclos)
    if enclos_type:
        count_query = count_query.where(Enclos.type.in_(enclos_type))
    if zone:
        count_query = count_query.where(Enclos.zone == zone)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()
    
    # Appliquer la pagination et exécuter
    query = query.offset(pagination.skip).limit(pagination.limit)
    result = await db.execute(query)
    enclos_list = result.scalars().all()
    
    # Pour chaque enclos, calculer l'occupation actuelle
    enclos_responses = []
    for enclos in enclos_list:        
        # Créer la réponse avec les champs calculés
        enclos_response = EnclosResponse(
            id=enclos.id,
            name=enclos.name,
            type=enclos.type,
            longueur=enclos.longueur,
            largeur=enclos.largeur,
            hauteur=enclos.hauteur,
            localisation_gps=enclos.localisation_gps,
            zone=enclos.zone,
            description=enclos.description,
            surface=enclos.surface,
            volume=enclos.volume,
            created_at=enclos.created_at,
            updated_at=enclos.updated_at
        )
        enclos_responses.append(enclos_response)
    
    return PaginatedResponse.create(
        items=enclos_responses,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )

@router.get("/length", response_model=EnclosLength)
async def get_enclos_stats(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_responsable_enclos_user)
):
    try:
        enclos_list = await enclos_service.get_enclos_list(db, limit=150)    
        return EnclosLength(length=len(enclos_list))
    
    except Exception as e:
        logger.error(f"Erreur dans get_enclos_stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des enclos: {str(e)}"
        )

@router.post("/create", response_model=EnclosResponse, status_code=status.HTTP_201_CREATED)
async def create_enclos(
    enclos_data: EnclosCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Créer un nouvel enclos
    """
    enclos, error = await enclos_service.create_enclos(db, enclos_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return enclos


@router.get("/{enclos_id}", response_model=EnclosResponse)
async def get_enclos_by_id(
    enclos_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Obtenir un enclos par son ID
    """
    enclos = await enclos_service.get_enclos(db, enclos_id)
    if not enclos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enclos non trouvé"
        )
    return enclos

@router.put("/update/{enclos_id}", response_model=EnclosResponse)
async def update_enclos(
    enclos_id: int,
    enclos_data: EnclosUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Mettre à jour un enclos
    """
    enclos, error = await enclos_service.update_enclos(db, enclos_id, enclos_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return enclos
