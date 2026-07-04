# backend/app/api/v1/avicoles.py
"""
Routes de gestion des avicoles (volailles)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ...database import get_db
from ...schemas.avicole import AvicoleCreate, AvicoleUpdate, AvicoleResponse, EggProductionRecord
from ...services.avicole_service import avicole_service
from ...services.pesee_service import pesee_service
from ...api.dependencies.auth import can_read_avicoles, can_write_avicoles
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User
from ...models.animal import StatutAnimalEnum

router = APIRouter(prefix="/avicoles", tags=["Avicoles"])


@router.get("/", response_model=PaginatedResponse[AvicoleResponse])
async def get_avicoles(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    race: Optional[str] = Query(None, description="Filtrer par race"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    production_type: Optional[str] = Query(None, description="Type de production (viande/ponte/reproduction)"),
    statut: Optional[StatutAnimalEnum] = Query(None, description="Filtrer par statut"),
    current_user: User = Depends(can_read_avicoles),
):
    """
    Obtenir la liste des avicoles
    """
    avicoles = await avicole_service.get_avicoles(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        race=race,
        enclos_id=enclos_id,
        production_type=production_type,
        statut=statut
    )
    
    return PaginatedResponse.create(
        items=avicoles,
        total=len(avicoles),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{avicole_id}", response_model=AvicoleResponse)
async def get_avicole(
    avicole_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_avicoles),
):
    """
    Obtenir une volaille par son ID
    """
    avicole = await avicole_service.get_avicole(db, avicole_id)
    if not avicole:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avicole non trouvé"
        )
    return avicole


@router.post("/", response_model=AvicoleResponse, status_code=status.HTTP_201_CREATED)
async def create_avicole(
    avicole_data: AvicoleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_avicoles),
):
    """
    Créer une nouvelle volaille
    """
    avicole, error = await avicole_service.create_avicole(db, avicole_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return avicole


@router.put("/{avicole_id}", response_model=AvicoleResponse)
async def update_avicole(
    avicole_id: int,
    avicole_data: AvicoleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_avicoles),
):
    """
    Mettre à jour une volaille
    """
    avicole, error = await avicole_service.update_avicole(db, avicole_id, avicole_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return avicole


@router.post("/{avicole_id}/oeufs")
async def add_egg_production(
    avicole_id: int,
    request_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_avicoles),
):
    """
    Ajouter une production d'œufs
    """
    egg_count = request_data.get("egg_count")
    egg_weight_grams = request_data.get("egg_weight_grams")
    
    if not egg_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nombre d'œufs requis"
        )
    
    success = await avicole_service.add_egg_production(
        db, avicole_id, egg_count, egg_weight_grams or 0
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avicole non trouvé"
        )
    
    return {"message": f"Production d'œufs ajoutée: {egg_count} œufs"}


@router.get("/production/oeufs/stats")
async def get_egg_stats(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    days: int = Query(30, description="Nombre de jours"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_avicoles),
):
    """
    Obtenir les statistiques de ponte
    """
    stats = await avicole_service.get_egg_production_stats(db, enclos_id, days)
    return stats


@router.get("/stats/global")
async def get_avicole_stats(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_avicoles),
):
    """
    Obtenir des statistiques globales sur les avicoles
    """
    stats = await avicole_service.get_avicole_stats(db, enclos_id)
    return stats

@router.get("/production/oeufs/stats")
async def get_egg_production_stats(
    days: int = Query(30, ge=1, le=365, description="Nombre de jours"),
    enclos_id: Optional[int] = Query(None, description="ID de l'enclos"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(can_read_avicoles)
):
    """
    Obtenir les statistiques de production d'œufs
    """
    stats = await avicole_service.get_egg_production_stats(
        db, 
        enclos_id=enclos_id, 
        days=days
    )
    return stats


@router.get("/production/oeufs/history")
async def get_egg_production_history(
    days: int = Query(30, ge=1, le=365),
    enclos_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(can_read_avicoles)
):
    """
    Obtenir l'historique de production d'œufs
    """
    history = await avicole_service.get_egg_production_history(
        db,
        enclos_id=enclos_id,
        days=days
    )
    return {
        "history": history,
        "period_days": days
    }


@router.post("/production/oeufs/record")
async def record_egg_production(
    record: EggProductionRecord,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(can_write_avicoles)
):
    """
    Enregistrer une production d'œufs
    """
    result = await avicole_service.add_egg_production(
        db,
        avicole_id=record.avicole_id,
        egg_count=record.egg_count,
        egg_weight_grams=record.egg_weight_grams,
        recorded_by=current_user.id
    )
    
    if not result[0]:
        raise HTTPException(status_code=400, detail=result[1])
    
    return {"success": True, "message": "Production d'œufs enregistrée"}

