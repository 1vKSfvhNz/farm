# backend/app/api/v1/odoni.py
"""
Routes de gestion des nuisibles (odoni)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import datetime, date

from ...database import get_db
from ...schemas.odoni import PiegeOdoniCreate, PiegeOdoniUpdate, PiegeOdoniResponse, ComptageOdoniCreate, ComptageOdoniResponse
from ...services.odoni_service import odoni_service
from ...api.dependencies.auth import get_current_user, get_current_technicien_user, get_current_admin_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/odoni", tags=["Nuisibles (Odoni)"])


@router.get("/pieges", response_model=PaginatedResponse[PiegeOdoniResponse])
async def get_pieges(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    compost_id: Optional[int] = Query(None, description="Filtrer par compost"),
    is_active: Optional[bool] = Query(None, description="Filtrer par statut actif"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des pièges à nuisibles
    """
    pieges = await odoni_service.get_pieges(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        enclos_id=enclos_id,
        compost_id=compost_id,
        is_active=is_active
    )
    
    all_pieges = await odoni_service.get_pieges(db, limit=10000)
    
    return PaginatedResponse.create(
        items=pieges,
        total=len(all_pieges),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/pieges/{piege_id}", response_model=PiegeOdoniResponse)
async def get_piege(
    piege_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir un piège par son ID
    """
    piege = await odoni_service.get_piege(db, piege_id)
    if not piege:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Piège non trouvé"
        )
    return piege


@router.post("/pieges", response_model=PiegeOdoniResponse, status_code=status.HTTP_201_CREATED)
async def create_piege(
    piege_data: PiegeOdoniCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Créer un nouveau piège
    """
    piege, error = await odoni_service.create_piege(db, piege_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return piege


@router.put("/pieges/{piege_id}", response_model=PiegeOdoniResponse)
async def update_piege(
    piege_id: int,
    piege_data: PiegeOdoniUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Mettre à jour un piège
    """
    piege, error = await odoni_service.update_piege(db, piege_id, piege_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return piege


@router.delete("/pieges/{piege_id}")
async def delete_piege(
    piege_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Supprimer un piège (admin uniquement)
    """
    success = await odoni_service.delete_piege(db, piege_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Piège non trouvé"
        )
    return {"message": "Piège supprimé avec succès"}


@router.get("/pieges/{piege_id}/comptages", response_model=PaginatedResponse[ComptageOdoniResponse])
async def get_comptages(
    piege_id: int,
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir l'historique des comptages d'un piège
    """
    comptages = await odoni_service.get_comptages(
        db,
        piege_id,
        skip=pagination.skip,
        limit=pagination.limit,
        start_date=start_date,
        end_date=end_date
    )
    
    return PaginatedResponse.create(
        items=comptages,
        total=len(comptages),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.post("/pieges/{piege_id}/comptages", response_model=ComptageOdoniResponse)
async def add_comptage(
    piege_id: int,
    comptage_data: ComptageOdoniCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_technicien_user),
):
    """
    Ajouter un comptage de nuisibles
    """
    piege = await odoni_service.get_piege(db, piege_id)
    if not piege:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Piège non trouvé"
        )
    
    comptage_data.piege_id = piege_id
    comptage = await odoni_service.add_comptage(db, comptage_data, current_user.id)
    return comptage


@router.get("/dashboard/infestation")
async def get_infestation_level(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le niveau d'infestation actuel
    """
    level = await odoni_service.get_current_infestation_level(db, enclos_id)
    return level


@router.get("/dashboard/evolution")
async def get_infestation_evolution(
    days: int = Query(30, description="Nombre de jours"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir l'évolution de l'infestation
    """
    evolution = await odoni_service.get_infestation_evolution(db, days, enclos_id)
    return {
        "period_days": days,
        "data": evolution,
        "total_count": sum(e["count"] for e in evolution),
        "average_per_day": round(sum(e["count"] for e in evolution) / days, 1) if days > 0 else 0
    }


@router.get("/alerts/threshold")
async def check_thresholds(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Vérifier si les seuils d'alerte sont dépassés
    """
    alerts = await odoni_service.check_thresholds(db, enclos_id)
    
    return {
        "alerts": alerts,
        "critical_count": len([a for a in alerts if a["severity"] == "critical"]),
        "warning_count": len([a for a in alerts if a["severity"] == "warning"])
    }


@router.get("/pieges/{piege_id}/statistiques")
async def get_piege_statistics(
    piege_id: int,
    days: int = Query(30, description="Nombre de jours"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les statistiques détaillées d'un piège
    """
    stats = await odoni_service.get_piege_statistics(db, piege_id, days)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Piège non trouvé ou aucune donnée"
        )
    
    return stats


@router.get("/species/repartition")
async def get_species_repartition(
    days: int = Query(30, description="Nombre de jours"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la répartition par espèce de nuisibles
    """
    repartition = await odoni_service.get_species_repartition(db, days, enclos_id)
    
    return {
        "species": repartition,
        "total": sum(repartition.values()),
        "most_common": max(repartition, key=repartition.get) if repartition else None
    }


@router.post("/alerts/generate")
async def generate_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Générer des alertes basées sur les seuils (admin uniquement)
    """
    count = await odoni_service.generate_alerts(db)
    
    return {
        "message": f"{count} alertes générées",
        "alert_count": count
    }


@router.get("/recommendations")
async def get_treatment_recommendations(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir des recommandations de traitement
    """
    recommendations = await odoni_service.get_treatment_recommendations(db, enclos_id)
    
    return {
        "recommendations": recommendations,
        "has_critical": any(r["urgency"] == "critical" for r in recommendations)
    }