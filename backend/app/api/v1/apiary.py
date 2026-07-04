# backend/app/api/v1/apiculture.py
"""
Routes de gestion apicole (ruches, miel)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import date, datetime

from ...database import get_db
from ...schemas.apiary import (
    RucheCreate, RucheUpdate, RucheResponse,
    RecolteMielCreate, RecolteMielResponse,
    RucheStatsResponse
)
from ...services.apiary_service import apiary_service
from ..dependencies.auth import can_read_apiculture, can_write_apiculture
from ..dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/apiary", tags=["Apiculture"])

# ============ RUCHES ============

@router.get("/ruches", response_model=PaginatedResponse[RucheResponse])
async def get_ruches(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    statut: Optional[str] = Query(None, description="Filtrer par statut (active/orpheline/en_essaimage/morte)"),
    emplacement: Optional[str] = Query(None, description="Filtrer par emplacement"),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir la liste des ruches
    """
    ruches = await apiary_service.get_ruches(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        statut=statut,
        emplacement=emplacement
    )
    
    # Récupérer le total pour la pagination
    all_ruches = await apiary_service.get_ruches(
        db,
        statut=statut,
        emplacement=emplacement
    )
    
    return PaginatedResponse.create(
        items=ruches,
        total=len(all_ruches),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/ruches/{ruche_id}", response_model=RucheResponse)
async def get_ruche(
    ruche_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir une ruche par son ID
    """
    ruche = await apiary_service.get_ruche(db, ruche_id)
    if not ruche:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ruche non trouvée"
        )
    return ruche


@router.get("/ruches/identification/{identification}", response_model=RucheResponse)
async def get_ruche_by_identification(
    identification: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir une ruche par son identification
    """
    ruche = await apiary_service.get_ruche_by_identification(db, identification)
    if not ruche:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ruche non trouvée"
        )
    return ruche


@router.post("/ruches", response_model=RucheResponse, status_code=status.HTTP_201_CREATED)
async def create_ruche(
    ruche_data: RucheCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_apiculture),
):
    """
    Créer une nouvelle ruche
    """
    ruche, error = await apiary_service.create_ruche(db, ruche_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return ruche


@router.put("/ruches/{ruche_id}", response_model=RucheResponse)
async def update_ruche(
    ruche_id: int,
    ruche_data: RucheUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_apiculture),
):
    """
    Mettre à jour une ruche
    """
    ruche, error = await apiary_service.update_ruche(db, ruche_id, ruche_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return ruche


@router.delete("/ruches/{ruche_id}")
async def delete_ruche(
    ruche_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_apiculture),
    soft_delete: bool = Query(True, description="Suppression logique ou définitive"),
):
    """
    Supprimer une ruche
    """
    success = await apiary_service.delete_ruche(db, ruche_id, current_user.id, soft_delete)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ruche non trouvée"
        )
    return {"message": "Ruche supprimée avec succès"}

# ============ RÉCOLTES DE MIEL ============

@router.get("/ruches/{ruche_id}/recoltes", response_model=PaginatedResponse[RecolteMielResponse])
async def get_recoltes(
    ruche_id: int,
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    year: Optional[int] = Query(None, description="Filtrer par année"),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir l'historique des récoltes de miel d'une ruche
    """
    recoltes = await apiary_service.get_recoltes(
        db,
        ruche_id=ruche_id,
        skip=pagination.skip,
        limit=pagination.limit,
        year=year
    )
    
    all_recoltes = await apiary_service.get_recoltes(
        db,
        ruche_id=ruche_id,
        year=year
    )
    
    return PaginatedResponse.create(
        items=recoltes,
        total=len(all_recoltes),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/recoltes/{recolte_id}", response_model=RecolteMielResponse)
async def get_recolte(
    recolte_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir une récolte par son ID
    """
    recolte = await apiary_service.get_recolte(db, recolte_id)
    if not recolte:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Récolte non trouvée"
        )
    return recolte


@router.post("/ruches/{ruche_id}/recoltes", response_model=RecolteMielResponse, status_code=status.HTTP_201_CREATED)
async def add_recolte(
    ruche_id: int,
    recolte_data: RecolteMielCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_apiculture),
):
    """
    Ajouter une récolte de miel
    """
    # S'assurer que la ruche_id correspond
    recolte_data.ruche_id = ruche_id
    
    recolte, error = await apiary_service.add_recolte(db, recolte_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return recolte


@router.put("/recoltes/{recolte_id}", response_model=RecolteMielResponse)
async def update_recolte(
    recolte_id: int,
    recolte_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_apiculture),
):
    """
    Mettre à jour une récolte
    """
    recolte, error = await apiary_service.update_recolte(db, recolte_id, recolte_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return recolte


# ============ STATISTIQUES ET TABLEAU DE BORD ============

@router.get("/stats/production")
async def get_production_stats(
    year: Optional[int] = Query(None, description="Année"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir les statistiques de production de miel
    """
    if not year:
        year = date.today().year
    
    total_miel = await apiary_service.get_total_honey_production(db, year)
    monthly_production = await apiary_service.get_monthly_production(db, year)
    ruches = await apiary_service.get_ruches(db, limit=10000)
    
    return {
        "year": year,
        "total_honey_kg": round(total_miel, 1),
        "monthly_production": monthly_production,
        "average_per_ruche": round(total_miel / max(1, len(ruches)), 1)
    }


@router.get("/stats/ruches", response_model=RucheStatsResponse)
async def get_ruches_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir les statistiques sur les ruches
    """
    return await apiary_service.get_ruches_stats(db)


@router.get("/stats/reines")
async def get_reines_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir les statistiques sur les reines
    """
    return await apiary_service.get_reines_stats(db)


@router.get("/dashboard")
async def get_apiculture_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir le tableau de bord apicole complet
    """
    ruches_stats = await apiary_service.get_ruches_stats(db)
    production_stats = await apiary_service.get_production_stats(db)
    reines_stats = await apiary_service.get_reines_stats(db)
    
    # Dernières récoltes
    dernieres_recoltes = await apiary_service.get_last_recoltes(db, limit=5)
    
    # Alertes (reines à remplacer, etc.)
    alerts = await apiary_service.get_alerts(db)
    
    return {
        "ruches": ruches_stats,
        "production": production_stats,
        "reines": reines_stats,
        "dernieres_recoltes": dernieres_recoltes,
        "alertes": alerts,
        "derniere_mise_a_jour": datetime.now().isoformat()
    }


@router.get("/alerts")
async def get_apiculture_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir les alertes apicoles
    """
    alerts = await apiary_service.get_alerts(db)
    
    return {
        "alerts": alerts,
        "critical_count": len([a for a in alerts if a.get("severity") == "critical"]),
        "warning_count": len([a for a in alerts if a.get("severity") == "warning"])
    }


@router.get("/production/evolution")
async def get_production_evolution(
    years: int = Query(5, description="Nombre d'années"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir l'évolution de la production sur plusieurs années
    """
    evolution = []
    current_year = date.today().year
    
    for i in range(years):
        year = current_year - i
        total = await apiary_service.get_total_honey_production(db, year)
        evolution.append({
            "year": year,
            "total_honey_kg": round(total, 1)
        })
    
    # Déterminer la tendance
    trend = "stable"
    if len(evolution) >= 2:
        if evolution[0]["total_honey_kg"] > evolution[-1]["total_honey_kg"]:
            trend = "decreasing"
        elif evolution[0]["total_honey_kg"] < evolution[-1]["total_honey_kg"]:
            trend = "increasing"
    
    return {
        "evolution": evolution,
        "trend": trend
    }


# ============ INSPECTIONS ============

@router.post("/ruches/{ruche_id}/inspections")
async def add_inspection(
    ruche_id: int,
    inspection_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_apiculture),
):
    """
    Ajouter une inspection de ruche
    """
    ruche = await apiary_service.get_ruche(db, ruche_id)
    if not ruche:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ruche non trouvée"
        )
    
    inspection = await apiary_service.add_inspection(
        db, ruche_id, inspection_data, current_user.id
    )
    
    return {"message": "Inspection ajoutée", "inspection": inspection}


@router.get("/ruches/{ruche_id}/inspections")
async def get_inspections(
    ruche_id: int,
    limit: int = Query(10, description="Nombre maximum d'inspections"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir l'historique des inspections d'une ruche
    """
    inspections = await apiary_service.get_inspections(db, ruche_id, limit)
    
    return {
        "ruche_id": ruche_id,
        "inspections": inspections,
        "count": len(inspections)
    }


# ============ ESSAIMAGE ============

@router.post("/ruches/{ruche_id}/essaimage")
async def record_swarming(
    ruche_id: int,
    swarming_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_apiculture),
):
    """
    Enregistrer un essaimage
    """
    result = await apiary_service.record_swarming(
        db, ruche_id, swarming_data, current_user.id
    )
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.get("message", "Ruche non trouvée")
        )
    
    return result


@router.get("/ruches/{ruche_id}/historique/essaimages")
async def get_swarming_history(
    ruche_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_apiculture),
):
    """
    Obtenir l'historique des essaimages d'une ruche
    """
    history = await apiary_service.get_swarming_history(db, ruche_id)
    
    return {
        "ruche_id": ruche_id,
        "swarming_events": history,
        "count": len(history)
    }