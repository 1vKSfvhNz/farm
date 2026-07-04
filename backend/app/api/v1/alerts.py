# backend/app/api/v1/alerts.py
"""
Routes des alertes
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...schemas.alerts import AlertCreate, AlertResponse
from ...services.alert_service import alert_service
from ...api.dependencies.auth import get_current_user, get_current_admin_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/alerts", tags=["Alertes"])


@router.get("/", response_model=PaginatedResponse[AlertResponse])
async def get_alerts(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    niveau: Optional[str] = Query(None, description="Filtrer par niveau (info/warning/critical)"),
    est_lue: Optional[bool] = Query(None, description="Filtrer par statut de lecture"),
    espece: Optional[str] = Query(None, description="Filtrer par espèce"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des alertes
    """
    alerts = await alert_service.get_alerts(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        niveau=niveau,
        est_lue=est_lue,
        espece=espece,
        user_id=current_user.id
    )
    
    return PaginatedResponse.create(
        items=alerts,
        total=len(alerts),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/unread/count")
async def get_unread_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le nombre d'alertes non lues
    """
    count = await alert_service.get_unread_count(db, current_user.id)
    return {"unread_count": count}


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir une alerte par son ID
    """
    alert = await alert_service.get_alert(db, alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alerte non trouvée"
        )
    
    # Vérifier que l'utilisateur a accès à cette alerte
    if alert.utilisateur_id and alert.utilisateur_id != current_user.id:
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Accès non autorisé"
            )
    
    return alert


@router.post("/{alert_id}/read")
async def mark_as_read(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Marquer une alerte comme lue
    """
    success = await alert_service.mark_as_read(db, alert_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alerte non trouvée"
        )
    return {"message": "Alerte marquée comme lue"}


@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    resolution_note: Optional[str] = Query(None, description="Note de résolution"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Résoudre une alerte (admin uniquement)
    """
    success = await alert_service.resolve_alert(db, alert_id, current_user.id, resolution_note)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alerte non trouvée"
        )
    return {"message": "Alerte résolue"}


@router.post("/mark-all-read")
async def mark_all_as_read(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Marquer toutes les alertes comme lues
    """
    count = await alert_service.mark_all_as_read(db, current_user.id)
    return {"message": f"{count} alertes marquées comme lues"}


@router.post("/generate/vaccination")
async def generate_vaccination_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Générer des alertes pour les vaccinations à venir (admin uniquement)
    """
    count = await alert_service.generate_vaccination_alerts(db)
    return {"message": f"{count} alertes de vaccination générées"}


@router.post("/generate/water-quality")
async def generate_water_quality_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Générer des alertes de qualité d'eau (admin uniquement)
    """
    count = await alert_service.generate_water_quality_alerts(db)
    return {"message": f"{count} alertes de qualité d'eau générées"}