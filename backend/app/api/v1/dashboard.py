# backend/app/api/v1/dashboard.py
"""
Routes du tableau de bord
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
from datetime import date

from ...database import get_db
from ...services.dashboard_service import dashboard_service
from ...api.dependencies.auth import get_current_user
from ...models.user import User

router = APIRouter(prefix="/dashboard", tags=["Tableau de bord"])


@router.get("/")
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les indicateurs du tableau de bord principal
    """
    dashboard_data = await dashboard_service.get_main_dashboard(db)
    return dashboard_data

@router.get("/recent-activities")
async def get_recent_activities(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Obtenir les activités récentes pour le dashboard
    """
    activities = await dashboard_service.get_recent_activities(
        db,
        limit=limit,
        user_id=current_user.id
    )
    return activities


@router.get("/animals")
async def get_animals_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le résumé des animaux par espèce
    """
    summary = await dashboard_service.get_animals_summary(db)
    return summary


@router.get("/production")
async def get_production_summary(
    days: int = Query(30, description="Nombre de jours"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le résumé de production
    """
    summary = await dashboard_service.get_production_summary(db, days)
    return summary


@router.get("/financial")
async def get_financial_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le résumé financier
    """
    summary = await dashboard_service.get_financial_summary(db)
    return summary


@router.get("/alerts")
async def get_recent_alerts(
    limit: int = Query(10, description="Nombre d'alertes"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les alertes récentes
    """
    alerts = await dashboard_service.get_recent_alerts(db, limit)
    return {"alerts": alerts}


@router.get("/health")
async def get_health_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le statut de santé global
    """
    health = await dashboard_service.get_health_status(db)
    return health


@router.get("/water-quality")
async def get_water_quality_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le résumé de la qualité de l'eau
    """
    summary = await dashboard_service.get_water_quality_summary(db)
    return summary


@router.get("/compost")
async def get_compost_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le résumé du compostage
    """
    summary = await dashboard_service.get_compost_summary(db)
    return summary