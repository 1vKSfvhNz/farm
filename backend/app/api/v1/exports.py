# backend/app/api/v1/exports.py
"""
Routes d'export de données
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import date
import io

from ...database import get_db
from ...schemas.exports import ExportFilter, ExportResponse
from ...services.export_service import export_service
from ...api.dependencies.auth import get_current_user, get_current_responsable_enclos_user
from ...models.user import User

router = APIRouter(prefix="/exports", tags=["Exports"])


@router.get("/animals/csv")
async def export_animals_csv(
    espece: Optional[str] = Query(None, description="Filtrer par espèce"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Exporter la liste des animaux au format CSV
    """
    csv_data = await export_service.export_animals(db, espece, enclos_id)
    
    filename = f"animaux_{date.today()}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/financial/csv")
async def export_financial_csv(
    start_date: date = Query(..., description="Date de début"),
    end_date: date = Query(..., description="Date de fin"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Exporter les données financières au format CSV
    """
    csv_data = await export_service.export_financial(db, start_date, end_date)
    
    filename = f"financial_{start_date}_{end_date}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/pesees/csv")
async def export_weighings_csv(
    espece: Optional[str] = Query(None, description="Filtrer par espèce"),
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Exporter l'historique des pesées au format CSV
    """
    csv_data = await export_service.export_weighings(db, espece, start_date, end_date)
    
    filename = f"pesees_{date.today()}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/mortality/csv")
async def export_mortality_csv(
    espece: Optional[str] = Query(None, description="Filtrer par espèce"),
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Exporter les données de mortalité au format CSV
    """
    csv_data = await export_service.export_mortality(db, espece, start_date, end_date)
    
    filename = f"mortalite_{date.today()}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/vaccinations/csv")
async def export_vaccinations_csv(
    espece: Optional[str] = Query(None, description="Filtrer par espèce"),
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_responsable_enclos_user),
):
    """
    Exporter le calendrier des vaccinations au format CSV
    """
    csv_data = await export_service.export_vaccinations(db, espece, start_date, end_date)
    
    filename = f"vaccinations_{date.today()}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )