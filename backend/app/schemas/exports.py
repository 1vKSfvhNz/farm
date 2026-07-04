# backend/app/schemas/exports.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime


class ExportFilter(BaseModel):
    espece: Optional[str] = None
    enclos_id: Optional[int] = None
    date_debut: Optional[date] = None
    date_fin: Optional[date] = None
    animal_ids: Optional[List[int]] = []
    include_financial: bool = True
    include_health: bool = True
    include_growth: bool = True
    format: str = "csv"  # csv, pdf


class ExportResponse(BaseModel):
    export_id: str
    status: str  # pending, processing, completed, failed
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    created_at: datetime
    expires_at: Optional[datetime] = None