# backend/app/schemas/avicole.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
from .animal import AnimalBase


class AvicoleBase(AnimalBase):
    production_viande: bool = False
    production_reproduction: bool = False
    production_ponte: bool = False
    oeufs_pondus_total: int = 0
    poids_oeufs_total: float = 0


class AvicoleCreate(AvicoleBase):
    pass


class AvicoleUpdate(BaseModel):
    production_viande: Optional[bool] = None
    production_reproduction: Optional[bool] = None
    production_ponte: Optional[bool] = None
    oeufs_pondus_total: Optional[int] = Field(None, ge=0)
    poids_oeufs_total: Optional[float] = Field(None, ge=0)


class AvicoleResponse(AvicoleBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class EggProductionRecord(BaseModel):
    """Enregistrement d'une production d'œufs"""
    avicole_id: int
    egg_count: int = Field(..., gt=0, description="Nombre d'œufs")
    egg_weight_grams: float = Field(..., gt=0, description="Poids total en grammes")
    date_record: Optional[date] = Field(None, description="Date de la production (défaut aujourd'hui)")


class EggProductionStatsResponse(BaseModel):
    """Statistiques de production d'œufs"""
    moyenne_par_jour: float = Field(..., description="Moyenne d'œufs par jour")
    total_oeufs: int = Field(..., description="Total d'œufs sur la période")
    total_poids_kg: float = Field(..., description="Poids total en kg")
    moyenne_poids_par_oeuf_g: float = Field(..., description="Poids moyen par œuf en grammes")
    nombre_poules_pondeuses: int = Field(..., description="Nombre de poules pondeuses")
    periode_jours: int = Field(..., description="Période en jours")
    date_debut: date
    date_fin: date
    tendance: str = Field(..., description="Tendance: hausse, baisse, stable")
    variation_pourcent: float = Field(..., description="Variation en pourcentage")


class EggProductionHistoryPoint(BaseModel):
    """Point d'historique de production"""
    date: date
    oeufs: int
    poids_kg: float
    moyenne_par_poule: float


class DashboardRecentActivity(BaseModel):
    """Activité récente pour le dashboard"""
    id: int
    type: str  # naissance, mortalite, vaccination, recolte, vente, alerte
    title: str
    description: str
    date: datetime
    entity_id: Optional[int] = None
    entity_type: Optional[str] = None
    severity: Optional[str] = None  # info, warning, critical