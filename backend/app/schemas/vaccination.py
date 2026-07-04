# backend/app/schemas/vaccination.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime


class MaladieBase(BaseModel):
    nom: str = Field(..., min_length=1, max_length=100)
    especes_concernees: str
    description: Optional[str] = None
    vaccin_disponible: Optional[str] = None


class MaladieCreate(MaladieBase):
    pass


class MaladieResponse(MaladieBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class VaccinBase(BaseModel):
    nom: str = Field(..., min_length=1, max_length=100)
    fabricant: Optional[str] = None
    lot: Optional[str] = None
    maladie_id: int


class VaccinCreate(VaccinBase):
    pass


class VaccinResponse(VaccinBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class VaccinationBase(BaseModel):
    animal_id: int
    maladie_id: int
    vaccin_id: Optional[int] = None
    date_prevue: date
    date_realisee: Optional[date] = None
    dose: Optional[str] = None
    rappel_necessaire: bool = False
    date_prochain_rappel: Optional[date] = None
    veterinaire_responsable: Optional[str] = None
    cout: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None


class VaccinationCreate(VaccinationBase):
    pass


class VaccinationUpdate(BaseModel):
    date_prevue: Optional[date] = None
    date_realisee: Optional[date] = None
    dose: Optional[str] = None
    rappel_necessaire: Optional[bool] = None
    date_prochain_rappel: Optional[date] = None
    veterinaire_responsable: Optional[str] = None
    cout: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None


class VaccinationResponse(VaccinationBase):
    id: int
    est_a_jour: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True