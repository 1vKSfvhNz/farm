# backend/app/schemas/alimentation.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime


class AlimentationBase(BaseModel):
    animal_id: Optional[int] = None
    lot_entomo_id: Optional[int] = None
    date: date
    poids_nourriture: float = Field(..., gt=0)
    type_nourriture: str = Field(..., min_length=1, max_length=100)
    composition: Optional[str] = None
    cout: Optional[float] = Field(None, ge=0)


class AlimentationCreate(AlimentationBase):
    pass


class AlimentationUpdate(BaseModel):
    date: Optional[date] = None
    poids_nourriture: Optional[float] = Field(None, gt=0)
    type_nourriture: Optional[str] = Field(None, min_length=1, max_length=100)
    composition: Optional[str] = None
    cout: Optional[float] = Field(None, ge=0)


class AlimentationResponse(AlimentationBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RationAlimentaireBase(BaseModel):
    espece: str = Field(..., min_length=1, max_length=50)
    race: Optional[str] = None
    categorie: str = Field(..., min_length=1, max_length=50)
    proteines_pourcent: Optional[float] = Field(None, ge=0, le=100)
    energie_kcal: Optional[float] = Field(None, ge=0)
    calcium_g: Optional[float] = Field(None, ge=0)
    phosphore_g: Optional[float] = Field(None, ge=0)
    quantite_recommandee_kg: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None


class RationAlimentaireCreate(RationAlimentaireBase):
    pass


class RationAlimentaireResponse(RationAlimentaireBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True