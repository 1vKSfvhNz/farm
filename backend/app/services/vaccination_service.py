# backend/app/services/vaccination_service.py
"""
Service de gestion des vaccinations
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..models.animal import Animal

from ..models.vaccination import Vaccination, Maladie, Vaccin
from ..schemas.vaccination import *
from ..services.notification_service import notification_service

logger = logging.getLogger(__name__)


class VaccinationService:
    """Service de gestion des vaccinations"""
    
    async def create_vaccination(
        self,
        db: AsyncSession,
        vacc_data: VaccinationCreate,
        created_by: int
    ) -> Vaccination:
        """Créer une vaccination"""
        vaccination = Vaccination(
            animal_id=vacc_data.animal_id,
            maladie_id=vacc_data.maladie_id,
            vaccin_id=vacc_data.vaccin_id,
            date_prevue=vacc_data.date_prevue,
            date_realisee=vacc_data.date_realisee,
            dose=vacc_data.dose,
            rappel_necessaire=vacc_data.rappel_necessaire,
            date_prochain_rappel=vacc_data.date_prochain_rappel,
            veterinaire_responsable=vacc_data.veterinaire_responsable,
            cout=vacc_data.cout,
            notes=vacc_data.notes
        )
        db.add(vaccination)
        await db.commit()
        
        logger.info(f"Vaccination created for animal {vacc_data.animal_id}")
        return vaccination
    
    async def get_upcoming_vaccinations(
        self,
        db: AsyncSession,
        days_ahead: int = 7
    ) -> List[Vaccination]:
        """Obtenir les vaccinations à venir"""
        today = date.today()
        deadline = today + timedelta(days=days_ahead)
        
        stmt = select(Vaccination).where(
            Vaccination.date_realisee.is_(None),
            Vaccination.date_prevue >= today,
            Vaccination.date_prevue <= deadline
        ).order_by(Vaccination.date_prevue)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_overdue_vaccinations(
        self,
        db: AsyncSession
    ) -> List[Vaccination]:
        """Obtenir les vaccinations en retard"""
        stmt = select(Vaccination).where(
            Vaccination.date_realisee.is_(None),
            Vaccination.date_prevue < date.today()
        )
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def check_vaccination_status(
        self,
        db: AsyncSession,
        animal_id: int
    ) -> dict[str, any]:
        """Vérifier le statut vaccinal d'un animal"""
        stmt = select(Vaccination).where(Vaccination.animal_id == animal_id)
        result = await db.execute(stmt)
        vaccinations = result.scalars().all()
        
        total = len(vaccinations)
        completed = len([v for v in vaccinations if v.date_realisee])
        pending = total - completed
        overdue = len([v for v in vaccinations if not v.date_realisee and v.date_prevue < date.today()])
        
        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "overdue": overdue,
            "is_up_to_date": overdue == 0
        }

    # backend/app/services/vaccination_service.py
# Ajoutez ces méthodes à la classe VaccinationService

    async def get_vaccination(
        self,
        db: AsyncSession,
        vaccination_id: int
    ) -> Optional[Vaccination]:
        """Obtenir une vaccination par son ID"""
        stmt = select(Vaccination).where(Vaccination.id == vaccination_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_vaccinations_by_animal(
        self,
        db: AsyncSession,
        animal_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[Vaccination]:
        """Obtenir toutes les vaccinations d'un animal"""
        stmt = select(Vaccination).where(
            Vaccination.animal_id == animal_id
        ).order_by(Vaccination.date_prevue.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_vaccination(
        self,
        db: AsyncSession,
        vaccination_id: int,
        vacc_data: VaccinationUpdate,
        updated_by: int
    ) -> Tuple[Optional[Vaccination], Optional[str]]:
        """Mettre à jour une vaccination"""
        vaccination = await self.get_vaccination(db, vaccination_id)
        if not vaccination:
            return None, "Vaccination non trouvée"
        
        if vacc_data.date_realisee is not None:
            vaccination.date_realisee = vacc_data.date_realisee
        if vacc_data.dose is not None:
            vaccination.dose = vacc_data.dose
        if vacc_data.notes is not None:
            vaccination.notes = vacc_data.notes
        if vacc_data.veterinaire_responsable is not None:
            vaccination.veterinaire_responsable = vacc_data.veterinaire_responsable
        if vacc_data.cout is not None:
            vaccination.cout = vacc_data.cout
        
        await db.commit()
        logger.info(f"Vaccination updated: {vaccination_id} by {updated_by}")
        return vaccination, None
    
    async def delete_vaccination(
        self,
        db: AsyncSession,
        vaccination_id: int,
        deleted_by: int
    ) -> bool:
        """Supprimer une vaccination"""
        vaccination = await self.get_vaccination(db, vaccination_id)
        if not vaccination:
            return False
        
        await db.delete(vaccination)
        await db.commit()
        logger.info(f"Vaccination deleted: {vaccination_id} by {deleted_by}")
        return True
    
    async def get_vaccination_stats(
        self,
        db: AsyncSession,
        espece: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques des vaccinations"""
        stmt = select(Vaccination)
        if espece:
            stmt = stmt.join(Animal).where(Animal.type_espece == espece)
        
        result = await db.execute(stmt)
        vaccinations = result.scalars().all()
        
        total = len(vaccinations)
        realisees = len([v for v in vaccinations if v.date_realisee])
        en_retard = len([v for v in vaccinations if not v.date_realisee and v.date_prevue < date.today()])
        a_venir = len([v for v in vaccinations if not v.date_realisee and v.date_prevue >= date.today()])
        
        # Coût total
        cout_total = sum(v.cout or 0 for v in vaccinations if v.date_realisee)
        
        return {
            "total_vaccinations": total,
            "realisees": realisees,
            "en_retard": en_retard,
            "a_venir": a_venir,
            "taux_realisation": round(realisees / total * 100, 1) if total > 0 else 0,
            "cout_total": round(cout_total, 2)
        }
    
vaccination_service = VaccinationService()