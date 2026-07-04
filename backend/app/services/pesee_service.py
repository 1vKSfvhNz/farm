# backend/app/services/pesee_service.py
"""
Service de gestion des pesées
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.pesee import Pesee
from ..models.animal import Animal
from ..schemas.pesee import *

logger = logging.getLogger(__name__)


class PeseeService:
    """Service de gestion des pesées"""
    
    async def create_pesee(
        self,
        db: AsyncSession,
        pesee_data: PeseeCreate,
        created_by: int
    ) -> Pesee:
        """Créer une pesée"""
        pesee = Pesee(
            animal_id=pesee_data.animal_id,
            lot_entomo_id=pesee_data.lot_entomo_id,
            lot_avicole_id=pesee_data.lot_avicole_id,
            date_pesee=pesee_data.date_pesee,
            poids=pesee_data.poids,
            methode=pesee_data.methode,
            video_record_id=pesee_data.video_record_id,
            notes=pesee_data.notes
        )
        db.add(pesee)
        await db.commit()
        await db.refresh(pesee)
        
        logger.info(f"Pesee created for animal {pesee_data.animal_id}: {pesee_data.poids}kg by {created_by}")
        return pesee
    
    async def get_last_pesee(
        self,
        db: AsyncSession,
        animal_id: int
    ) -> Optional[Pesee]:
        """Obtenir la dernière pesée d'un animal"""
        stmt = select(Pesee).where(
            Pesee.animal_id == animal_id
        ).order_by(Pesee.date_pesee.desc()).limit(1)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_pesees(
        self,
        db: AsyncSession,
        animal_id: Optional[int] = None,
        lot_entomo_id: Optional[int] = None,
        lot_avicole_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Pesee]:
        """Obtenir la liste des pesées avec filtres"""
        stmt = select(Pesee)
        
        if animal_id is not None:
            stmt = stmt.where(Pesee.animal_id == animal_id)
        if lot_entomo_id is not None:
            stmt = stmt.where(Pesee.lot_entomo_id == lot_entomo_id)
        if lot_avicole_id is not None:
            stmt = stmt.where(Pesee.lot_avicole_id == lot_avicole_id)
        
        stmt = stmt.order_by(Pesee.date_pesee.asc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def count_pesees(
        self,
        db: AsyncSession,
        animal_id: Optional[int] = None,
        lot_entomo_id: Optional[int] = None,
        lot_avicole_id: Optional[int] = None
    ) -> int:
        """Compter le nombre total de pesées avec filtres"""
        stmt = select(func.count(Pesee.id))
        
        if animal_id is not None:
            stmt = stmt.where(Pesee.animal_id == animal_id)
        if lot_entomo_id is not None:
            stmt = stmt.where(Pesee.lot_entomo_id == lot_entomo_id)
        if lot_avicole_id is not None:
            stmt = stmt.where(Pesee.lot_avicole_id == lot_avicole_id)
        
        result = await db.execute(stmt)
        return result.scalar() or 0
    
    async def get_pesee(
        self,
        db: AsyncSession,
        pesee_id: int
    ) -> Optional[Pesee]:
        """Obtenir une pesée par son ID"""
        stmt = select(Pesee).where(Pesee.id == pesee_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update_pesee(
        self,
        db: AsyncSession,
        pesee_id: int,
        pesee_data: PeseeUpdate,
        updated_by: int
    ) -> Tuple[Optional[Pesee], Optional[str]]:
        """Mettre à jour une pesée"""
        pesee = await self.get_pesee(db, pesee_id)
        if not pesee:
            return None, "Pesée non trouvée"
        
        update_data = pesee_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(pesee, field, value)
        
        await db.commit()
        await db.refresh(pesee)
        
        logger.info(f"Pesee updated: {pesee_id} by {updated_by}")
        return pesee, None
    
    async def delete_pesee(
        self,
        db: AsyncSession,
        pesee_id: int,
        deleted_by: int
    ) -> bool:
        """Supprimer une pesée"""
        pesee = await self.get_pesee(db, pesee_id)
        if not pesee:
            return False
        
        await db.delete(pesee)
        await db.commit()
        
        logger.info(f"Pesee deleted: {pesee_id} by {deleted_by}")
        return True
    
    async def get_growth_curve(
        self,
        db: AsyncSession,
        animal_id: int
    ) -> List[Dict[str, Any]]:
        """Obtenir la courbe de croissance d'un animal"""
        pesees = await self.get_pesees(db, animal_id=animal_id, limit=100)
        
        # Récupérer l'animal pour calculer l'âge
        stmt = select(Animal).where(Animal.id == animal_id)
        result = await db.execute(stmt)
        animal = result.scalar_one_or_none()
        
        curve = []
        for i, p in enumerate(pesees):
            point = {
                "date": p.date_pesee.isoformat(),
                "poids_kg": p.poids,
                "methode": p.methode
            }
            
            if animal and animal.date_naissance:
                point["age_jours"] = (p.date_pesee - animal.date_naissance).days
            
            if i > 0:
                prev = pesees[i-1]
                days = (p.date_pesee - prev.date_pesee).days
                if days > 0:
                    point["gain_journalier"] = round((p.poids - prev.poids) / days, 2)
            
            curve.append(point)
        
        # Trier par date croissante
        curve.sort(key=lambda x: x["date"])
        
        return curve


pesee_service = PeseeService()