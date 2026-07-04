# backend/app/services/alimentation_service.py
"""
Service de gestion de l'alimentation
"""

import logging
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.alimentation import Alimentation
from ..schemas.alimentation import *

logger = logging.getLogger(__name__)


class AlimentationService:
    """Service de gestion de l'alimentation"""
    
    async def create_alimentation(
        self,
        db: AsyncSession,
        alimentation_data: AlimentationCreate,
        created_by: int
    ) -> Alimentation:
        """Créer un enregistrement d'alimentation"""
        alimentation = Alimentation(
            animal_id=alimentation_data.animal_id,
            lot_entomo_id=alimentation_data.lot_entomo_id,
            date=alimentation_data.date,
            poids_nourriture=alimentation_data.poids_nourriture,
            type_nourriture=alimentation_data.type_nourriture,
            composition=alimentation_data.composition,
            cout=alimentation_data.cout
        )
        db.add(alimentation)
        await db.commit()
        
        logger.info(f"Alimentation created: {alimentation_data.poids_nourriture}kg for animal {alimentation_data.animal_id}")
        return alimentation
    
    async def get_total_consumption(
        self,
        db: AsyncSession,
        animal_id: int,
        days: int = 30
    ) -> float:
        """Obtenir la consommation totale sur une période"""
        from datetime import date, timedelta
        start_date = date.today() - timedelta(days=days)
        
        stmt = select(func.sum(Alimentation.poids_nourriture)).where(
            Alimentation.animal_id == animal_id,
            Alimentation.date >= start_date
        )
        result = await db.execute(stmt)
        return result.scalar() or 0.0


alimentation_service = AlimentationService()