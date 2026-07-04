# backend/app/services/odoni_service.py
"""
Service de gestion des nuisibles (odoni)
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.odoni import PiegeOdoni, ComptageOdoni
from ..schemas.odoni import *

logger = logging.getLogger(__name__)


class OdoniService:
    """Service de gestion des nuisibles"""
    
    async def add_count(
        self,
        db: AsyncSession,
        piege_id: int,
        nombre: int,
        espece: Optional[str] = None
    ) -> ComptageOdoni:
        """Ajouter un comptage"""
        comptage = ComptageOdoni(
            piege_id=piege_id,
            nombre=nombre,
            espece=espece,
            methode="manuel"
        )
        db.add(comptage)
        await db.commit()
        
        # Vérifier le seuil d'alerte
        if nombre > 50:
            logger.warning(f"Alerte odoni: {nombre} nuisibles détectés sur piège {piege_id}")
        
        return comptage
    
    async def get_current_infestation_level(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir le niveau d'infestation actuel"""
        stmt = select(PiegeOdoni)
        if enclos_id:
            stmt = stmt.where(PiegeOdoni.enclos_id == enclos_id)
        result = await db.execute(stmt)
        pieges = result.scalars().all()
        
        total_count = 0
        for piege in pieges:
            stmt = select(ComptageOdoni).where(
                ComptageOdoni.piege_id == piege.id
            ).order_by(ComptageOdoni.timestamp.desc()).limit(1)
            result = await db.execute(stmt)
            last_count = result.scalar_one_or_none()
            if last_count:
                total_count += last_count.nombre
        
        if total_count > 100:
            level = "critical"
        elif total_count > 50:
            level = "high"
        elif total_count > 20:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "total_count": total_count,
            "pieges_actifs": len(pieges)
        }


odoni_service = OdoniService()