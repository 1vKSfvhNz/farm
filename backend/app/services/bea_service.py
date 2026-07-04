# backend/app/services/bea_service.py
"""
Service de bien-être animal (BEA)
"""

import logging
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.bien_etre import BienEtreIndice
from ..schemas.bea import *

logger = logging.getLogger(__name__)


class BeaService:
    """Service de bien-être animal"""
    
    async def calculate_daily_index(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> float:
        """Calculer l'indice de bien-être quotidien"""
        # Facteurs à prendre en compte:
        # - Propreté
        # - Accès à l'eau
        # - Densité
        # - Comportement filmé
        
        # Implémentation simplifiée
        base_score = 80.0
        
        # À compléter avec des données réelles
        return base_score
    
    async def create_daily_index(
        self,
        db: AsyncSession,
        enclos_id: int,
        indice: float
    ) -> BienEtreIndice:
        """Créer l'indice de bien-être quotidien"""
        from datetime import date
        
        bea = BienEtreIndice(
            enclos_id=enclos_id,
            date=date.today(),
            indice_global=indice
        )
        db.add(bea)
        await db.commit()
        
        return bea


bea_service = BeaService()