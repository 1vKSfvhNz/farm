# backend/app/services/piscicole_service.py
"""
Service de gestion des piscicoles
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.animal import StatutAnimalEnum

from ..models.piscicole import Piscicole
from ..schemas.piscicole import *
from .animal_service import animal_service
from ..core.constants import SEUILS_QUALITE_EAU

logger = logging.getLogger(__name__)


class PiscicoleService:
    """Service de gestion des piscicoles"""
    
    async def create_piscicole(
        self,
        db: AsyncSession,
        piscicole_data: PiscicoleCreate,
        created_by: int
    ) -> Tuple[Optional[Piscicole], Optional[str]]:
        """Créer un nouveau poisson"""
        animal, error = await animal_service.create_animal(
            db, piscicole_data, "piscicole", created_by
        )
        if error:
            return None, error
        
        piscicole = Piscicole(
            id=animal.id,
            production_viande=piscicole_data.production_viande,
            production_reproduction=piscicole_data.production_reproduction,
            taille_moyenne=piscicole_data.taille_moyenne
        )
        db.add(piscicole)
        await db.commit()
        
        logger.info(f"Piscicole created: {piscicole.identification} by {created_by}")
        return piscicole, None
    
    async def get_piscicole(
        self,
        db: AsyncSession,
        piscicole_id: int
    ) -> Optional[Piscicole]:
        """Obtenir un poisson par son ID"""
        stmt = select(Piscicole).where(
            Piscicole.id == piscicole_id,
            Piscicole.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_biomass(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> float:
        """Calculer la biomasse totale dans un bassin"""
        stmt = select(Piscicole).where(
            Piscicole.enclos_id == enclos_id,
            Piscicole.statut == "vivant",
            Piscicole.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        fish = result.scalars().all()
        
        biomass = sum(f.dernier_poids or 0 for f in fish)
        return biomass

    async def get_piscicole_stats(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques des piscicoles"""
        stmt = select(Piscicole).where(
            Piscicole.statut == StatutAnimalEnum.VIVANT
        )
        if enclos_id:
            stmt = stmt.where(Piscicole.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        piscicoles = result.scalars().all()
        
        total = len(piscicoles)
        
        # Biomasse totale
        biomasses = [p.dernier_poids or 0 for p in piscicoles]
        biomasse_totale = sum(biomasses)
        
        # Tailles
        tailles = [p.taille_moyenne for p in piscicoles if p.taille_moyenne]
        taille_moyenne = sum(tailles) / len(tailles) if tailles else 0
        
        # Types de production
        production_viande = len([p for p in piscicoles if p.production_viande])
        production_reproduction = len([p for p in piscicoles if p.production_reproduction])
        
        # Races/espèces
        races = {}
        for p in piscicoles:
            races[p.race] = races.get(p.race, 0) + 1
        
        # Densité (si enclos spécifié)
        densite = None
        if enclos_id:
            from ..models.enclos import Enclos
            enclos = await db.get(Enclos, enclos_id)
            if enclos:
                volume = enclos.volume or (enclos.longueur * enclos.largeur * (enclos.hauteur or 1))
                if volume > 0:
                    densite = round(biomasse_totale / volume, 2)
        
        return {
            "total": total,
            "races": races,
            "biomasse_totale_kg": round(biomasse_totale, 1),
            "poids_moyen_kg": round(biomasse_totale / total, 2) if total > 0 else 0,
            "taille_moyenne_cm": round(taille_moyenne, 1),
            "production": {
                "viande": production_viande,
                "reproduction": production_reproduction
            },
            "densite_kg_m3": densite
        }
    
    async def get_all_piscicoles(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Piscicole]:
        """Obtenir tous les piscicoles"""
        stmt = select(Piscicole).offset(skip).limit(limit).order_by(Piscicole.identification)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_piscicole_by_identification(
        self,
        db: AsyncSession,
        identification: str
    ) -> Optional[Piscicole]:
        """Obtenir un piscicole par son identification"""
        stmt = select(Piscicole).where(
            Piscicole.identification == identification,
            Piscicole.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update_biomass(
        self,
        db: AsyncSession,
        enclos_id: int,
        recorded_by: int
    ) -> Tuple[bool, str]:
        """Mettre à jour la biomasse d'un bassin"""
        # Calculer la biomasse totale
        stmt = select(Piscicole).where(
            Piscicole.enclos_id == enclos_id,
            Piscicole.deleted_at.is_(None),
            Piscicole.statut == StatutAnimalEnum.VIVANT
        )
        result = await db.execute(stmt)
        piscicoles = result.scalars().all()
        
        biomasse = sum(p.dernier_poids or 0 for p in piscicoles)
        
        # Enregistrer dans l'historique (à implémenter)
        logger.info(f"Biomass updated for enclosure {enclos_id}: {biomasse} kg by {recorded_by}")
        return True, f"Biomasse calculée: {round(biomasse, 1)} kg"
    
    async def get_biomass(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> float:
        """Calculer la biomasse totale dans un bassin"""
        stmt = select(Piscicole).where(
            Piscicole.enclos_id == enclos_id,
            Piscicole.statut == StatutAnimalEnum.VIVANT,
            Piscicole.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        fish = result.scalars().all()
        
        biomass = sum(f.dernier_poids or 0 for f in fish)
        return biomass
    
piscicole_service = PiscicoleService()