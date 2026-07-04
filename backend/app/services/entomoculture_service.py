# backend/app/services/entomoculture_service.py
"""
Service de gestion de l'entomoculture (insectes)
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.entomoculture import EntomocultureLot, EntomocultureCycle, StadeInsecteEnum
from ..schemas.entomoculture import *

logger = logging.getLogger(__name__)


class EntomocultureService:
    """Service de gestion de l'entomoculture"""
    
    async def create_lot(
        self,
        db: AsyncSession,
        lot_data: EntomocultureLotCreate,
        created_by: int
    ) -> Tuple[Optional[EntomocultureLot], Optional[str]]:
        """Créer un nouveau lot d'insectes"""
        # Vérifier si l'identification existe
        stmt = select(EntomocultureLot).where(EntomocultureLot.identification == lot_data.identification)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            return None, "Cette identification existe déjà"
        
        lot = EntomocultureLot(
            identification=lot_data.identification,
            espece=lot_data.espece,
            stade_actuel=lot_data.stade_actuel,
            date_arrivee=lot_data.date_arrivee,
            provenance=lot_data.provenance,
            prix_achat=lot_data.prix_achat,
            poids_initial=lot_data.poids_initial,
            quantite_estimative=lot_data.quantite_estimative,
            enclos_id=lot_data.enclos_id,
            type_production=lot_data.type_production,
            notes=lot_data.notes
        )
        db.add(lot)
        await db.commit()
        
        logger.info(f"Entomoculture lot created: {lot.identification} by {created_by}")
        return lot, None
    
    async def get_lot(
        self,
        db: AsyncSession,
        lot_id: int
    ) -> Optional[EntomocultureLot]:
        """Obtenir un lot par son ID"""
        stmt = select(EntomocultureLot).where(
            EntomocultureLot.id == lot_id
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def add_cycle(
        self,
        db: AsyncSession,
        cycle_data: EntomocultureCycleCreate
    ) -> Tuple[Optional[EntomocultureCycle], Optional[str]]:
        """Ajouter un cycle pour un lot"""
        lot = await self.get_lot(db, cycle_data.lot_id)
        if not lot:
            return None, "Lot non trouvé"
        
        cycle = EntomocultureCycle(
            lot_id=cycle_data.lot_id,
            date_debut=cycle_data.date_debut,
            date_fin=cycle_data.date_fin,
            stade_debut=cycle_data.stade_debut,
            stade_fin=cycle_data.stade_fin,
            production_grammes=cycle_data.production_grammes,
            taux_mortalite=cycle_data.taux_mortalite,
            substrat_utilise=cycle_data.substrat_utilise
        )
        db.add(cycle)
        
        # Mettre à jour le stade actuel du lot
        if cycle_data.stade_fin:
            lot.stade_actuel = cycle_data.stade_fin
        
        await db.commit()
        
        logger.info(f"Cycle added for lot {lot.identification}")
        return cycle, None

    async def get_stats(
        self,
        db: AsyncSession,
        espece: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques de l'entomoculture"""
        return await self.get_entomoculture_stats(db, espece)
        
    async def get_entomoculture_stats(
        self,
        db: AsyncSession,
        espece: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques de l'entomoculture"""
        stmt = select(EntomocultureLot)
        if espece:
            stmt = stmt.where(EntomocultureLot.espece == espece)
        
        result = await db.execute(stmt)
        lots = result.scalars().all()
        
        total_lots = len(lots)
        
        # Lots actifs (non adultes)
        actifs = len([l for l in lots if l.stade_actuel != StadeInsecteEnum.ADULTE])
        termines = len([l for l in lots if l.stade_actuel == StadeInsecteEnum.ADULTE])
        
        # Production totale
        cycles = []
        for lot in lots:
            cycles.extend(lot.cycles)
        
        production_totale = sum(c.production_grammes or 0 for c in cycles)
        taux_mortalite_moyen = sum(c.taux_mortalite or 0 for c in cycles) / len(cycles) if cycles else 0
        
        # Par espèce
        by_espece = {}
        for lot in lots:
            by_espece[lot.espece] = by_espece.get(lot.espece, 0) + 1
        
        return {
            "total_lots": total_lots,
            "actifs": actifs,
            "termines": termines,
            "total_cycles": len(cycles),
            "production_totale_grammes": round(production_totale, 1),
            "production_totale_kg": round(production_totale / 1000, 2),
            "taux_mortalite_moyen": round(taux_mortalite_moyen, 1),
            "especes": by_espece,
            "nombre_especes": len(by_espece)
        }
    
    async def get_all_lots(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[EntomocultureLot]:
        """Obtenir tous les lots"""
        stmt = select(EntomocultureLot).offset(skip).limit(limit).order_by(EntomocultureLot.identification)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_lot_by_identification(
        self,
        db: AsyncSession,
        identification: str
    ) -> Optional[EntomocultureLot]:
        """Obtenir un lot par son identification"""
        stmt = select(EntomocultureLot).where(
            EntomocultureLot.identification == identification
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update_lot_stage(
        self,
        db: AsyncSession,
        lot_id: int,
        new_stage: StadeInsecteEnum,
        updated_by: int
    ) -> Tuple[bool, str]:
        """Mettre à jour le stade d'un lot"""
        lot = await self.get_lot(db, lot_id)
        if not lot:
            return False, "Lot non trouvé"
        
        old_stage = lot.stade_actuel
        lot.stade_actuel = new_stage
        
        await db.commit()
        logger.info(f"Lot {lot.identification} stage updated: {old_stage} -> {new_stage}")
        return True, f"Stade mis à jour: {new_stage.value}"
    
    async def get_cycles_by_lot(
        self,
        db: AsyncSession,
        lot_id: int
    ) -> List[EntomocultureCycle]:
        """Obtenir tous les cycles d'un lot"""
        stmt = select(EntomocultureCycle).where(
            EntomocultureCycle.lot_id == lot_id
        ).order_by(EntomocultureCycle.date_debut.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def close_cycle(
        self,
        db: AsyncSession,
        cycle_id: int,
        production_grammes: float,
        taux_mortalite: float,
        updated_by: int
    ) -> Tuple[bool, str]:
        """Fermer un cycle et enregistrer la production"""
        stmt = select(EntomocultureCycle).where(EntomocultureCycle.id == cycle_id)
        result = await db.execute(stmt)
        cycle = result.scalar_one_or_none()
        
        if not cycle:
            return False, "Cycle non trouvé"
        
        cycle.date_fin = date.today()
        cycle.production_grammes = production_grammes
        cycle.taux_mortalite = taux_mortalite
        
        # Mettre à jour le stade du lot
        lot = await self.get_lot(db, cycle.lot_id)
        if lot and cycle.stade_fin:
            lot.stade_actuel = cycle.stade_fin
        
        await db.commit()
        logger.info(f"Cycle {cycle_id} closed: production {production_grammes}g, mortality {taux_mortalite}%")
        return True, "Cycle fermé avec succès"
    
entomoculture_service = EntomocultureService()