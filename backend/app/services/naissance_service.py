# backend/app/services/naissance_service.py
"""
Service de gestion des naissances
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from ..models.naissance import Naissance
from ..models.animal import Animal, StatutAnimalEnum, SexeEnum
from ..models.bovin import Bovin
from ..models.ovin import Ovin
from ..models.caprin import Caprin
from ..schemas.naissance import *
from .animal_service import animal_service

logger = logging.getLogger(__name__)


class NaissanceService:
    """Service de gestion des naissances"""
    
    async def create_naissance(
        self,
        db: AsyncSession,
        naissance_data: NaissanceCreate,
        created_by: int
    ) -> Tuple[Optional[Naissance], Optional[str]]:
        """
        Créer un enregistrement de naissance
        """
        # Vérifier que la mère existe
        mere = await db.get(Animal, naissance_data.mere_id)
        if not mere:
            return None, "Mère non trouvée"
        
        if mere.sexe != SexeEnum.FEMELLE:
            return None, "L'animal sélectionné n'est pas une femelle"
        
        # Vérifier que le nouveau-né existe
        nouveau_ne = await db.get(Animal, naissance_data.animal_ne_id)
        if not nouveau_ne:
            return None, "Nouveau-né non trouvé"
        
        # Créer la naissance
        naissance = Naissance(
            mere_id=naissance_data.mere_id,
            pere_bovin_id=naissance_data.pere_bovin_id,
            pere_ovin_id=naissance_data.pere_ovin_id,
            pere_caprin_id=naissance_data.pere_caprin_id,
            animal_ne_id=naissance_data.animal_ne_id,
            date_naissance=naissance_data.date_naissance,
            poids_naissance=naissance_data.poids_naissance,
            sexe=naissance_data.sexe,
            complications=naissance_data.complications,
            notes=naissance_data.notes
        )
        db.add(naissance)
        
        # Mettre à jour la date de naissance de l'animal si non définie
        if not nouveau_ne.date_naissance:
            nouveau_ne.date_naissance = naissance_data.date_naissance
        
        await db.commit()
        await db.refresh(naissance)
        
        logger.info(f"Naissance créée: mère={naissance.mere_id}, nouveau-né={naissance.animal_ne_id} par {created_by}")
        return naissance, None
    
    async def get_naissance(
        self,
        db: AsyncSession,
        naissance_id: int
    ) -> Optional[Naissance]:
        """Obtenir une naissance par son ID"""
        stmt = select(Naissance).where(Naissance.id == naissance_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_naissances_by_mere(
        self,
        db: AsyncSession,
        mere_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Naissance]:
        """Obtenir toutes les naissances d'une mère"""
        stmt = select(Naissance).where(
            Naissance.mere_id == mere_id
        ).order_by(Naissance.date_naissance.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_naissances_by_period(
        self,
        db: AsyncSession,
        start_date: date,
        end_date: date,
        espece: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Naissance]:
        """Obtenir les naissances sur une période"""
        stmt = select(Naissance).where(
            Naissance.date_naissance >= start_date,
            Naissance.date_naissance <= end_date
        )
        
        if espece:
            if espece == "bovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="bovin"))
            elif espece == "ovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="ovin"))
            elif espece == "caprin":
                stmt = stmt.where(Naissance.mere.has(type_espece="caprin"))
        
        stmt = stmt.order_by(Naissance.date_naissance.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_naissance(
        self,
        db: AsyncSession,
        naissance_id: int,
        naissance_data: NaissanceUpdate,
        updated_by: int
    ) -> Tuple[Optional[Naissance], Optional[str]]:
        """Mettre à jour une naissance"""
        naissance = await self.get_naissance(db, naissance_id)
        if not naissance:
            return None, "Naissance non trouvée"
        
        if naissance_data.poids_naissance is not None:
            naissance.poids_naissance = naissance_data.poids_naissance
        if naissance_data.complications is not None:
            naissance.complications = naissance_data.complications
        if naissance_data.notes is not None:
            naissance.notes = naissance_data.notes
        
        await db.commit()
        await db.refresh(naissance)
        
        logger.info(f"Naissance mise à jour: id={naissance_id} par {updated_by}")
        return naissance, None
    
    async def delete_naissance(
        self,
        db: AsyncSession,
        naissance_id: int,
        deleted_by: int
    ) -> bool:
        """Supprimer une naissance"""
        naissance = await self.get_naissance(db, naissance_id)
        if not naissance:
            return False
        
        await db.delete(naissance)
        await db.commit()
        
        logger.info(f"Naissance supprimée: id={naissance_id} par {deleted_by}")
        return True
    
    async def get_naissances_stats(
        self,
        db: AsyncSession,
        year: Optional[int] = None,
        espece: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtenir des statistiques sur les naissances"""
        if not year:
            year = date.today().year
        
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        stmt = select(Naissance).where(
            Naissance.date_naissance >= start_date,
            Naissance.date_naissance <= end_date
        )
        
        if espece:
            if espece == "bovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="bovin"))
            elif espece == "ovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="ovin"))
            elif espece == "caprin":
                stmt = stmt.where(Naissance.mere.has(type_espece="caprin"))
        
        result = await db.execute(stmt)
        naissances = result.scalars().all()
        
        # Statistiques
        total = len(naissances)
        
        # Par sexe
        males = len([n for n in naissances if n.sexe == "male"])
        females = len([n for n in naissances if n.sexe == "femelle"])
        
        # Par mois
        by_month = {i: 0 for i in range(1, 13)}
        for n in naissances:
            by_month[n.date_naissance.month] += 1
        
        # Complications
        complications = len([n for n in naissances if n.complications])
        
        # Poids moyen à la naissance
        poids_valides = [n.poids_naissance for n in naissances if n.poids_naissance]
        poids_moyen = sum(poids_valides) / len(poids_valides) if poids_valides else 0
        
        # Taux de mortalité néonatale (à implémenter avec les mortalités)
        mortalite_neonatale = 0
        
        return {
            "year": year,
            "espece": espece or "toutes",
            "total": total,
            "males": males,
            "femelles": females,
            "ratio_m_f": round(males / females, 2) if females > 0 else 0,
            "by_month": by_month,
            "complications": complications,
            "taux_complications": round(complications / total * 100, 1) if total > 0 else 0,
            "poids_moyen_naissance_kg": round(poids_moyen, 2),
            "mortalite_neonatale_percent": round(mortalite_neonatale, 1)
        }
    
    async def get_last_naissances(
        self,
        db: AsyncSession,
        limit: int = 10,
        espece: Optional[str] = None
    ) -> List[Naissance]:
        """Obtenir les dernières naissances"""
        stmt = select(Naissance).order_by(Naissance.date_naissance.desc()).limit(limit)
        
        if espece:
            if espece == "bovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="bovin"))
            elif espece == "ovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="ovin"))
            elif espece == "caprin":
                stmt = stmt.where(Naissance.mere.has(type_espece="caprin"))
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_birth_calendar(
        self,
        db: AsyncSession,
        start_date: date,
        end_date: date,
        espece: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Obtenir un calendrier des naissances prévues (basé sur les gestations)"""
        # Pour les gestations en cours, il faudrait une table dédiée
        # Cette fonction est un placeholder
        return []
    
    async def get_naissances_stats(
        self,
        db: AsyncSession,
        year: Optional[int] = None,
        espece: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques des naissances"""
        if not year:
            year = date.today().year
        
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        stmt = select(Naissance).where(
            Naissance.date_naissance >= start_date,
            Naissance.date_naissance <= end_date
        )
        
        if espece:
            if espece == "bovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="bovin"))
            elif espece == "ovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="ovin"))
            elif espece == "caprin":
                stmt = stmt.where(Naissance.mere.has(type_espece="caprin"))
        
        result = await db.execute(stmt)
        naissances = result.scalars().all()
        
        total = len(naissances)
        males = len([n for n in naissances if n.sexe == "male"])
        females = len([n for n in naissances if n.sexe == "femelle"])
        
        # Par mois
        by_month = {i: 0 for i in range(1, 13)}
        for n in naissances:
            by_month[n.date_naissance.month] += 1
        
        # Complications
        complications = len([n for n in naissances if n.complications])
        
        # Poids moyen
        poids_valides = [n.poids_naissance for n in naissances if n.poids_naissance]
        poids_moyen = sum(poids_valides) / len(poids_valides) if poids_valides else 0
        
        return {
            "year": year,
            "espece": espece or "toutes",
            "total": total,
            "males": males,
            "femelles": females,
            "ratio_m_f": round(males / females, 2) if females > 0 else 0,
            "by_month": by_month,
            "complications": complications,
            "taux_complications": round(complications / total * 100, 1) if total > 0 else 0,
            "poids_moyen_naissance_kg": round(poids_moyen, 2)
        }
    
    async def get_last_naissances(
        self,
        db: AsyncSession,
        limit: int = 10,
        espece: Optional[str] = None
    ) -> List[Naissance]:
        """Obtenir les dernières naissances"""
        stmt = select(Naissance).order_by(Naissance.date_naissance.desc()).limit(limit)
        
        if espece:
            if espece == "bovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="bovin"))
            elif espece == "ovin":
                stmt = stmt.where(Naissance.mere.has(type_espece="ovin"))
            elif espece == "caprin":
                stmt = stmt.where(Naissance.mere.has(type_espece="caprin"))
        
        result = await db.execute(stmt)
        return result.scalars().all()


naissance_service = NaissanceService()