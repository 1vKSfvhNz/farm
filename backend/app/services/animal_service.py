# backend/app/services/animal_service.py
"""
Service de base pour la gestion des animaux
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.pesee import Pesee
from app.services.id_service import log_action

from ..models.animal import Animal, SexeEnum, StatutAnimalEnum
from ..models.enclos import Enclos
from ..schemas.animal import *

logger = logging.getLogger(__name__)


class AnimalService:
    """Service de base pour la gestion des animaux"""

    async def get_enclos_name_by_id(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> Optional[str]:
        """Trouve un enclos par son ID"""
        stmt = select(Enclos.name).where(Enclos.id == enclos_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_last_weight(
        self,
        db: AsyncSession,
        animal_id: int
    ) -> Optional[float]:
        """Récupère le dernier poids d'un animal de manière asynchrone"""
        from sqlalchemy import desc
        
        stmt = (
            select(Pesee.poids)
            .where(Pesee.animal_id == animal_id)
            .order_by(desc(Pesee.date_pesee))
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
        
    
    async def get_animal(
        self,
        db: AsyncSession,
        animal_id: int
    ) -> Optional[Animal]:
        """Obtenir un animal par son ID"""
        stmt = select(Animal).where(Animal.id == animal_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_animal_by_identification(
        self,
        db: AsyncSession,
        identification: str
    ) -> Optional[Animal]:
        """Obtenir un animal par son identification"""
        stmt = select(Animal).where(
            Animal.identification == identification
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_animals(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        espece: Optional[str] = None,
        enclos_id: Optional[int] = None,
        race: Optional[str] = None,
        statut: Optional[StatutAnimalEnum] = None,
        sexe: Optional[SexeEnum] = None
    ) -> List[Animal]:
        """Obtenir la liste des animaux avec filtres"""
        stmt = select(Animal)
        
        if espece:
            stmt = stmt.where(Animal.type_espece == espece)
        if enclos_id:
            stmt = stmt.where(Animal.enclos_id == enclos_id)
        if race:
            stmt = stmt.where(Animal.race == race)
        if statut:
            stmt = stmt.where(Animal.statut == statut)
        if sexe:
            stmt = stmt.where(Animal.sexe == sexe)
        
        stmt = stmt.offset(skip).limit(limit).order_by(Animal.date_arrivee.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    
    async def get_animals_by_enclos(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> List[Animal]:
        """Obtenir tous les animaux d'un enclos"""
        stmt = select(Animal).where(
            Animal.enclos_id == enclos_id,
            Animal.statut == StatutAnimalEnum.VIVANT
        )
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_statistics(
        self,
        db: AsyncSession,
        espece: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtenir des statistiques sur les animaux"""
        stmt = select(Animal)
        if espece:
            stmt = stmt.where(Animal.type_espece == espece)
        result = await db.execute(stmt)
        animals = result.scalars().all()
        
        total = len(animals)
        by_sexe = {}
        by_race = {}
        
        for animal in animals:
            # Par sexe
            sexe_key = animal.sexe.value if animal.sexe else "unknown"
            by_sexe[sexe_key] = by_sexe.get(sexe_key, 0) + 1
            
            # Par race
            by_race[animal.race] = by_race.get(animal.race, 0) + 1
        
        return {
            "total": total,
            "by_sexe": by_sexe,
            "by_race": by_race,
            "alive": len([a for a in animals if a.statut == StatutAnimalEnum.VIVANT]),
            "sold": len([a for a in animals if a.statut == StatutAnimalEnum.VENDU]),
            "deceased": len([a for a in animals if a.statut == StatutAnimalEnum.DECEDE])
        }
    
    async def enregistrer_vente(
        self,
        db: AsyncSession,
        animal_id: int,
        action: str,
        espece: str,
        vente_data: AnimalVenteCreate,
        user_id: int
    ) -> Tuple[Optional[Animal], Optional[str]]:
        """
        Enregistrer la vente d'un animal
        
        Args:
            db: Session de base de données
            animal_id: ID du bovin à vendre
            vente_data: Données de la vente
            user_id: ID de l'utilisateur qui enregistre la vente
        
        Returns:
            Tuple[Optional[Bovin], Optional[str]]: (bovin_updated, error_message)
        """
        # Récupérer le bovin
        animal = await self.get_animal(db, animal_id)
        if not animal:
            return None, "Bovin non trouvé"
        
        # Vérifier que le animal n'est pas déjà vendu ou décédé
        if animal.statut == StatutAnimalEnum.VENDU:
            return None, "Ce animal est déjà vendu"
        if animal.statut == StatutAnimalEnum.DECEDE:
            return None, "Impossible de vendre un animal décédé"
        
        # Mettre à jour les informations de vente
        old_status = animal.statut.value if animal.statut else None
        animal.prix_vente = vente_data.prix_vente
        animal.date_vente = vente_data.date_vente or date.today()
        animal.client_acheteur = vente_data.client_acheteur
        animal.note_vente = vente_data.note_vente
        
        # Mettre à jour le statut si spécifié dans le schéma
        if vente_data.statut:
            animal.statut = vente_data.statut
        else:
            animal.statut = StatutAnimalEnum.VENDU
        
        await db.commit()
        await db.refresh(animal)
        
        # Journaliser la vente
        await log_action(
            db, 
            user_id, 
            action, 
            espece, 
            animal_id,
            {
                "prix_vente": vente_data.prix_vente,
                "date_vente": animal.date_vente.isoformat(),
                "client": vente_data.client_acheteur,
                "ancien_statut": old_status,
                "nouveau_statut": animal.statut.value
            }
        )
        
        logger.info(f"Bovin {animal.identification} vendu pour {vente_data.prix_vente}€ à {vente_data.client_acheteur}")
        return animal, None
    

animal_service = AnimalService()