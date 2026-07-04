# backend/app/services/enclos_service.py
"""
Service de gestion des enclos
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, select, func

from app.services.id_service import log_action

from ..models.enclos import Enclos, EnclosType
from ..models.animal import Animal, StatutAnimalEnum
from ..schemas.enclos import EnclosCreate, EnclosUpdate, EnclosResponse

logger = logging.getLogger(__name__)


class EnclosService:
    """Service de gestion des enclos"""
    
    async def _get_occupation_actuelle(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> int:
        """Calcule l'occupation actuelle d'un enclos"""
        stmt = select(func.count(Animal.id)).where(
            Animal.enclos_id == enclos_id,
            Animal.is_deleted == False
        )
        result = await db.execute(stmt)
        return result.scalar() or 0
        
    async def _get_densite_animale(self, db: AsyncSession, enclos: Enclos) -> Dict[str, float]:
        """
        Calcule la densité d'animaux :
        - Pour les enclos terrestres : nombre d'animaux par m²
        - Pour les bassins/bacs : nombre de poissons par m³
        """
        occupation = await self._get_occupation_actuelle(db, enclos.id)
        
        if enclos.type in [EnclosType.BASSIN, EnclosType.BAC]:
            # Densité volumique (poissons/m³)
            volume = enclos.volume
            densite = occupation / volume if volume > 0 else 0.0
            unite = "poissons/m³"
        else:
            # Densité surfacique (animaux/m²)
            surface = enclos.surface
            densite = occupation / surface if surface > 0 else 0.0
            unite = "animaux/m²"
        
        return {
            "densite": round(densite, 2),
            "unite": unite,
            "occupation": occupation,
            "surface_m2": enclos.surface,
            "volume_m3": enclos.volume
        }
        
    async def create_enclos(
        self,
        db: AsyncSession,
        enclos_data: EnclosCreate,
        created_by: int
    ) -> Tuple[Optional[Enclos], Optional[str]]:
        """Créer un nouvel enclos"""
        # Vérifier si le nom existe déjà
        stmt = select(Enclos).where(Enclos.name == enclos_data.name)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            return None, "Un enclos avec ce nom existe déjà"
        
        enclos = Enclos(
            name=enclos_data.name,
            type=enclos_data.type,
            longueur=enclos_data.longueur,
            largeur=enclos_data.largeur,
            hauteur=enclos_data.hauteur,
            localisation_gps=enclos_data.localisation_gps,
            zone=enclos_data.zone,
            description=enclos_data.description
        )
        db.add(enclos)
        await db.flush()
        
        # Journaliser
        await log_action(db, created_by, "CREATE_ENCLOS", "enclos", enclos.id, {"name": enclos.name})
        await db.commit()
        
        # Rafraîchir et enrichir l'objet
        await db.refresh(enclos)
        
        logger.info(f"Enclos created: {enclos.name} by {created_by}")
        return enclos, None
    
    async def get_enclos(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> Optional[Enclos]:
        """Obtenir un enclos par son ID"""
        stmt = select(Enclos).where(Enclos.id == enclos_id)
        result = await db.execute(stmt)
        enclos = result.scalar_one_or_none()
                
        return enclos
    
    async def get_enclos_by_name(
        self,
        db: AsyncSession,
        name: str
    ) -> Optional[Enclos]:
        """Obtenir un enclos par son nom"""
        stmt = select(Enclos).where(Enclos.name == name)
        result = await db.execute(stmt)
        enclos = result.scalar_one_or_none()
                
        return enclos
    
    async def get_enclos_list(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        enclos_type: Optional[EnclosType] = None,
        zone: Optional[str] = None
    ) -> List[Enclos]:
        """Obtenir la liste des enclos"""
        # Filtrer les enclos non supprimés
        stmt = select(Enclos)
        
        if enclos_type:
            stmt = stmt.where(Enclos.type == enclos_type)
        if zone:
            stmt = stmt.where(Enclos.zone == zone)
        
        stmt = stmt.offset(skip).limit(limit).order_by(Enclos.name)
        result = await db.execute(stmt)
        enclos_list = result.scalars().all()
                
        return enclos_list
    
    async def update_enclos(
        self,
        db: AsyncSession,
        enclos_id: int,
        enclos_data: EnclosUpdate,
        updated_by: int
    ) -> Tuple[Optional[Enclos], Optional[str]]:
        """Mettre à jour un enclos"""
        enclos = await self.get_enclos(db, enclos_id)
        if not enclos:
            return None, "Enclos non trouvé"
        
        # Sauvegarder les anciennes valeurs pour le log
        old_data = {
            "name": enclos.name,
            "type": enclos.type.value,
            "longueur": enclos.longueur,
            "largeur": enclos.largeur,
            "hauteur": enclos.hauteur,
            "zone": enclos.zone
        }
                
        if enclos_data.name is not None:
            # Vérifier que le nom n'est pas déjà utilisé
            stmt = select(Enclos).where(
                Enclos.name == enclos_data.name,
                Enclos.id != enclos_id,
                Enclos.is_deleted == False
            )
            result = await db.execute(stmt)
            if result.scalar_one_or_none():
                return None, "Ce nom est déjà utilisé"
            enclos.name = enclos_data.name
        
        if enclos_data.type is not None:
            enclos.type = enclos_data.type
        if enclos_data.longueur is not None:
            enclos.longueur = enclos_data.longueur
        if enclos_data.largeur is not None:
            enclos.largeur = enclos_data.largeur
        if enclos_data.hauteur is not None:
            enclos.hauteur = enclos_data.hauteur
        if enclos_data.localisation_gps is not None:
            enclos.localisation_gps = enclos_data.localisation_gps
        if enclos_data.zone is not None:
            enclos.zone = enclos_data.zone
        if enclos_data.description is not None:
            enclos.description = enclos_data.description
        
        await db.commit()
        await db.refresh(enclos)
                
        await log_action(db, updated_by, "UPDATE_ENCLOS", "enclos", enclos_id, {"old": old_data})
        
        logger.info(f"Enclos updated: {enclos.name} by {updated_by}")
        return enclos, None
            
    async def get_enclos_stats(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> Dict[str, Any]:
        """Obtenir les statistiques d'un enclos"""
        enclos = await self.get_enclos(db, enclos_id)
        if not enclos:
            return {}
        
        # Compter les animaux par espèce
        stmt = select(Animal.type_espece, func.count()).where(
            Animal.enclos_id == enclos_id,
            Animal.statut == StatutAnimalEnum.VIVANT,
            Animal.is_deleted == False
        ).group_by(Animal.type_espece)
        result = await db.execute(stmt)
        by_espece = {row[0]: row[1] for row in result}
        
        # Calculer la densité
        densite_data = await self._get_densite_animale(db, enclos)
        
        return {
            "enclos_id": enclos.id,
            "name": enclos.name,
            "type": enclos.type.value,
            "surface_m2": enclos.surface,
            "volume_m3": enclos.volume,
            "animaux_par_espece": by_espece,
            "densite": densite_data["densite"],
            "densite_unite": densite_data["unite"],
            "occupation_actuelle": densite_data["occupation"]
        }
    
    async def get_all_enclos_stats(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Obtenir les statistiques globales des enclos"""
        enclos_list = await self.get_enclos_list(db, limit=1000)
        
        total_enclos = len(enclos_list)
        total_surface = sum(e.surface for e in enclos_list)
        total_volume = sum(e.volume for e in enclos_list if e.type in [EnclosType.BASSIN, EnclosType.BAC])
        
        total_animaux = 0
        for e in enclos_list:
            total_animaux += await self._get_occupation_actuelle(db, e.id)
        
        stats_by_type = {}
        for e in enclos_list:
            type_key = e.type.value
            if type_key not in stats_by_type:
                stats_by_type[type_key] = {
                    "count": 0,
                    "surface_totale_m2": 0,
                    "volume_total_m3": 0,
                    "animaux_actuels": 0
                }
            stats_by_type[type_key]["count"] += 1
            stats_by_type[type_key]["surface_totale_m2"] += e.surface
            if e.type in [EnclosType.BASSIN, EnclosType.BAC]:
                stats_by_type[type_key]["volume_total_m3"] += e.volume
            stats_by_type[type_key]["animaux_actuels"] += await self._get_occupation_actuelle(db, e.id)
        
        # Densité moyenne globale
        densite_moyenne_surfacique = total_animaux / total_surface if total_surface > 0 else 0
        densite_moyenne_volumique = total_animaux / total_volume if total_volume > 0 else 0
        
        return {
            "total_enclos": total_enclos,
            "surface_totale_m2": total_surface,
            "volume_total_m3": total_volume,
            "animaux_totaux": total_animaux,
            "densite_moyenne_surfacique": round(densite_moyenne_surfacique, 2),
            "densite_moyenne_volumique": round(densite_moyenne_volumique, 2),
            "statistiques_par_type": stats_by_type
        }
    
    async def get_enclos_detail_stats(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> Optional[Dict[str, Any]]:
        """Obtenir les statistiques détaillées d'un enclos"""
        enclos = await self.get_enclos(db, enclos_id)
        if not enclos:
            return None
        
        from ..models.animal import Animal, StatutAnimalEnum
        
        # Compter les animaux par espèce
        stmt = select(Animal.type_espece, func.count()).where(
            Animal.enclos_id == enclos_id,
            Animal.statut == StatutAnimalEnum.VIVANT,
            Animal.is_deleted == False
        ).group_by(Animal.type_espece)
        result = await db.execute(stmt)
        animaux_par_espece = {row[0]: row[1] for row in result}
        
        # Compter par sexe
        stmt = select(Animal.sexe, func.count()).where(
            Animal.enclos_id == enclos_id,
            Animal.statut == StatutAnimalEnum.VIVANT,
            Animal.is_deleted == False
        ).group_by(Animal.sexe)
        result = await db.execute(stmt)
        animaux_par_sexe = {row[0].value if row[0] else "unknown": row[1] for row in result}
        
        # Densité
        densite_data = await self._get_densite_animale(db, enclos)
        
        return {
            "id": enclos.id,
            "name": enclos.name,
            "type": enclos.type.value,
            "surface_m2": enclos.surface,
            "volume_m3": enclos.volume,
            "occupation_actuelle": densite_data["occupation"],
            "densite": densite_data["densite"],
            "densite_unite": densite_data["unite"],
            "animaux_par_espece": animaux_par_espece,
            "animaux_par_sexe": animaux_par_sexe,
            "zone": enclos.zone,
            "localisation_gps": enclos.localisation_gps,
            "description": enclos.description
        }

enclos_service = EnclosService()