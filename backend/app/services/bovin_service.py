# backend/app/services/bovin_service.py
"""
Service de gestion des bovins
"""

import base64
from datetime import date
import logging
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.core.constants import SexeEnum
from app.core.validators import validate_animal_age
from app.models.pesee import Pesee
from .media_service import media_service
from .animal_service import animal_service
from .id_service import generate_identification, log_action

from ..models.bovin import Bovin
from ..models.animal import StatutAnimalEnum
from ..schemas.bovin import BovinCreate, BovinUpdate


logger = logging.getLogger(__name__)


class BovinService:
    """Service de gestion des bovins"""
    
    async def create_bovin(
        self,
        db: AsyncSession,
        bovin_data: BovinCreate,
        created_by: int
    ) -> Tuple[Optional[Bovin], Optional[str]]:
        """Créer un nouveau bovin avec photo et poids initial"""
        
        # Vérifier que l'enclos existe
        enclos = await animal_service.get_enclos_by_name(db, bovin_data.enclos_name)
        if not enclos:
            return None, f"Enclos '{bovin_data.enclos_name}' non trouvé"
        
        # Générer l'identification
        identification = await generate_identification(db, "bov")
        
        # Vérifier l'âge si date de naissance fournie
        if bovin_data.date_naissance:
            valid, error = validate_animal_age(bovin_data.date_naissance)
            if not valid:
                return None, error
        
        # Créer le bovin
        bovin = Bovin(
            type_espece="bovin",
            identification=identification,
            race=bovin_data.race,
            sexe=bovin_data.sexe,
            date_naissance=bovin_data.date_naissance,
            date_arrivee=bovin_data.date_arrivee or date.today(),
            provenance=bovin_data.provenance,
            prix_achat=bovin_data.prix_achat,
            enclos_id=enclos.id,
            statut=bovin_data.statut or StatutAnimalEnum.VIVANT,
            notes=bovin_data.notes,
            production_laitiere=bovin_data.production_laitiere,
            production_viande=bovin_data.production_viande,
            production_reproduction=bovin_data.production_reproduction,
            lactation_en_cours=bovin_data.lactation_en_cours,
            production_lait_quotidienne=bovin_data.production_lait_quotidienne
        )
        
        db.add(bovin)
        await db.flush()
        
        # Sauvegarder la photo si présente
        if bovin_data.photo_base64:
            photo_url = media_service.save_base64_photo(
                bovin_data.photo_base64, "bovin", bovin.id
            )
            if photo_url:
                bovin.photo_url = photo_url
                await db.flush()
        
        # Ajouter le poids initial si fourni
        if bovin_data.poids_initial:
            pesee = Pesee(
                animal_id=bovin.id,
                date_pesee=bovin_data.date_arrivee or date.today(),
                poids=bovin_data.poids_initial,
                methode="Initiale",
                notes="Poids à l'arrivée"
            )
            db.add(pesee)
        
        await db.commit()
        await db.refresh(bovin)
        
        # Journaliser
        await log_action(
            db, created_by, "CREATE_BOVIN", "bovin", bovin.id,
            {"identification": bovin.identification, "race": bovin_data.race}
        )
        
        return bovin, None
    
    async def update_bovin(
        self,
        db: AsyncSession,
        bovin_id: int,
        bovin_data: BovinUpdate,
        updated_by: int
    ) -> Tuple[Optional[Bovin], Optional[str]]:
        """Mettre à jour un bovin avec photo et poids"""
        
        # Récupérer le bovin avec son animal
        stmt = select(Bovin).where(Bovin.id == bovin_id)
        result = await db.execute(stmt)
        bovin = result.scalar_one_or_none()
        
        if not bovin:
            return None, "Bovin non trouvé"
        
        # Sauvegarder l'ancien statut pour le log
        old_status = bovin.statut.value if bovin.statut else None
        
        # Mettre à jour les champs bovin
        update_bovin_fields = False
        if bovin_data.production_laitiere is not None:
            bovin.production_laitiere = bovin_data.production_laitiere
            update_bovin_fields = True
        if bovin_data.production_viande is not None:
            bovin.production_viande = bovin_data.production_viande
            update_bovin_fields = True
        if bovin_data.production_reproduction is not None:
            bovin.production_reproduction = bovin_data.production_reproduction
            update_bovin_fields = True
        if bovin_data.lactation_en_cours is not None:
            bovin.lactation_en_cours = bovin_data.lactation_en_cours
            update_bovin_fields = True
        if bovin_data.production_lait_quotidienne is not None:
            bovin.production_lait_quotidienne = bovin_data.production_lait_quotidienne
            update_bovin_fields = True
        
        # Mettre à jour l'animal de base
        if bovin_data.race is not None:
            bovin.race = bovin_data.race
        if bovin_data.sexe is not None:
            bovin.sexe = bovin_data.sexe
        if bovin_data.date_naissance is not None:
            bovin.date_naissance = bovin_data.date_naissance
        if bovin_data.date_arrivee is not None:
            bovin.date_arrivee = bovin_data.date_arrivee
        if bovin_data.provenance is not None:
            bovin.provenance = bovin_data.provenance
        if bovin_data.prix_achat is not None:
            bovin.prix_achat = bovin_data.prix_achat
        if bovin_data.enclos_id is not None:
            enclos_name = await animal_service.get_enclos_name_by_id(db, bovin_data.enclos_id)
            if enclos_name:
                bovin.enclos_id = bovin_data.enclos_id
        if bovin_data.statut is not None:
            bovin.statut = bovin_data.statut
        if bovin_data.notes is not None:
            bovin.notes = bovin_data.notes
        
        # Traiter la nouvelle photo
        if bovin_data.photo_base64:
            # Supprimer l'ancienne photo
            if bovin.photo_url:
                media_service.delete_photo(bovin.photo_url)
            
            # Sauvegarder la nouvelle
            photo_url = media_service.save_base64_photo(
                bovin_data.photo_base64, "bovin", bovin_id
            )
            if photo_url:
                bovin.photo_url = photo_url
                await db.flush()
        
        # Ajouter un nouveau poids si fourni
        if bovin_data.poids_initial:
            pesee = Pesee(
                animal_id=bovin_id,
                poids=bovin_data.poids_initial,
                date_pesee=date.today(),
                methode="Manuelle",
                notes=f"Mise à jour le {date.today()}",
                created_by=updated_by
            )
            db.add(pesee)
        
        await db.commit()
        await db.refresh(bovin)
        
        # Journaliser la mise à jour
        await log_action(
            db, updated_by, "UPDATE_BOVIN", "bovin", bovin_id,
            {
                "old_status": old_status, 
                "new_status": bovin_data.statut.value if bovin_data.statut else None
            }
        )
        
        logger.info(f"Bovin updated: {bovin.identification} by {updated_by}")
        return bovin, None
    
    async def get_bovin(
        self,
        db: AsyncSession,
        bovin_id: int
    ) -> Optional[Bovin]:
        """Obtenir un bovin par son ID"""
        stmt = select(Bovin).where(Bovin.id == bovin_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all_bovins_with_filters(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        race: Optional[str] = None,
        production_type: Optional[str] = None,
        enclos_id: Optional[int] = None,
        sexe: Optional[str] = None,
        statut: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Bovin]:
        """Obtenir tous les bovins avec filtres (sans pagination pour compter)"""
        # ✅ Convertir la chaîne en liste
        stmt = select(Bovin)

        statut_list = None
        if statut:
            if isinstance(statut, str) and ',' in statut:
                statut_list = [s.strip() for s in statut.split(',') if s.strip()]
            elif isinstance(statut, str):
                statut_list = [statut]
            elif isinstance(statut, list):
                statut_list = statut
            stmt = stmt.where(Bovin.statut.in_(statut_list))

        sexe_list = None
        if sexe:
            if isinstance(sexe, str) and ',' in sexe:
                sexe_list = [s.strip() for s in sexe.split(',') if s.strip()]
            elif isinstance(sexe, str):
                sexe_list = [sexe]
            elif isinstance(sexe, list):
                sexe_list = sexe
            stmt = stmt.where(Bovin.sexe.in_(sexe_list))
        
        if race:
            stmt = stmt.where(Bovin.race == race)
        if enclos_id:
            stmt = stmt.where(Bovin.enclos_id == enclos_id)
        if production_type:
            if production_type == "lait":
                stmt = stmt.where(Bovin.production_laitiere == True)
            elif production_type == "viande":
                stmt = stmt.where(Bovin.production_viande == True)
            elif production_type == "reproduction":
                stmt = stmt.where(Bovin.production_reproduction == True)
        if search:
            search_pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    Bovin.identification.ilike(search_pattern),
                    Bovin.race.ilike(search_pattern)
                )
            )
        
        stmt = stmt.offset(skip).limit(limit).order_by(Bovin.created_at.asc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_weight_progression(
        self,
        db: AsyncSession,
        bovin_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtenir la progression des poids d'un bovin"""
        stmt = (
            select(Pesee)
            .where(Pesee.animal_id == bovin_id)
            .order_by(Pesee.date_pesee)
            .limit(100)
        )
        result = await db.execute(stmt)
        pesees = result.scalars().all()
        
        return [
            {
                "date": pesee.date_pesee.isoformat(),
                "poids": pesee.poids,
                "methode": pesee.methode
            }
            for pesee in pesees
        ]
    
    async def get_lactating_cows(
        self,
        db: AsyncSession
    ) -> List[Bovin]:
        """Obtenir les vaches en lactation"""
        stmt = select(Bovin).where(
            Bovin.lactation_en_cours == True,
            Bovin.production_laitiere == True,
            Bovin.statut == StatutAnimalEnum.VIVANT
        )
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_milk_production(
        self,
        db: AsyncSession,
        bovin_id: int,
        daily_liters: float,
        updated_by: int
    ) -> Tuple[bool, str]:
        """Mettre à jour la production laitière quotidienne"""
        bovin = await self.get_bovin(db, bovin_id)
        if not bovin:
            return False, "Bovin non trouvé"
        
        if not bovin.production_laitiere:
            return False, "Ce bovin n'est pas une vache laitière"
        
        old_liters = bovin.production_lait_quotidienne
        bovin.production_lait_quotidienne = daily_liters
        
        await db.commit()
        
        # Journaliser la mise à jour de production
        await log_action(
            db, updated_by, "UPDATE_MILK_PRODUCTION", "bovin", bovin_id,
            {"old_liters": old_liters, "new_liters": daily_liters}
        )
        
        logger.info(f"Milk production updated for bovin {bovin_id}: {daily_liters} L/day")
        return True, "Production laitière mise à jour"
    
    async def record_milk_harvest(
        self,
        db: AsyncSession,
        bovin_id: int,
        liters: float,
        date_harvest: date,
        recorded_by: int
    ) -> Tuple[bool, str]:
        """Enregistrer une traite quotidienne"""
        bovin = await self.get_bovin(db, bovin_id)
        if not bovin:
            return False, "Bovin non trouvé"
        
        if not bovin.production_laitiere:
            return False, "Ce bovin n'est pas une vache laitière"
        
        # Mettre à jour la production quotidienne
        bovin.production_lait_quotidienne = liters
        
        await db.commit()
        
        # Journaliser la traite
        await log_action(
            db, recorded_by, "MILK_HARVEST", "bovin", bovin_id,
            {"liters": liters, "date": date_harvest.isoformat()}
        )
        
        logger.info(f"Milk harvest recorded for bovin {bovin_id}: {liters} L on {date_harvest}")
        return True, "Traite enregistrée"
    
    async def get_milk_production_history(
        self,
        db: AsyncSession,
        bovin_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtenir l'historique de production laitière"""
        # À implémenter avec une table MilkProduction
        return []
    
    async def get_bovin_stats(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques des bovins"""
        stmt = select(Bovin).where(
            Bovin.statut == StatutAnimalEnum.VIVANT
        )
        if enclos_id:
            stmt = stmt.where(Bovin.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        bovins = result.scalars().all()
        
        total = len(bovins)
        males = len([b for b in bovins if b.sexe == SexeEnum.MALE])
        females = len([b for b in bovins if b.sexe == SexeEnum.FEMELLE])
        
        production_laitiere = len([b for b in bovins if b.production_laitiere])
        production_viande = len([b for b in bovins if b.production_viande])
        production_reproduction = len([b for b in bovins if b.production_reproduction])
        lactation_en_cours = len([b for b in bovins if b.lactation_en_cours])
        
        lait_quotidien = [b.production_lait_quotidienne for b in bovins if b.production_lait_quotidienne]
        production_lait_moyenne = sum(lait_quotidien) / len(lait_quotidien) if lait_quotidien else 0
        
        total_weight = 0
        weight_count = 0
        for b in bovins:
            # Récupérer le dernier poids via le service
            dernier_poids = await animal_service.get_last_weight(db, b.id)
            if dernier_poids:
                total_weight += dernier_poids
                weight_count += 1
        
        avg_weight = total_weight / weight_count if weight_count > 0 else 0
        
        races = {}
        for b in bovins:
            races[b.race] = races.get(b.race, 0) + 1
        
        return {
            "total": total,
            "males": males,
            "femelles": females,
            "ratio_m_f": round(males / females, 2) if females > 0 else 0,
            "races": races,
            "poids_moyen_kg": round(avg_weight, 1),
            "production": {
                "laitiere": production_laitiere,
                "viande": production_viande,
                "reproduction": production_reproduction,
                "lactation_en_cours": lactation_en_cours,
                "production_lait_moyenne_l_jour": round(production_lait_moyenne, 1)
            },
            "taux_occupation_utile": round((production_laitiere + production_viande + production_reproduction) / total * 100, 1) if total > 0 else 0
        }
    
    async def get_bovin_by_identification(
        self,
        db: AsyncSession,
        identification: str
    ) -> Optional[Bovin]:
        """Obtenir un bovin par son identification"""
        stmt = select(Bovin).where(Bovin.identification == identification)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all_bovins(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Bovin]:
        """Obtenir tous les bovins"""
        stmt = select(Bovin).offset(skip).limit(limit).order_by(Bovin.identification)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_reproduction_status(
        self,
        db: AsyncSession,
        bovin_id: int
    ) -> Dict[str, Any]:
        """Obtenir le statut reproducteur d'un bovin"""
        bovin = await self.get_bovin(db, bovin_id)
        if not bovin:
            return {"error": "Bovin non trouvé"}
        
        from ..models.naissance import Naissance
        
        stmt = select(Naissance).where(Naissance.mere_id == bovin_id)
        result = await db.execute(stmt)
        naissances = result.scalars().all()
        
        derniere_naissance = max(naissances, key=lambda n: n.date_naissance) if naissances else None
        
        return {
            "production_reproduction": bovin.production_reproduction,
            "nombre_naissances": len(naissances),
            "derniere_naissance": derniere_naissance.date_naissance if derniere_naissance else None,
            "intervalle_moyen_jours": None
        }
    
    async def get_ventes_stats(
        self,
        db: AsyncSession,
        date_debut: Optional[date] = None,
        date_fin: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Obtenir les statistiques des ventes de bovins
        
        Args:
            db: Session de base de données
            date_debut: Date de début pour filtrer les ventes
            date_fin: Date de fin pour filtrer les ventes
        
        Returns:
            Dict contenant les statistiques des ventes
        """
        
        # Construction de la requête de base
        stmt = select(Bovin).where(Bovin.prix_vente.isnot(None))
        
        if date_debut:
            stmt = stmt.where(Bovin.date_vente >= date_debut)
        if date_fin:
            stmt = stmt.where(Bovin.date_vente <= date_fin)
        
        result = await db.execute(stmt)
        bovins_vendus = result.scalars().all()
        
        if not bovins_vendus:
            return {
                "total_ventes": 0,
                "montant_total": 0,
                "prix_moyen": 0,
                "prix_min": None,
                "prix_max": None,
                "par_statut": {},
                "par_client": {},
                "par_mois": {}
            }
        
        # Statistiques de base
        prix = [b.prix_vente for b in bovins_vendus if b.prix_vente]
        
        # Statistiques par statut (pour les bovins vendus avec d'autres statuts)
        par_statut = {}
        for b in bovins_vendus:
            statut_key = b.statut.value if b.statut else "unknown"
            par_statut[statut_key] = par_statut.get(statut_key, 0) + 1
        
        # Statistiques par client
        par_client = {}
        for b in bovins_vendus:
            client = b.client_acheteur or "Inconnu"
            par_client[client] = par_client.get(client, 0) + 1
        
        # Statistiques par mois
        par_mois = {}
        for b in bovins_vendus:
            if b.date_vente:
                mois_key = b.date_vente.strftime("%Y-%m")
                par_mois[mois_key] = par_mois.get(mois_key, 0) + b.prix_vente
        
        return {
            "total_ventes": len(bovins_vendus),
            "montant_total": sum(prix),
            "prix_moyen": sum(prix) / len(prix) if prix else 0,
            "prix_min": min(prix) if prix else None,
            "prix_max": max(prix) if prix else None,
            "par_statut": par_statut,
            "par_client": par_client,
            "par_mois": par_mois
        }

bovin_service = BovinService()