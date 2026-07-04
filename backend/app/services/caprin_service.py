# backend/app/services/caprin_service.py
"""
Service de gestion des caprins (chèvres)
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from datetime import date, timedelta

from ..models.caprin import Caprin
from ..models.animal import StatutAnimalEnum, SexeEnum
from ..models.pesee import Pesee
from ..models.naissance import Naissance
from ..models.mortalite import Mortalite
from ..schemas.caprin import CaprinCreate, CaprinUpdate
from .animal_service import animal_service
from app.services.id_service import generate_identification, log_action
from app.core.validators import validate_animal_age
from app.services.media_service import media_service

logger = logging.getLogger(__name__)


class CaprinService:
    """Service de gestion des caprins (chèvres)"""
    
    # Durée de gestation en jours
    GESTATION_DAYS = 150
    
    async def create_caprin(
        self,
        db: AsyncSession,
        caprin_data: CaprinCreate,
        created_by: int
    ) -> Tuple[Optional[Caprin], Optional[str]]:
        """Créer un nouveau caprin"""
        
        # Vérifier que l'enclos existe
        enclos_name = await animal_service.get_enclos_name_by_id(db, caprin_data.enclos_id)
        if not enclos_name:
            return None, f"Enclos non trouvé"
        
        # Générer l'identification
        identification = await generate_identification(db, "cap")
        
        # Vérifier l'âge si date de naissance fournie
        if caprin_data.date_naissance:
            valid, error = validate_animal_age(caprin_data.date_naissance)
            if not valid:
                return None, error
        
        # Créer le caprin
        caprin = Caprin(
            type_espece="caprin",
            identification=identification,
            race=caprin_data.race,
            sexe=caprin_data.sexe,
            date_naissance=caprin_data.date_naissance,
            date_arrivee=caprin_data.date_arrivee or date.today(),
            provenance=caprin_data.provenance,
            prix_achat=caprin_data.prix_achat,
            enclos_id=caprin_data.enclos_id,
            statut=caprin_data.statut or StatutAnimalEnum.VIVANT,
            notes=caprin_data.notes,
            production_viande=caprin_data.production_viande,
            production_reproduction=caprin_data.production_reproduction
        )
        
        db.add(caprin)
        await db.flush()
        
        # Sauvegarder la photo si présente
        if caprin_data.photo_base64:
            photo_url = media_service.save_base64_photo(
                caprin_data.photo_base64, "caprin", caprin.id
            )
            if photo_url:
                caprin.photo_url = photo_url
                await db.flush()
        
        # Ajouter le poids initial si fourni
        if caprin_data.poids_initial:
            pesee = Pesee(
                animal_id=caprin.id,
                date_pesee=caprin_data.date_arrivee or date.today(),
                poids=caprin_data.poids_initial,
                methode="Initiale",
                notes="Poids à l'arrivée"
            )
            db.add(pesee)
        
        await db.commit()
        await db.refresh(caprin)
        
        # Journaliser
        await log_action(
            db, created_by, "CREATE_CAPRIN", "caprin", caprin.id,
            {"identification": caprin.identification, "race": caprin_data.race}
        )
        
        logger.info(f"Caprin created: {caprin.identification} by {created_by}")
        return caprin, None
    
    async def get_caprin(
        self,
        db: AsyncSession,
        caprin_id: int
    ) -> Optional[Caprin]:
        """Obtenir un caprin par son ID"""
        stmt = select(Caprin).where(Caprin.id == caprin_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_caprin_by_identification(
        self,
        db: AsyncSession,
        identification: str
    ) -> Optional[Caprin]:
        """Obtenir un caprin par son identification"""
        stmt = select(Caprin).where(Caprin.identification == identification)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_caprins(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        race: Optional[str] = None,
        sexe: Optional[SexeEnum] = None,
        enclos_id: Optional[int] = None,
        production_type: Optional[str] = None,
        statut: Optional[StatutAnimalEnum] = None,
        search: Optional[str] = None
    ) -> List[Caprin]:
        """Obtenir la liste des caprins avec filtres"""
        stmt = select(Caprin)
        
        if race:
            stmt = stmt.where(Caprin.race == race)
        if sexe:
            stmt = stmt.where(Caprin.sexe == sexe)
        if enclos_id:
            stmt = stmt.where(Caprin.enclos_id == enclos_id)
        if statut:
            stmt = stmt.where(Caprin.statut == statut)
        if production_type:
            if production_type == "viande":
                stmt = stmt.where(Caprin.production_viande == True)
            elif production_type == "reproduction":
                stmt = stmt.where(Caprin.production_reproduction == True)
        if search:
            stmt = stmt.where(
                or_(
                    Caprin.identification.ilike(f"%{search}%"),
                    Caprin.race.ilike(f"%{search}%")
                )
            )
        
        stmt = stmt.offset(skip).limit(limit).order_by(Caprin.date_arrivee.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_all_caprins_with_filters(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        race: Optional[str] = None,
        production_type: Optional[str] = None,
        enclos_id: Optional[int] = None,
        sexe: Optional[str] = None,
        statut: Optional[StatutAnimalEnum] = None,
        search: Optional[str] = None
    ) -> List[Caprin]:
        """Obtenir tous les caprins avec filtres (sans pagination pour compter)"""
        stmt = select(Caprin)
        statut_list = None
        if statut:
            if isinstance(statut, str) and ',' in statut:
                statut_list = [s.strip() for s in statut.split(',') if s.strip()]
            elif isinstance(statut, str):
                statut_list = [statut]
            elif isinstance(statut, list):
                statut_list = statut
            stmt = stmt.where(Caprin.statut.in_(statut_list))

        sexe_list = None
        if sexe:
            if isinstance(sexe, str) and ',' in sexe:
                sexe_list = [s.strip() for s in sexe.split(',') if s.strip()]
            elif isinstance(sexe, str):
                sexe_list = [sexe]
            elif isinstance(sexe, list):
                sexe_list = sexe
            stmt = stmt.where(Caprin.sexe.in_(sexe_list))

        if race:
            stmt = stmt.where(Caprin.race == race)
        if enclos_id:
            stmt = stmt.where(Caprin.enclos_id == enclos_id)
        if production_type:
            if production_type == "viande":
                stmt = stmt.where(Caprin.production_viande == True)
            elif production_type == "reproduction":
                stmt = stmt.where(Caprin.production_reproduction == True)
        if search:
            search_pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    Caprin.identification.ilike(search_pattern),
                    Caprin.race.ilike(search_pattern)
                )
            )
        
        stmt = stmt.offset(skip).limit(limit).order_by(Caprin.created_at.asc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_caprin(
        self,
        db: AsyncSession,
        caprin_id: int,
        caprin_data: CaprinUpdate,
        updated_by: int
    ) -> Tuple[Optional[Caprin], Optional[str]]:
        """Mettre à jour un caprin"""
        
        # Récupérer le caprin
        caprin = await self.get_caprin(db, caprin_id)
        if not caprin:
            return None, "Caprin non trouvé"
        
        # Sauvegarder l'ancien statut pour le log
        old_status = caprin.statut.value if caprin.statut else None
        
        # Mettre à jour les champs spécifiques aux caprins
        if caprin_data.production_viande is not None:
            caprin.production_viande = caprin_data.production_viande
        if caprin_data.production_reproduction is not None:
            caprin.production_reproduction = caprin_data.production_reproduction
        
        # Mettre à jour l'animal de base
        if caprin_data.race is not None:
            caprin.race = caprin_data.race
        if caprin_data.sexe is not None:
            caprin.sexe = caprin_data.sexe
        if caprin_data.date_naissance is not None:
            caprin.date_naissance = caprin_data.date_naissance
        if caprin_data.date_arrivee is not None:
            caprin.date_arrivee = caprin_data.date_arrivee
        if caprin_data.provenance is not None:
            caprin.provenance = caprin_data.provenance
        if caprin_data.prix_achat is not None:
            caprin.prix_achat = caprin_data.prix_achat
        if caprin_data.enclos_id is not None:
            enclos_name = await animal_service.get_enclos_name_by_id(db, caprin_data.enclos_id)
            if enclos_name:
                caprin.enclos_id = caprin_data.enclos_id
        if caprin_data.statut is not None:
            caprin.statut = caprin_data.statut
        if caprin_data.notes is not None:
            caprin.notes = caprin_data.notes
        
        # Traiter la nouvelle photo
        if caprin_data.photo_base64:
            if caprin.photo_url:
                media_service.delete_photo(caprin.photo_url)
            
            photo_url = media_service.save_base64_photo(
                caprin_data.photo_base64, "caprin", caprin_id
            )
            if photo_url:
                caprin.photo_url = photo_url
                await db.flush()
        
        # Ajouter un nouveau poids si fourni
        if caprin_data.poids_initial:
            pesee = Pesee(
                animal_id=caprin_id,
                poids=caprin_data.poids_initial,
                date_pesee=date.today(),
                methode="Manuelle",
                notes=f"Mise à jour le {date.today()}",
                created_by=updated_by
            )
            db.add(pesee)
        
        await db.commit()
        await db.refresh(caprin)
        
        # Journaliser la mise à jour
        await log_action(
            db, updated_by, "UPDATE_CAPRIN", "caprin", caprin_id,
            {
                "old_status": old_status,
                "new_status": caprin_data.statut.value if caprin_data.statut else None
            }
        )
        
        logger.info(f"Caprin updated: {caprin.identification} by {updated_by}")
        return caprin, None
    
    async def get_caprin_stats(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir des statistiques sur les caprins"""
        stmt = select(Caprin).where(
            Caprin.statut == StatutAnimalEnum.VIVANT
        )
        if enclos_id:
            stmt = stmt.where(Caprin.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        caprins = result.scalars().all()
        
        total = len(caprins)
        males = len([c for c in caprins if c.sexe == SexeEnum.MALE])
        females = len([c for c in caprins if c.sexe == SexeEnum.FEMELLE])
        
        production_viande = len([c for c in caprins if c.production_viande])
        production_reproduction = len([c for c in caprins if c.production_reproduction])
        
        # Poids moyen
        total_weight = 0
        weight_count = 0
        for c in caprins:
            dernier_poids = await animal_service.get_last_weight(db, c.id)
            if dernier_poids:
                total_weight += dernier_poids
                weight_count += 1
        
        avg_weight = total_weight / weight_count if weight_count > 0 else 0
        
        races = {}
        for c in caprins:
            races[c.race] = races.get(c.race, 0) + 1
        
        return {
            "total": total,
            "males": males,
            "femelles": females,
            "ratio_m_f": round(males / females, 2) if females > 0 else 0,
            "races": races,
            "production": {
                "viande": production_viande,
                "reproduction": production_reproduction
            },
            "poids_moyen_kg": round(avg_weight, 1),
            "nombre_enclos": len(set(c.enclos_id for c in caprins if c.enclos_id))
        }
    
    async def get_weight_progression(
        self,
        db: AsyncSession,
        caprin_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtenir la progression des poids d'un caprin"""
        stmt = (
            select(Pesee)
            .where(Pesee.animal_id == caprin_id)
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
    
    async def get_pregnant_females(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> List[Caprin]:
        """Obtenir les femelles gestantes"""
        min_age_date = date.today() - timedelta(days=180)
        max_age_date = date.today() - timedelta(days=8*365)
        
        stmt = select(Caprin).where(
            Caprin.sexe == SexeEnum.FEMELLE,
            Caprin.statut == StatutAnimalEnum.VIVANT,
            Caprin.production_reproduction == True,
            Caprin.date_naissance <= min_age_date,
            Caprin.date_naissance >= max_age_date
        )
        if enclos_id:
            stmt = stmt.where(Caprin.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_kids(
        self,
        db: AsyncSession,
        age_days_max: int = 90,
        enclos_id: Optional[int] = None
    ) -> List[Caprin]:
        """Obtenir les chevreaux (jeunes caprins)"""
        cutoff_date = date.today() - timedelta(days=age_days_max)
        
        stmt = select(Caprin).where(
            Caprin.date_naissance >= cutoff_date,
            Caprin.statut == StatutAnimalEnum.VIVANT
        )
        if enclos_id:
            stmt = stmt.where(Caprin.enclos_id == enclos_id)
        
        stmt = stmt.order_by(Caprin.date_naissance.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_breeding_males(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> List[Caprin]:
        """Obtenir les boucs reproducteurs"""
        stmt = select(Caprin).where(
            Caprin.sexe == SexeEnum.MALE,
            Caprin.statut == StatutAnimalEnum.VIVANT,
            Caprin.production_reproduction == True
        )
        if enclos_id:
            stmt = stmt.where(Caprin.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def estimate_kidding_date(
        self,
        mating_date: date,
        doe_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Estimer la date de mise bas (chevrotage)"""
        estimated_date = mating_date + timedelta(days=self.GESTATION_DAYS)
        
        return {
            "mating_date": mating_date,
            "estimated_kidding_date": estimated_date,
            "gestation_days": self.GESTATION_DAYS,
            "adjustment_days": 0,
            "confidence": "medium",
            "earliest_date": estimated_date - timedelta(days=5),
            "latest_date": estimated_date + timedelta(days=5)
        }
    
    async def record_kidding(
        self,
        db: AsyncSession,
        doe_id: int,
        buck_id: Optional[int],
        kids_data: List[Dict[str, Any]],
        recorded_by: int
    ) -> Tuple[List[Caprin], Optional[str]]:
        """Enregistrer une mise bas (chevrotage)"""
        doe = await self.get_caprin(db, doe_id)
        if not doe:
            return None, "Chèvre non trouvée"
        
        if doe.sexe != SexeEnum.FEMELLE:
            return None, "L'animal sélectionné n'est pas une femelle"
        
        kids = []
        for kid_data in kids_data:
            import random
            
            identification = f"KID-{doe.identification}-{random.randint(1000, 9999)}"
            
            # Déterminer le sexe
            sexe_value = kid_data.get("sexe", "female")
            if isinstance(sexe_value, str):
                sexe = SexeEnum.MALE if sexe_value.lower() == "male" else SexeEnum.FEMELLE
            else:
                sexe = sexe_value
            
            # Créer les données du chevreau
            kid_create = CaprinCreate(
                identification=identification,
                race=doe.race,
                sexe=sexe,
                date_naissance=date.today(),
                date_arrivee=date.today(),
                provenance=f"Naissance à la ferme - Mère: {doe.identification}",
                enclos_name=doe.enclos.name if doe.enclos else None,
                statut=StatutAnimalEnum.VIVANT,
                production_viande=True,
                production_reproduction=(sexe == SexeEnum.FEMELLE),
                poids_initial=kid_data.get("poids_naissance"),
                notes=kid_data.get("notes")
            )
            
            kid, error = await self.create_caprin(db, kid_create, recorded_by)
            if error:
                return None, f"Erreur création chevreau: {error}"
            
            # Enregistrer la naissance
            naissance = Naissance(
                mere_id=doe_id,
                pere_caprin_id=buck_id,
                animal_ne_id=kid.id,
                date_naissance=date.today(),
                poids_naissance=kid_data.get("poids_naissance"),
                sexe=sexe.value,
                notes=kid_data.get("notes")
            )
            db.add(naissance)
            
            kids.append(kid)
        
        await db.commit()
        
        logger.info(f"Kidding recorded for doe {doe.identification} by {recorded_by}: {len(kids)} kids")
        return kids, None
    
    async def record_death(
        self,
        db: AsyncSession,
        caprin_id: int,
        death_date: date,
        cause: Optional[str],
        recorded_by: int,
        autopsie_realisee: bool = False
    ) -> Tuple[bool, str]:
        """Enregistrer le décès d'un caprin"""
        caprin = await self.get_caprin(db, caprin_id)
        if not caprin:
            return False, "Caprin non trouvé"
        
        mortalite = Mortalite(
            animal_id=caprin_id,
            date_mort=death_date,
            cause=cause,
            autopsie_realisee=autopsie_realisee
        )
        db.add(mortalite)
        
        caprin.statut = StatutAnimalEnum.DECEDE
        
        await db.commit()
        
        logger.info(f"Death recorded for caprin {caprin.identification} by {recorded_by}: {cause}")
        return True, "Décès enregistré avec succès"
    
    # === NOUVELLES MÉTHODES POUR LES VENTES ===
    async def get_ventes_stats(
        self,
        db: AsyncSession,
        date_debut: Optional[date] = None,
        date_fin: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Obtenir les statistiques des ventes de caprins
        
        Args:
            db: Session de base de données
            date_debut: Date de début pour filtrer les ventes
            date_fin: Date de fin pour filtrer les ventes
        
        Returns:
            Dict contenant les statistiques des ventes
        """
        
        # Construction de la requête de base
        stmt = select(Caprin).where(Caprin.prix_vente.isnot(None))
        
        if date_debut:
            stmt = stmt.where(Caprin.date_vente >= date_debut)
        if date_fin:
            stmt = stmt.where(Caprin.date_vente <= date_fin)
        
        result = await db.execute(stmt)
        caprins_vendus = result.scalars().all()
        
        if not caprins_vendus:
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
        prix = [c.prix_vente for c in caprins_vendus if c.prix_vente]
        
        # Statistiques par statut
        par_statut = {}
        for c in caprins_vendus:
            statut_key = c.statut.value if c.statut else "unknown"
            par_statut[statut_key] = par_statut.get(statut_key, 0) + 1
        
        # Statistiques par client
        par_client = {}
        for c in caprins_vendus:
            client = c.client_acheteur or "Inconnu"
            par_client[client] = par_client.get(client, 0) + 1
        
        # Statistiques par mois
        par_mois = {}
        for c in caprins_vendus:
            if c.date_vente:
                mois_key = c.date_vente.strftime("%Y-%m")
                par_mois[mois_key] = par_mois.get(mois_key, 0) + c.prix_vente
        
        return {
            "total_ventes": len(caprins_vendus),
            "montant_total": sum(prix),
            "prix_moyen": sum(prix) / len(prix) if prix else 0,
            "prix_min": min(prix) if prix else None,
            "prix_max": max(prix) if prix else None,
            "par_statut": par_statut,
            "par_client": par_client,
            "par_mois": par_mois
        }
    
    async def get_all_caprins(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Caprin]:
        """Obtenir tous les caprins"""
        stmt = select(Caprin).offset(skip).limit(limit).order_by(Caprin.identification)
        result = await db.execute(stmt)
        return result.scalars().all()


caprin_service = CaprinService()