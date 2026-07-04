# backend/app/services/ovin_service.py
"""
Service de gestion des ovins (moutons)
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from datetime import date, timedelta

from ..models.ovin import Ovin
from ..models.animal import StatutAnimalEnum, SexeEnum
from ..models.pesee import Pesee
from ..models.naissance import Naissance
from ..models.mortalite import Mortalite
from ..schemas.ovin import OvinCreate, OvinUpdate
from .animal_service import animal_service
from app.services.id_service import generate_identification, log_action
from app.core.validators import validate_animal_age
from app.services.media_service import media_service

logger = logging.getLogger(__name__)


class OvinService:
    """Service de gestion des ovins (moutons)"""
    
    # Durée de gestation en jours
    GESTATION_DAYS = 150
    
    async def create_ovin(
        self,
        db: AsyncSession,
        ovin_data: OvinCreate,
        created_by: int
    ) -> Tuple[Optional[Ovin], Optional[str]]:
        """Créer un nouvel ovin"""
        
        # Vérifier que l'enclos existe
        enclos_name = await animal_service.get_enclos_name_by_id(db, ovin_data.enclos_id)
        if not enclos_name:
            return None, f"Enclos non trouvé"
        
        # Générer l'identification
        identification = await generate_identification(db, "ovi")
        
        # Vérifier l'âge si date de naissance fournie
        if ovin_data.date_naissance:
            valid, error = validate_animal_age(ovin_data.date_naissance)
            if not valid:
                return None, error
        
        # Créer l'ovin
        ovin = Ovin(
            type_espece="ovin",
            identification=identification,
            race=ovin_data.race,
            sexe=ovin_data.sexe,
            date_naissance=ovin_data.date_naissance,
            date_arrivee=ovin_data.date_arrivee or date.today(),
            provenance=ovin_data.provenance,
            prix_achat=ovin_data.prix_achat,
            enclos_id=ovin_data.enclos_id,
            statut=ovin_data.statut or StatutAnimalEnum.VIVANT,
            notes=ovin_data.notes,
            production_viande=ovin_data.production_viande,
            production_reproduction=ovin_data.production_reproduction,
            production_laine=ovin_data.production_laine,
            qualite_laine=ovin_data.qualite_laine
        )
        
        db.add(ovin)
        await db.flush()
        
        # Sauvegarder la photo si présente
        if ovin_data.photo_base64:
            photo_url = media_service.save_base64_photo(
                ovin_data.photo_base64, "ovin", ovin.id
            )
            if photo_url:
                ovin.photo_url = photo_url
                await db.flush()
        
        # Ajouter le poids initial si fourni
        if ovin_data.poids_initial:
            pesee = Pesee(
                animal_id=ovin.id,
                date_pesee=ovin_data.date_arrivee or date.today(),
                poids=ovin_data.poids_initial,
                methode="Initiale",
                notes="Poids à l'arrivée"
            )
            db.add(pesee)
        
        await db.commit()
        await db.refresh(ovin)
        
        # Journaliser
        await log_action(
            db, created_by, "CREATE_OVIN", "ovin", ovin.id,
            {"identification": ovin.identification, "race": ovin_data.race}
        )
        
        logger.info(f"Ovin created: {ovin.identification} by {created_by}")
        return ovin, None
    
    async def get_ovin(
        self,
        db: AsyncSession,
        ovin_id: int
    ) -> Optional[Ovin]:
        """Obtenir un ovin par son ID"""
        stmt = select(Ovin).where(Ovin.id == ovin_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_ovin_by_identification(
        self,
        db: AsyncSession,
        identification: str
    ) -> Optional[Ovin]:
        """Obtenir un ovin par son identification"""
        stmt = select(Ovin).where(Ovin.identification == identification)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_ovins(
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
    ) -> List[Ovin]:
        """Obtenir la liste des ovins avec filtres"""
        stmt = select(Ovin)
        
        if race:
            stmt = stmt.where(Ovin.race == race)
        if sexe:
            stmt = stmt.where(Ovin.sexe == sexe)
        if enclos_id:
            stmt = stmt.where(Ovin.enclos_id == enclos_id)
        if statut:
            stmt = stmt.where(Ovin.statut == statut)
        if production_type:
            if production_type == "viande":
                stmt = stmt.where(Ovin.production_viande == True)
            elif production_type == "laine":
                stmt = stmt.where(Ovin.production_laine == True)
            elif production_type == "reproduction":
                stmt = stmt.where(Ovin.production_reproduction == True)
        if search:
            stmt = stmt.where(
                or_(
                    Ovin.identification.ilike(f"%{search}%"),
                    Ovin.race.ilike(f"%{search}%")
                )
            )
        
        stmt = stmt.offset(skip).limit(limit).order_by(Ovin.date_arrivee.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_all_ovins_with_filters(
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
    ) -> List[Ovin]:
        """Obtenir tous les ovins avec filtres (sans pagination pour compter)"""
        stmt = select(Ovin)

        statut_list = None
        if statut:
            if isinstance(statut, str) and ',' in statut:
                statut_list = [s.strip() for s in statut.split(',') if s.strip()]
            elif isinstance(statut, str):
                statut_list = [statut]
            elif isinstance(statut, list):
                statut_list = statut
            stmt = stmt.where(Ovin.statut.in_(statut_list))

        sexe_list = None
        if sexe:
            if isinstance(sexe, str) and ',' in sexe:
                sexe_list = [s.strip() for s in sexe.split(',') if s.strip()]
            elif isinstance(sexe, str):
                sexe_list = [sexe]
            elif isinstance(sexe, list):
                sexe_list = sexe
            stmt = stmt.where(Ovin.sexe.in_(sexe_list))

        if race:
            stmt = stmt.where(Ovin.race == race)
        if enclos_id:
            stmt = stmt.where(Ovin.enclos_id == enclos_id)
        if production_type:
            if production_type == "viande":
                stmt = stmt.where(Ovin.production_viande == True)
            elif production_type == "laine":
                stmt = stmt.where(Ovin.production_laine == True)
            elif production_type == "reproduction":
                stmt = stmt.where(Ovin.production_reproduction == True)
        if search:
            search_pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    Ovin.identification.ilike(search_pattern),
                    Ovin.race.ilike(search_pattern)
                )
            )
        
        stmt = stmt.offset(skip).limit(limit).order_by(Ovin.created_at.asc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_ovin(
        self,
        db: AsyncSession,
        ovin_id: int,
        ovin_data: OvinUpdate,
        updated_by: int
    ) -> Tuple[Optional[Ovin], Optional[str]]:
        """Mettre à jour un ovin"""
        
        # Récupérer l'ovin
        ovin = await self.get_ovin(db, ovin_id)
        if not ovin:
            return None, "Ovin non trouvé"
        
        # Sauvegarder l'ancien statut pour le log
        old_status = ovin.statut.value if ovin.statut else None
        
        # Mettre à jour les champs spécifiques aux ovins
        if ovin_data.production_viande is not None:
            ovin.production_viande = ovin_data.production_viande
        if ovin_data.production_reproduction is not None:
            ovin.production_reproduction = ovin_data.production_reproduction
        if ovin_data.production_laine is not None:
            ovin.production_laine = ovin_data.production_laine
        if ovin_data.qualite_laine is not None:
            ovin.qualite_laine = ovin_data.qualite_laine
        
        # Mettre à jour l'animal de base
        if ovin_data.race is not None:
            ovin.race = ovin_data.race
        if ovin_data.sexe is not None:
            ovin.sexe = ovin_data.sexe
        if ovin_data.date_naissance is not None:
            ovin.date_naissance = ovin_data.date_naissance
        if ovin_data.date_arrivee is not None:
            ovin.date_arrivee = ovin_data.date_arrivee
        if ovin_data.provenance is not None:
            ovin.provenance = ovin_data.provenance
        if ovin_data.prix_achat is not None:
            ovin.prix_achat = ovin_data.prix_achat
        if ovin_data.enclos_id is not None:
            enclos_name = await animal_service.get_enclos_name_by_id(db, ovin_data.enclos_id)
            if enclos_name:
                ovin.enclos_id = ovin_data.enclos_id
        if ovin_data.statut is not None:
            ovin.statut = ovin_data.statut
        if ovin_data.notes is not None:
            ovin.notes = ovin_data.notes
        
        # Traiter la nouvelle photo
        if ovin_data.photo_base64:
            if ovin.photo_url:
                media_service.delete_photo(ovin.photo_url)
            
            photo_url = media_service.save_base64_photo(
                ovin_data.photo_base64, "ovin", ovin_id
            )
            if photo_url:
                ovin.photo_url = photo_url
                await db.flush()
        
        # Ajouter un nouveau poids si fourni
        if ovin_data.poids_initial:
            pesee = Pesee(
                animal_id=ovin_id,
                poids=ovin_data.poids_initial,
                date_pesee=date.today(),
                methode="Manuelle",
                notes=f"Mise à jour le {date.today()}",
                created_by=updated_by
            )
            db.add(pesee)
        
        await db.commit()
        await db.refresh(ovin)
        
        # Journaliser la mise à jour
        await log_action(
            db, updated_by, "UPDATE_OVIN", "ovin", ovin_id,
            {
                "old_status": old_status,
                "new_status": ovin_data.statut.value if ovin_data.statut else None
            }
        )
        
        logger.info(f"Ovin updated: {ovin.identification} by {updated_by}")
        return ovin, None
    
    async def get_ovin_stats(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir des statistiques sur les ovins"""
        stmt = select(Ovin).where(
            Ovin.statut == StatutAnimalEnum.VIVANT
        )
        if enclos_id:
            stmt = stmt.where(Ovin.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        ovins = result.scalars().all()
        
        total = len(ovins)
        males = len([o for o in ovins if o.sexe == SexeEnum.MALE])
        females = len([o for o in ovins if o.sexe == SexeEnum.FEMELLE])
        
        production_viande = len([o for o in ovins if o.production_viande])
        production_laine = len([o for o in ovins if o.production_laine])
        production_reproduction = len([o for o in ovins if o.production_reproduction])
        
        # Poids moyen
        total_weight = 0
        weight_count = 0
        for o in ovins:
            dernier_poids = await animal_service.get_last_weight(db, o.id)
            if dernier_poids:
                total_weight += dernier_poids
                weight_count += 1
        
        avg_weight = total_weight / weight_count if weight_count > 0 else 0
        
        races = {}
        for o in ovins:
            races[o.race] = races.get(o.race, 0) + 1
        
        return {
            "total": total,
            "males": males,
            "femelles": females,
            "ratio_m_f": round(males / females, 2) if females > 0 else 0,
            "races": races,
            "production": {
                "viande": production_viande,
                "laine": production_laine,
                "reproduction": production_reproduction
            },
            "poids_moyen_kg": round(avg_weight, 1),
            "nombre_enclos": len(set(o.enclos_id for o in ovins if o.enclos_id))
        }
    
    async def get_weight_progression(
        self,
        db: AsyncSession,
        ovin_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtenir la progression des poids d'un ovin"""
        stmt = (
            select(Pesee)
            .where(Pesee.animal_id == ovin_id)
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
    
    async def get_lambs(
        self,
        db: AsyncSession,
        age_days_max: int = 120,
        enclos_id: Optional[int] = None
    ) -> List[Ovin]:
        """Obtenir les agneaux (jeunes ovins)"""
        cutoff_date = date.today() - timedelta(days=age_days_max)
        
        stmt = select(Ovin).where(
            Ovin.date_naissance >= cutoff_date,
            Ovin.statut == StatutAnimalEnum.VIVANT
        )
        if enclos_id:
            stmt = stmt.where(Ovin.enclos_id == enclos_id)
        
        stmt = stmt.order_by(Ovin.date_naissance.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_ewes_for_breeding(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> List[Ovin]:
        """Obtenir les brebis aptes à la reproduction"""
        min_age_date = date.today() - timedelta(days=180)
        max_age_date = date.today() - timedelta(days=8*365)
        
        stmt = select(Ovin).where(
            Ovin.sexe == SexeEnum.FEMELLE,
            Ovin.statut == StatutAnimalEnum.VIVANT,
            Ovin.production_reproduction == True,
            Ovin.date_naissance <= min_age_date,
            Ovin.date_naissance >= max_age_date
        )
        if enclos_id:
            stmt = stmt.where(Ovin.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_rams(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> List[Ovin]:
        """Obtenir les béliers reproducteurs"""
        stmt = select(Ovin).where(
            Ovin.sexe == SexeEnum.MALE,
            Ovin.statut == StatutAnimalEnum.VIVANT,
            Ovin.production_reproduction == True
        )
        if enclos_id:
            stmt = stmt.where(Ovin.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def estimate_lambing_date(
        self,
        mating_date: date,
        ewe_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Estimer la date d'agnelage"""
        estimated_date = mating_date + timedelta(days=self.GESTATION_DAYS)
        
        return {
            "mating_date": mating_date,
            "estimated_lambing_date": estimated_date,
            "gestation_days": self.GESTATION_DAYS,
            "adjustment_days": 0,
            "confidence": "medium",
            "earliest_date": estimated_date - timedelta(days=5),
            "latest_date": estimated_date + timedelta(days=5)
        }
    
    async def record_lambing(
        self,
        db: AsyncSession,
        ewe_id: int,
        ram_id: Optional[int],
        lambs_data: List[Dict[str, Any]],
        recorded_by: int
    ) -> Tuple[List[Ovin], Optional[str]]:
        """Enregistrer un agnelage"""
        ewe = await self.get_ovin(db, ewe_id)
        if not ewe:
            return None, "Brebis non trouvée"
        
        if ewe.sexe != SexeEnum.FEMELLE:
            return None, "L'animal sélectionné n'est pas une femelle"
        
        lambs = []
        for lamb_data in lambs_data:
            import random
            
            identification = f"LAMB-{ewe.identification}-{random.randint(1000, 9999)}"
            
            sexe_value = lamb_data.get("sexe", "female")
            if isinstance(sexe_value, str):
                sexe = SexeEnum.MALE if sexe_value.lower() == "male" else SexeEnum.FEMELLE
            else:
                sexe = sexe_value
            
            lamb_create = OvinCreate(
                identification=identification,
                race=ewe.race,
                sexe=sexe,
                date_naissance=date.today(),
                date_arrivee=date.today(),
                provenance=f"Naissance à la ferme - Mère: {ewe.identification}",
                enclos_name=ewe.enclos.name if ewe.enclos else None,
                statut=StatutAnimalEnum.VIVANT,
                production_viande=True,
                production_reproduction=False,
                production_laine=True,
                qualite_laine=ewe.qualite_laine,
                poids_initial=lamb_data.get("poids_naissance"),
                notes=lamb_data.get("notes")
            )
            
            lamb, error = await self.create_ovin(db, lamb_create, recorded_by)
            if error:
                return None, f"Erreur création agneau: {error}"
            
            naissance = Naissance(
                mere_id=ewe_id,
                pere_ovin_id=ram_id,
                animal_ne_id=lamb.id,
                date_naissance=date.today(),
                poids_naissance=lamb_data.get("poids_naissance"),
                sexe=sexe.value,
                notes=lamb_data.get("notes")
            )
            db.add(naissance)
            
            lambs.append(lamb)
        
        await db.commit()
        
        logger.info(f"Lambing recorded for ewe {ewe.identification} by {recorded_by}: {len(lambs)} lambs")
        return lambs, None
    
    async def record_death(
        self,
        db: AsyncSession,
        ovin_id: int,
        death_date: date,
        cause: Optional[str],
        recorded_by: int,
        autopsie_realisee: bool = False
    ) -> Tuple[bool, str]:
        """Enregistrer le décès d'un ovin"""
        ovin = await self.get_ovin(db, ovin_id)
        if not ovin:
            return False, "Ovin non trouvé"
        
        mortalite = Mortalite(
            animal_id=ovin_id,
            date_mort=death_date,
            cause=cause,
            autopsie_realisee=autopsie_realisee
        )
        db.add(mortalite)
        
        ovin.statut = StatutAnimalEnum.DECEDE
        
        await db.commit()
        
        logger.info(f"Death recorded for ovin {ovin.identification} by {recorded_by}: {cause}")
        return True, "Décès enregistré avec succès"
    
    async def get_wool_production(
        self,
        db: AsyncSession,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques de production de laine"""
        from datetime import datetime
        
        if not year:
            year = datetime.now().year
        
        min_birth_date = date(year - 1, 1, 1)
        
        stmt = select(Ovin).where(
            Ovin.date_naissance <= min_birth_date,
            Ovin.production_laine == True,
            Ovin.statut == StatutAnimalEnum.VIVANT
        )
        result = await db.execute(stmt)
        ovins = result.scalars().all()
        
        wool_yield = {
            "merino": 4.5,
            "dorset": 3.5,
            "suffolk": 2.5,
            "rambouillet": 4.0,
            "default": 3.0
        }
        
        total_wool = 0
        by_race = {}
        
        for o in ovins:
            yield_kg = wool_yield.get(o.race.lower(), wool_yield["default"])
            total_wool += yield_kg
            by_race[o.race] = by_race.get(o.race, 0) + yield_kg
        
        return {
            "year": year,
            "total_wool_kg": round(total_wool, 1),
            "average_per_animal_kg": round(total_wool / len(ovins), 1) if ovins else 0,
            "number_of_animals": len(ovins),
            "by_race": by_race,
            "estimated_value_eur": round(total_wool * 3.5, 0)
        }
    
    # === NOUVELLES MÉTHODES POUR LES VENTES ===
    async def get_ventes_stats(
        self,
        db: AsyncSession,
        date_debut: Optional[date] = None,
        date_fin: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Obtenir les statistiques des ventes d'ovins
        
        Args:
            db: Session de base de données
            date_debut: Date de début pour filtrer les ventes
            date_fin: Date de fin pour filtrer les ventes
        
        Returns:
            Dict contenant les statistiques des ventes
        """
        
        # Construction de la requête de base
        stmt = select(Ovin).where(Ovin.prix_vente.isnot(None))
        
        if date_debut:
            stmt = stmt.where(Ovin.date_vente >= date_debut)
        if date_fin:
            stmt = stmt.where(Ovin.date_vente <= date_fin)
        
        result = await db.execute(stmt)
        ovins_vendus = result.scalars().all()
        
        if not ovins_vendus:
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
        prix = [o.prix_vente for o in ovins_vendus if o.prix_vente]
        
        # Statistiques par statut
        par_statut = {}
        for o in ovins_vendus:
            statut_key = o.statut.value if o.statut else "unknown"
            par_statut[statut_key] = par_statut.get(statut_key, 0) + 1
        
        # Statistiques par client
        par_client = {}
        for o in ovins_vendus:
            client = o.client_acheteur or "Inconnu"
            par_client[client] = par_client.get(client, 0) + 1
        
        # Statistiques par mois
        par_mois = {}
        for o in ovins_vendus:
            if o.date_vente:
                mois_key = o.date_vente.strftime("%Y-%m")
                par_mois[mois_key] = par_mois.get(mois_key, 0) + o.prix_vente
        
        return {
            "total_ventes": len(ovins_vendus),
            "montant_total": sum(prix),
            "prix_moyen": sum(prix) / len(prix) if prix else 0,
            "prix_min": min(prix) if prix else None,
            "prix_max": max(prix) if prix else None,
            "par_statut": par_statut,
            "par_client": par_client,
            "par_mois": par_mois
        }
    
    async def get_all_ovins(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Ovin]:
        """Obtenir tous les ovins"""
        stmt = select(Ovin).offset(skip).limit(limit).order_by(Ovin.identification)
        result = await db.execute(stmt)
        return result.scalars().all()


ovin_service = OvinService()