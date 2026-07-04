# backend/app/services/avicole_service.py
"""
Service de gestion des avicoles (volailles)
"""

import logging
from datetime import date, timedelta
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.avicole import Avicole, StatutLotAvicoleEnum, TypeProductionAvicoleEnum
from ..models.enclos import Enclos
from ..models.pesee import Pesee
from ..schemas.avicole import *
from ..services.id_service import generate_identification, log_action

logger = logging.getLogger(__name__)


class AvicoleService:
    """Service de gestion des avicoles (lots de volailles)"""
    
    async def get_enclos_by_name(self, db: AsyncSession, name: str) -> Optional[Enclos]:
        """Trouve un enclos par son nom"""
        stmt = select(Enclos).where(Enclos.name == name)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create_avicole(
        self,
        db: AsyncSession,
        avicole_data: AvicoleCreate,
        created_by: int
    ) -> Tuple[Optional[Avicole], Optional[str]]:
        """Créer un nouveau lot avicole"""
        
        # Vérifier que l'enclos existe
        enclos = await self.get_enclos_by_name(db, avicole_data.enclos_name)
        if not enclos:
            return None, f"Enclos '{avicole_data.enclos_name}' non trouvé"
        
        # Générer l'identification
        identification = await generate_identification(db, "avi")
        
        # Créer le lot avicole
        avicole = Avicole(
            identification=identification,
            espece=avicole_data.espece,
            race=avicole_data.race,
            type_production=avicole_data.type_production,
            quantite_initial=avicole_data.quantite_initial,
            quantite_actuelle=avicole_data.quantite_initial,
            date_arrivee=avicole_data.date_arrivee or date.today(),
            provenance=avicole_data.provenance,
            prix_achat_unitaire=avicole_data.prix_achat_unitaire,
            prix_achat_total=avicole_data.prix_achat_unitaire * avicole_data.quantite_initial if avicole_data.prix_achat_unitaire else None,
            production_viande=avicole_data.type_production in [TypeProductionAvicoleEnum.VIANDE, TypeProductionAvicoleEnum.MIXTE],
            production_reproduction=avicole_data.type_production == TypeProductionAvicoleEnum.REPRODUCTION,
            production_ponte=avicole_data.type_production in [TypeProductionAvicoleEnum.PONTE, TypeProductionAvicoleEnum.MIXTE],
            enclos_id=enclos.id,
            poids_moyen_initial=avicole_data.poids_moyen_initial,
            poids_moyen_actuel=avicole_data.poids_moyen_initial,
            notes=avicole_data.notes,
            photo_url=avicole_data.photo_url
        )
        
        db.add(avicole)
        await db.flush()
        
        # Ajouter le poids initial si fourni
        if avicole_data.poids_initial_total:
            pesee = Pesee(
                lot_avicole_id=avicole.id,
                date_pesee=avicole_data.date_arrivee or date.today(),
                poids=avicole_data.poids_initial_total,
                methode="Initiale",
                notes=f"Poids total du lot à l'arrivée: {avicole_data.quantite_initial} individus"
            )
            db.add(pesee)
        
        await db.commit()
        await db.refresh(avicole)
        
        # Journaliser l'action
        await log_action(
            db, created_by, "CREATE_AVICOLE", "avicole", avicole.id,
            {"identification": avicole.identification, "espece": avicole.espece, "quantite": avicole.quantite_initial}
        )
        
        logger.info(f"Avicole lot created: {avicole.identification} ({avicole.quantite_initial} {avicole.espece}s) by {created_by}")
        return avicole, None
    
    async def get_avicole(
        self,
        db: AsyncSession,
        avicole_id: int,
        include_closed: bool = False
    ) -> Optional[Avicole]:
        """Obtenir un lot avicole par son ID"""
        stmt = select(Avicole).where(Avicole.id == avicole_id)
                
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_avicole_by_identification(
        self,
        db: AsyncSession,
        identification: str
    ) -> Optional[Avicole]:
        """Obtenir un lot avicole par son identification"""
        stmt = select(Avicole).where(Avicole.identification == identification)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_avicoles(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        enclos_id: Optional[int] = None,
        espece: Optional[str] = None,
        type_production: Optional[TypeProductionAvicoleEnum] = None,
        statut: Optional[StatutLotAvicoleEnum] = None
    ) -> List[Avicole]:
        """Obtenir la liste des lots avicoles avec filtres"""
        stmt = select(Avicole)
        
        if enclos_id:
            stmt = stmt.where(Avicole.enclos_id == enclos_id)
        if espece:
            stmt = stmt.where(Avicole.espece == espece)
        if type_production:
            stmt = stmt.where(Avicole.type_production == type_production)
        if statut:
            stmt = stmt.where(Avicole.statut == statut)
        
        stmt = stmt.offset(skip).limit(limit).order_by(Avicole.date_arrivee.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_avicole(
        self,
        db: AsyncSession,
        avicole_id: int,
        avicole_data: AvicoleUpdate,
        updated_by: int
    ) -> Tuple[Optional[Avicole], Optional[str]]:
        """Mettre à jour un lot avicole"""
        avicole = await self.get_avicole(db, avicole_id)
        if not avicole:
            return None, "Lot avicole non trouvé"
        
        # Sauvegarder l'ancien statut pour le log
        old_status = avicole.statut.value if avicole.statut else None
        
        # Mettre à jour les champs
        if avicole_data.espece is not None:
            avicole.espece = avicole_data.espece
        if avicole_data.race is not None:
            avicole.race = avicole_data.race
        if avicole_data.type_production is not None:
            avicole.type_production = avicole_data.type_production
            avicole.production_viande = avicole_data.type_production in [TypeProductionAvicoleEnum.VIANDE, TypeProductionAvicoleEnum.MIXTE]
            avicole.production_reproduction = avicole_data.type_production == TypeProductionAvicoleEnum.REPRODUCTION
            avicole.production_ponte = avicole_data.type_production in [TypeProductionAvicoleEnum.PONTE, TypeProductionAvicoleEnum.MIXTE]
        
        if avicole_data.quantite_actuelle is not None:
            avicole.quantite_actuelle = avicole_data.quantite_actuelle
        if avicole_data.enclos_name is not None:
            enclos = await self.get_enclos_by_name(db, avicole_data.enclos_name)
            if enclos:
                avicole.enclos_id = enclos.id
        if avicole_data.poids_moyen_actuel is not None:
            avicole.poids_moyen_actuel = avicole_data.poids_moyen_actuel
        if avicole_data.statut is not None:
            avicole.statut = avicole_data.statut
            if avicole_data.statut != StatutLotAvicoleEnum.ACTIF:
                avicole.date_fermeture = date.today()
        if avicole_data.notes is not None:
            avicole.notes = avicole_data.notes
        if avicole_data.photo_url is not None:
            avicole.photo_url = avicole_data.photo_url
        
        await db.commit()
        await db.refresh(avicole)
        
        # Journaliser la mise à jour
        await log_action(
            db, updated_by, "UPDATE_AVICOLE", "avicole", avicole_id,
            {"old_status": old_status, "new_status": avicole.statut.value if avicole.statut else None}
        )
        
        logger.info(f"Avicole updated: {avicole.identification} by {updated_by}")
        return avicole, None
    
    async def delete_avicole(
        self,
        db: AsyncSession,
        avicole_id: int,
        deleted_by: int
    ) -> bool:
        """Supprimer (soft delete) un lot avicole"""
        avicole = await self.get_avicole(db, avicole_id)
        if not avicole:
            return False
        
        # Soft delete via la colonne deleted_at (hérité de SoftDeleteMixin)
        avicole.deleted_at = date.today()
        
        await log_action(
            db, deleted_by, "DELETE_AVICOLE", "avicole", avicole_id,
            {"identification": avicole.identification}
        )
        await db.commit()
        
        logger.info(f"Avicole deleted: {avicole.identification} by {deleted_by}")
        return True
    
    async def add_mortality(
        self,
        db: AsyncSession,
        avicole_id: int,
        nombre_morts: int,
        cause: Optional[str] = None,
        date_mortalite: Optional[date] = None,
        recorded_by: int = None
    ) -> Tuple[bool, str]:
        """Ajouter des mortalités à un lot"""
        avicole = await self.get_avicole(db, avicole_id)
        if not avicole:
            return False, "Lot avicole non trouvé"
        
        if avicole.statut != StatutLotAvicoleEnum.ACTIF:
            return False, "Le lot n'est plus actif"
        
        if nombre_morts > avicole.quantite_actuelle:
            return False, f"Le nombre de morts ({nombre_morts}) dépasse la quantité actuelle ({avicole.quantite_actuelle})"
        
        # Mettre à jour les compteurs
        avicole.mortalite_total += nombre_morts
        avicole.quantite_actuelle -= nombre_morts
        avicole.taux_mortalite = (avicole.mortalite_total / avicole.quantite_initial) * 100
        
        # Enregistrer la mortalité
        from ..models.avicole import AvicoleMortalite
        mortalite = AvicoleMortalite(
            lot_id=avicole.id,
            date=date_mortalite or date.today(),
            nombre_morts=nombre_morts,
            cause=cause,
            notes=f"Enregistré par {recorded_by}" if recorded_by else None
        )
        db.add(mortalite)
        
        # Si tout le lot est mort, fermer le lot
        if avicole.quantite_actuelle <= 0:
            avicole.statut = StatutLotAvicoleEnum.DECEDE
            avicole.date_fermeture = date.today()
        
        await db.commit()
        
        logger.info(f"Mortality added for {avicole.identification}: {nombre_morts} deaths, remaining: {avicole.quantite_actuelle}")
        return True, f"{nombre_morts} mortalité(s) enregistrée(s)"
    
    async def add_egg_production(
        self,
        db: AsyncSession,
        avicole_id: int,
        nombre_oeufs: int,
        poids_oeufs_kg: Optional[float] = None,
        date_production: Optional[date] = None,
        recorded_by: int = None
    ) -> Tuple[bool, str]:
        """Ajouter une production d'œufs quotidienne"""
        avicole = await self.get_avicole(db, avicole_id)
        if not avicole:
            return False, "Lot avicole non trouvé"
        
        if not avicole.production_ponte:
            return False, "Ce lot n'est pas destiné à la production d'œufs"
        
        if avicole.statut != StatutLotAvicoleEnum.ACTIF:
            return False, "Le lot n'est plus actif"
        
        # Calculer le poids si non fourni (estimation: 60g par œuf)
        if poids_oeufs_kg is None:
            poids_oeufs_kg = nombre_oeufs * 0.060
        
        # Mettre à jour les totaux
        avicole.oeufs_pondus_total += nombre_oeufs
        avicole.oeufs_pondus_jour = nombre_oeufs
        avicole.poids_oeufs_total += poids_oeufs_kg
        
        # Mettre à jour le poids moyen par œuf
        if avicole.oeufs_pondus_total > 0:
            avicole.poids_oeufs_moyen = (avicole.poids_oeufs_total / avicole.oeufs_pondus_total) * 1000  # en grammes
        
        # Enregistrer la production quotidienne
        from ..models.avicole import AvicoleProduction
        production = AvicoleProduction(
            lot_id=avicole.id,
            date=date_production or date.today(),
            nombre_oeufs=nombre_oeufs,
            poids_oeufs_kg=poids_oeufs_kg,
            notes=f"Enregistré par {recorded_by}" if recorded_by else None
        )
        db.add(production)
        
        await db.commit()
        
        logger.info(f"Egg production added for {avicole.identification}: {nombre_oeufs} eggs, {poids_oeufs_kg} kg")
        return True, f"Production de {nombre_oeufs} œufs enregistrée"
    
    async def update_quantity(
        self,
        db: AsyncSession,
        avicole_id: int,
        nouvelle_quantite: int,
        reason: str,
        updated_by: int
    ) -> Tuple[bool, str]:
        """Mettre à jour la quantité actuelle (vente, abattage, etc.)"""
        avicole = await self.get_avicole(db, avicole_id)
        if not avicole:
            return False, "Lot avicole non trouvé"
        
        if nouvelle_quantite > avicole.quantite_actuelle:
            return False, f"La nouvelle quantité ({nouvelle_quantite}) ne peut pas dépasser la quantité actuelle ({avicole.quantite_actuelle})"
        
        difference = avicole.quantite_actuelle - nouvelle_quantite
        
        if "vente" in reason.lower():
            avicole.vendus_total += difference
            avicole.date_derniere_vente = date.today()
        elif "abattage" in reason.lower():
            avicole.abattus_total += difference
        
        avicole.quantite_actuelle = nouvelle_quantite
        
        # Si le lot est vide, fermer
        if nouvelle_quantite == 0:
            avicole.statut = StatutLotAvicoleEnum.VENDU if "vente" in reason.lower() else StatutLotAvicoleEnum.ABATTU
            avicole.date_fermeture = date.today()
        
        await db.commit()
        
        await log_action(
            db, updated_by, "UPDATE_AVICOLE_QUANTITY", "avicole", avicole_id,
            {"old_quantity": avicole.quantite_actuelle + difference, "new_quantity": nouvelle_quantite, "reason": reason}
        )
        
        logger.info(f"Quantity updated for {avicole.identification}: {nouvelle_quantite} (-{difference})")
        return True, f"Quantité mise à jour: {nouvelle_quantite}"
    
    async def get_avicole_stats(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques des lots avicoles"""
        stmt = select(Avicole)
        if enclos_id:
            stmt = stmt.where(Avicole.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        avicoles = result.scalars().all()
        
        total_lots = len(avicoles)
        lots_actifs = len([a for a in avicoles if a.statut == StatutLotAvicoleEnum.ACTIF])
        
        total_individus = sum(a.quantite_actuelle for a in avicoles)
        total_individus_initial = sum(a.quantite_initial for a in avicoles)
        
        # Par espèce
        par_espece = {}
        for a in avicoles:
            par_espece[a.espece] = par_espece.get(a.espece, 0) + a.quantite_actuelle
        
        # Par type de production
        production_viande = sum(a.quantite_actuelle for a in avicoles if a.production_viande)
        production_ponte = sum(a.quantite_actuelle for a in avicoles if a.production_ponte)
        production_reproduction = sum(a.quantite_actuelle for a in avicoles if a.production_reproduction)
        
        # Production d'œufs
        total_oeufs = sum(a.oeufs_pondus_total for a in avicoles)
        total_poids_oeufs = sum(a.poids_oeufs_total for a in avicoles)
        
        # Mortalité
        total_morts = sum(a.mortalite_total for a in avicoles)
        taux_mortalite_moyen = (total_morts / total_individus_initial * 100) if total_individus_initial > 0 else 0
        
        # Ventes/Abattages
        total_vendus = sum(a.vendus_total for a in avicoles)
        total_abattus = sum(a.abattus_total for a in avicoles)
        
        return {
            "total_lots": total_lots,
            "lots_actifs": lots_actifs,
            "total_individus": total_individus,
            "total_individus_initial": total_individus_initial,
            "par_espece": par_espece,
            "production": {
                "viande": production_viande,
                "ponte": production_ponte,
                "reproduction": production_reproduction
            },
            "oeufs": {
                "total_pondus": total_oeufs,
                "total_poids_kg": round(total_poids_oeufs, 1),
                "moyenne_par_lot": round(total_oeufs / total_lots, 1) if total_lots > 0 else 0
            },
            "mortalite": {
                "total": total_morts,
                "taux_moyen": round(taux_mortalite_moyen, 1)
            },
            "ventes_abattages": {
                "vendus": total_vendus,
                "abattus": total_abattus,
                "total_sortis": total_vendus + total_abattus
            }
        }
    
    async def get_egg_production_stats(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Obtenir les statistiques de production d'œufs sur une période"""
        from ..models.avicole import AvicoleProduction
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Récupérer les lots pondeurs
        stmt = select(Avicole).where(
            Avicole.production_ponte == True,
            Avicole.statut == StatutLotAvicoleEnum.ACTIF,
            
        )
        if enclos_id:
            stmt = stmt.where(Avicole.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        lots_pondeurs = result.scalars().all()
        
        # Récupérer les productions sur la période
        prod_stmt = select(AvicoleProduction).where(
            AvicoleProduction.date >= start_date,
            AvicoleProduction.date <= end_date
        )
        prod_result = await db.execute(prod_stmt)
        productions = prod_result.scalars().all()
        
        # Filtrer par lots pondeurs
        lots_ids = [lot.id for lot in lots_pondeurs]
        productions_filtrees = [p for p in productions if p.lot_id in lots_ids]
        
        total_oeufs = sum(p.nombre_oeufs for p in productions_filtrees)
        total_poids = sum(p.poids_oeufs_kg for p in productions_filtrees)
        
        # Calculer la moyenne par jour
        moyenne_par_jour = total_oeufs / days if days > 0 else 0
        
        return {
            "moyenne_par_jour": round(moyenne_par_jour, 1),
            "total_oeufs": total_oeufs,
            "total_poids_kg": round(total_poids, 1),
            "moyenne_poids_par_oeuf_g": round((total_poids / total_oeufs * 1000) if total_oeufs > 0 else 60, 1),
            "nombre_lots_pondeurs": len(lots_pondeurs),
            "nombre_poules_pondeuses": sum(lot.quantite_actuelle for lot in lots_pondeurs),
            "periode_jours": days,
            "date_debut": start_date,
            "date_fin": end_date,
            "production_par_poule": round(total_oeufs / sum(lot.quantite_actuelle for lot in lots_pondeurs) if sum(lot.quantite_actuelle for lot in lots_pondeurs) > 0 else 0, 1)
        }
    
    async def get_egg_production_history(
        self,
        db: AsyncSession,
        avicole_id: Optional[int] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtenir l'historique de production d'œufs"""
        from ..models.avicole import AvicoleProduction
        from datetime import timedelta
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        stmt = select(AvicoleProduction).where(
            AvicoleProduction.date >= start_date,
            AvicoleProduction.date <= end_date
        ).order_by(AvicoleProduction.date)
        
        if avicole_id:
            stmt = stmt.where(AvicoleProduction.lot_id == avicole_id)
        
        result = await db.execute(stmt)
        productions = result.scalars().all()
        
        history = []
        for prod in productions:
            history.append({
                "date": prod.date,
                "lot_id": prod.lot_id,
                "oeufs": prod.nombre_oeufs,
                "poids_kg": prod.poids_oeufs_kg,
                "notes": prod.notes
            })
        
        return history
    
    async def get_mortality_history(
        self,
        db: AsyncSession,
        avicole_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtenir l'historique des mortalités"""
        from ..models.avicole import AvicoleMortalite
        from datetime import timedelta
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        stmt = select(AvicoleMortalite).where(
            AvicoleMortalite.lot_id == avicole_id,
            AvicoleMortalite.date >= start_date,
            AvicoleMortalite.date <= end_date
        ).order_by(AvicoleMortalite.date.desc())
        
        result = await db.execute(stmt)
        mortalites = result.scalars().all()
        
        history = []
        for mort in mortalites:
            history.append({
                "date": mort.date,
                "nombre_morts": mort.nombre_morts,
                "cause": mort.cause,
                "notes": mort.notes
            })
        
        return history
    
    async def get_lots_by_espece(
        self,
        db: AsyncSession,
        espece: str
    ) -> List[Avicole]:
        """Obtenir tous les lots d'une espèce spécifique"""
        stmt = select(Avicole).where(
            Avicole.espece == espece,
        ).order_by(Avicole.date_arrivee.desc())
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_active_lots(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> List[Avicole]:
        """Obtenir tous les lots actifs"""
        stmt = select(Avicole).where(
            Avicole.statut == StatutLotAvicoleEnum.ACTIF,   
        )
        if enclos_id:
            stmt = stmt.where(Avicole.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        return result.scalars().all()


avicole_service = AvicoleService()