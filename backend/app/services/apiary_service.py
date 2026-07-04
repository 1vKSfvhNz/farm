# backend/app/services/apiary_service.py
"""
Service de gestion apicole (ruches, miel)
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import date

from app.schemas.apiary import InspectionRucheResponse

from ..models.apiary import (
    Ruche,
    RecolteMiel
)
from ..schemas.apiary import (
    RucheCreate, RucheUpdate, RecolteMielCreate
)
from ..services.id_service import generate_identification

logger = logging.getLogger(__name__)


class ApiaryService:
    """Service de gestion apicole"""
    
    # ============ RUCHES ============
    
    async def create_ruche(
        self,
        db: AsyncSession,
        ruche_data: RucheCreate,
        created_by: int
    ) -> Tuple[Optional[Ruche], Optional[str]]:
        """Créer une nouvelle ruche"""
        # Générer l'identification
        identification = await generate_identification(db, "RUC")
        
        ruche = Ruche(
            identification=identification,
            emplacement=ruche_data.emplacement,
            date_installation=ruche_data.date_installation,
            race=ruche_data.race,
            statut=ruche_data.statut,
            nombre_cadres=ruche_data.nombre_cadres,
            notes=ruche_data.notes
        )
        db.add(ruche)
        await db.commit()
        await db.refresh(ruche)
        
        logger.info(f"Ruche created: {identification} by {created_by}")
        return ruche, None
    
    async def get_ruche(
        self,
        db: AsyncSession,
        ruche_id: int
    ) -> Optional[Ruche]:
        """Obtenir une ruche par son ID"""
        stmt = select(Ruche).where(
            Ruche.id == ruche_id,
            Ruche.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_ruche_by_identification(
        self,
        db: AsyncSession,
        identification: str
    ) -> Optional[Ruche]:
        """Obtenir une ruche par son identification"""
        stmt = select(Ruche).where(
            Ruche.identification == identification,
            Ruche.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_ruches(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        statut: Optional[str] = None,
        emplacement: Optional[str] = None
    ) -> List[Ruche]:
        """Obtenir la liste des ruches avec filtres"""
        stmt = select(Ruche).where(Ruche.deleted_at.is_(None))
        
        if statut:
            stmt = stmt.where(Ruche.statut == statut)
        if emplacement:
            stmt = stmt.where(Ruche.emplacement.ilike(f"%{emplacement}%"))
        
        stmt = stmt.order_by(Ruche.identification).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_ruche(
        self,
        db: AsyncSession,
        ruche_id: int,
        ruche_data: RucheUpdate,
        updated_by: int
    ) -> Tuple[Optional[Ruche], Optional[str]]:
        """Mettre à jour une ruche"""
        ruche = await self.get_ruche(db, ruche_id)
        if not ruche:
            return None, "Ruche non trouvée"
        
        update_data = ruche_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(ruche, field, value)
        
        await db.commit()
        await db.refresh(ruche)
        
        logger.info(f"Ruche updated: {ruche.identification} by {updated_by}")
        return ruche, None
    
    async def delete_ruche(
        self,
        db: AsyncSession,
        ruche_id: int,
        deleted_by: int,
        soft_delete: bool = True
    ) -> bool:
        """Supprimer une ruche (soft delete par défaut)"""
        ruche = await self.get_ruche(db, ruche_id)
        if not ruche:
            return False
        
        if soft_delete:
            from datetime import datetime
            ruche.deleted_at = datetime.utcnow()
            await db.commit()
            logger.info(f"Ruche soft deleted: {ruche.identification} by {deleted_by}")
        else:
            await db.delete(ruche)
            await db.commit()
            logger.info(f"Ruche hard deleted: {ruche.identification} by {deleted_by}")
        
        return True
        
    # ============ RÉCOLTES ============
    
    async def add_recolte(
        self,
        db: AsyncSession,
        recolte_data: RecolteMielCreate,
        created_by: int
    ) -> Tuple[Optional[RecolteMiel], Optional[str]]:
        """Ajouter une récolte de miel"""
        # Vérifier que la ruche existe
        ruche = await self.get_ruche(db, recolte_data.ruche_id)
        if not ruche:
            return None, "Ruche non trouvée"
        
        recolte = RecolteMiel(
            ruche_id=recolte_data.ruche_id,
            date_recolte=recolte_data.date_recolte,
            poids_kg=recolte_data.poids_kg,
            qualite=recolte_data.qualite,
            taux_eau=recolte_data.taux_eau,
            notes=recolte_data.notes
        )
        db.add(recolte)
        await db.commit()
        await db.refresh(recolte)
        
        logger.info(f"Recolte added to ruche {ruche.identification}: {recolte_data.poids_kg}kg by {created_by}")
        return recolte, None
    
    async def get_recolte(
        self,
        db: AsyncSession,
        recolte_id: int
    ) -> Optional[RecolteMiel]:
        """Obtenir une récolte par son ID"""
        stmt = select(RecolteMiel).where(RecolteMiel.id == recolte_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_recoltes(
        self,
        db: AsyncSession,
        ruche_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
        year: Optional[int] = None
    ) -> List[RecolteMiel]:
        """Obtenir l'historique des récoltes"""
        stmt = select(RecolteMiel)
        
        if ruche_id is not None:
            stmt = stmt.where(RecolteMiel.ruche_id == ruche_id)
        if year is not None:
            stmt = stmt.where(func.extract('year', RecolteMiel.date_recolte) == year)
        
        stmt = stmt.order_by(RecolteMiel.date_recolte.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_last_recoltes(
        self,
        db: AsyncSession,
        limit: int = 5
    ) -> List[RecolteMiel]:
        """Obtenir les dernières récoltes"""
        stmt = select(RecolteMiel).order_by(
            RecolteMiel.date_recolte.desc()
        ).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_recolte(
        self,
        db: AsyncSession,
        recolte_id: int,
        recolte_data: dict,
        updated_by: int
    ) -> Tuple[Optional[RecolteMiel], Optional[str]]:
        """Mettre à jour une récolte"""
        recolte = await self.get_recolte(db, recolte_id)
        if not recolte:
            return None, "Récolte non trouvée"
        
        for field, value in recolte_data.items():
            if hasattr(recolte, field):
                setattr(recolte, field, value)
        
        await db.commit()
        await db.refresh(recolte)
        
        logger.info(f"Recolte updated: {recolte_id} by {updated_by}")
        return recolte, None
    
    # ============ STATISTIQUES ============
    
    async def get_ruches_stats(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Obtenir les statistiques sur les ruches"""
        ruches = await self.get_ruches(db, limit=10000)
        
        active = len([r for r in ruches if r.statut == "active"])
        orphelines = len([r for r in ruches if r.statut == "orpheline"])
        en_essaimage = len([r for r in ruches if r.statut == "en_essaimage"])
        mortes = len([r for r in ruches if r.statut == "morte"])
        
        # Âge moyen des ruches
        ages = []
        for r in ruches:
            if r.date_installation:
                age_days = (date.today() - r.date_installation).days
                ages.append(age_days)
        
        avg_age_days = sum(ages) / len(ages) if ages else 0
        
        return {
            "total": len(ruches),
            "active": active,
            "orphelines": orphelines,
            "en_essaimage": en_essaimage,
            "mortes": mortes,
            "taux_activite": round(active / len(ruches) * 100, 1) if ruches else 0,
            "age_moyen_jours": round(avg_age_days, 0),
            "age_moyen_annees": round(avg_age_days / 365, 1)
        }
    
    async def get_production_stats(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Obtenir les statistiques de production"""
        from datetime import date
        
        current_year = date.today().year
        total_miel = await self.get_total_honey_production(db, current_year)
        monthly_production = await self.get_monthly_production(db, current_year)
        ruches = await self.get_ruches(db, limit=10000)
        
        return {
            "total_honey_kg": round(total_miel, 1),
            "monthly_production": monthly_production,
            "average_per_ruche": round(total_miel / max(1, len(ruches)), 1),
            "year": current_year
        }
    
    async def get_total_honey_production(
        self,
        db: AsyncSession,
        year: int
    ) -> float:
        """Obtenir la production totale de miel pour une année"""
        stmt = select(func.sum(RecolteMiel.poids_kg)).where(
            func.extract('year', RecolteMiel.date_recolte) == year
        )
        result = await db.execute(stmt)
        return result.scalar() or 0.0
    
    async def get_monthly_production(
        self,
        db: AsyncSession,
        year: int
    ) -> List[Dict[str, Any]]:
        """Obtenir la production mensuelle de miel"""
        monthly = []
        for month in range(1, 13):
            stmt = select(func.sum(RecolteMiel.poids_kg)).where(
                and_(
                    func.extract('year', RecolteMiel.date_recolte) == year,
                    func.extract('month', RecolteMiel.date_recolte) == month
                )
            )
            result = await db.execute(stmt)
            total = result.scalar() or 0.0
            
            monthly.append({
                "month": month,
                "month_name": date(year, month, 1).strftime("%B"),
                "total_kg": round(total, 1)
            })
        
        return monthly
    
    async def get_reines_stats(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Obtenir les statistiques sur les reines"""
        reines = await self.get_all_reines(db)
        
        # Âge moyen des reines
        ages = []
        for r in reines:
            if r.annee_naissance:
                age = date.today().year - r.annee_naissance
                ages.append(age)
        
        avg_age = sum(ages) / len(ages) if ages else 0
        jeunes = len([r for r in reines if r.annee_naissance and date.today().year - r.annee_naissance <= 1])
        vieilles = len([r for r in reines if r.annee_naissance and date.today().year - r.annee_naissance >= 3])
        
        return {
            "total_reines_enregistrees": len(reines),
            "age_moyen_annees": round(avg_age, 1),
            "reines_de_moins_d_un_an": jeunes,
            "reines_de_plus_de_3_ans": vieilles,
            "taux_renouvellement": round(jeunes / len(reines) * 100, 1) if reines else 0
        }
    
    # ============ ALERTES ============
    
    async def get_alerts(
        self,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Obtenir les alertes apicoles"""
        alerts = []
        
        # Vérifier les ruches orphelines
        ruches = await self.get_ruches(db, limit=10000)
        orphelines = [r for r in ruches if r.statut == "orpheline"]
        if orphelines:
            alerts.append({
                "type": "ruche_orpheline",
                "severity": "critical",
                "message": f"{len(orphelines)} ruche(s) orpheline(s)",
                "ruches": [{"id": r.id, "identification": r.identification} for r in orphelines]
            })
        
        # Vérifier les ruches en essaimage
        essaimage = [r for r in ruches if r.statut == "en_essaimage"]
        if essaimage:
            alerts.append({
                "type": "essaimage",
                "severity": "warning",
                "message": f"{len(essaimage)} ruche(s) en essaimage",
                "ruches": [{"id": r.id, "identification": r.identification} for r in essaimage]
            })
        
        # Vérifier les reines âgées (plus de 3 ans)
        reines = await self.get_all_reines(db)
        vieilles_reines = [r for r in reines if r.annee_naissance and date.today().year - r.annee_naissance >= 3]
        if vieilles_reines:
            alerts.append({
                "type": "reine_agee",
                "severity": "warning",
                "message": f"{len(vieilles_reines)} reine(s) de plus de 3 ans",
                "reines": [{"id": r.id, "ruche_id": r.ruche_id} for r in vieilles_reines[:5]]
            })
        
        return alerts
    
    # ============ INSPECTIONS ============
    
    async def add_inspection(
        self,
        db: AsyncSession,
        ruche_id: int,
        inspection_data: dict,
        created_by: int
    ) -> InspectionRucheResponse:
        """Ajouter une inspection de ruche"""
        inspection = InspectionRucheResponse(
            ruche_id=ruche_id,
            date_inspection=inspection_data.get("date_inspection", date.today()),
            inspecteur=inspection_data.get("inspecteur"),
            cadre_occupe=inspection_data.get("cadre_occupe"),
            cadre_couvert=inspection_data.get("cadre_couvert"),
            presence_couvain=inspection_data.get("presence_couvain"),
            presence_miel=inspection_data.get("presence_miel"),
            presence_pollen=inspection_data.get("presence_pollen"),
            varroa=inspection_data.get("varroa"),
            loque=inspection_data.get("loque"),
            notes=inspection_data.get("notes")
        )
        db.add(inspection)
        await db.commit()
        await db.refresh(inspection)
        
        logger.info(f"InspectionRucheResponse added for ruche {ruche_id} by {created_by}")
        return inspection
    
    async def get_inspections(
        self,
        db: AsyncSession,
        ruche_id: int,
        limit: int = 10
    ) -> List[InspectionRucheResponse]:
        """Obtenir l'historique des inspections d'une ruche"""
        stmt = select(InspectionRucheResponse).where(
            InspectionRucheResponse.ruche_id == ruche_id
        ).order_by(InspectionRucheResponse.date_inspection.desc()).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    # ============ ESSAIMAGE ============
    
    async def record_swarming(
        self,
        db: AsyncSession,
        ruche_id: int,
        swarming_data: dict,
        created_by: int
    ) -> Dict[str, Any]:
        """Enregistrer un essaimage"""
        # Mettre à jour le statut de la ruche
        ruche = await self.get_ruche(db, ruche_id)
        if not ruche:
            return {"success": False, "message": "Ruche non trouvée"}
        
        ruche.statut = "en_essaimage"
        await db.commit()
        
        # Créer un enregistrement d'essaimage
        # TODO: Créer une table Essaimage si nécessaire
        
        logger.info(f"Swarming recorded for ruche {ruche.identification} by {created_by}")
        return {
            "success": True,
            "message": "Essaimage enregistré",
            "ruche": ruche.identification
        }
    
    async def get_swarming_history(
        self,
        db: AsyncSession,
        ruche_id: int
    ) -> List[Dict[str, Any]]:
        """Obtenir l'historique des essaimages d'une ruche"""
        # TODO: Implémenter avec une table Essaimage
        return []


apiary_service = ApiaryService()