# backend/app/services/compost_service.py (CORRIGÉ)
"""
Service de gestion du compostage
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import date, timedelta

from ..models.compost import Compost, CompostType, RetournementCompost
from ..schemas.compost import *
from ..prediction.compost_predictor import compost_predictor

logger = logging.getLogger(__name__)


class CompostService:
    """Service de gestion du compostage"""
    
    # Températures optimales par phase
    TEMP_RANGES = {
        "mesophile": (20, 40),
        "thermophile": (55, 65),
        "refroidissement": (40, 50),
        "maturation": (25, 35)
    }
    
    # Humidité optimale (%)
    OPTIMAL_HUMIDITY = 55
    HUMIDITY_RANGE = (45, 65)
    
    async def create_compost(
        self,
        db: AsyncSession,
        compost_data: CompostCreate,
        created_by: int
    ) -> Tuple[Optional[Compost], Optional[str]]:
        """Créer un nouveau tas de compost"""
        # Vérifier si le nom existe déjà
        stmt = select(Compost).where(Compost.name == compost_data.name)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            return None, "Un compost avec ce nom existe déjà"
        
        compost = Compost(
            name=compost_data.name,
            type=compost_data.type,
            date_demarrage=compost_data.date_demarrage,
            volume_initial=compost_data.volume_initial,
            volume_final=compost_data.volume_final,
            date_maturite_estimee=compost_data.date_maturite_estimee,
            date_maturite_reelle=compost_data.date_maturite_reelle,
            utilisation_finale=compost_data.utilisation_finale,
            notes=compost_data.notes
        )
        db.add(compost)
        await db.flush()
        
        # Estimer la date de maturité si non fournie
        if not compost.date_maturite_estimee:
            maturity_prediction = await compost_predictor.predict_maturity_date(
                compost_type=compost.type.value,
                start_date=compost.date_demarrage,
                temperature_readings=[],
                humidity_readings=[],
                turning_dates=[],
                volume_m3=compost.volume_initial
            )
            compost.date_maturite_estimee = maturity_prediction["estimated_maturity_date"]
        
        await db.commit()
        
        logger.info(f"Compost created: {compost.name} by {created_by}")
        return compost, None
    
    async def get_compost(
        self,
        db: AsyncSession,
        compost_id: int
    ) -> Optional[Compost]:
        """Obtenir un compost par son ID"""
        stmt = select(Compost).where(
            Compost.id == compost_id,
            Compost.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_composts(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        compost_type: Optional[CompostType] = None,
        is_mature: Optional[bool] = None
    ) -> List[Compost]:
        """Obtenir la liste des composts"""
        stmt = select(Compost).where(Compost.deleted_at.is_(None))
        
        if compost_type:
            stmt = stmt.where(Compost.type == compost_type)
        if is_mature is not None:
            if is_mature:
                stmt = stmt.where(Compost.date_maturite_reelle.is_not(None))
            else:
                stmt = stmt.where(Compost.date_maturite_reelle.is_(None))
        
        stmt = stmt.offset(skip).limit(limit).order_by(Compost.date_demarrage.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update_compost(
        self,
        db: AsyncSession,
        compost_id: int,
        compost_data: CompostUpdate,
        updated_by: int
    ) -> Tuple[Optional[Compost], Optional[str]]:
        """Mettre à jour un compost"""
        compost = await self.get_compost(db, compost_id)
        if not compost:
            return None, "Compost non trouvé"
        
        if compost_data.name is not None:
            # Vérifier que le nom n'est pas déjà utilisé
            stmt = select(Compost).where(
                Compost.name == compost_data.name,
                Compost.id != compost_id,
                Compost.deleted_at.is_(None)
            )
            result = await db.execute(stmt)
            if result.scalar_one_or_none():
                return None, "Ce nom est déjà utilisé"
            compost.name = compost_data.name
        
        if compost_data.type is not None:
            compost.type = compost_data.type
        if compost_data.volume_initial is not None:
            compost.volume_initial = compost_data.volume_initial
        if compost_data.volume_final is not None:
            compost.volume_final = compost_data.volume_final
        if compost_data.date_maturite_estimee is not None:
            compost.date_maturite_estimee = compost_data.date_maturite_estimee
        if compost_data.date_maturite_reelle is not None:
            compost.date_maturite_reelle = compost_data.date_maturite_reelle
        if compost_data.utilisation_finale is not None:
            compost.utilisation_finale = compost_data.utilisation_finale
        if compost_data.notes is not None:
            compost.notes = compost_data.notes
        
        await db.commit()
        
        logger.info(f"Compost updated: {compost.name} by {updated_by}")
        return compost, None
    
    async def delete_compost(
        self,
        db: AsyncSession,
        compost_id: int,
        deleted_by: int
    ) -> Tuple[bool, str]:
        """Supprimer un compost (soft delete)"""
        compost = await self.get_compost(db, compost_id)
        if not compost:
            return False, "Compost non trouvé"
        
        from datetime import datetime
        compost.deleted_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"Compost deleted: {compost.name} by {deleted_by}")
        return True, "Compost supprimé avec succès"
    
    async def add_turning(
        self,
        db: AsyncSession,
        turning_data: RetournementCompostCreate,
        created_by: int
    ) -> Tuple[Optional[RetournementCompost], Optional[str]]:
        """Ajouter un retournement de compost"""
        compost = await self.get_compost(db, turning_data.compost_id)
        if not compost:
            return None, "Compost non trouvé"
        
        turning = RetournementCompost(
            compost_id=turning_data.compost_id,
            date_retournement=turning_data.date_retournement,
            responsable=turning_data.responsable,
            temperature_avant=turning_data.temperature_avant,
            temperature_apres=turning_data.temperature_apres,
            humidite_avant=turning_data.humidite_avant,
            humidite_apres=turning_data.humidite_apres,
            notes=turning_data.notes
        )
        db.add(turning)
        await db.commit()
        
        logger.info(f"Turning added for compost {compost.name} by {created_by}")
        return turning, None
    
    async def get_turnings(
        self,
        db: AsyncSession,
        compost_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[RetournementCompost]:
        """Obtenir l'historique des retournements"""
        stmt = select(RetournementCompost).where(
            RetournementCompost.compost_id == compost_id
        ).order_by(RetournementCompost.date_retournement.desc())
        
        stmt = stmt.offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_compost_status(
        self,
        db: AsyncSession,
        compost_id: int,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None
    ) -> Dict[str, Any]:
        """Obtenir le statut avancé d'un compost"""
        compost = await self.get_compost(db, compost_id)
        if not compost:
            return {}
        
        turnings = await self.get_turnings(db, compost_id, limit=10)
        
        # Déterminer la phase actuelle
        days_since_start = (date.today() - compost.date_demarrage).days
        
        current_phase = self._get_current_phase(compost.type.value, days_since_start)
        
        # Vérifier les anomalies
        anomalies = []
        if temperature is not None:
            if current_phase == "thermophile" and temperature < 50:
                anomalies.append({
                    "type": "temperature_low",
                    "severity": "warning",
                    "message": "Température trop basse pour la phase thermophile",
                    "action": "Retourner le compost pour réactiver l'aération"
                })
            elif current_phase == "thermophile" and temperature > 70:
                anomalies.append({
                    "type": "temperature_high",
                    "severity": "warning",
                    "message": "Température excessive - risque de destruction des micro-organismes",
                    "action": "Retourner d'urgence pour refroidir"
                })
        
        if humidity is not None:
            if humidity < 40:
                anomalies.append({
                    "type": "humidity_low",
                    "severity": "warning",
                    "message": "Compost trop sec - activité microbienne ralentie",
                    "action": "Arroser légèrement lors du prochain retournement"
                })
            elif humidity > 70:
                anomalies.append({
                    "type": "humidity_high",
                    "severity": "warning",
                    "message": "Compost trop humide - risque d'anaérobie",
                    "action": "Retourner plus fréquemment pour assécher"
                })
        
        # Vérifier la fréquence des retournements
        if turnings:
            last_turning = turnings[0]
            days_since_last_turning = (date.today() - last_turning.date_retournement).days
            
            if current_phase == "thermophile" and days_since_last_turning > 14:
                anomalies.append({
                    "type": "turning_delayed",
                    "severity": "info",
                    "message": f"Pas de retournement depuis {days_since_last_turning} jours",
                    "action": "Programmer un retournement dans les 3 jours"
                })
        
        # Prédire la date de maturité
        temp_readings = [(t.date_retournement, t.temperature_apres or 0) for t in turnings if t.temperature_apres]
        humidity_readings = [(t.date_retournement, t.humidite_apres or 0) for t in turnings if t.humidite_apres]
        turning_dates = [t.date_retournement for t in turnings]
        
        maturity_prediction = await compost_predictor.predict_maturity_date(
            compost_type=compost.type.value,
            start_date=compost.date_demarrage,
            temperature_readings=temp_readings,
            humidity_readings=humidity_readings,
            turning_dates=turning_dates,
            volume_m3=compost.volume_initial
        )
        
        # Prédire le volume final
        volume_prediction = await compost_predictor.predict_final_volume(
            initial_volume_m3=compost.volume_initial,
            compost_type=compost.type.value,
            turning_count=len(turnings),
            duration_days=days_since_start
        )
        
        return {
            "id": compost.id,
            "name": compost.name,
            "type": compost.type.value,
            "days_since_start": days_since_start,
            "current_phase": current_phase,
            "temperature": temperature,
            "humidity": humidity,
            "anomalies": anomalies,
            "turnings_count": len(turnings),
            "maturity": {
                "estimated_date": maturity_prediction["estimated_maturity_date"],
                "days_remaining": maturity_prediction["days_remaining"],
                "confidence": maturity_prediction["confidence_percent"],
                "recommendations": maturity_prediction["recommendations"]
            },
            "volume": {
                "initial_m3": compost.volume_initial,
                "predicted_final_m3": volume_prediction["predicted_final_volume_m3"],
                "reduction_percent": volume_prediction["volume_reduction_percent"]
            },
            "needs_intervention": len(anomalies) > 0
        }
    
    async def mark_as_mature(
        self,
        db: AsyncSession,
        compost_id: int,
        updated_by: int,
        volume_final: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Marquer un compost comme mature"""
        compost = await self.get_compost(db, compost_id)
        if not compost:
            return False, "Compost non trouvé"
        
        compost.date_maturite_reelle = date.today()
        if volume_final:
            compost.volume_final = volume_final
        
        await db.commit()
        
        logger.info(f"Compost marked as mature: {compost.name} by {updated_by}")
        return True, "Compost marqué comme mature"
    
    async def get_compost_stats(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Obtenir des statistiques globales sur les composts"""
        stmt = select(Compost).where(Compost.deleted_at.is_(None))
        result = await db.execute(stmt)
        composts = result.scalars().all()
        
        active = [c for c in composts if not c.date_maturite_reelle]
        mature = [c for c in composts if c.date_maturite_reelle]
        
        total_volume_initial = sum(c.volume_initial for c in composts)
        total_volume_final = sum(c.volume_final for c in composts if c.volume_final)
        
        # Compost par type
        by_type = {}
        for c in composts:
            key = c.type.value
            by_type[key] = by_type.get(key, 0) + 1
        
        return {
            "total_composts": len(composts),
            "active": len(active),
            "mature": len(mature),
            "total_volume_initial_m3": round(total_volume_initial, 1),
            "total_volume_final_m3": round(total_volume_final, 1),
            "volume_reduction_percent": round((1 - total_volume_final / total_volume_initial) * 100, 1) if total_volume_initial > 0 else 0,
            "by_type": by_type
        }
    
    def _get_current_phase(self, compost_type: str, days: int) -> str:
        """Déterminer la phase actuelle du compost en fonction des jours"""
        phases = {
            "déchets verts": [
                ("mesophile", 0, 3),
                ("thermophile", 3, 21),
                ("refroidissement", 21, 42),
                ("maturation", 42, 90)
            ],
            "fumier": [
                ("mesophile", 0, 5),
                ("thermophile", 5, 35),
                ("refroidissement", 35, 60),
                ("maturation", 60, 120)
            ],
            "mixte": [
                ("mesophile", 0, 4),
                ("thermophile", 4, 28),
                ("refroidissement", 28, 56),
                ("maturation", 56, 105)
            ]
        }
        
        type_phases = phases.get(compost_type, phases["mixte"])
        
        for phase_name, start, end in type_phases:
            if start <= days <= end:
                return phase_name
        
        return "maturation"


compost_service = CompostService()