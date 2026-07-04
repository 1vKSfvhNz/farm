# backend/app/services/alert_service.py
"""
Service de gestion des alertes
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import case, select, update, func

from ..models.alerts import Alert
from ..models.vaccination import Vaccination
from ..models.water_quality import WaterQuality, WaterQualityAlerte
from ..models.animal import Animal
from ..services.notification_service import notification_service
from ..schemas.alerts import *

logger = logging.getLogger(__name__)


class AlertService:
    """Service de gestion des alertes"""
    
    async def get_alerts(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        niveau: Optional[str] = None,
        est_lue: Optional[bool] = None,
        espece: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> List[Alert]:
        """Obtenir la liste des alertes"""
        
        stmt = select(Alert)
        
        if niveau:
            stmt = stmt.where(Alert.niveau == niveau)
        if est_lue is not None:
            stmt = stmt.where(Alert.est_lue == est_lue)
        if espece:
            stmt = stmt.where(Alert.espece == espece)
        if user_id:
            stmt = stmt.where(Alert.utilisateur_id == user_id)
        
        stmt = stmt.order_by(Alert.date_alerte.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_alert(
        self,
        db: AsyncSession,
        alert_id: int
    ) -> Optional[Alert]:
        """Obtenir une alerte par son ID"""
        stmt = select(Alert).where(Alert.id == alert_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_unread_count(
        self,
        db: AsyncSession,
        user_id: int
    ) -> int:
        """Obtenir le nombre d'alertes non lues"""
        stmt = select(func.count()).select_from(Alert).where(
            Alert.utilisateur_id == user_id,
            Alert.est_lue == False
        )
        result = await db.execute(stmt)
        return result.scalar() or 0
    
    async def mark_as_read(
        self,
        db: AsyncSession,
        alert_id: int,
        user_id: int
    ) -> bool:
        """Marquer une alerte comme lue"""
        stmt = update(Alert).where(
            Alert.id == alert_id,
            Alert.utilisateur_id == user_id
        ).values(est_lue=True, date_lue=datetime.now())
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0
    
    async def resolve_alert(
        self,
        db: AsyncSession,
        alert_id: int,
        user_id: int,
        resolution_note: Optional[str] = None
    ) -> bool:
        """Résoudre une alerte"""
        stmt = update(Alert).where(Alert.id == alert_id).values(
            est_traitee=True,
            utilisateur_traitement_id=user_id,
            date_traitement=datetime.now(),
            resolution_note=resolution_note
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0
    
    async def mark_all_as_read(
        self,
        db: AsyncSession,
        user_id: int
    ) -> int:
        """Marquer toutes les alertes comme lues"""
        stmt = update(Alert).where(
            Alert.utilisateur_id == user_id,
            Alert.est_lue == False
        ).values(est_lue=True, date_lue=datetime.now())
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount
    
    async def create_alert(
        self,
        db: AsyncSession,
        alert_data: Dict[str, Any]
    ) -> Alert:
        """Créer une alerte"""
        alert = Alert(
            type=alert_data.get("type"),
            niveau=alert_data.get("niveau", "info"),
            message=alert_data.get("message"),
            espece=alert_data.get("espece"),
            animal_id=alert_data.get("animal_id"),
            enclos_id=alert_data.get("enclos_id"),
            utilisateur_id=alert_data.get("utilisateur_id"),
            date_alerte=datetime.now(),
            date_limite=alert_data.get("date_limite")
        )
        db.add(alert)
        await db.commit()
        
        # Envoyer la notification
        await notification_service.send_alert(
            user_id=alert.utilisateur_id or 0,
            alert_type=alert.type,
            title=f"Alerte {alert.niveau}: {alert.type}",
            message=alert.message,
            severity=alert.niveau
        )
        
        return alert
    
    async def generate_vaccination_alerts(
        self,
        db: AsyncSession
    ) -> int:
        """Générer des alertes pour les vaccinations à venir"""
        
        today = date.today()
        deadline = today + timedelta(days=7)
        
        stmt = select(Vaccination).where(
            Vaccination.date_realisee.is_(None),
            Vaccination.date_prevue >= today,
            Vaccination.date_prevue <= deadline
        )
        result = await db.execute(stmt)
        vaccinations = result.scalars().all()
        
        count = 0
        for vacc in vaccinations:
            days_left = (vacc.date_prevue - today).days
            
            alert = await self.create_alert(
                db,
                {
                    "type": "vaccination_reminder",
                    "niveau": "warning" if days_left <= 3 else "info",
                    "message": f"Vaccination prévue dans {days_left} jours",
                    "animal_id": vacc.animal_id,
                    "date_limite": vacc.date_prevue
                }
            )
            count += 1
        
        return count
    
    async def generate_water_quality_alerts(
        self,
        db: AsyncSession
    ) -> int:
        """Générer des alertes de qualité d'eau"""
        
        # Récupérer les alertes non traitées
        stmt = select(WaterQualityAlerte).where(
            WaterQualityAlerte.traitee == False
        )
        result = await db.execute(stmt)
        alerts = result.scalars().all()
        
        for alert in alerts:
            await self.create_alert(
                db,
                {
                    "type": "water_quality",
                    "niveau": alert.niveau,
                    "message": alert.message,
                    "enclos_id": alert.water_quality.enclos_id if alert.water_quality else None
                }
            )
        
        return len(alerts)

    async def get_alert_stats(
        self,
        db: AsyncSession,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir les statistiques des alertes"""
        stmt = select(
            func.count(Alert.id).label('total'),
            func.sum(case((Alert.est_lue == False, 1), else_=0)).label('non_lues'),
            func.sum(case((Alert.est_traitee == False, 1), else_=0)).label('non_traitees'),
            func.sum(case((Alert.niveau == 'critical', 1), else_=0)).label('critical'),
            func.sum(case((Alert.niveau == 'warning', 1), else_=0)).label('warning'),
            func.sum(case((Alert.niveau == 'info', 1), else_=0)).label('info')
        )
        
        if user_id:
            stmt = stmt.where(Alert.utilisateur_id == user_id)
        
        result = await db.execute(stmt)
        row = result.one()
        
        return {
            "total": row.total or 0,
            "non_lues": row.non_lues or 0,
            "non_traitees": row.non_traitees or 0,
            "critical": row.critical or 0,
            "warning": row.warning or 0,
            "info": row.info or 0
        }
    
    async def get_alert_by_id(
        self,
        db: AsyncSession,
        alert_id: int
    ) -> Optional[Alert]:
        """Obtenir une alerte par son ID"""
        stmt = select(Alert).where(Alert.id == alert_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_alerts_by_entity(
        self,
        db: AsyncSession,
        entity_type: str,
        entity_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[Alert]:
        """Obtenir les alertes pour une entité spécifique"""
        if entity_type == "animal":
            stmt = select(Alert).where(Alert.animal_id == entity_id)
        elif entity_type == "enclos":
            stmt = select(Alert).where(Alert.enclos_id == entity_id)
        elif entity_type == "compost":
            stmt = select(Alert).where(Alert.compost_id == entity_id)
        else:
            return []
        
        stmt = stmt.order_by(Alert.date_alerte.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def archive_old_alerts(
        self,
        db: AsyncSession,
        days_old: int = 30
    ) -> int:
        """Archiver les alertes anciennes (soft delete)"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        stmt = select(Alert).where(
            Alert.date_alerte < cutoff_date,
            Alert.est_traitee == True
        )
        result = await db.execute(stmt)
        alerts = result.scalars().all()
        
        for alert in alerts:
            alert.deleted_at = datetime.now()
        
        await db.commit()
        return len(alerts)

alert_service = AlertService()