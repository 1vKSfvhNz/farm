# backend/app/services/water_quality_service.py
"""
Service de gestion de la qualité de l'eau
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.water_quality import WaterQuality, WaterQualityAlerte
from ..schemas.water_quality import *
from ..services.notification_service import notification_service
from ..core.constants import SEUILS_QUALITE_EAU

logger = logging.getLogger(__name__)


class WaterQualityService:
    """Service de gestion de la qualité de l'eau"""
    
    async def create_measurement(
        self,
        db: AsyncSession,
        measurement: WaterQualityCreate
    ) -> WaterQuality:
        """Créer une mesure de qualité d'eau"""
        water_quality = WaterQuality(
            enclos_id=measurement.enclos_id,
            timestamp=measurement.timestamp,
            ph=measurement.ph,
            temperature=measurement.temperature,
            oxygene_dissous=measurement.oxygene_dissous,
            ammoniac=measurement.ammoniac,
            nitrites=measurement.nitrites,
            nitrates=measurement.nitrates,
            conductivite=measurement.conductivite,
            turbidite=measurement.turbidite,
            source=measurement.source
        )
        db.add(water_quality)
        await db.flush()
        
        # Vérifier les seuils et générer des alertes
        alerts = await self._check_thresholds(db, water_quality)
        water_quality.alerte_generee = len(alerts) > 0
        
        await db.commit()
        
        return water_quality
    
    async def _check_thresholds(
        self,
        db: AsyncSession,
        measurement: WaterQuality
    ) -> List[WaterQualityAlerte]:
        """Vérifier les seuils et créer des alertes"""
        alerts = []
        
        # Vérifier l'oxygène
        if measurement.oxygene_dissous and measurement.oxygene_dissous < 4.0:
            alert = WaterQualityAlerte(
                water_quality_id=measurement.id,
                parametre="oxygen",
                valeur=measurement.oxygene_dissous,
                seuil=4.0,
                niveau="warning" if measurement.oxygene_dissous >= 3.0 else "critical",
                message=f"Oxygène dissous bas: {measurement.oxygene_dissous} mg/L"
            )
            alerts.append(alert)
        
        # Vérifier le pH
        if measurement.ph:
            if measurement.ph < 6.0 or measurement.ph > 9.0:
                alert = WaterQualityAlerte(
                    water_quality_id=measurement.id,
                    parametre="ph",
                    valeur=measurement.ph,
                    seuil=6.0 if measurement.ph < 6.0 else 9.0,
                    niveau="critical",
                    message=f"pH hors norme: {measurement.ph}"
                )
                alerts.append(alert)
        
        # Vérifier l'ammoniac
        if measurement.ammoniac and measurement.ammoniac > 0.1:
            alert = WaterQualityAlerte(
                water_quality_id=measurement.id,
                parametre="ammonia",
                valeur=measurement.ammoniac,
                seuil=0.1,
                niveau="critical",
                message=f"Ammoniac élevé: {measurement.ammoniac} mg/L"
            )
            alerts.append(alert)
        
        for alert in alerts:
            db.add(alert)
        
        return alerts
    
    async def get_last_measurement(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> Optional[WaterQuality]:
        """Obtenir la dernière mesure pour un enclos"""
        stmt = select(WaterQuality).where(
            WaterQuality.enclos_id == enclos_id
        ).order_by(WaterQuality.timestamp.desc()).limit(1)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


water_quality_service = WaterQualityService()