# backend/app/services/dashboard_service.py
"""
Service pour le tableau de bord
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.animal import Animal, StatutAnimalEnum
from ..models.enclos import Enclos
from ..models.compost import Compost
from ..models.accounting import Depense, Recette
from ..models.water_quality import WaterQuality

logger = logging.getLogger(__name__)


class DashboardService:
    """Service pour les données du tableau de bord"""
    
    async def get_main_dashboard(self, db: AsyncSession) -> Dict[str, Any]:
        """Obtenir les indicateurs principaux du tableau de bord"""
        
        # Statistiques animaux
        animals = await self._get_animals_stats(db)
        
        # Statistiques enclos
        enclos = await self._get_enclos_stats(db)
        
        # Statistiques financières
        financial = await self._get_financial_stats(db)
        
        # Alertes récentes
        alerts = await self._get_recent_alerts_count(db)
        
        # Production du jour
        production = await self._get_daily_production(db)
        
        return {
            "animals": animals,
            "enclos": enclos,
            "financial": financial,
            "alerts": alerts,
            "production": production,
            "last_update": date.today()
        }
    
    async def get_animals_summary(self, db: AsyncSession) -> Dict[str, Any]:
        """Résumé des animaux par espèce"""
        
        species = ["bovin", "ovin", "caprin", "avicole", "piscicole"]
        result = {}
        
        for espece in species:
            stmt = select(func.count()).select_from(Animal).where(
                Animal.type_espece == espece,
                Animal.statut == StatutAnimalEnum.VIVANT
            )
            total = (await db.execute(stmt)).scalar() or 0
            
            stmt = select(func.count()).select_from(Animal).where(
                Animal.type_espece == espece,
                Animal.sexe == "male",
                Animal.statut == StatutAnimalEnum.VIVANT
            )
            males = (await db.execute(stmt)).scalar() or 0
            
            stmt = select(func.count()).select_from(Animal).where(
                Animal.type_espece == espece,
                Animal.sexe == "femelle",
                Animal.statut == StatutAnimalEnum.VIVANT
            )
            females = (await db.execute(stmt)).scalar() or 0
            
            result[espece] = {
                "total": total,
                "males": males,
                "femelles": females,
                "naissance_30j": 0,  # À implémenter
                "mortalite_30j": 0   # À implémenter
            }
        
        return result
    
    async def get_production_summary(self, db: AsyncSession, days: int = 30) -> Dict[str, Any]:
        """Résumé de production"""
        
        # Production laitière (à implémenter avec table dédiée)
        milk_production = 0
        
        # Production d'œufs (à implémenter avec table dédiée)
        egg_production = 0
        
        return {
            "milk_liters": milk_production,
            "eggs_count": egg_production,
            "eggs_weight_kg": 0,
            "compost_m3": 0,
            "period_days": days
        }
    
    async def get_financial_summary(self, db: AsyncSession) -> Dict[str, Any]:
        """Résumé financier"""
        
        current_month = date.today().replace(day=1)
        last_month = (current_month - timedelta(days=1)).replace(day=1)
        
        # Dépenses du mois
        stmt = select(func.sum(Depense.montant)).where(
            Depense.date >= current_month
        )
        depenses_mois = (await db.execute(stmt)).scalar() or 0
        
        # Recettes du mois
        stmt = select(func.sum(Recette.montant)).where(
            Recette.date >= current_month
        )
        recettes_mois = (await db.execute(stmt)).scalar() or 0
        
        # Dépenses mois dernier
        stmt = select(func.sum(Depense.montant)).where(
            Depense.date >= last_month,
            Depense.date < current_month
        )
        depenses_mois_dernier = (await db.execute(stmt)).scalar() or 0
        
        # Recettes mois dernier
        stmt = select(func.sum(Recette.montant)).where(
            Recette.date >= last_month,
            Recette.date < current_month
        )
        recettes_mois_dernier = (await db.execute(stmt)).scalar() or 0
        
        return {
            "depenses_mois": round(depenses_mois, 2),
            "recettes_mois": round(recettes_mois, 2),
            "benefice_mois": round(recettes_mois - depenses_mois, 2),
            "depenses_mois_dernier": round(depenses_mois_dernier, 2),
            "recettes_mois_dernier": round(recettes_mois_dernier, 2),
            "evolution_depenses": round(((depenses_mois - depenses_mois_dernier) / max(depenses_mois_dernier, 1)) * 100, 1),
            "evolution_recettes": round(((recettes_mois - recettes_mois_dernier) / max(recettes_mois_dernier, 1)) * 100, 1)
        }
    
    async def get_recent_alerts(self, db: AsyncSession, limit: int = 10) -> List[Dict]:
        """Obtenir les alertes récentes"""
        # À implémenter avec la table d'alertes
        return []
    
    async def get_health_status(self, db: AsyncSession) -> Dict[str, Any]:
        """Statut de santé global"""
        return {
            "global_score": 85,
            "by_species": {},
            "critical_alerts": 0,
            "warning_alerts": 0
        }
    
    async def get_water_quality_summary(self, db: AsyncSession) -> Dict[str, Any]:
        """Résumé qualité de l'eau"""
        
        # Dernières mesures pour chaque bassin
        stmt = select(WaterQuality.enclos_id, func.max(WaterQuality.timestamp)).group_by(WaterQuality.enclos_id)
        result = await db.execute(stmt)
        
        return {
            "bassins_monitorés": len(result.all()),
            "alertes_actives": 0,
            "moyenne_ph": 0,
            "moyenne_oxygene": 0
        }
    
    async def get_compost_summary(self, db: AsyncSession) -> Dict[str, Any]:
        """Résumé compostage"""
        
        stmt = select(Compost)
        result = await db.execute(stmt)
        composts = result.scalars().all()
        
        actifs = [c for c in composts if not c.date_maturite_reelle]
        mature = [c for c in composts if c.date_maturite_reelle]
        
        return {
            "total_andains": len(composts),
            "andains_actifs": len(actifs),
            "andains_matures": len(mature),
            "volume_total_m3": sum(c.volume_initial for c in composts)
        }
    
    async def _get_animals_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Statistiques animaux"""
        
        stmt = select(func.count()).select_from(Animal).where(
            Animal.statut == StatutAnimalEnum.VIVANT
        )
        total = (await db.execute(stmt)).scalar() or 0
        
        stmt = select(Animal.type_espece, func.count()).group_by(Animal.type_espece).where(
            Animal.statut == StatutAnimalEnum.VIVANT
        )
        result = await db.execute(stmt)
        by_species = {row[0]: row[1] for row in result}
        
        return {
            "total": total,
            "by_species": by_species
        }
    
    async def _get_enclos_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Statistiques enclos"""
        
        stmt = select(func.count()).select_from(Enclos)
        total = (await db.execute(stmt)).scalar() or 0
        
        return {
            "total": total,
            "occupation_moyenne": 0
        }
    
    async def _get_financial_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Statistiques financières"""
        
        # CA du mois
        current_month = date.today().replace(day=1)
        stmt = select(func.sum(Recette.montant)).where(Recette.date >= current_month)
        ca_mois = (await db.execute(stmt)).scalar() or 0
        
        return {
            "ca_mois": round(ca_mois, 2),
            "ca_mois_dernier": 0,
            "depenses_mois": 0
        }
    
    async def _get_recent_alerts_count(self, db: AsyncSession) -> Dict[str, int]:
        """Nombre d'alertes récentes"""
        return {
            "critical": 0,
            "warning": 0,
            "info": 0,
            "total": 0
        }
    
    async def _get_daily_production(self, db: AsyncSession) -> Dict[str, Any]:
        """Production du jour"""
        return {
            "milk_liters": 0,
            "eggs_count": 0,
            "eggs_weight_kg": 0
        }

    async def get_recent_activities(
        self,
        db: AsyncSession,
        limit: int = 10,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtenir les activités récentes pour le dashboard
        """
        from ..models.naissance import Naissance
        from ..models.vaccination import Vaccination
        from ..models.alerts import Alert
        
        activities = []
        
        # 1. Naissances récentes
        stmt = select(Naissance).order_by(Naissance.date_naissance.desc()).limit(limit)
        result = await db.execute(stmt)
        naissances = result.scalars().all()
        
        for n in naissances:
            animal = await db.get(Animal, n.animal_ne_id)
            activities.append({
                "id": n.id,
                "type": "naissance",
                "title": "Nouvelle naissance",
                "description": f"Naissance d'un {animal.type_espece if animal else 'animal'} - {n.sexe}",
                "date": n.created_at,
                "entity_id": n.animal_ne_id,
                "entity_type": "animal"
            })
        
        # 2. Vaccinations récentes
        stmt = select(Vaccination).where(
            Vaccination.date_realisee.isnot(None)
        ).order_by(Vaccination.date_realisee.desc()).limit(limit)
        result = await db.execute(stmt)
        vaccinations = result.scalars().all()
        
        for v in vaccinations:
            animal = await db.get(Animal, v.animal_id)
            activities.append({
                "id": v.id,
                "type": "vaccination",
                "title": "Vaccination effectuée",
                "description": f"Vaccination pour {animal.identification if animal else 'animal'}",
                "date": v.date_realisee,
                "entity_id": v.animal_id,
                "entity_type": "animal"
            })
        
        # 3. Alertes critiques récentes
        stmt = select(Alert).where(
            Alert.niveau.in_(["critical", "warning"]),
            Alert.est_traitee == False
        ).order_by(Alert.date_alerte.desc()).limit(limit)
        result = await db.execute(stmt)
        alerts = result.scalars().all()
        
        for a in alerts:
            activities.append({
                "id": a.id,
                "type": "alerte",
                "title": a.title,
                "description": a.message,
                "date": a.date_alerte,
                "entity_id": a.id,
                "entity_type": "alert",
                "severity": a.niveau
            })
        
        # 4. Transactions récentes
        stmt = select(Recette).order_by(Recette.date.desc()).limit(limit)
        result = await db.execute(stmt)
        recettes = result.scalars().all()
        
        for r in recettes:
            activities.append({
                "id": r.id,
                "type": "vente",
                "title": "Vente enregistrée",
                "description": f"{r.categorie.value}: {r.montant}€",
                "date": r.date,
                "entity_id": r.id,
                "entity_type": "recette"
            })
        
        # 5. Trier par date et limiter
        activities.sort(key=lambda x: x["date"], reverse=True)
        
        # Ajouter des informations supplémentaires
        for act in activities[:limit]:
            if act["type"] == "naissance":
                act["icon"] = "🎉"
                act["color"] = "green"
            elif act["type"] == "vaccination":
                act["icon"] = "💉"
                act["color"] = "blue"
            elif act["type"] == "alerte":
                act["icon"] = "⚠️" if act.get("severity") == "warning" else "🔴"
                act["color"] = "orange" if act.get("severity") == "warning" else "red"
            elif act["type"] == "vente":
                act["icon"] = "💰"
                act["color"] = "green"
            else:
                act["icon"] = "📋"
                act["color"] = "gray"
        
        return activities[:limit]
    

dashboard_service = DashboardService()