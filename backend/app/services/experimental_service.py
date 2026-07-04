# backend/app/services/experimental_service.py
"""
Service pour le mode expérimental - Auto-apprentissage des références
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from ..models.experimental import ReferenceGenerale, DonneeExperimentale
from ..models.reference import ReferenceHypothese
from ..models.pesee import Pesee
from ..models.mortalite import Mortalite
from ..models.alimentation import Alimentation
from ..schemas.experimental import *
from ..config import settings

logger = logging.getLogger(__name__)


class ExperimentalService:
    """
    Service pour le mode expérimental
    Gère l'auto-apprentissage des références et le niveau de confiance
    """
    
    async def get_experimental_mode_status(
        self,
        db: AsyncSession,
        espece: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtenir le statut du mode expérimental
        """
        # Compter les données collectées
        pesee_count = await self._count_pesees(db, espece)
        mortalite_count = await self._count_mortalites(db, espece)
        alimentation_count = await self._count_alimentations(db, espece)
        
        total_data = pesee_count + mortalite_count + alimentation_count
        
        # Déterminer le mode
        if total_data < 50:
            mode = "experimental"
            confidence = min(0.2, total_data / 250)
        elif total_data < 200:
            mode = "hybride"
            confidence = 0.4 + (total_data - 50) / 500
        else:
            mode = "complet"
            confidence = min(0.9, 0.6 + (total_data - 200) / 1000)
        
        # Jours de collecte
        first_data_date = await self._get_first_data_date(db, espece)
        days_collected = (date.today() - first_data_date).days if first_data_date else 0
        
        # Recommandations
        recommendations = []
        if total_data < 50:
            recommendations.append(f"Collectez au moins {50 - total_data} données supplémentaires pour des prédictions fiables")
        if pesee_count < 10:
            recommendations.append("Ajoutez des pesées régulières pour améliorer les prédictions de croissance")
        if days_collected < 60:
            recommendations.append(f"Continuez la collecte pendant {60 - days_collected} jours")
        
        return {
            "mode": mode,
            "jours_collecte": days_collected,
            "nombre_donnees": {
                "pesees": pesee_count,
                "mortalites": mortalite_count,
                "alimentations": alimentation_count,
                "total": total_data
            },
            "confiance_moyenne": round(confidence * 100, 1),
            "seuils_atteints": self._get_thresholds_status(total_data, days_collected),
            "recommandations": recommendations
        }
    
    async def generate_reference(
        self,
        db: AsyncSession,
        espece: str,
        type_reference: str,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Générer une référence à partir des données collectées
        """
        if type_reference == "croissance":
            return await self._generate_growth_reference(db, espece, force_regenerate)
        elif type_reference == "mortalite":
            return await self._generate_mortality_reference(db, espece, force_regenerate)
        elif type_reference == "conversion":
            return await self._generate_conversion_reference(db, espece, force_regenerate)
        else:
            return {"error": f"Type de référence inconnu: {type_reference}"}
    
    async def _generate_growth_reference(
        self,
        db: AsyncSession,
        espece: str,
        force_regenerate: bool
    ) -> Dict[str, Any]:
        """Générer une courbe de croissance à partir des pesées"""
        
        # Récupérer toutes les pesées
        stmt = select(Pesee).join(Pesee.animal).where(
            Pesee.animal.has(type_espece=espece),
            Pesee.deleted_at.is_(None)
        ).order_by(Pesee.date_pesee)
        
        result = await db.execute(stmt)
        pesees = result.scalars().all()
        
        if len(pesees) < 10:
            return {
                "success": False,
                "message": f"Données insuffisantes: {len(pesees)} pesées (minimum 10)",
                "donnees_utilisees": len(pesees)
            }
        
        # Grouper par âge
        age_groups = {}
        for pesee in pesees:
            if pesee.animal and pesee.animal.date_naissance:
                age = (pesee.date_pesee - pesee.animal.date_naissance).days
                if age > 0:
                    if age not in age_groups:
                        age_groups[age] = []
                    age_groups[age].append(pesee.poids)
        
        # Calculer les moyennes par tranche d'âge
        references = []
        for age, weights in sorted(age_groups.items()):
            if len(weights) >= 3:
                references.append({
                    "age_jours": age,
                    "poids_min": min(weights),
                    "poids_moyen": sum(weights) / len(weights),
                    "poids_max": max(weights)
                })
        
        # Sauvegarder la référence
        existing = await db.execute(
            select(ReferenceGenerale).where(
                ReferenceGenerale.espece == espece,
                ReferenceGenerale.type_reference == "croissance"
            )
        )
        existing_ref = existing.scalar_one_or_none()
        
        if existing_ref and not force_regenerate:
            return {
                "success": False,
                "message": "Une référence existe déjà. Utilisez force_regenerate=True pour la remplacer",
                "donnees_utilisees": len(pesees)
            }
        
        if existing_ref:
            existing_ref.donnees = references
            existing_ref.nombre_donnees = len(pesees)
            existing_ref.confiance = min(0.9, len(pesees) / 100)
            existing_ref.date_derniere_mise_a_jour = date.today()
        else:
            new_ref = ReferenceGenerale(
                espece=espece,
                type_reference="croissance",
                donnees=references,
                nombre_donnees=len(pesees),
                confiance=min(0.9, len(pesees) / 100),
                date_derniere_mise_a_jour=date.today(),
                is_active=True
            )
            db.add(new_ref)
        
        await db.commit()
        
        return {
            "success": True,
            "message": f"Référence de croissance générée pour {espece}",
            "donnees_utilisees": len(pesees),
            "points_generes": len(references),
            "confiance": round(min(0.9, len(pesees) / 100) * 100, 1)
        }
    
    async def _generate_mortality_reference(
        self,
        db: AsyncSession,
        espece: str,
        force_regenerate: bool
    ) -> Dict[str, Any]:
        """Générer une référence de mortalité"""
        
        # Compter les animaux et les mortalités
        from ..models.animal import Animal
        
        stmt = select(func.count()).select_from(Animal).where(
            Animal.type_espece == espece,
            Animal.deleted_at.is_(None)
        )
        total_animals = (await db.execute(stmt)).scalar() or 1
        
        stmt = select(func.count()).select_from(Animal).where(
            Animal.type_espece == espece,
            Animal.statut == "decede",
            Animal.deleted_at.is_(None)
        )
        deaths = (await db.execute(stmt)).scalar() or 0
        
        mortality_rate = (deaths / total_animals) * 100 if total_animals > 0 else 0
        
        reference_data = {
            "taux_mortalite_moyen": round(mortality_rate, 2),
            "total_animaux": total_animals,
            "total_deces": deaths
        }
        
        # Sauvegarder
        existing = await db.execute(
            select(ReferenceGenerale).where(
                ReferenceGenerale.espece == espece,
                ReferenceGenerale.type_reference == "mortalite"
            )
        )
        existing_ref = existing.scalar_one_or_none()
        
        if existing_ref and not force_regenerate:
            return {
                "success": False,
                "message": "Une référence existe déjà",
                "donnees_utilisees": total_animals
            }
        
        if existing_ref:
            existing_ref.donnees = reference_data
            existing_ref.nombre_donnees = total_animals
        else:
            new_ref = ReferenceGenerale(
                espece=espece,
                type_reference="mortalite",
                donnees=reference_data,
                nombre_donnees=total_animals,
                confiance=min(0.9, total_animals / 100),
                date_derniere_mise_a_jour=date.today()
            )
            db.add(new_ref)
        
        await db.commit()
        
        return {
            "success": True,
            "message": f"Référence de mortalité générée pour {espece}",
            "donnees_utilisees": total_animals,
            "taux_mortalite": round(mortality_rate, 2),
            "confiance": round(min(0.9, total_animals / 100) * 100, 1)
        }
    
    async def _generate_conversion_reference(
        self,
        db: AsyncSession,
        espece: str,
        force_regenerate: bool
    ) -> Dict[str, Any]:
        """Générer une référence de conversion alimentaire"""
        
        from ..models.animal import Animal
        from ..models.pesee import Pesee
        from ..models.alimentation import Alimentation
        
        # Calculer le FCR moyen
        fcr_values = []
        
        stmt = select(Animal).where(
            Animal.type_espece == espece,
            Animal.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        animals = result.scalars().all()
        
        for animal in animals:
            # Obtenir les pesées
            stmt = select(Pesee).where(Pesee.animal_id == animal.id).order_by(Pesee.date_pesee)
            pesees = (await db.execute(stmt)).scalars().all()
            
            if len(pesees) >= 2:
                weight_gain = pesees[-1].poids - pesees[0].poids
                days = (pesees[-1].date_pesee - pesees[0].date_pesee).days
                
                if days > 0 and weight_gain > 0:
                    # Obtenir la consommation
                    stmt = select(func.sum(Alimentation.poids_nourriture)).where(
                        Alimentation.animal_id == animal.id,
                        Alimentation.date >= pesees[0].date_pesee,
                        Alimentation.date <= pesees[-1].date_pesee
                    )
                    total_feed = (await db.execute(stmt)).scalar() or 0
                    
                    if total_feed > 0:
                        fcr = total_feed / weight_gain
                        fcr_values.append(fcr)
        
        if not fcr_values:
            return {
                "success": False,
                "message": "Données insuffisantes pour calculer le FCR",
                "donnees_utilisees": 0
            }
        
        avg_fcr = sum(fcr_values) / len(fcr_values)
        
        reference_data = {
            "fcr_moyen": round(avg_fcr, 2),
            "nombre_animaux_analyse": len(fcr_values),
            "fcr_min": min(fcr_values),
            "fcr_max": max(fcr_values)
        }
        
        existing = await db.execute(
            select(ReferenceGenerale).where(
                ReferenceGenerale.espece == espece,
                ReferenceGenerale.type_reference == "conversion"
            )
        )
        existing_ref = existing.scalar_one_or_none()
        
        if existing_ref and not force_regenerate:
            return {
                "success": False,
                "message": "Une référence existe déjà",
                "donnees_utilisees": len(fcr_values)
            }
        
        if existing_ref:
            existing_ref.donnees = reference_data
            existing_ref.nombre_donnees = len(fcr_values)
        else:
            new_ref = ReferenceGenerale(
                espece=espece,
                type_reference="conversion",
                donnees=reference_data,
                nombre_donnees=len(fcr_values),
                confiance=min(0.9, len(fcr_values) / 50),
                date_derniere_mise_a_jour=date.today()
            )
            db.add(new_ref)
        
        await db.commit()
        
        return {
            "success": True,
            "message": f"Référence de conversion générée pour {espece}",
            "donnees_utilisees": len(fcr_values),
            "fcr_moyen": round(avg_fcr, 2),
            "confiance": round(min(0.9, len(fcr_values) / 50) * 100, 1)
        }
    
    async def save_experimental_data(
        self,
        db: AsyncSession,
        entity_type: str,
        entity_id: int,
        user_id: int,
        is_test: bool = False
    ) -> DonneeExperimentale:
        """Sauvegarder une donnée comme expérimentale"""
        exp_data = DonneeExperimentale(
            utilisateur_id=user_id,
            entite_type=entity_type,
            entite_id=entity_id,
            est_essai=is_test,
            est_production=not is_test,
            date_essai=date.today()
        )
        db.add(exp_data)
        await db.commit()
        return exp_data
    
    async def get_confidence(
        self,
        db: AsyncSession,
        espece: str,
        prediction_type: str
    ) -> Dict[str, Any]:
        """Obtenir le niveau de confiance pour un type de prédiction"""
        
        # Récupérer la référence existante
        stmt = select(ReferenceGenerale).where(
            ReferenceGenerale.espece == espece,
            ReferenceGenerale.type_reference == prediction_type,
            ReferenceGenerale.is_active == True
        )
        result = await db.execute(stmt)
        reference = result.scalar_one_or_none()
        
        if not reference:
            return {
                "confidence": 0,
                "confidence_label": "Aucune",
                "facteurs": ["Pas assez de données collectées"],
                "donnees_manquantes": ["Pesées", "Historique"],
                "recommandations": ["Collectez au moins 10 données pour commencer"]
            }
        
        confidence = reference.confiance
        
        if confidence >= 0.7:
            label = "Élevée"
            facteurs = ["Nombre suffisant de données", "Données récentes"]
        elif confidence >= 0.4:
            label = "Moyenne"
            facteurs = ["Données modérées", "Peut être améliorée"]
        else:
            label = "Faible"
            facteurs = ["Données insuffisantes", "Période de collecte courte"]
        
        return {
            "confidence": round(confidence * 100, 1),
            "confidence_label": label,
            "facteurs": facteurs,
            "donnees_manquantes": self._get_missing_data(reference),
            "recommandations": self._get_recommendations(reference)
        }
    
    def _count_pesees(self, db: AsyncSession, espece: Optional[str]) -> int:
        """Compter le nombre de pesées"""
        # Implémentation simplifiée
        return 0
    
    def _count_mortalites(self, db: AsyncSession, espece: Optional[str]) -> int:
        """Compter le nombre de mortalités"""
        return 0
    
    def _count_alimentations(self, db: AsyncSession, espece: Optional[str]) -> int:
        """Compter le nombre d'enregistrements d'alimentation"""
        return 0
    
    async def _get_first_data_date(self, db: AsyncSession, espece: Optional[str]) -> Optional[date]:
        """Obtenir la date de la première donnée"""
        return None
    
    def _get_thresholds_status(self, total_data: int, days_collected: int) -> List[str]:
        """Obtenir le statut des seuils"""
        thresholds = []
        if total_data >= 50:
            thresholds.append("Données minimales atteintes")
        if total_data >= 200:
            thresholds.append("Données suffisantes pour prédictions fiables")
        if days_collected >= 60:
            thresholds.append("Période de collecte suffisante")
        return thresholds
    
    def _get_missing_data(self, reference: ReferenceGenerale) -> List[str]:
        """Identifier les données manquantes"""
        missing = []
        if reference.nombre_donnees < 20:
            missing.append("Plus de pesées")
        if reference.nombre_donnees < 10:
            missing.append("Historique de mortalité")
        return missing
    
    def _get_recommendations(self, reference: ReferenceGenerale) -> List[str]:
        """Générer des recommandations"""
        recommendations = []
        if reference.confiance < 0.5:
            recommendations.append("Continuez à collecter des données régulièrement")
        if reference.nombre_donnees < 30:
            recommendations.append("Ajoutez des pesées bi-mensuelles")
        return recommendations


experimental_service = ExperimentalService()