# backend/app/prediction/growth_predictor.py
"""
Prédiction de croissance animale
Modèles: Gompertz, Logistique, Von Bertalanffy
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

from ..cpp_bindings.growth_binding import growth_predictor as cpp_growth
from ..models.reference import ReferenceCroissance
from ..database import get_db

logger = logging.getLogger(__name__)


@dataclass
class GrowthPredictionPoint:
    """Point de prédiction de croissance"""
    day: int
    weight_min: float
    weight_mean: float
    weight_max: float
    confidence: float


class GrowthPredictor:
    """
    Prédicteur de croissance animale
    Utilise les modèles C++ quand disponibles, sinon fallback Python
    """
    
    def __init__(self):
        self.cpp_predictor = cpp_growth
    
    async def predict_animal_growth(
        self,
        espece: str,
        race: str,
        age_jours: int,
        poids_actuel_kg: float,
        pesees_historiques: Optional[List[Tuple[int, float]]] = None,
        horizon_jours: int = 90,
        model_type: str = "gompertz"
    ) -> List[GrowthPredictionPoint]:
        """
        Prédire la croissance d'un animal
        
        Args:
            espece: bovin, ovin, caprin, avicole, piscicole
            race: Race de l'animal
            age_jours: Âge actuel en jours
            poids_actuel_kg: Poids actuel en kg
            pesees_historiques: Liste des (âge_jours, poids_kg) historiques
            horizon_jours: Nombre de jours à prédire
            model_type: Type de modèle
        
        Returns:
            Liste des points de prédiction
        """
        # Récupérer les références de croissance
        reference_weights = await self._get_reference_weights(espece, race)
        
        # Ajuster le modèle avec les données réelles si disponibles
        params = await self._estimate_parameters(
            espece, race, age_jours, poids_actuel_kg, pesees_historiques
        )
        
        # Jours cibles pour la prédiction
        target_days = list(range(age_jours, age_jours + horizon_jours + 1, 7))
        
        # Utiliser le prédicteur C++ ou Python
        predictions = self.cpp_predictor.predict_gompertz(
            weight_initial=poids_actuel_kg,
            age_initial_days=age_jours,
            target_days=target_days,
            weight_inf=params.get("weight_inf"),
            growth_rate=params.get("growth_rate")
        )
        
        # Calculer la confiance de chaque prédiction
        confidence = await self._calculate_confidence(
            espece, race, len(pesees_historiques) if pesees_historiques else 0
        )
        
        results = []
        for pred in predictions:
            # Ajuster la confiance en fonction de l'horizon
            horizon_factor = 1.0 - (pred.day - age_jours) / horizon_jours * 0.5
            point_confidence = confidence * horizon_factor
            
            results.append(GrowthPredictionPoint(
                day=pred.day,
                weight_min=pred.weight_min,
                weight_mean=pred.weight_mean,
                weight_max=pred.weight_max,
                confidence=point_confidence
            ))
        
        return results
    
    async def get_age_at_weight(
        self,
        espece: str,
        race: str,
        poids_cible_kg: float,
        poids_actuel_kg: float,
        age_jours: int
    ) -> Optional[int]:
        """
        Estimer l'âge auquel un animal atteindra un poids cible
        
        Returns:
            Nombre de jours depuis la naissance, ou None si impossible
        """
        predictions = await self.predict_animal_growth(
            espece, race, age_jours, poids_actuel_kg,
            horizon_jours=365
        )
        
        for pred in predictions:
            if pred.weight_mean >= poids_cible_kg:
                return pred.day
        
        return None
    
    async def detect_growth_anomaly(
        self,
        espece: str,
        race: str,
        age_jours: int,
        poids_kg: float,
        pesees_historiques: List[Tuple[int, float]]
    ) -> Dict[str, any]:
        """
        Détecter les anomalies de croissance
        
        Returns:
            Dictionnaire avec anomalie détectée, écart et recommandations
        """
        # Obtenir le poids attendu
        expected = await self._get_expected_weight(espece, race, age_jours)
        
        if expected is None:
            return {
                "anomaly_detected": False,
                "deviation_percent": 0,
                "severity": "unknown",
                "recommendations": ["Données de référence insuffisantes"]
            }
        
        deviation = ((poids_kg - expected) / expected) * 100
        
        if deviation < -20:
            severity = "critical"
            recommendations = [
                "Consulter un vétérinaire",
                "Vérifier l'alimentation",
                "Vérifier les conditions sanitaires"
            ]
        elif deviation < -10:
            severity = "warning"
            recommendations = [
                "Augmenter la ration alimentaire",
                "Vérifier la qualité de l'aliment",
                "Surveiller le comportement"
            ]
        elif deviation > 20:
            severity = "info"
            recommendations = ["Poids supérieur à la moyenne, maintenir le suivi"]
        else:
            severity = "normal"
            recommendations = ["Croissance normale"]
        
        return {
            "anomaly_detected": abs(deviation) > 15,
            "deviation_percent": round(deviation, 1),
            "expected_weight": round(expected, 1),
            "severity": severity,
            "recommendations": recommendations
        }
    
    async def _get_reference_weights(
        self,
        espece: str,
        race: str
    ) -> Dict[int, Dict[str, float]]:
        """Récupérer les poids de référence depuis la base"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy import select
        
        async for db in get_db():
            stmt = select(ReferenceCroissance).where(
                ReferenceCroissance.espece == espece,
                ReferenceCroissance.race == race,
                ReferenceCroissance.is_active == True
            )
            result = await db.execute(stmt)
            references = result.scalars().all()
            
            return {
                ref.age_jours: {
                    "min": ref.poids_min,
                    "mean": ref.poids_moyen,
                    "max": ref.poids_max
                } for ref in references
            }
    
    async def _get_expected_weight(
        self,
        espece: str,
        race: str,
        age_jours: int
    ) -> Optional[float]:
        """Obtenir le poids attendu pour un âge donné"""
        references = await self._get_reference_weights(espece, race)
        
        # Trouver la référence la plus proche
        closest_age = min(references.keys(), key=lambda x: abs(x - age_jours))
        if abs(closest_age - age_jours) <= 30:
            return references[closest_age]["mean"]
        
        return None
    
    async def _estimate_parameters(
        self,
        espece: str,
        race: str,
        age_jours: int,
        poids_actuel_kg: float,
        pesees_historiques: Optional[List[Tuple[int, float]]]
    ) -> Dict[str, float]:
        """Estimer les paramètres du modèle de croissance"""
        params = {
            "weight_inf": None,
            "growth_rate": None,
        }
        
        # Essayer d'estimer à partir des données historiques
        if pesees_historiques and len(pesees_historiques) >= 3:
            ages = [p[0] for p in pesees_historiques]
            weights = [p[1] for p in pesees_historiques]
            
            estimated = self.cpp_predictor.estimate_parameters(ages, weights)
            params["weight_inf"] = estimated.get("weight_inf")
            params["growth_rate"] = estimated.get("growth_rate")
        
        # Sinon utiliser les références
        if not params["weight_inf"]:
            references = await self._get_reference_weights(espece, race)
            if references:
                max_age = max(references.keys())
                params["weight_inf"] = references[max_age]["max"] * 1.1
                params["growth_rate"] = 0.01  # Valeur par défaut
        
        return params
    
    async def _calculate_confidence(
        self,
        espece: str,
        race: str,
        nb_pesees: int
    ) -> float:
        """Calculer le niveau de confiance de la prédiction"""
        # Facteurs de confiance
        confidence = 0.5  # Base
        
        # Plus de pesées = plus de confiance
        if nb_pesees >= 10:
            confidence += 0.3
        elif nb_pesees >= 5:
            confidence += 0.15
        
        # Références disponibles ?
        references = await self._get_reference_weights(espece, race)
        if references:
            confidence += 0.2
        
        return min(confidence, 0.95)


growth_predictor = GrowthPredictor()