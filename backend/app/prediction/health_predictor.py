# backend/app/prediction/health_predictor.py
"""
Prédiction sanitaire: risques de maladie, mortalité
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import date, timedelta

logger = logging.getLogger(__name__)


class HealthPredictor:
    """
    Prédicteur de risques sanitaires
    Détection précoce de problèmes de santé
    """
    
    # Facteurs de risque par espèce
    RISK_FACTORS = {
        "bovin": {
            "vaccination_delayed": 1.5,
            "high_density": 1.3,
            "poor_hygiene": 1.4,
            "feed_change": 1.2,
            "heat_stress": 1.3
        },
        "avicole": {
            "vaccination_delayed": 1.6,
            "high_density": 1.5,
            "poor_hygiene": 1.5,
            "ammonia_high": 1.4,
            "temperature_extreme": 1.4
        },
        "piscicole": {
            "oxygen_low": 1.8,
            "ammonia_high": 1.6,
            "temperature_extreme": 1.4,
            "high_density": 1.3
        }
    }
    
    # Seuils de mortalité acceptables par espèce (%)
    MORTALITY_THRESHOLDS = {
        "bovin": {"warning": 2.0, "critical": 5.0},
        "ovin": {"warning": 3.0, "critical": 6.0},
        "caprin": {"warning": 3.0, "critical": 6.0},
        "avicole": {"warning": 5.0, "critical": 10.0},
        "piscicole": {"warning": 1.0, "critical": 3.0},
        "entomoculture": {"warning": 10.0, "critical": 20.0}
    }
    
    async def predict_mortality_risk(
        self,
        espece: str,
        enclos_id: int,
        mortality_history: List[float],
        environmental_factors: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Prédire le risque de mortalité
        
        Returns:
            Dictionnaire avec risque, probabilité, recommandations
        """
        # Calculer le taux de mortalité actuel
        current_mortality = sum(mortality_history[-30:]) / len(mortality_history[-30:]) if mortality_history else 0
        
        # Risque de base
        base_risk = current_mortality / self.MORTALITY_THRESHOLDS.get(espece, {}).get("warning", 5.0)
        
        # Ajuster selon les facteurs environnementaux
        risk_multiplier = 1.0
        risk_factors_applied = []
        
        for factor, weight in self.RISK_FACTORS.get(espece, {}).items():
            if environmental_factors.get(factor, False):
                risk_multiplier *= weight
                risk_factors_applied.append(factor)
        
        # Risque total
        total_risk = min(base_risk * risk_multiplier, 2.0)
        
        # Déterminer le niveau de risque
        if total_risk >= 1.5:
            risk_level = "critical"
            probability = min(total_risk * 50, 90)
        elif total_risk >= 1.0:
            risk_level = "high"
            probability = total_risk * 30
        elif total_risk >= 0.5:
            risk_level = "medium"
            probability = total_risk * 20
        else:
            risk_level = "low"
            probability = total_risk * 10
        
        # Recommandations
        recommendations = self._get_mortality_recommendations(risk_level, risk_factors_applied)
        
        # Date estimée du dépassement de seuil
        days_to_threshold = self._estimate_days_to_threshold(
            current_mortality,
            total_risk,
            self.MORTALITY_THRESHOLDS.get(espece, {}).get("critical", 5.0)
        )
        
        return {
            "risk_level": risk_level,
            "probability_percent": round(probability, 1),
            "current_mortality_percent": round(current_mortality, 2),
            "total_risk_score": round(total_risk, 2),
            "risk_factors": risk_factors_applied,
            "days_to_critical_threshold": days_to_threshold,
            "recommendations": recommendations
        }
    
    async def predict_disease_risk(
        self,
        espece: str,
        vaccination_status: Dict[str, date],
        population_density: float,
        hygiene_score: float
    ) -> Dict[str, any]:
        """
        Prédire le risque d'épidémie
        
        Returns:
            Dictionnaire avec risque par maladie
        """
        diseases_risk = {}
        
        for disease, last_vaccination in vaccination_status.items():
            # Calculer le risque pour chaque maladie
            days_since_vaccination = (date.today() - last_vaccination).days if last_vaccination else 999
            
            if days_since_vaccination > 365:
                vaccine_risk = 0.8
            elif days_since_vaccination > 180:
                vaccine_risk = 0.4
            elif days_since_vaccination > 90:
                vaccine_risk = 0.2
            else:
                vaccine_risk = 0.05
            
            density_risk = min(population_density / 10, 0.3)
            hygiene_risk = max(0, (1 - hygiene_score) * 0.3)
            
            total_risk = min(vaccine_risk + density_risk + hygiene_risk, 1.0)
            
            diseases_risk[disease] = {
                "risk_percent": round(total_risk * 100, 1),
                "vaccine_protection": round((1 - vaccine_risk) * 100, 1),
                "recommendation": "Vaccination recommandée" if vaccine_risk > 0.5 else "Surveillance normale"
            }
        
        return diseases_risk
    
    async def detect_health_anomaly(
        self,
        espece: str,
        weight_history: List[float],
        feed_consumption: List[float],
        production: List[float]
    ) -> Dict[str, any]:
        """
        Détecter les anomalies de santé via les données de production
        
        Returns:
            Anomalies détectées avec sévérité
        """
        anomalies = []
        
        # Vérifier la perte de poids
        if len(weight_history) >= 3:
            weight_trend = weight_history[-1] - weight_history[-2]
            if weight_trend < 0:
                anomalies.append({
                    "type": "weight_loss",
                    "severity": "warning" if abs(weight_trend) < 5 else "critical",
                    "message": f"Perte de poids de {abs(weight_trend):.1f} kg"
                })
        
        # Vérifier la baisse de consommation
        if len(feed_consumption) >= 3:
            avg_consumption = sum(feed_consumption[-7:]) / 7
            if feed_consumption[-1] < avg_consumption * 0.7:
                anomalies.append({
                    "type": "reduced_feed_intake",
                    "severity": "warning",
                    "message": "Baisse significative de la consommation alimentaire"
                })
        
        # Vérifier la baisse de production
        if len(production) >= 3:
            avg_production = sum(production[-7:]) / 7
            if production[-1] < avg_production * 0.8:
                anomalies.append({
                    "type": "reduced_production",
                    "severity": "warning",
                    "message": "Baisse de production détectée"
                })
        
        return {
            "anomalies": anomalies,
            "health_score": self._calculate_health_score(anomalies),
            "requires_vet_attention": len([a for a in anomalies if a["severity"] == "critical"]) > 0
        }
    
    def _get_mortality_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Générer des recommandations selon le niveau de risque"""
        recommendations = {
            "critical": [
                "URGENT: Contacter un vétérinaire immédiatement",
                "Mettre en quarantaine les animaux malades",
                "Renforcer les mesures d'hygiène",
                "Vérifier tous les paramètres environnementaux"
            ],
            "high": [
                "Planifier une visite vétérinaire sous 48h",
                "Surveiller attentivement les animaux",
                "Vérifier l'alimentation et l'eau"
            ],
            "medium": [
                "Renforcer la surveillance",
                "Vérifier les vaccinations",
                "Améliorer l'hygiène si nécessaire"
            ],
            "low": [
                "Maintenir les bonnes pratiques",
                "Surveillance normale recommandée"
            ]
        }
        
        base_recs = recommendations.get(risk_level, recommendations["low"])
        
        # Ajouter des recommandations spécifiques aux facteurs de risque
        specific_recs = []
        if "high_density" in risk_factors:
            specific_recs.append("Réduire la densité animale dans l'enclos")
        if "poor_hygiene" in risk_factors:
            specific_recs.append("Intensifier le nettoyage et la désinfection")
        if "vaccination_delayed" in risk_factors:
            specific_recs.append("Planifier les vaccinations manquantes")
        
        return base_recs + specific_recs
    
    def _estimate_days_to_threshold(self, current_rate: float, risk: float, threshold: float) -> Optional[int]:
        """Estimer le nombre de jours avant d'atteindre le seuil critique"""
        if current_rate >= threshold:
            return 0
        
        if risk <= 0:
            return None
        
        # Taux de progression estimé par jour
        daily_increase = risk * 0.1
        
        if daily_increase <= 0:
            return None
        
        days = int((threshold - current_rate) / daily_increase)
        return max(days, 1)
    
    def _calculate_health_score(self, anomalies: List[Dict]) -> int:
        """Calculer un score de santé global (0-100)"""
        base_score = 100
        penalty = 0
        
        for anomaly in anomalies:
            if anomaly["severity"] == "critical":
                penalty += 30
            elif anomaly["severity"] == "warning":
                penalty += 15
        
        return max(0, base_score - penalty)


health_predictor = HealthPredictor()