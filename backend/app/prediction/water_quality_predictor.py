# backend/app/prediction/water_quality_predictor.py
"""
Prédiction de la qualité de l'eau pour pisciculture
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WaterQualityPrediction:
    """Prédiction de qualité d'eau"""
    timestamp: datetime
    parameter: str
    predicted_value: float
    lower_bound: float
    upper_bound: float
    risk_level: str  # normal, warning, critical
    confidence: float


class WaterQualityPredictor:
    """
    Prédicteur de qualité d'eau pour bassins piscicoles
    Basé sur l'historique, la densité de poissons, la température, etc.
    """
    
    # Seuils critiques par paramètre
    THRESHOLDS = {
        "oxygen": {"critical_min": 3.0, "warning_min": 4.0, "optimal_min": 5.0},
        "ph": {"critical_min": 5.5, "warning_min": 6.0, "critical_max": 9.5, "warning_max": 9.0, "optimal_min": 6.5, "optimal_max": 8.5},
        "ammonia": {"critical_max": 0.1, "warning_max": 0.05, "optimal_max": 0.02},
        "nitrite": {"critical_max": 1.0, "warning_max": 0.5, "optimal_max": 0.2},
        "temperature": {"critical_min": 4, "warning_min": 8, "critical_max": 35, "warning_max": 30}
    }
    
    async def predict_oxygen_level(
        self,
        current_oxygen: float,
        fish_biomass_kg: float,
        water_temperature: float,
        feeding_rate_kg: float,
        aeration_active: bool,
        hours_ahead: int = 24
    ) -> List[WaterQualityPrediction]:
        """
        Prédire l'évolution de l'oxygène dissous
        
        L'oxygène est le paramètre le plus critique pour la survie des poissons
        """
        predictions = []
        
        # Facteurs affectant l'oxygène
        consumption_rate = 0.05 * (fish_biomass_kg / 1000)  # kg O2/heure par tonne de biomasse
        temperature_factor = 1.0 + (water_temperature - 20) * 0.05  # +5% par °C au-dessus de 20
        feeding_factor = 1.0 + (feeding_rate_kg / 10)  # +10% par 10kg de nourriture
        aeration_factor = 0.3 if aeration_active else 0  # kg O2/heure apporté par aération
        
        net_consumption = consumption_rate * temperature_factor * feeding_factor - aeration_factor
        
        for h in range(hours_ahead):
            predicted_oxygen = current_oxygen - (net_consumption * h)
            
            # Ajouter une incertitude croissante avec l'horizon
            uncertainty = 0.1 + h * 0.02
            lower_bound = max(0, predicted_oxygen - uncertainty)
            upper_bound = predicted_oxygen + uncertainty
            
            # Déterminer le niveau de risque
            risk_level = "normal"
            if predicted_oxygen < self.THRESHOLDS["oxygen"]["critical_min"]:
                risk_level = "critical"
            elif predicted_oxygen < self.THRESHOLDS["oxygen"]["warning_min"]:
                risk_level = "warning"
            
            confidence = max(0.3, 0.9 - h * 0.03)
            
            predictions.append(WaterQualityPrediction(
                timestamp=datetime.now() + timedelta(hours=h),
                parameter="oxygen",
                predicted_value=round(predicted_oxygen, 2),
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2),
                risk_level=risk_level,
                confidence=round(confidence, 2)
            ))
            
            # Alerte critique imminente
            if predicted_oxygen < 2.5 and h < 6:
                predictions.append(WaterQualityPrediction(
                    timestamp=datetime.now() + timedelta(hours=h),
                    parameter="oxygen_alert",
                    predicted_value=predicted_oxygen,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    risk_level="critical",
                    confidence=0.95
                ))
        
        return predictions
    
    async def predict_ammonia_peak(
        self,
        current_ammonia: float,
        fish_biomass_kg: float,
        feeding_time: datetime,
        feeding_amount_kg: float,
        filtration_efficiency: float
    ) -> Dict[str, any]:
        """
        Prédire le pic d'ammoniac après nourrissage
        """
        # L'ammoniac monte généralement 2-4 heures après le nourrissage
        hours_to_peak = 3
        peak_time = feeding_time + timedelta(hours=hours_to_peak)
        
        # Calcul du pic estimé
        ammonia_production = feeding_amount_kg * 0.03  # ~3% de l'aliment devient ammoniac
        peak_ammonia = current_ammonia + (ammonia_production / fish_biomass_kg) * (1 - filtration_efficiency)
        
        # Niveau de risque
        risk_level = "normal"
        if peak_ammonia > self.THRESHOLDS["ammonia"]["critical_max"]:
            risk_level = "critical"
        elif peak_ammonia > self.THRESHOLDS["ammonia"]["warning_max"]:
            risk_level = "warning"
        
        # Recommandations
        recommendations = []
        if risk_level == "critical":
            recommendations = [
                "URGENT: Réduire immédiatement la quantité de nourriture",
                "Augmenter l'aération",
                "Effectuer un renouvellement d'eau partiel (20-30%)"
            ]
        elif risk_level == "warning":
            recommendations = [
                "Surveiller l'ammoniac toutes les heures",
                "Réduire la prochaine ration de 50%",
                "Augmenter la filtration si possible"
            ]
        
        return {
            "current_ammonia": round(current_ammonia, 3),
            "predicted_peak_ammonia": round(peak_ammonia, 3),
            "peak_time": peak_time,
            "hours_to_peak": hours_to_peak,
            "risk_level": risk_level,
            "threshold_warning": self.THRESHOLDS["ammonia"]["warning_max"],
            "threshold_critical": self.THRESHOLDS["ammonia"]["critical_max"],
            "recommendations": recommendations
        }
    
    async def predict_ph_evolution(
        self,
        current_ph: float,
        water_hardness: float,
        fish_density: float,
        rainfall_forecast: float,
        hours_ahead: int = 48
    ) -> List[WaterQualityPrediction]:
        """
        Prédire l'évolution du pH
        """
        predictions = []
        
        # Facteurs influençant le pH
        respiration_factor = -0.01 * (fish_density / 10)  # CO2 des poissons baisse le pH
        rainfall_factor = -0.002 * rainfall_forecast  # Pluie acide
        hardness_buffer = min(0.5, water_hardness / 200)  # Eau dure tamponne mieux
        
        net_change_per_hour = (respiration_factor + rainfall_factor) * (1 - hardness_buffer)
        
        for h in range(hours_ahead):
            predicted_ph = current_ph + net_change_per_hour * h
            
            # Ajouter une incertitude
            uncertainty = 0.05 + h * 0.005
            lower_bound = max(0, predicted_ph - uncertainty)
            upper_bound = min(14, predicted_ph + uncertainty)
            
            # Niveau de risque
            risk_level = "normal"
            if predicted_ph < self.THRESHOLDS["ph"]["critical_min"] or predicted_ph > self.THRESHOLDS["ph"]["critical_max"]:
                risk_level = "critical"
            elif predicted_ph < self.THRESHOLDS["ph"]["warning_min"] or predicted_ph > self.THRESHOLDS["ph"]["warning_max"]:
                risk_level = "warning"
            
            confidence = max(0.4, 0.85 - h * 0.01)
            
            predictions.append(WaterQualityPrediction(
                timestamp=datetime.now() + timedelta(hours=h),
                parameter="ph",
                predicted_value=round(predicted_ph, 1),
                lower_bound=round(lower_bound, 1),
                upper_bound=round(upper_bound, 1),
                risk_level=risk_level,
                confidence=round(confidence, 2)
            ))
        
        return predictions
    
    async def get_critical_alerts(
        self,
        current_quality: Dict[str, float],
        fish_species: str
    ) -> List[Dict[str, any]]:
        """
        Vérifier si des paramètres sont critiques et générer des alertes
        """
        alerts = []
        
        # Seuils spécifiques par espèce
        species_thresholds = self._get_species_thresholds(fish_species)
        
        for param, value in current_quality.items():
            if param in species_thresholds:
                thresholds = species_thresholds[param]
                
                if thresholds.get("critical_min") and value < thresholds["critical_min"]:
                    alerts.append({
                        "parameter": param,
                        "value": value,
                        "threshold": thresholds["critical_min"],
                        "severity": "critical",
                        "direction": "below",
                        "message": f"{param.upper()} trop bas: {value} (seuil critique: {thresholds['critical_min']})",
                        "recommended_action": self._get_action_for_parameter(param, "low")
                    })
                
                if thresholds.get("critical_max") and value > thresholds["critical_max"]:
                    alerts.append({
                        "parameter": param,
                        "value": value,
                        "threshold": thresholds["critical_max"],
                        "severity": "critical",
                        "direction": "above",
                        "message": f"{param.upper()} trop élevé: {value} (seuil critique: {thresholds['critical_max']})",
                        "recommended_action": self._get_action_for_parameter(param, "high")
                    })
                
                elif thresholds.get("warning_min") and value < thresholds["warning_min"]:
                    alerts.append({
                        "parameter": param,
                        "value": value,
                        "threshold": thresholds["warning_min"],
                        "severity": "warning",
                        "direction": "below",
                        "message": f"{param.upper()} bas: {value} (seuil d'alerte: {thresholds['warning_min']})",
                        "recommended_action": self._get_action_for_parameter(param, "low")
                    })
                
                elif thresholds.get("warning_max") and value > thresholds["warning_max"]:
                    alerts.append({
                        "parameter": param,
                        "value": value,
                        "threshold": thresholds["warning_max"],
                        "severity": "warning",
                        "direction": "above",
                        "message": f"{param.upper()} élevé: {value} (seuil d'alerte: {thresholds['warning_max']})",
                        "recommended_action": self._get_action_for_parameter(param, "high")
                    })
        
        return alerts
    
    def _get_species_thresholds(self, species: str) -> Dict:
        """Obtenir les seuils spécifiques par espèce de poisson"""
        if species.lower() == "tilapia":
            return {
                "oxygen": {"critical_min": 3.0, "warning_min": 4.0, "optimal_min": 5.0},
                "ph": {"critical_min": 6.0, "warning_min": 6.5, "critical_max": 9.0, "warning_max": 8.5},
                "ammonia": {"critical_max": 0.1, "warning_max": 0.05},
                "temperature": {"critical_min": 14, "warning_min": 20, "critical_max": 36, "warning_max": 32}
            }
        elif species.lower() == "truite":
            return {
                "oxygen": {"critical_min": 5.0, "warning_min": 6.0, "optimal_min": 8.0},
                "ph": {"critical_min": 5.5, "warning_min": 6.0, "critical_max": 8.5, "warning_max": 8.0},
                "ammonia": {"critical_max": 0.02, "warning_max": 0.01},
                "temperature": {"critical_min": 4, "warning_min": 6, "critical_max": 18, "warning_max": 15}
            }
        else:  # clarias ou autre
            return self.THRESHOLDS
    
    def _get_action_for_parameter(self, parameter: str, direction: str) -> str:
        """Obtenir l'action recommandée pour un paramètre hors norme"""
        actions = {
            "oxygen": {
                "low": "Activer l'aération d'urgence, réduire l'alimentation, renouveler l'eau"
            },
            "ammonia": {
                "high": "Réduire l'alimentation, augmenter la filtration, renouveler l'eau (30-50%)"
            },
            "ph": {
                "low": "Ajouter du bicarbonate de sodium progressivement",
                "high": "Ajouter de l'acide phosphorique ou du CO2"
            },
            "temperature": {
                "low": "Activer le chauffage ou réduire le renouvellement d'eau",
                "high": "Augmenter l'aération, ombrager le bassin, renouveler l'eau"
            },
            "nitrite": {
                "high": "Ajouter du sel (NaCl) à 0.1-0.3%, réduire l'alimentation"
            }
        }
        
        return actions.get(parameter, {}).get(direction, "Contacter le technicien")


water_quality_predictor = WaterQualityPredictor()