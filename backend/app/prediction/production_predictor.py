# backend/app/prediction/production_predictor.py
"""
Prédiction de production: lait, œufs, larves
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProductionPrediction:
    """Prédiction de production"""
    daily_estimate: float
    weekly_estimate: float
    monthly_estimate: float
    confidence: float
    seasonal_factor: float
    trend: str  # increasing, decreasing, stable


class ProductionPredictor:
    """
    Prédicteur de production agricole
    Lait (bovins, ovins, caprins), œufs (avicoles), larves (entomoculture)
    """
    
    # Courbes de lactation par défaut (litres/jour)
    LACTATION_CURVES = {
        "bovin": {
            "holstein": {"peak": 45, "duration": 305, "decline_rate": 0.08},
            "normande": {"peak": 25, "duration": 280, "decline_rate": 0.07},
            "default": {"peak": 30, "duration": 300, "decline_rate": 0.08}
        },
        "ovin": {
            "default": {"peak": 2.5, "duration": 180, "decline_rate": 0.1}
        },
        "caprin": {
            "default": {"peak": 4.0, "duration": 240, "decline_rate": 0.09}
        }
    }
    
    # Courbes de ponte (œufs/jour)
    EGG_CURVES = {
        "leghorn": {"peak": 0.95, "peak_age_weeks": 30, "decline_rate": 0.01},
        "rhode_island": {"peak": 0.85, "peak_age_weeks": 32, "decline_rate": 0.008},
        "default": {"peak": 0.8, "peak_age_weeks": 30, "decline_rate": 0.01}
    }
    
    # Facteurs saisonniers (impact sur production)
    SEASONAL_FACTORS = {
        "lait": {
            "spring": 1.05,
            "summer": 0.95,
            "autumn": 1.0,
            "winter": 1.0
        },
        "oeufs": {
            "spring": 1.1,
            "summer": 0.9,
            "autumn": 1.0,
            "winter": 0.85
        },
        "larves": {
            "spring": 1.05,
            "summer": 1.15,
            "autumn": 1.0,
            "winter": 0.8
        }
    }
    
    async def predict_milk_production(
        self,
        espece: str,
        race: str,
        lactation_day: int,
        daily_production: float,
        production_history: Optional[List[Tuple[int, float]]] = None,
        horizon_jours: int = 90
    ) -> ProductionPrediction:
        """
        Prédire la production laitière
        
        Args:
            espece: bovin, ovin, caprin
            race: Race de l'animal
            lactation_day: Jour de lactation (1-305)
            daily_production: Production quotidienne actuelle (litres)
            production_history: Historique des productions
            horizon_jours: Horizon de prédiction
        
        Returns:
            Prédiction de production
        """
        # Obtenir la courbe de lactation
        curve = self.LACTATION_CURVES.get(espece, {}).get(
            race.lower() if race else "default",
            self.LACTATION_CURVES.get(espece, {}).get("default", {"peak": 30, "duration": 300, "decline_rate": 0.08})
        )
        
        # Calculer la production estimée sur l'horizon
        peak_day = 60  # Pic de lactation généralement à 60 jours
        if lactation_day <= peak_day:
            # Phase de montée vers le pic
            remaining_peak = max(0, peak_day - lactation_day)
            factor = 1.0 + (remaining_peak / peak_day) * 0.2
            daily_estimate = daily_production * factor
        else:
            # Phase de déclin
            days_past_peak = lactation_day - peak_day
            decline_factor = max(0.3, 1.0 - (days_past_peak * curve["decline_rate"] / 30))
            daily_estimate = daily_production * decline_factor
        
        # Appliquer le facteur saisonnier
        seasonal_factor = self._get_seasonal_factor("lait")
        daily_estimate *= seasonal_factor
        
        # Calculer les agrégats
        weekly_estimate = daily_estimate * 7
        monthly_estimate = daily_estimate * 30
        
        # Déterminer la tendance
        if lactation_day < 60:
            trend = "increasing"
        elif lactation_day > 200:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Calculer la confiance
        confidence = self._calculate_confidence(len(production_history) if production_history else 0)
        
        return ProductionPrediction(
            daily_estimate=round(daily_estimate, 1),
            weekly_estimate=round(weekly_estimate, 1),
            monthly_estimate=round(monthly_estimate, 0),
            confidence=confidence,
            seasonal_factor=seasonal_factor,
            trend=trend
        )
    
    async def predict_egg_production(
        self,
        race: str,
        age_weeks: int,
        daily_production: float,
        horizon_jours: int = 30
    ) -> ProductionPrediction:
        """
        Prédire la production d'œufs
        
        Args:
            race: Race de la poule
            age_weeks: Âge en semaines
            daily_production: Production quotidienne actuelle (œufs/jour)
            horizon_jours: Horizon de prédiction
        
        Returns:
            Prédiction de production
        """
        curve = self.EGG_CURVES.get(race.lower() if race else "default", self.EGG_CURVES["default"])
        
        peak_age = curve["peak_age_weeks"]
        
        if age_weeks <= peak_age:
            # Phase de montée
            progress = age_weeks / peak_age
            daily_estimate = curve["peak"] * progress
        else:
            # Phase de déclin
            weeks_past_peak = age_weeks - peak_age
            decline = 1.0 - (weeks_past_peak * curve["decline_rate"])
            daily_estimate = curve["peak"] * max(0.5, decline)
        
        # Ajuster selon la production actuelle si disponible
        if daily_production > 0:
            ratio = daily_production / daily_estimate
            daily_estimate = daily_production * 0.9 + daily_estimate * 0.1  # Lissage
        
        # Facteur saisonnier
        seasonal_factor = self._get_seasonal_factor("oeufs")
        daily_estimate *= seasonal_factor
        
        # Poids moyen d'un œuf (g)
        egg_weight = 55 if race.lower() == "leghorn" else 60
        
        return ProductionPrediction(
            daily_estimate=round(daily_estimate, 2),
            weekly_estimate=round(daily_estimate * 7, 1),
            monthly_estimate=round(daily_estimate * 30, 0),
            confidence=0.7,
            seasonal_factor=seasonal_factor,
            trend="decreasing" if age_weeks > peak_age else "increasing"
        )
    
    async def predict_larvae_production(
        self,
        espece: str,
        temperature: float,
        humidity: float,
        substrate_quality: float = 0.8,
        horizon_jours: int = 30
    ) -> ProductionPrediction:
        """
        Prédire la production de larves (entomoculture)
        
        Args:
            espece: Hermetia illucens, Tenebrio molitor, etc.
            temperature: Température actuelle (°C)
            humidity: Humidité relative (%)
            substrate_quality: Qualité du substrat (0-1)
            horizon_jours: Horizon de prédiction
        """
        # Taux de croissance de base par espèce (g/jour par kg de substrat)
        base_rates = {
            "hermetia": 0.08,
            "tenebrio": 0.03,
            "grillon": 0.05
        }
        
        species_key = "hermetia" if "hermetia" in espece.lower() else "tenebrio" if "tenebrio" in espece.lower() else "grillon"
        base_rate = base_rates.get(species_key, 0.04)
        
        # Facteurs environnementaux
        temp_factor = 1.0
        if 25 <= temperature <= 32:
            temp_factor = 1.0
        elif 20 <= temperature < 25:
            temp_factor = 0.7
        elif temperature > 35:
            temp_factor = 0.5
        else:
            temp_factor = 0.3
        
        humidity_factor = 1.0 if 60 <= humidity <= 80 else 0.7 if 40 <= humidity <= 90 else 0.4
        
        # Production quotidienne
        daily_estimate = base_rate * temp_factor * humidity_factor * substrate_quality
        
        # Facteur saisonnier
        seasonal_factor = self._get_seasonal_factor("larves")
        daily_estimate *= seasonal_factor
        
        return ProductionPrediction(
            daily_estimate=round(daily_estimate, 3),
            weekly_estimate=round(daily_estimate * 7, 2),
            monthly_estimate=round(daily_estimate * 30, 1),
            confidence=0.6,
            seasonal_factor=seasonal_factor,
            trend="stable"
        )
    
    def _get_seasonal_factor(self, production_type: str) -> float:
        """Obtenir le facteur saisonnier selon la saison actuelle"""
        from datetime import datetime
        month = datetime.now().month
        
        if month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        elif month in [9, 10, 11]:
            season = "autumn"
        else:
            season = "winter"
        
        return self.SEASONAL_FACTORS.get(production_type, {}).get(season, 1.0)
    
    def _calculate_confidence(self, history_length: int) -> float:
        """Calculer le niveau de confiance"""
        if history_length >= 30:
            return 0.85
        elif history_length >= 15:
            return 0.7
        elif history_length >= 7:
            return 0.55
        else:
            return 0.4


production_predictor = ProductionPredictor()