# backend/app/prediction/feed_predictor.py
"""
Prédiction de consommation alimentaire et gestion des stocks
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeedConsumption:
    """Consommation alimentaire quotidienne"""
    date: date
    amount_kg: float
    cost_eur: float


class FeedPredictor:
    """
    Prédicteur de consommation alimentaire
    Basé sur le poids des animaux, la production, la saison
    """
    
    # Consommation moyenne par espèce (kg/jour par animal)
    BASE_CONSUMPTION = {
        "bovin_viande": 8.0,
        "bovin_lait": 12.0,
        "ovin": 1.5,
        "caprin": 1.5,
        "avicole": 0.12,
        "piscicole": 0.02,
        "entomoculture": 0.001
    }
    
    async def predict_feed_consumption(
        self,
        espece: str,
        animal_count: int,
        avg_weight_kg: float,
        production_type: Optional[str] = None,
        horizon_days: int = 30
    ) -> Dict[str, any]:
        """
        Prédire la consommation alimentaire future
        """
        base_consumption = self.BASE_CONSUMPTION.get(f"{espece}_{production_type}" if production_type else espece, 
                                                      self.BASE_CONSUMPTION.get(espece, 1.0))
        
        # Ajustement basé sur le poids
        weight_factor = avg_weight_kg / 500 if espece == "bovin" else avg_weight_kg / 50 if espece in ["ovin", "caprin"] else 1.0
        weight_factor = min(max(weight_factor, 0.5), 1.5)
        
        # Ajustement saisonnier
        seasonal_factor = self._get_seasonal_factor()
        
        daily_consumption_per_animal = base_consumption * weight_factor * seasonal_factor
        total_daily_consumption = daily_consumption_per_animal * animal_count
        
        # Coût estimé (€/kg)
        feed_cost_per_kg = self._get_feed_cost(espece)
        daily_cost = total_daily_consumption * feed_cost_per_kg
        
        # Projections sur l'horizon
        projections = []
        total_consumption = 0
        total_cost = 0
        
        for day in range(horizon_days):
            current_date = date.today() + timedelta(days=day)
            projections.append(FeedConsumption(
                date=current_date,
                amount_kg=round(total_daily_consumption, 2),
                cost_eur=round(daily_cost, 2)
            ))
            total_consumption += total_daily_consumption
            total_cost += daily_cost
        
        # Stock recommandé (30 jours de consommation)
        recommended_stock_kg = total_daily_consumption * 30
        
        # Date de rupture de stock estimée
        current_stock_kg = None  # À récupérer depuis la base
        days_to_depletion = None
        if current_stock_kg:
            days_to_depletion = int(current_stock_kg / total_daily_consumption)
        
        return {
            "daily_consumption_kg": round(total_daily_consumption, 2),
            "daily_cost_eur": round(daily_cost, 2),
            "monthly_consumption_kg": round(total_consumption, 0),
            "monthly_cost_eur": round(total_cost, 0),
            "yearly_consumption_kg": round(total_consumption * 12, 0),
            "yearly_cost_eur": round(total_cost * 12, 0),
            "recommended_stock_kg": round(recommended_stock_kg, 0),
            "projections": projections,
            "days_to_depletion": days_to_depletion,
            "feed_cost_per_kg": feed_cost_per_kg,
            "conversion_rate": await self.calculate_conversion_rate(espece, avg_weight_kg, total_daily_consumption)
        }
    
    async def predict_stock_alert(
        self,
        current_stock_kg: float,
        espece: str,
        animal_count: int,
        avg_weight_kg: float
    ) -> Dict[str, any]:
        """
        Prédire quand le stock d'aliment sera épuisé
        """
        consumption_data = await self.predict_feed_consumption(espece, animal_count, avg_weight_kg)
        daily_consumption = consumption_data["daily_consumption_kg"]
        
        if daily_consumption <= 0:
            return {
                "alert_level": "no_alert",
                "days_remaining": None,
                "recommended_order_date": None,
                "recommended_quantity_kg": None
            }
        
        days_remaining = int(current_stock_kg / daily_consumption)
        
        if days_remaining <= 3:
            alert_level = "critical"
            recommended_order_date = date.today()
        elif days_remaining <= 7:
            alert_level = "high"
            recommended_order_date = date.today()
        elif days_remaining <= 14:
            alert_level = "medium"
            recommended_order_date = date.today() + timedelta(days=days_remaining - 10)
        else:
            alert_level = "low"
            recommended_order_date = date.today() + timedelta(days=days_remaining - 14)
        
        recommended_quantity_kg = daily_consumption * 30  # Commande pour 30 jours
        
        return {
            "alert_level": alert_level,
            "days_remaining": days_remaining,
            "recommended_order_date": recommended_order_date,
            "recommended_quantity_kg": round(recommended_quantity_kg, 0),
            "daily_consumption_kg": round(daily_consumption, 2)
        }
    
    async def calculate_conversion_rate(
        self,
        espece: str,
        avg_weight_kg: float,
        daily_feed_kg: float
    ) -> Dict[str, any]:
        """
        Calculer le taux de conversion alimentaire (FCR)
        """
        # Gain de poids quotidien estimé (à améliorer avec données réelles)
        daily_gain_kg = self._estimate_daily_gain(espece, avg_weight_kg)
        
        if daily_gain_kg <= 0:
            fcr = None
        else:
            fcr = daily_feed_kg / daily_gain_kg
        
        # Évaluation
        if fcr:
            if espece == "bovin":
                if fcr < 6:
                    evaluation = "excellent"
                elif fcr < 8:
                    evaluation = "standard"
                else:
                    evaluation = "mauvais"
            elif espece == "avicole":
                if fcr < 1.8:
                    evaluation = "excellent"
                elif fcr < 2.2:
                    evaluation = "standard"
                else:
                    evaluation = "mauvais"
            else:
                evaluation = "standard" if fcr < 5 else "mauvais"
        else:
            evaluation = "inconnu"
        
        return {
            "fcr": round(fcr, 2) if fcr else None,
            "daily_gain_kg": round(daily_gain_kg, 2),
            "daily_feed_kg": round(daily_feed_kg, 2),
            "evaluation": evaluation,
            "recommendation": self._get_fcr_recommendation(evaluation, espece)
        }
    
    def _estimate_daily_gain(self, espece: str, weight_kg: float) -> float:
        """Estimer le gain de poids quotidien"""
        if espece == "bovin":
            if weight_kg < 200:
                return 0.8
            elif weight_kg < 400:
                return 1.2
            else:
                return 1.0
        elif espece == "ovin" or espece == "caprin":
            return 0.15
        elif espece == "avicole":
            return 0.05
        elif espece == "piscicole":
            return 0.002
        else:
            return 0.1
    
    def _get_seasonal_factor(self) -> float:
        """Facteur saisonnier pour la consommation"""
        from datetime import datetime
        month = datetime.now().month
        
        if month in [11, 12, 1, 2]:  # Hiver
            return 1.1
        elif month in [6, 7, 8]:  # Été
            return 0.9
        else:
            return 1.0
    
    def _get_feed_cost(self, espece: str) -> float:
        """Coût moyen de l'aliment par kg"""
        costs = {
            "bovin": 0.25,
            "ovin": 0.30,
            "caprin": 0.30,
            "avicole": 0.35,
            "piscicole": 0.80,
            "entomoculture": 0.50
        }
        return costs.get(espece, 0.40)
    
    def _get_fcr_recommendation(self, evaluation: str, espece: str) -> str:
        """Recommandation basée sur le FCR"""
        if evaluation == "excellent":
            return "Taux de conversion excellent - maintenir les pratiques actuelles"
        elif evaluation == "standard":
            return "Taux de conversion standard - peut être amélioré avec une meilleure ration"
        elif evaluation == "mauvais":
            return "Taux de conversion élevé - consulter un nutritionniste pour optimiser la ration"
        else:
            return "Collecter plus de données pour évaluer le taux de conversion"


feed_predictor = FeedPredictor()