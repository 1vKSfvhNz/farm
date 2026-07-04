# backend/app/prediction/consequence_analyzer.py
"""
Analyse des conséquences des actions (scénarios "si... alors...")
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, timedelta

logger = logging.getLogger(__name__)


class ConsequenceAnalyzer:
    """
    Analyseur de conséquences pour les scénarios "si... alors..."
    Permet d'évaluer l'impact des décisions avant de les prendre
    """
    
    async def analyze_vaccination_delay(
        self,
        espece: str,
        delay_days: int,
        population_size: int,
        current_vaccination_rate: float
    ) -> Dict[str, any]:
        """
        Analyser les conséquences d'un retard de vaccination
        """
        # Risque épidémique accru
        risk_increase = min(0.5, delay_days / 30 * 0.3)
        epidemic_risk = min(0.8, current_vaccination_rate * risk_increase + 0.1)
        
        # Perte économique estimée (par animal)
        loss_per_animal = self._estimate_vaccination_loss(espece, epidemic_risk)
        total_loss = loss_per_animal * population_size
        
        # Mortalité estimée
        mortality_rate = min(0.3, epidemic_risk * 0.5)
        estimated_deaths = int(population_size * mortality_rate)
        
        # Recommandations
        recommendations = []
        if delay_days > 30:
            recommendations.append("URGENT: Vacciner dans les 48 heures")
        elif delay_days > 14:
            recommendations.append("Planifier la vaccination sous 7 jours")
        
        if epidemic_risk > 0.3:
            recommendations.append("Mettre en place une quarantaine préventive")
            recommendations.append("Surveiller les signes cliniques quotidiennement")
        
        return {
            "epidemic_risk_percent": round(epidemic_risk * 100, 1),
            "estimated_mortality_rate_percent": round(mortality_rate * 100, 1),
            "estimated_deaths": estimated_deaths,
            "estimated_economic_loss_eur": round(total_loss, 0),
            "loss_per_animal_eur": round(loss_per_animal, 2),
            "severity": "critical" if epidemic_risk > 0.5 else "high" if epidemic_risk > 0.2 else "medium",
            "recommendations": recommendations,
            "quarantine_recommended": epidemic_risk > 0.3
        }
    
    async def analyze_cleaning_skip(
        self,
        enclosure_name: str,
        days_since_last_cleaning: int,
        animal_count: int
    ) -> Dict[str, any]:
        """
        Analyser les conséquences d'un nettoyage non effectué
        """
        # Risque parasitaire
        parasite_risk = min(0.8, days_since_last_cleaning / 7 * 0.2)
        
        # Impact sur le taux de conversion alimentaire
        fcr_impact = 1.0 + (days_since_last_cleaning / 30 * 0.1)
        
        # Perte de production estimée
        production_loss_percent = min(0.15, days_since_last_cleaning / 60 * 0.1)
        
        recommendations = []
        if days_since_last_cleaning > 14:
            recommendations.append("Nettoyage urgent requis")
            recommendations.append("Programmer une désinfection après nettoyage")
        elif days_since_last_cleaning > 7:
            recommendations.append("Nettoyage à effectuer dans les 48 heures")
        
        return {
            "parasite_risk_percent": round(parasite_risk * 100, 1),
            "fcr_impact_factor": round(fcr_impact, 2),
            "estimated_production_loss_percent": round(production_loss_percent * 100, 1),
            "severity": "critical" if parasite_risk > 0.5 else "high" if parasite_risk > 0.3 else "medium",
            "recommendations": recommendations,
            "veterinary_intervention_recommended": parasite_risk > 0.4
        }
    
    async def analyze_overfeeding(
        self,
        espece: str,
        excess_percent: float,
        duration_days: int,
        animal_count: int
    ) -> Dict[str, any]:
        """
        Analyser les conséquences d'une suralimentation
        """
        # Problèmes de santé attendus
        health_issues = []
        
        if espece == "bovin":
            if excess_percent > 20:
                health_issues.append("Risque d'acidose ruminale")
                health_issues.append("Risque de fourbure")
            if duration_days > 30:
                health_issues.append("Stéatose hépatique")
        
        # Coût supplémentaire
        extra_feed_cost = 0  # À calculer avec les prix réels
        
        # Perte de performance de reproduction
        reproduction_impact = min(0.4, excess_percent / 100 * duration_days / 60)
        
        recommendations = [
            "Réduire immédiatement la ration pour revenir aux besoins réels",
            "Consulter un nutritionniste pour rééquilibrer la ration"
        ]
        
        return {
            "excess_percent": excess_percent,
            "duration_days": duration_days,
            "health_risks": health_issues,
            "reproduction_impact_percent": round(reproduction_impact * 100, 1),
            "severity": "critical" if excess_percent > 30 else "high" if excess_percent > 15 else "medium",
            "recommendations": recommendations
        }
    
    async def analyze_underfeeding(
        self,
        espece: str,
        deficit_percent: float,
        duration_days: int,
        current_weight_kg: float
    ) -> Dict[str, any]:
        """
        Analyser les conséquences d'une sous-alimentation
        """
        # Perte de poids estimée
        daily_weight_loss = current_weight_kg * (deficit_percent / 100) * 0.01
        total_weight_loss = daily_weight_loss * duration_days
        
        # Retard de croissance
        growth_delay_days = int(duration_days * (deficit_percent / 100) * 1.5)
        
        # Mortalité potentielle
        mortality_risk = min(0.3, deficit_percent / 100 * duration_days / 30 * 0.1)
        
        recommendations = [
            f"Ajouter {deficit_percent}% de nourriture immédiatement",
            "Vérifier la qualité de l'aliment"
        ]
        
        if mortality_risk > 0.1:
            recommendations.append("Surveillance vétérinaire recommandée")
        
        return {
            "estimated_weight_loss_kg": round(total_weight_loss, 1),
            "growth_delay_days": growth_delay_days,
            "mortality_risk_percent": round(mortality_risk * 100, 1),
            "severity": "critical" if mortality_risk > 0.2 else "high" if mortality_risk > 0.1 else "medium",
            "recommendations": recommendations
        }
    
    async def analyze_overcrowding(
        self,
        current_count: int,
        max_capacity: int,
        espece: str,
        duration_weeks: int
    ) -> Dict[str, any]:
        """
        Analyser les conséquences d'une surpopulation
        """
        overcrowding_ratio = current_count / max_capacity if max_capacity > 0 else 1.0
        
        # Impact sur la croissance
        growth_impact = min(0.3, (overcrowding_ratio - 1) * 0.5) if overcrowding_ratio > 1 else 0
        growth_loss_percent = growth_impact * 100
        
        # Impact sur la mortalité
        mortality_increase = min(0.15, (overcrowding_ratio - 1) * 0.2) if overcrowding_ratio > 1 else 0
        
        # Recommandations
        recommendations = []
        if overcrowding_ratio > 1.2:
            recommendations.append("Transfert d'animaux vers un autre enclos sous 7 jours")
            recommendations.append("Envisager une vente anticipée")
        elif overcrowding_ratio > 1.0:
            recommendations.append("Surveiller le comportement et la santé")
            recommendations.append("Planifier un transfert dans 30 jours")
        
        return {
            "overcrowding_ratio": round(overcrowding_ratio, 2),
            "growth_impact_percent": round(growth_loss_percent, 1),
            "mortality_increase_percent": round(mortality_increase * 100, 1),
            "critical_date": date.today() + timedelta(weeks=duration_weeks) if overcrowding_ratio > 1.1 else None,
            "severity": "critical" if overcrowding_ratio > 1.3 else "high" if overcrowding_ratio > 1.1 else "medium" if overcrowding_ratio > 1.0 else "low",
            "recommendations": recommendations
        }
    
    async def analyze_weather_impact(
        self,
        forecast: Dict[str, any],
        espece: str,
        has_shelter: bool
    ) -> Dict[str, any]:
        """
        Analyser l'impact des conditions météorologiques
        """
        temperature = forecast.get("temperature", 20)
        rain_mm = forecast.get("rain_mm", 0)
        wind_kmh = forecast.get("wind_kmh", 10)
        
        impacts = []
        recommendations = []
        
        # Impact de la chaleur
        if espece == "bovin" and temperature > 28:
            impacts.append("Stress thermique - baisse de production laitière")
            recommendations.append("Fournir de l'ombre et de l'eau fraîche")
            recommendations.append("Aérer les bâtiments")
        
        # Impact du froid
        if temperature < 5:
            impacts.append("Augmentation des besoins énergétiques")
            recommendations.append("Augmenter la ration alimentaire de 10-15%")
            if not has_shelter:
                recommendations.append("Mettre à disposition un abri")
        
        # Impact de la pluie
        if rain_mm > 20:
            impacts.append("Risque de boue - augmentation des maladies podales")
            recommendations.append("Surveiller l'état des sabots/onglons")
        
        # Impact du vent
        if wind_kmh > 50:
            impacts.append("Stress dû au vent - isolement recommandé")
            recommendations.append("Protéger les animaux du vent")
        
        return {
            "impacts": impacts,
            "recommendations": recommendations,
            "severity": "high" if len(impacts) >= 2 else "medium" if len(impacts) == 1 else "low",
            "feed_adjustment_percent": 15 if temperature < 5 else 0,
            "water_requirement_increase": 50 if temperature > 30 else 0
        }
    
    async def analyze_purchase_impact(
        self,
        new_animals_count: int,
        espece: str,
        current_population: int,
        current_enclosure_capacity: int,
        feed_stock_kg: float,
        daily_feed_kg: float
    ) -> Dict[str, any]:
        """
        Analyser l'impact d'un achat massif d'animaux
        """
        # Impact sur l'encombrement
        new_total = current_population + new_animals_count
        overcrowding = new_total > current_enclosure_capacity
        
        # Impact sur le stock alimentaire
        new_daily_feed = daily_feed_kg * (new_total / current_population) if current_population > 0 else daily_feed_kg * 2
        days_of_feed_remaining = feed_stock_kg / new_daily_feed if new_daily_feed > 0 else 0
        
        # Impact financier
        additional_costs = new_animals_count * self._get_daily_cost_per_animal(espece) * 30  # Coût sur 30 jours
        
        return {
            "new_total_animals": new_total,
            "overcrowding": overcrowding,
            "overcrowding_percent": round((new_total / current_enclosure_capacity) * 100, 1) if current_enclosure_capacity > 0 else 100,
            "days_of_feed_remaining": round(days_of_feed_remaining, 0),
            "additional_monthly_cost_eur": round(additional_costs, 0),
            "feed_alert": days_of_feed_remaining < 30,
            "recommendations": self._get_purchase_recommendations(overcrowding, days_of_feed_remaining)
        }
    
    def _estimate_vaccination_loss(self, espece: str, epidemic_risk: float) -> float:
        """Estimer la perte économique par animal due à un retard de vaccination"""
        base_values = {
            "bovin": 1500,
            "ovin": 200,
            "caprin": 250,
            "avicole": 5,
            "piscicole": 10
        }
        base_value = base_values.get(espece, 100)
        return base_value * epidemic_risk
    
    def _get_daily_cost_per_animal(self, espece: str) -> float:
        """Coût quotidien par animal"""
        costs = {
            "bovin": 3.0,
            "ovin": 0.8,
            "caprin": 0.8,
            "avicole": 0.05,
            "piscicole": 0.02
        }
        return costs.get(espece, 1.0)
    
    def _get_purchase_recommendations(self, overcrowding: bool, days_of_feed: float) -> List[str]:
        """Recommandations pour un achat d'animaux"""
        recommendations = []
        
        if overcrowding:
            recommendations.append("ATTENTION: Risque de surpopulation - agrandir ou vendre avant l'achat")
        
        if days_of_feed < 14:
            recommendations.append("URGENT: Commander de l'aliment avant l'arrivée des animaux")
        elif days_of_feed < 30:
            recommendations.append("Prévoir une commande d'aliment supplémentaire")
        
        if not recommendations:
            recommendations.append("Achat réalisable dans les conditions actuelles")
        
        return recommendations


consequence_analyzer = ConsequenceAnalyzer()