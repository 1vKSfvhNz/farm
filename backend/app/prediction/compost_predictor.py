# backend/app/prediction/compost_predictor.py
"""
Prédiction de maturité du compost et suivi du processus
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompostPhase:
    """Phase du processus de compostage"""
    name: str
    start_day: int
    end_day: int
    temperature_range: Tuple[float, float]
    expected_humidity: float


class CompostPredictor:
    """
    Prédicteur de maturité du compost
    Basé sur la température, l'humidité, les retournements et le type de matière
    """
    
    # Phases typiques du compostage (en jours)
    COMPOST_PHASES = {
        "dechets_verts": [
            CompostPhase("Mésophile", 0, 3, (20, 40), 60),
            CompostPhase("Thermophile", 3, 21, (55, 65), 55),
            CompostPhase("Refroidissement", 21, 42, (40, 50), 50),
            CompostPhase("Maturation", 42, 90, (25, 35), 45)
        ],
        "fumier": [
            CompostPhase("Mésophile", 0, 5, (20, 40), 65),
            CompostPhase("Thermophile", 5, 35, (55, 65), 55),
            CompostPhase("Refroidissement", 35, 60, (40, 50), 50),
            CompostPhase("Maturation", 60, 120, (25, 35), 45)
        ],
        "mixte": [
            CompostPhase("Mésophile", 0, 4, (20, 40), 60),
            CompostPhase("Thermophile", 4, 28, (55, 65), 55),
            CompostPhase("Refroidissement", 28, 56, (40, 50), 50),
            CompostPhase("Maturation", 56, 105, (25, 35), 45)
        ]
    }
    
    async def predict_maturity_date(
        self,
        compost_type: str,
        start_date: date,
        temperature_readings: List[Tuple[date, float]],
        humidity_readings: List[Tuple[date, float]],
        turning_dates: List[date],
        volume_m3: float
    ) -> Dict[str, any]:
        """
        Prédire la date de maturité du compost
        
        Returns:
            Dictionnaire avec date estimée, phase actuelle, confiance
        """
        phases = self.COMPOST_PHASES.get(compost_type, self.COMPOST_PHASES["mixte"])
        
        # Déterminer la phase actuelle en fonction de la température
        current_temp = temperature_readings[-1][1] if temperature_readings else 30
        current_day = (date.today() - start_date).days
        
        current_phase = None
        for phase in phases:
            if phase.start_day <= current_day <= phase.end_day:
                current_phase = phase
                break
        
        if not current_phase:
            current_phase = phases[-1]  # Dernière phase par défaut
        
        # Ajuster la durée en fonction des conditions réelles
        adjustment_factor = self._calculate_adjustment_factor(
            temperature_readings,
            humidity_readings,
            turning_dates,
            volume_m3
        )
        
        # Estimer le jour de maturité
        total_days = phases[-1].end_day
        adjusted_total_days = int(total_days * adjustment_factor)
        
        maturity_date = start_date + timedelta(days=adjusted_total_days)
        
        # Niveau de confiance
        confidence = self._calculate_maturity_confidence(
            current_day,
            len(temperature_readings),
            len(turning_dates)
        )
        
        # Recommandations pour accélérer ou corriger
        recommendations = self._get_compost_recommendations(
            current_phase,
            temperature_readings[-1][1] if temperature_readings else 30,
            humidity_readings[-1][1] if humidity_readings else 55,
            len(turning_dates)
        )
        
        return {
            "estimated_maturity_date": maturity_date,
            "days_remaining": max(0, adjusted_total_days - current_day),
            "current_phase": current_phase.name,
            "phase_progress_percent": min(100, int((current_day - current_phase.start_day) / max(1, current_phase.end_day - current_phase.start_day) * 100)),
            "adjustment_factor": round(adjustment_factor, 2),
            "confidence_percent": round(confidence * 100, 1),
            "recommendations": recommendations,
            "temperature_optimal": current_phase.temperature_range[0] <= current_temp <= current_phase.temperature_range[1],
            "current_temperature": round(current_temp, 1),
            "target_temperature": current_phase.temperature_range
        }
    
    async def predict_final_volume(
        self,
        initial_volume_m3: float,
        compost_type: str,
        turning_count: int,
        duration_days: int
    ) -> Dict[str, float]:
        """
        Prédire le volume final de compost après maturation
        """
        # Facteurs de réduction de volume par type
        reduction_factors = {
            "dechets_verts": 0.6,  # Perte de 40% de volume
            "fumier": 0.7,         # Perte de 30%
            "mixte": 0.65          # Perte de 35%
        }
        
        base_factor = reduction_factors.get(compost_type, 0.65)
        
        # Plus de retournements = plus de perte
        turning_adjustment = 1.0 - (turning_count * 0.02)
        
        # Durée plus longue = plus de perte
        duration_adjustment = 1.0 - (duration_days / 365) * 0.1
        
        final_volume = initial_volume_m3 * base_factor * turning_adjustment * duration_adjustment
        
        return {
            "initial_volume_m3": initial_volume_m3,
            "predicted_final_volume_m3": round(max(initial_volume_m3 * 0.3, final_volume), 2),
            "volume_reduction_percent": round((1 - final_volume / initial_volume_m3) * 100, 1),
            "reduction_factor": round(base_factor, 2)
        }
    
    async def detect_compost_anomaly(
        self,
        temperature: float,
        humidity: float,
        phase: str,
        days_since_last_turn: int
    ) -> Dict[str, any]:
        """
        Détecter les anomalies dans le processus de compostage
        """
        anomalies = []
        
        # Vérifier la température
        if phase == "Thermophile" and temperature < 50:
            anomalies.append({
                "type": "temperature_low",
                "severity": "warning",
                "message": "Température trop basse pour la phase thermophile",
                "action": "Retourner le compost pour réactiver l'aération"
            })
        elif phase == "Thermophile" and temperature > 70:
            anomalies.append({
                "type": "temperature_high",
                "severity": "warning",
                "message": "Température excessive - risque de destruction des micro-organismes",
                "action": "Retourner d'urgence pour refroidir"
            })
        
        # Vérifier l'humidité
        if humidity < 40:
            anomalies.append({
                "type": "humidity_low",
                "severity": "warning",
                "message": "Compost trop sec - activité microbienne ralentie",
                "action": "Arroser légèrement lors du prochain retournement"
            })
        elif humidity > 70:
            anomalies.append({
                "type": "humidity_high",
                "severity": "warning",
                "message": "Compost trop humide - risque d'anaérobie",
                "action": "Retourner plus fréquemment pour assécher"
            })
        
        # Vérifier la fréquence des retournements
        if phase == "Thermophile" and days_since_last_turn > 14:
            anomalies.append({
                "type": "turning_delayed",
                "severity": "info",
                "message": f"Pas de retournement depuis {days_since_last_turn} jours",
                "action": "Programmer un retournement dans les 3 jours"
            })
        
        return {
            "anomalies": anomalies,
            "process_health": self._calculate_process_health(anomalies),
            "needs_intervention": len(anomalies) > 0
        }
    
    def _calculate_adjustment_factor(
        self,
        temperature_readings: List[Tuple[date, float]],
        humidity_readings: List[Tuple[date, float]],
        turning_dates: List[date],
        volume_m3: float
    ) -> float:
        """Calculer le facteur d'ajustement basé sur les conditions réelles"""
        factor = 1.0
        
        # Ajustement basé sur la température
        if temperature_readings:
            avg_temp = sum(t[1] for t in temperature_readings) / len(temperature_readings)
            if avg_temp < 45:
                factor *= 1.2  # Plus lent
            elif avg_temp > 65:
                factor *= 0.9  # Plus rapide
        
        # Ajustement basé sur l'humidité
        if humidity_readings:
            avg_humidity = sum(h[1] for h in humidity_readings) / len(humidity_readings)
            if avg_humidity < 45:
                factor *= 1.15
            elif avg_humidity > 65:
                factor *= 1.1
        
        # Ajustement basé sur les retournements
        turning_count = len(turning_dates)
        if turning_count > 4:
            factor *= 0.85  # Plus de retournements = plus rapide
        elif turning_count < 2:
            factor *= 1.2
        
        # Volume important = plus long
        if volume_m3 > 50:
            factor *= 1.1
        
        return min(max(factor, 0.7), 1.5)
    
    def _calculate_maturity_confidence(self, days: int, temp_readings: int, turnings: int) -> float:
        """Calculer la confiance dans la prédiction de maturité"""
        confidence = 0.5  # Base
        
        if days > 30:
            confidence += 0.2
        if temp_readings > 20:
            confidence += 0.15
        if turnings > 3:
            confidence += 0.15
        
        return min(confidence, 0.9)
    
    def _get_compost_recommendations(self, phase: CompostPhase, temp: float, humidity: float, turnings: int) -> List[str]:
        """Générer des recommandations pour optimiser le compostage"""
        recommendations = []
        
        if phase.name == "Thermophile":
            if temp < 50:
                recommendations.append("Augmenter l'aération - température trop basse")
            if humidity < 45:
                recommendations.append("Ajouter de l'eau lors du prochain retournement")
            if turnings < 2:
                recommendations.append("Programmer un retournement dans les 7 jours")
        
        elif phase.name == "Maturation":
            recommendations.append("Surveiller la température - le compost approche de la maturité")
            recommendations.append("Préparer l'espace de stockage pour le compost mûr")
        
        return recommendations
    
    def _calculate_process_health(self, anomalies: List[Dict]) -> int:
        """Calculer un score de santé du processus (0-100)"""
        base = 100
        penalty = sum(10 if a["severity"] == "warning" else 5 for a in anomalies)
        return max(0, base - penalty)


compost_predictor = CompostPredictor()