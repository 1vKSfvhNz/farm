# backend/app/prediction/reproduction_predictor.py
"""
Prédiction des mises bas et du taux de fertilité
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BirthPrediction:
    """Prédiction de mise bas"""
    animal_id: int
    identification: str
    estimated_birth_date: date
    confidence: float
    expected_offspring: int
    risk_factors: List[str]


class ReproductionPredictor:
    """
    Prédicteur de reproduction
    Gestation, mises bas, fertilité
    """
    
    # Durée de gestation par espèce (jours)
    GESTATION_LENGTH = {
        "bovin": 283,
        "ovin": 150,
        "caprin": 150,
        "piscicole": 30  # Variable selon espèce
    }
    
    # Taux de fertilité par défaut
    DEFAULT_FERTILITY_RATES = {
        "bovin": 0.65,
        "ovin": 0.80,
        "caprin": 0.75,
        "avicole": 0.85
    }
    
    async def predict_birth_date(
        self,
        espece: str,
        insemination_date: date,
        previous_gestations: Optional[List[Dict]] = None
    ) -> BirthPrediction:
        """
        Prédire la date de mise bas
        """
        gestation_days = self.GESTATION_LENGTH.get(espece, 280)
        
        # Ajustement basé sur l'historique
        adjustment = 0
        if previous_gestations and len(previous_gestations) >= 2:
            avg_duration = sum(g["duration_days"] for g in previous_gestations) / len(previous_gestations)
            adjustment = avg_duration - gestation_days
        
        estimated_birth_date = insemination_date + timedelta(days=int(gestation_days + adjustment))
        
        # Calcul de confiance
        confidence = 0.7
        if previous_gestations:
            confidence += min(0.2, len(previous_gestations) * 0.05)
        
        # Facteurs de risque
        risk_factors = self._assess_gestation_risks(espece, previous_gestations)
        
        return BirthPrediction(
            animal_id=0,  # À remplir
            identification="",
            estimated_birth_date=estimated_birth_date,
            confidence=round(confidence, 2),
            expected_offspring=self._get_expected_offspring(espece),
            risk_factors=risk_factors
        )
    
    async def predict_fertility_rate(
        self,
        espece: str,
        insemination_history: List[Dict],
        animal_age_months: int,
        health_status: str
    ) -> Dict[str, any]:
        """
        Prédire le taux de fertilité
        """
        base_rate = self.DEFAULT_FERTILITY_RATES.get(espece, 0.7)
        
        # Ajustement par âge
        if espece == "bovin":
            if animal_age_months < 18:
                age_factor = 0.8
            elif animal_age_months > 120:
                age_factor = 0.7
            else:
                age_factor = 1.0
        elif espece in ["ovin", "caprin"]:
            if animal_age_months < 12:
                age_factor = 0.7
            elif animal_age_months > 96:
                age_factor = 0.6
            else:
                age_factor = 1.0
        else:
            age_factor = 1.0
        
        # Ajustement par historique
        if insemination_history:
            success_count = sum(1 for i in insemination_history if i["success"])
            history_rate = success_count / len(insemination_history) if insemination_history else base_rate
            history_factor = (history_rate + base_rate) / 2 / base_rate
        else:
            history_factor = 1.0
        
        # Ajustement par santé
        health_factors = {
            "excellent": 1.1,
            "good": 1.0,
            "fair": 0.8,
            "poor": 0.5
        }
        health_factor = health_factors.get(health_status, 1.0)
        
        predicted_rate = base_rate * age_factor * history_factor * health_factor
        predicted_rate = min(max(predicted_rate, 0.1), 0.95)
        
        # Nombre d'inséminations estimées pour une gestation
        estimated_inseminations = 1 / predicted_rate if predicted_rate > 0 else 10
        
        return {
            "predicted_fertility_rate": round(predicted_rate * 100, 1),
            "estimated_inseminations_per_gestation": round(estimated_inseminations, 1),
            "base_rate": round(base_rate * 100, 1),
            "age_factor": round(age_factor, 2),
            "history_factor": round(history_factor, 2),
            "health_factor": round(health_factor, 2),
            "recommendation": self._get_fertility_recommendation(predicted_rate, espece)
        }
    
    async def detect_abortion_risk(
        self,
        espece: str,
        gestation_week: int,
        health_history: List[Dict],
        environmental_stress: bool
    ) -> Dict[str, any]:
        """
        Détecter le risque d'avortement
        """
        base_risk = 0.05  # 5% de risque de base
        
        # Facteurs de risque
        risk_factors = []
        
        # Stress environnemental
        if environmental_stress:
            base_risk += 0.1
            risk_factors.append("stress_environnemental")
        
        # Santé antérieure
        for event in health_history:
            if event["type"] == "maladie" and (date.today() - event["date"]).days < 90:
                base_risk += 0.15
                risk_factors.append("maladie_recente")
        
        # Stade de gestation (plus de risques en début et fin)
        if gestation_week < 8:
            base_risk += 0.05
        elif gestation_week > 30 and espece == "bovin":
            base_risk += 0.08
        
        total_risk = min(base_risk, 0.5)
        
        # Niveau de risque
        if total_risk > 0.3:
            risk_level = "critical"
            recommendation = "URGENT: Consultation vétérinaire immédiate"
        elif total_risk > 0.15:
            risk_level = "high"
            recommendation = "Surveillance rapprochée, éviter tout stress"
        elif total_risk > 0.08:
            risk_level = "medium"
            recommendation = "Surveillance normale, suivre l'alimentation"
        else:
            risk_level = "low"
            recommendation = "Gestation normale, poursuivre le suivi"
        
        return {
            "abortion_risk_percent": round(total_risk * 100, 1),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "recommended_monitoring_frequency": self._get_monitoring_frequency(risk_level)
        }
    
    def _assess_gestation_risks(self, espece: str, previous_gestations: Optional[List[Dict]]) -> List[str]:
        """Évaluer les facteurs de risque pour la gestation"""
        risks = []
        
        if previous_gestations:
            # Vérifier les antécédents d'avortement
            abortions = [g for g in previous_gestations if g.get("abortion", False)]
            if abortions:
                risks.append(f"Antécédent d'avortement ({len(abortions)} fois)")
            
            # Vérifier les gestations difficiles
            difficulties = [g for g in previous_gestations if g.get("difficulty", False)]
            if difficulties:
                risks.append("Antécédent de mise bas difficile")
        
        return risks
    
    def _get_expected_offspring(self, espece: str) -> int:
        """Nombre estimé de petits par portée"""
        if espece == "bovin":
            return 1
        elif espece in ["ovin", "caprin"]:
            return 2  # Moyenne
        elif espece == "piscicole":
            return 1000  # Variable selon espèce
        else:
            return 1
    
    def _get_fertility_recommendation(self, rate: float, espece: str) -> str:
        """Recommandation basée sur le taux de fertilité"""
        if rate >= 0.7:
            return "Fertilité excellente - maintenir les pratiques"
        elif rate >= 0.5:
            return "Fertilité moyenne - vérifier l'alimentation et la santé"
        else:
            return "Fertilité faible - consulter un vétérinaire spécialiste en reproduction"
    
    def _get_monitoring_frequency(self, risk_level: str) -> str:
        """Fréquence de surveillance recommandée"""
        frequencies = {
            "critical": "quotidienne",
            "high": "hebdomadaire",
            "medium": "bimensuelle",
            "low": "mensuelle"
        }
        return frequencies.get(risk_level, "mensuelle")


reproduction_predictor = ReproductionPredictor()