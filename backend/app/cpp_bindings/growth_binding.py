# backend/app/cpp_bindings/growth_binding.py
"""
Liaison Python vers le module C++ de prédiction de croissance
Utilise pybind11 pour appeler les fonctions C++ optimisées
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math

from ..cpp_bindings import GROWTH_CPP_AVAILABLE

logger = logging.getLogger(__name__)

# Tentative d'import du module compilé
_growth_cpp = None
if GROWTH_CPP_AVAILABLE:
    try:
        # Ce module est généré par pybind11 à partir du C++
        # Nom: growth_cpp.so (Linux) ou growth_cpp.pyd (Windows)
        import growth_cpp as _growth_cpp
        logger.info("Successfully imported growth_cpp module")
    except ImportError as e:
        logger.warning(f"Could not import growth_cpp: {e}")
        GROWTH_CPP_AVAILABLE = False


@dataclass
class GrowthPredictionPoint:
    """Point de prédiction de croissance"""
    day: int
    weight_min: float
    weight_mean: float
    weight_max: float


class GrowthPredictorCpp:
    """
    Prédicteur de croissance utilisant le module C++.
    Implémente les modèles de croissance: Gompertz, Logistic, Von Bertalanffy
    """
    
    def __init__(self):
        self.available = GROWTH_CPP_AVAILABLE and _growth_cpp is not None
        
        if self.available:
            try:
                self.predictor = _growth_cpp.GrowthPredictor()
                logger.info("C++ GrowthPredictor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize C++ GrowthPredictor: {e}")
                self.available = False
    
    def predict_gompertz(
        self,
        weight_initial: float,
        age_initial_days: int,
        target_days: List[int],
        weight_inf: Optional[float] = None,
        growth_rate: Optional[float] = None
    ) -> List[GrowthPredictionPoint]:
        """
        Prédiction avec modèle de Gompertz
        
        Args:
            weight_initial: Poids initial (kg)
            age_initial_days: Âge initial (jours)
            target_days: Jours cibles pour la prédiction
            weight_inf: Poids asymptotique (optionnel)
            growth_rate: Taux de croissance (optionnel)
        
        Returns:
            Liste des points de prédiction
        """
        if self.available:
            try:
                results = self.predictor.predict_gompertz(
                    weight_initial, age_initial_days, target_days,
                    weight_inf or 0, growth_rate or 0
                )
                return [
                    GrowthPredictionPoint(
                        day=r[0], weight_min=r[1], weight_mean=r[2], weight_max=r[3]
                    ) for r in results
                ]
            except Exception as e:
                logger.error(f"C++ prediction failed: {e}, using Python fallback")
        
        # Fallback Python
        return self._predict_gompertz_python(
            weight_initial, age_initial_days, target_days,
            weight_inf, growth_rate
        )
    
    def predict_logistic(
        self,
        weight_initial: float,
        age_initial_days: int,
        target_days: List[int],
        carrying_capacity: Optional[float] = None,
        growth_rate: Optional[float] = None
    ) -> List[GrowthPredictionPoint]:
        """Prédiction avec modèle Logistique"""
        if self.available:
            try:
                results = self.predictor.predict_logistic(
                    weight_initial, age_initial_days, target_days,
                    carrying_capacity or 0, growth_rate or 0
                )
                return [
                    GrowthPredictionPoint(
                        day=r[0], weight_min=r[1], weight_mean=r[2], weight_max=r[3]
                    ) for r in results
                ]
            except Exception as e:
                logger.error(f"C++ logistic prediction failed: {e}")
        
        return self._predict_logistic_python(
            weight_initial, age_initial_days, target_days,
            carrying_capacity, growth_rate
        )
    
    def predict_von_bertalanffy(
        self,
        weight_initial: float,
        age_initial_days: int,
        target_days: List[int],
        asymptotic_weight: Optional[float] = None,
        metabolic_rate: Optional[float] = None
    ) -> List[GrowthPredictionPoint]:
        """Prédiction avec modèle de Von Bertalanffy (souvent utilisé pour les poissons)"""
        if self.available:
            try:
                results = self.predictor.predict_von_bertalanffy(
                    weight_initial, age_initial_days, target_days,
                    asymptotic_weight or 0, metabolic_rate or 0
                )
                return [
                    GrowthPredictionPoint(
                        day=r[0], weight_min=r[1], weight_mean=r[2], weight_max=r[3]
                    ) for r in results
                ]
            except Exception as e:
                logger.error(f"C++ von Bertalanffy prediction failed: {e}")
        
        return self._predict_von_bertalanffy_python(
            weight_initial, age_initial_days, target_days,
            asymptotic_weight, metabolic_rate
        )
    
    def estimate_parameters(
        self,
        ages_days: List[int],
        weights_kg: List[float],
        model_type: str = "gompertz"
    ) -> Dict[str, float]:
        """
        Estimer les paramètres du modèle à partir des données historiques
        
        Returns:
            Dictionnaire des paramètres estimés
        """
        if self.available and len(ages_days) >= 3:
            try:
                params = self.predictor.estimate_parameters(
                    ages_days, weights_kg, model_type
                )
                return {
                    "weight_inf": params[0],
                    "growth_rate": params[1],
                    "r_squared": params[2],
                }
            except Exception as e:
                logger.error(f"C++ parameter estimation failed: {e}")
        
        return self._estimate_parameters_python(ages_days, weights_kg, model_type)
    
    # ============ IMPLÉMENTATIONS PYTHON (FALLBACK) ============
    
    def _gompertz_function(self, t: float, A: float, mu: float, lambda_: float) -> float:
        """Fonction de Gompertz: w = A * exp(-exp(-mu * (t - lambda)))"""
        return A * math.exp(-math.exp(-mu * (t - lambda_)))
    
    def _predict_gompertz_python(
        self,
        weight_initial: float,
        age_initial_days: int,
        target_days: List[int],
        weight_inf: Optional[float],
        growth_rate: Optional[float]
    ) -> List[GrowthPredictionPoint]:
        """Implémentation Python de fallback pour Gompertz"""
        # Paramètres par défaut (bovin)
        A = weight_inf or 800.0  # Poids asymptotique (kg)
        mu = growth_rate or 0.01  # Taux de croissance
        lambda_ = age_initial_days + (1 / mu) * math.log(-math.log(weight_initial / A)) if weight_initial > 0 else age_initial_days
        
        results = []
        for day in target_days:
            if day <= age_initial_days:
                weight = weight_initial
            else:
                weight = self._gompertz_function(day, A, mu, lambda_)
            
            # Ajouter une incertitude de ±10%
            uncertainty = weight * 0.1
            results.append(GrowthPredictionPoint(
                day=day,
                weight_min=max(0, weight - uncertainty),
                weight_mean=weight,
                weight_max=weight + uncertainty
            ))
        
        return results
    
    def _predict_logistic_python(
        self,
        weight_initial: float,
        age_initial_days: int,
        target_days: List[int],
        carrying_capacity: Optional[float],
        growth_rate: Optional[float]
    ) -> List[GrowthPredictionPoint]:
        """Implémentation Python de fallback pour Logistique"""
        K = carrying_capacity or 800.0  # Capacité limite
        r = growth_rate or 0.02  # Taux de croissance
        
        # Calculer le point d'inflexion
        if weight_initial > 0 and weight_initial < K:
            t0 = age_initial_days - (1/r) * math.log((K - weight_initial) / weight_initial)
        else:
            t0 = age_initial_days
        
        results = []
        for day in target_days:
            if day <= age_initial_days:
                weight = weight_initial
            else:
                weight = K / (1 + math.exp(-r * (day - t0)))
            
            uncertainty = weight * 0.12
            results.append(GrowthPredictionPoint(
                day=day,
                weight_min=max(0, weight - uncertainty),
                weight_mean=weight,
                weight_max=weight + uncertainty
            ))
        
        return results
    
    def _predict_von_bertalanffy_python(
        self,
        weight_initial: float,
        age_initial_days: int,
        target_days: List[int],
        asymptotic_weight: Optional[float],
        metabolic_rate: Optional[float]
    ) -> List[GrowthPredictionPoint]:
        """Implémentation Python de fallback pour Von Bertalanffy (poissons)"""
        W_inf = asymptotic_weight or 500.0  # Poids asymptotique (g)
        k = metabolic_rate or 0.01  # Taux métabolique
        
        # Convertir les jours en années pour le modèle
        age_years = age_initial_days / 365.0
        
        # Calculer t0 (âge à poids théorique 0)
        if weight_initial > 0:
            t0 = age_years - (1/k) * math.log(1 - (weight_initial / W_inf)**(1/3))
        else:
            t0 = age_years
        
        results = []
        for day in target_days:
            t = day / 365.0
            if day <= age_initial_days:
                weight = weight_initial
            else:
                weight = W_inf * (1 - math.exp(-k * (t - t0)))**3
            
            uncertainty = weight * 0.08
            results.append(GrowthPredictionPoint(
                day=day,
                weight_min=max(0, weight - uncertainty),
                weight_mean=weight,
                weight_max=weight + uncertainty
            ))
        
        return results
    
    def _estimate_parameters_python(
        self,
        ages_days: List[int],
        weights_kg: List[float],
        model_type: str = "gompertz"
    ) -> Dict[str, float]:
        """Estimation simple des paramètres en Python"""
        if len(ages_days) < 3:
            return {"weight_inf": max(weights_kg) * 1.2, "growth_rate": 0.01, "r_squared": 0.0}
        
        # Estimation très simplifiée
        max_weight = max(weights_kg)
        weight_inf = max_weight * 1.2
        
        # Taux de croissance approximatif
        if len(weights_kg) > 1:
            growth_rate = (weights_kg[-1] - weights_kg[0]) / (ages_days[-1] - ages_days[0]) / max_weight
        else:
            growth_rate = 0.01
        
        # R² simplifié
        r_squared = 0.7 if len(ages_days) > 5 else 0.4
        
        return {"weight_inf": weight_inf, "growth_rate": growth_rate, "r_squared": r_squared}


# Instance globale
growth_predictor = GrowthPredictorCpp()