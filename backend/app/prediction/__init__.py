# backend/app/prediction/__init__.py
"""
Moteur de prédictions pour la ferme connectée
Prédictions basées sur les données historiques et les références
"""

from .growth_predictor import GrowthPredictor, growth_predictor
from .production_predictor import ProductionPredictor, production_predictor
from .health_predictor import HealthPredictor, health_predictor
from .feed_predictor import FeedPredictor, feed_predictor
from .reproduction_predictor import ReproductionPredictor, reproduction_predictor
from .compost_predictor import CompostPredictor, compost_predictor
from .overcrowding_predictor import OvercrowdingPredictor, overcrowding_predictor
from .cashflow_predictor import CashflowPredictor, cashflow_predictor
from .water_quality_predictor import WaterQualityPredictor, water_quality_predictor
from .consequence_analyzer import ConsequenceAnalyzer, consequence_analyzer

__all__ = [
    "GrowthPredictor",
    "growth_predictor",
    "ProductionPredictor", 
    "production_predictor",
    "HealthPredictor",
    "health_predictor",
    "FeedPredictor",
    "feed_predictor",
    "ReproductionPredictor",
    "reproduction_predictor",
    "CompostPredictor",
    "compost_predictor",
    "OvercrowdingPredictor",
    "overcrowding_predictor",
    "CashflowPredictor",
    "cashflow_predictor",
    "WaterQualityPredictor",
    "water_quality_predictor",
    "ConsequenceAnalyzer",
    "consequence_analyzer",
]