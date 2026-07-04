# backend/app/services/prediction_service.py
"""
Service de prédictions - Point d'entrée unique pour toutes les prédictions
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import date, timedelta, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.animal import Animal
from ..models.enclos import Enclos
from ..models.compost import Compost
from ..schemas.predictions import PredictionRequest, PredictionResponse, GrowthPredictionResponse, ProductionPredictionResponse, CashflowPredictionResponse
from ..prediction.growth_predictor import growth_predictor
from ..prediction.production_predictor import production_predictor
from ..prediction.health_predictor import health_predictor
from ..prediction.feed_predictor import feed_predictor
from ..prediction.reproduction_predictor import reproduction_predictor
from ..prediction.compost_predictor import compost_predictor
from ..prediction.overcrowding_predictor import overcrowding_predictor
from ..prediction.cashflow_predictor import cashflow_predictor
from ..prediction.water_quality_predictor import water_quality_predictor
from ..prediction.consequence_analyzer import consequence_analyzer
from ..services.accounting_service import accounting_service

logger = logging.getLogger(__name__)


class PredictionService:
    """Service unifié pour toutes les prédictions"""
    
    async def make_prediction(
        self,
        db: AsyncSession,
        request: PredictionRequest
    ) -> PredictionResponse:
        """Point d'entrée unique pour les prédictions"""
        
        if request.prediction_type == "growth":
            return await self.predict_growth_by_params(db, request)
        elif request.prediction_type == "production":
            return await self.predict_production_by_params(db, request)
        elif request.prediction_type == "health":
            return await self.predict_health_by_params(db, request)
        elif request.prediction_type == "cashflow":
            return await self.predict_cashflow_by_params(db, request)
        elif request.prediction_type == "compost":
            return await self.predict_compost_by_params(db, request)
        elif request.prediction_type == "water_quality":
            return await self.predict_water_quality_by_params(db, request)
        else:
            raise ValueError(f"Type de prédiction inconnu: {request.prediction_type}")
    
    async def predict_growth(
        self,
        db: AsyncSession,
        animal_id: int,
        horizon_jours: int = 90
    ) -> Optional[GrowthPredictionResponse]:
        """Prédire la croissance d'un animal"""
        animal = await db.get(Animal, animal_id)
        if not animal:
            return None
        
        # Récupérer l'historique des pesées
        from ..models.pesee import Pesee
        stmt = select(Pesee).where(Pesee.animal_id == animal_id).order_by(Pesee.date_pesee)
        result = await db.execute(stmt)
        pesees = result.scalars().all()
        
        historique = [(p.date_pesee, p.poids) for p in pesees]
        
        # Prédiction
        predictions = await growth_predictor.predict_animal_growth(
            espece=animal.type_espece,
            race=animal.race,
            age_jours=animal.age_jours or 0,
            poids_actuel_kg=animal.dernier_poids or 0,
            pesees_historiques=historique,
            horizon_jours=horizon_jours
        )
        
        # Détection d'anomalie
        anomaly = await growth_predictor.detect_growth_anomaly(
            espece=animal.type_espece,
            race=animal.race,
            age_jours=animal.age_jours or 0,
            poids_kg=animal.dernier_poids or 0,
            pesees_historiques=historique
        )
        
        # Date d'atteinte du poids de vente estimé
        poids_vente = 500 if animal.type_espece == "bovin" else 50 if animal.type_espece in ["ovin", "caprin"] else 2
        date_atteinte = await growth_predictor.get_age_at_weight(
            espece=animal.type_espece,
            race=animal.race,
            poids_cible_kg=poids_vente,
            poids_actuel_kg=animal.dernier_poids or 0,
            age_jours=animal.age_jours or 0
        )
        
        return GrowthPredictionResponse(
            animal_id=animal_id,
            age_actuel_jours=animal.age_jours or 0,
            poids_actuel_kg=animal.dernier_poids or 0,
            poids_prevu_jours=[{"jour": p.day, "poids_min": p.weight_min, "poids_max": p.weight_max, "poids_moyen": p.weight_mean} for p in predictions],
            date_atteinte_poids_vente=date.today() + timedelta(days=date_atteinte - (animal.age_jours or 0)) if date_atteinte else None,
            retard_croissance_detecte=anomaly["anomaly_detected"],
            recommandations=anomaly["recommendations"]
        )
    
    async def predict_production(
        self,
        db: AsyncSession,
        espece: str,
        race: Optional[str],
        enclos_id: Optional[int],
        horizon_jours: int = 30
    ) -> ProductionPredictionResponse:
        """Prédire la production"""
        
        if espece == "bovin":
            # Production laitière
            prediction = await production_predictor.predict_milk_production(
                espece=espece,
                race=race or "default",
                lactation_day=60,  # À récupérer depuis la base
                daily_production=20  # À récupérer depuis la base
            )
            type_production = "lait"
        elif espece == "avicole":
            # Production d'œufs
            prediction = await production_predictor.predict_egg_production(
                race=race or "default",
                age_weeks=30,  # À récupérer depuis la base
                daily_production=0.8
            )
            type_production = "oeufs"
        elif espece == "entomoculture":
            # Production de larves
            prediction = await production_predictor.predict_larvae_production(
                espece=race or "hermetia",
                temperature=25,
                humidity=70
            )
            type_production = "larves"
        else:
            raise ValueError(f"Espèce non supportée pour la prédiction de production: {espece}")
        
        return ProductionPredictionResponse(
            espece=espece,
            type_production=type_production,
            production_quotidienne_actuelle=prediction.daily_estimate,
            production_prevue_15j=prediction.weekly_estimate * 2,
            production_prevue_30j=prediction.monthly_estimate,
            production_prevue_90j=prediction.monthly_estimate * 3,
            saisonnalite_impact=prediction.seasonal_factor,
            recommandations=[f"Tendance: {prediction.trend}"]
        )
    
    async def predict_cashflow(
        self,
        db: AsyncSession,
        horizon_jours: int = 90
    ) -> CashflowPredictionResponse:
        """Prédire l'évolution de la trésorerie"""
        
        # Obtenir le solde actuel
        summary = await accounting_service.get_summary(db)
        current_balance = summary.benefice  # Simplifié
        
        # Ventes et achats prévus
        expected_sales = await self._get_expected_sales(db, horizon_jours)
        expected_purchases = await self._get_expected_purchases(db, horizon_jours)
        
        # Prédiction
        prediction = await cashflow_predictor.predict_cashflow(
            current_balance=current_balance,
            expected_sales=expected_sales,
            expected_purchases=expected_purchases,
            horizon_days=horizon_jours
        )
        
        return CashflowPredictionResponse(
            tresorerie_actuelle=current_balance,
            entrees_prevues_30j=prediction["projected_balance_30d"] - current_balance if prediction["projected_balance_30d"] > current_balance else 0,
            sorties_prevues_30j=current_balance - prediction["projected_balance_30d"] if prediction["projected_balance_30d"] < current_balance else 0,
            tresorerie_prevue_30j=prediction["projected_balance_30d"],
            tresorerie_prevue_60j=prediction["projected_balance_60d"],
            tresorerie_prevue_90j=prediction["projected_balance_90d"],
            seuil_alerte_atteint=prediction["risk_level"] in ["high", "critical"],
            recommandations=prediction["recommendations"]
        )
    
    async def predict_health_risk(
        self,
        db: AsyncSession,
        espece: str,
        enclos_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Prédire les risques sanitaires"""
        
        # Récupérer l'historique de mortalité
        from ..models.mortalite import Mortalite
        from ..models.animal import Animal
        
        stmt = select(Mortalite).join(Animal).where(Animal.type_espece == espece)
        if enclos_id:
            stmt = stmt.where(Animal.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        mortalites = result.scalars().all()
        
        # Compter par mois
        mortality_rates = []
        # Simplifié: à compléter avec les données réelles
        
        risk = await health_predictor.predict_mortality_risk(
            espece=espece,
            enclos_id=enclos_id or 0,
            mortality_history=mortality_rates,
            environmental_factors={}
        )
        
        return risk
    
    async def predict_compost_maturity(
        self,
        db: AsyncSession,
        compost_id: int
    ) -> Optional[Dict[str, Any]]:
        """Prédire la date de maturité d'un compost"""
        
        compost = await db.get(Compost, compost_id)
        if not compost:
            return None
        
        # Récupérer les retournements
        from ..models.compost import RetournementCompost
        stmt = select(RetournementCompost).where(RetournementCompost.compost_id == compost_id)
        result = await db.execute(stmt)
        turnings = result.scalars().all()
        
        prediction = await compost_predictor.predict_maturity_date(
            compost_type=compost.type.value,
            start_date=compost.date_demarrage,
            temperature_readings=[(t.date_retournement, t.temperature_apres or 0) for t in turnings],
            humidity_readings=[(t.date_retournement, t.humidite_apres or 0) for t in turnings],
            turning_dates=[t.date_retournement for t in turnings],
            volume_m3=compost.volume_initial
        )
        
        return prediction
    
    async def predict_overcrowding(
        self,
        db: AsyncSession,
        enclos_id: int,
        horizon_jours: int = 90
    ) -> Dict[str, Any]:
        """Prédire la date de surpopulation d'un enclos"""
        
        enclos = await db.get(Enclos, enclos_id)
        if not enclos:
            return {"error": "Enclos non trouvé"}
        
        # Naissances prévues
        from ..models.animal import Animal
        from ..models.naissance import Naissance
        
        # Simplifié
        birth_forecast = []
        purchase_forecast = []
        sale_forecast = []
        death_forecast = []
        
        status = await overcrowding_predictor.predict_overcrowding(
            enclosure_id=enclos_id,
            current_animals=enclos.occupation_actuelle,
            max_capacity=enclos.capacite_maximale,
            birth_forecast=birth_forecast,
            purchase_forecast=purchase_forecast,
            sale_forecast=sale_forecast,
            death_forecast=death_forecast,
            horizon_days=horizon_jours
        )
        
        return {
            "enclos_id": enclos_id,
            "name": enclos.name,
            "current_animals": status.current_animals,
            "max_capacity": status.max_capacity,
            "occupancy_rate": status.occupancy_rate,
            "projected_days_to_full": status.projected_days_to_full,
            "risk_level": status.risk_level
        }
    
    async def predict_water_quality(
        self,
        db: AsyncSession,
        enclos_id: int,
        hours_ahead: int = 24
    ) -> Dict[str, Any]:
        """Prédire l'évolution de la qualité de l'eau"""
        
        # Récupérer la dernière mesure
        from ..models.water_quality import WaterQuality
        stmt = select(WaterQuality).where(WaterQuality.enclos_id == enclos_id).order_by(WaterQuality.timestamp.desc()).limit(1)
        result = await db.execute(stmt)
        last_measure = result.scalar_one_or_none()
        
        if not last_measure:
            return {"error": "Aucune mesure trouvée"}
        
        # Prédiction de l'oxygène
        oxygen_predictions = await water_quality_predictor.predict_oxygen_level(
            current_oxygen=last_measure.oxygene_dissous or 5.0,
            fish_biomass_kg=100,  # À récupérer
            water_temperature=last_measure.temperature or 20,
            feeding_rate_kg=10,  # À récupérer
            aeration_active=False,
            hours_ahead=hours_ahead
        )
        
        return {
            "oxygen_predictions": [{"hour": i, "value": p.predicted_value, "risk": p.risk_level} for i, p in enumerate(oxygen_predictions)],
            "current_measurement": {
                "ph": last_measure.ph,
                "temperature": last_measure.temperature,
                "oxygen": last_measure.oxygene_dissous,
                "ammonia": last_measure.ammoniac
            }
        }
    
    async def _get_expected_sales(self, db: AsyncSession, days: int) -> List[Dict]:
        """Récupérer les ventes prévues"""
        # À implémenter avec les données réelles
        return []
    
    async def _get_expected_purchases(self, db: AsyncSession, days: int) -> List[Dict]:
        """Récupérer les achats prévus"""
        # À implémenter avec les données réelles
        return []
    
    async def predict_growth_by_params(
        self,
        db: AsyncSession,
        request: PredictionRequest
    ) -> PredictionResponse:
        """Prédiction de croissance par paramètres"""
        # Implémentation simplifiée
        return PredictionResponse(
            prediction_id="test",
            espece=request.espece,
            prediction_type=request.prediction_type,
            horizon_jours=request.horizon_jours,
            predictions={},
            confidence=50,
            confidence_level="moyenne",
            warnings=[],
            generated_at=datetime.now()
        )
    
    async def predict_production_by_params(
        self,
        db: AsyncSession,
        request: PredictionRequest
    ) -> PredictionResponse:
        """Prédiction de production par paramètres"""
        return PredictionResponse(
            prediction_id="test",
            espece=request.espece,
            prediction_type=request.prediction_type,
            horizon_jours=request.horizon_jours,
            predictions={},
            confidence=50,
            confidence_level="moyenne",
            warnings=[],
            generated_at=datetime.now()
        )
    
    async def predict_health_by_params(
        self,
        db: AsyncSession,
        request: PredictionRequest
    ) -> PredictionResponse:
        """Prédiction sanitaire par paramètres"""
        return PredictionResponse(
            prediction_id="test",
            espece=request.espece,
            prediction_type=request.prediction_type,
            horizon_jours=request.horizon_jours,
            predictions={},
            confidence=50,
            confidence_level="moyenne",
            warnings=[],
            generated_at=datetime.now()
        )
    
    async def predict_cashflow_by_params(
        self,
        db: AsyncSession,
        request: PredictionRequest
    ) -> PredictionResponse:
        """Prédiction de trésorerie par paramètres"""
        return PredictionResponse(
            prediction_id="test",
            espece=request.espece,
            prediction_type=request.prediction_type,
            horizon_jours=request.horizon_jours,
            predictions={},
            confidence=50,
            confidence_level="moyenne",
            warnings=[],
            generated_at=datetime.now()
        )
    
    async def predict_compost_by_params(
        self,
        db: AsyncSession,
        request: PredictionRequest
    ) -> PredictionResponse:
        """Prédiction de compost par paramètres"""
        return PredictionResponse(
            prediction_id="test",
            espece=request.espece,
            prediction_type=request.prediction_type,
            horizon_jours=request.horizon_jours,
            predictions={},
            confidence=50,
            confidence_level="moyenne",
            warnings=[],
            generated_at=datetime.now()
        )
    
    async def predict_water_quality_by_params(
        self,
        db: AsyncSession,
        request: PredictionRequest
    ) -> PredictionResponse:
        """Prédiction de qualité d'eau par paramètres"""
        return PredictionResponse(
            prediction_id="test",
            espece=request.espece,
            prediction_type=request.prediction_type,
            horizon_jours=request.horizon_jours,
            predictions={},
            confidence=50,
            confidence_level="moyenne",
            warnings=[],
            generated_at=datetime.now()
        )


prediction_service = PredictionService()