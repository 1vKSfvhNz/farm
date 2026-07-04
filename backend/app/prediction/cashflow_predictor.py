# backend/app/prediction/cashflow_predictor.py
"""
Prédiction de trésorerie et analyse financière
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DailyCashflow:
    """Trésorerie quotidienne"""
    date: date
    incoming: float
    outgoing: float
    balance: float
    cumulative: float


class CashflowPredictor:
    """
    Prédicteur de trésorerie
    Basé sur les dépenses récurrentes, les ventes prévues et les prédictions de production
    """
    
    # Dépenses fixes mensuelles par catégorie (€)
    FIXED_EXPENSES = {
        "personnel": 2000,
        "eau_electricite": 500,
        "entretien": 300,
        "assurances": 200
    }
    
    # Prix de vente moyens par catégorie (€/unité)
    SALE_PRICES = {
        "bovin_viande": 4.5,  # €/kg
        "bovin_lait": 0.45,   # €/L
        "ovin_viande": 6.0,   # €/kg
        "caprin_viande": 5.5, # €/kg
        "avicole_viande": 5.0, # €/kg
        "oeufs": 0.30,        # €/unité
        "piscicole": 5.0,     # €/kg
        "larves": 2.0         # €/kg
    }
    
    async def predict_cashflow(
        self,
        current_balance: float,
        expected_sales: List[Dict],
        expected_purchases: List[Dict],
        horizon_days: int = 90
    ) -> Dict[str, any]:
        """
        Prédire l'évolution de la trésorerie
        
        Args:
            current_balance: Solde actuel (€)
            expected_sales: Ventes prévues [{"date": date, "amount": float}]
            expected_purchases: Achats prévus [{"date": date, "amount": float}]
            horizon_days: Horizon de prédiction (jours)
        
        Returns:
            Prédictions de trésorerie
        """
        daily_cashflow = []
        
        # Créer des dictionnaires pour un accès rapide
        sales_by_date = {s["date"]: s["amount"] for s in expected_sales}
        purchases_by_date = {p["date"]: p["amount"] for p in expected_purchases}
        
        cumulative = current_balance
        min_balance = current_balance
        min_balance_date = date.today()
        
        for i in range(horizon_days):
            current_date = date.today() + timedelta(days=i)
            
            # Dépenses fixes quotidiennes
            daily_outgoing = sum(self.FIXED_EXPENSES.values()) / 30
            
            # Dépenses variables
            if current_date in purchases_by_date:
                daily_outgoing += purchases_by_date[current_date]
            
            # Recettes
            daily_incoming = 0
            if current_date in sales_by_date:
                daily_incoming += sales_by_date[current_date]
            
            # Mise à jour du solde
            cumulative += daily_incoming - daily_outgoing
            
            daily_cashflow.append(DailyCashflow(
                date=current_date,
                incoming=daily_incoming,
                outgoing=daily_outgoing,
                balance=cumulative - cumulative + daily_incoming - daily_outgoing,
                cumulative=cumulative
            ))
            
            if cumulative < min_balance:
                min_balance = cumulative
                min_balance_date = current_date
        
        # Déterminer le risque de trésorerie
        risk_level = self._assess_risk(current_balance, min_balance, cumulative)
        
        # Date de dépassement du seuil d'alerte (30 jours de charges fixes)
        monthly_charges = sum(self.FIXED_EXPENSES.values())
        warning_threshold = monthly_charges
        days_to_warning = self._days_to_threshold(daily_cashflow, warning_threshold)
        
        return {
            "daily_projections": daily_cashflow,
            "current_balance": current_balance,
            "projected_balance_30d": daily_cashflow[29].cumulative if len(daily_cashflow) > 29 else cumulative,
            "projected_balance_60d": daily_cashflow[59].cumulative if len(daily_cashflow) > 59 else cumulative,
            "projected_balance_90d": cumulative,
            "min_balance": round(min_balance, 2),
            "min_balance_date": min_balance_date,
            "risk_level": risk_level,
            "days_to_warning_threshold": days_to_warning,
            "recommendations": self._get_recommendations(risk_level, min_balance)
        }
    
    async def calculate_breakeven(
        self,
        fixed_costs: float,
        variable_cost_per_unit: float,
        selling_price_per_unit: float,
        estimated_demand: int
    ) -> Dict[str, any]:
        """
        Calculer le seuil de rentabilité pour une production
        
        Returns:
            Seuil de rentabilité en unités et en chiffre d'affaires
        """
        if selling_price_per_unit <= variable_cost_per_unit:
            return {
                "breakeven_units": None,
                "breakeven_revenue": None,
                "margin_per_unit": 0,
                "message": "Marge négative ou nulle - non rentable"
            }
        
        margin_per_unit = selling_price_per_unit - variable_cost_per_unit
        breakeven_units = fixed_costs / margin_per_unit
        breakeven_revenue = breakeven_units * selling_price_per_unit
        
        # Estimer la probabilité d'atteindre le seuil
        probability = min(breakeven_units / estimated_demand, 1.0) if estimated_demand > 0 else 0.5
        
        return {
            "breakeven_units": round(breakeven_units, 0),
            "breakeven_revenue": round(breakeven_revenue, 2),
            "margin_per_unit": round(margin_per_unit, 2),
            "probability_achievable": round(probability * 100, 1),
            "recommendation": self._get_breakeven_recommendation(breakeven_units, estimated_demand)
        }
    
    async def calculate_production_cost(
        self,
        feed_cost: float,
        veterinary_cost: float,
        labor_cost: float,
        energy_cost: float,
        other_costs: float,
        total_production_kg: float
    ) -> Dict[str, float]:
        """
        Calculer le coût de production par kg
        
        Returns:
            Dictionnaire avec les coûts détaillés
        """
        total_cost = feed_cost + veterinary_cost + labor_cost + energy_cost + other_costs
        
        if total_production_kg <= 0:
            return {
                "total_cost": total_cost,
                "cost_per_kg": None,
                "feed_cost_percent": 0,
                "veterinary_cost_percent": 0,
                "labor_cost_percent": 0,
                "message": "Production nulle - impossible de calculer le coût unitaire"
            }
        
        cost_per_kg = total_cost / total_production_kg
        
        return {
            "total_cost": round(total_cost, 2),
            "cost_per_kg": round(cost_per_kg, 2),
            "feed_cost_percent": round((feed_cost / total_cost) * 100, 1) if total_cost > 0 else 0,
            "veterinary_cost_percent": round((veterinary_cost / total_cost) * 100, 1) if total_cost > 0 else 0,
            "labor_cost_percent": round((labor_cost / total_cost) * 100, 1) if total_cost > 0 else 0,
            "energy_cost_percent": round((energy_cost / total_cost) * 100, 1) if total_cost > 0 else 0
        }
    
    def _assess_risk(self, current_balance: float, min_balance: float, projected_balance: float) -> str:
        """Évaluer le risque financier"""
        if min_balance < 0:
            return "critical"
        elif min_balance < current_balance * 0.3:
            return "high"
        elif projected_balance < current_balance * 0.5:
            return "medium"
        else:
            return "low"
    
    def _days_to_threshold(self, cashflow: List[DailyCashflow], threshold: float) -> Optional[int]:
        """Nombre de jours avant d'atteindre le seuil"""
        for i, day in enumerate(cashflow):
            if day.cumulative < threshold:
                return i
        return None
    
    def _get_recommendations(self, risk_level: str, min_balance: float) -> List[str]:
        """Générer des recommandations financières"""
        recommendations = {
            "critical": [
                "URGENT: Trésorerie négative imminente",
                "Contacter votre banquier immédiatement",
                "Réduire toutes les dépenses non essentielles",
                "Accélérer les recouvrements clients",
                "Reporter les investissements non urgents"
            ],
            "high": [
                "Surveillance rapprochée de la trésorerie",
                "Optimiser les délais de paiement fournisseurs",
                "Relancer les clients en retard",
                "Prévoir une ligne de crédit"
            ],
            "medium": [
                "Maintenir une réserve de trésorerie",
                "Planifier les dépenses importantes",
                "Suivre mensuellement les indicateurs"
            ],
            "low": [
                "Situation saine - continuer la gestion prudente",
                "Envisager des investissements de développement"
            ]
        }
        
        return recommendations.get(risk_level, recommendations["low"])
    
    def _get_breakeven_recommendation(self, breakeven_units: float, estimated_demand: int) -> str:
        """Recommandation basée sur le seuil de rentabilité"""
        if breakeven_units is None:
            return "Revoyez votre stratégie de prix - marge insuffisante"
        
        ratio = breakeven_units / estimated_demand if estimated_demand > 0 else 1.0
        
        if ratio > 1.5:
            return "Seuil de rentabilité difficile à atteindre - réduisez les coûts fixes"
        elif ratio > 1.0:
            return f"Besoin de vendre {int(breakeven_units - estimated_demand)} unités supplémentaires pour être rentable"
        elif ratio > 0.7:
            return "Seuil de rentabilité accessible - optimisez la production"
        else:
            return "Bonne rentabilité prévisionnelle - maintenez le cap"


cashflow_predictor = CashflowPredictor()       