# backend/app/prediction/overcrowding_predictor.py
"""
Prédiction d'encombrement des enclos et recommandations
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnclosureStatus:
    """Statut d'un enclos"""
    enclosure_id: int
    name: str
    current_animals: int
    max_capacity: int
    occupancy_rate: float
    projected_days_to_full: Optional[int]
    risk_level: str


class OvercrowdingPredictor:
    """
    Prédicteur de surpopulation des enclos
    Anticipe les dépassements de capacité et suggère des actions
    """
    
    # Densités recommandées par espèce (m²/animal)
    RECOMMENDED_DENSITIES = {
        "bovin": 10.0,
        "ovin": 5.0,
        "caprin": 4.0,
        "avicole": 0.07,  # m²/poulet (15 poulets/m²)
        "piscicole": 0.033  # m³/kg (30 kg/m³)
    }
    
    async def predict_overcrowding(
        self,
        enclosure_id: int,
        current_animals: int,
        max_capacity: int,
        birth_forecast: List[date],
        purchase_forecast: List[date],
        sale_forecast: List[date],
        death_forecast: List[date],
        horizon_days: int = 90
    ) -> EnclosureStatus:
        """
        Prédire quand un enclos deviendra surpeuplé
        """
        # Créer un calendrier des entrées/sorties
        daily_count = current_animals
        days_to_full = None
        risk_level = "normal"
        
        for day in range(horizon_days):
            current_date = date.today() + timedelta(days=day)
            
            # Naissances
            daily_count += sum(1 for d in birth_forecast if d == current_date)
            
            # Achats
            daily_count += sum(1 for d in purchase_forecast if d == current_date)
            
            # Ventes
            daily_count -= sum(1 for d in sale_forecast if d == current_date)
            
            # Décès
            daily_count -= sum(1 for d in death_forecast if d == current_date)
            
            # Vérifier le dépassement
            if daily_count >= max_capacity and days_to_full is None:
                days_to_full = day
                if day <= 7:
                    risk_level = "critical"
                elif day <= 30:
                    risk_level = "high"
                elif day <= 60:
                    risk_level = "medium"
                else:
                    risk_level = "low"
        
        occupancy_rate = (current_animals / max_capacity) * 100 if max_capacity > 0 else 0
        
        return EnclosureStatus(
            enclosure_id=enclosure_id,
            name="",
            current_animals=current_animals,
            max_capacity=max_capacity,
            occupancy_rate=round(occupancy_rate, 1),
            projected_days_to_full=days_to_full,
            risk_level=risk_level
        )
    
    async def suggest_actions(
        self,
        status: EnclosureStatus,
        available_enclosures: List[Dict]
    ) -> List[Dict[str, any]]:
        """
        Suggérer des actions pour résoudre la surpopulation
        """
        actions = []
        
        if status.risk_level in ["high", "critical"]:
            # Action 1: Vente anticipée
            actions.append({
                "type": "early_sale",
                "priority": "high" if status.risk_level == "critical" else "medium",
                "description": f"Vendre {int(status.current_animals * 0.2)} animaux dans les {status.projected_days_to_full} jours",
                "estimated_revenue": None  # Serait calculé avec les prix
            })
            
            # Action 2: Transfert vers d'autres enclos
            for enclosure in available_enclosures:
                if enclosure["available_capacity"] > 0:
                    actions.append({
                        "type": "transfer",
                        "priority": "medium",
                        "description": f"Transférer {min(int(status.current_animals * 0.3), enclosure['available_capacity'])} animaux vers {enclosure['name']}",
                        "target_enclosure": enclosure["id"]
                    })
                    break
            
            # Action 3: Agrandissement
            actions.append({
                "type": "expansion",
                "priority": "low",
                "description": "Planifier un agrandissement de l'enclos ou construire un nouvel enclos",
                "estimated_cost": None
            })
        
        return actions
    
    async def calculate_optimal_density(
        self,
        espece: str,
        enclosure_area_m2: float,
        animal_avg_weight_kg: float
    ) -> Dict[str, any]:
        """
        Calculer la densité optimale basée sur l'espèce et le poids
        """
        recommended_density = self.RECOMMENDED_DENSITIES.get(espece, 5.0)
        
        # Ajustement pour les jeunes animaux (plus petite surface)
        if animal_avg_weight_kg < 50 and espece in ["bovin", "ovin", "caprin"]:
            recommended_density *= 0.5
        # Ajustement pour les gros animaux
        elif animal_avg_weight_kg > 500 and espece == "bovin":
            recommended_density *= 1.5
        
        optimal_animals = int(enclosure_area_m2 / recommended_density) if recommended_density > 0 else 0
        
        return {
            "espece": espece,
            "enclosure_area_m2": enclosure_area_m2,
            "recommended_density_m2_per_animal": recommended_density,
            "optimal_animal_count": optimal_animals,
            "current_animal_count": None,  # À remplir par l'appelant
            "occupancy_rate": None
        }


overcrowding_predictor = OvercrowdingPredictor()