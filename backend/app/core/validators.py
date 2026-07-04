# backend/app/core/validators.py
from datetime import date, timedelta
from typing import Tuple, Optional, List
from ..models.enclos import Enclos


def validate_date_range(
    start_date: date,
    end_date: date,
    max_days: int = 365
) -> Tuple[bool, Optional[str]]:
    """Valider qu'une plage de dates est valide"""
    if start_date > end_date:
        return False, "La date de début doit être antérieure à la date de fin"
    
    if (end_date - start_date).days > max_days:
        return False, f"La plage de dates ne peut pas dépasser {max_days} jours"
    
    if start_date > date.today():
        return False, "La date de début ne peut pas être dans le futur"
    
    return True, None


def validate_weight_range(
    weight: float,
    espece: str,
    age_jours: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """Valider qu'un poids est plausible pour une espèce"""
    # Poids minimums et maximums par espèce (kg)
    limits = {
        "bovin": {"min": 0.5, "max": 1500},
        "ovin": {"min": 0.5, "max": 200},
        "caprin": {"min": 0.5, "max": 150},
        "avicole": {"min": 0.02, "max": 10},
        "piscicole": {"min": 0.001, "max": 100},
    }
    
    esp_limits = limits.get(espece, {"min": 0, "max": 10000})
    
    if weight < esp_limits["min"]:
        return False, f"Poids trop faible pour un {espece} (minimum: {esp_limits['min']} kg)"
    
    if weight > esp_limits["max"]:
        return False, f"Poids trop élevé pour un {espece} (maximum: {esp_limits['max']} kg)"
    
    return True, None


def validate_pesee_frequency(
    last_pesee_date: date,
    current_date: date,
    min_days: int = 3
) -> Tuple[bool, Optional[str]]:
    """Valider la fréquence des pesées"""
    days_diff = (current_date - last_pesee_date).days
    
    if days_diff < min_days:
        return False, f"Pesée trop fréquente (minimum {min_days} jours entre pesées)"
    
    return True, None



def validate_animal_age(date_naissance: date) -> Tuple[bool, Optional[str]]:
    """Valider qu'un âge animal est plausible"""
    if date_naissance > date.today():
        return False, "La date de naissance ne peut pas être dans le futur"
    
    age_days = (date.today() - date_naissance).days
    
    if age_days < 0:
        return False, "Âge négatif impossible"
    
    # Maximum 30 ans pour les bovins
    if age_days > 30 * 365:
        return False, "Âge trop élevé (maximum 30 ans)"
    
    return True, None


def validate_water_quality(
    ph: Optional[float] = None,
    temperature: Optional[float] = None,
    oxygene_dissous: Optional[float] = None,
    espece: str = "piscicole"
) -> List[str]:
    """Valider les paramètres de qualité d'eau"""
    warnings = []
    
    # Seuils pour pisciculture
    if espece == "piscicole":
        if ph is not None:
            if ph < 6.0:
                warnings.append(f"pH trop bas: {ph} (minimum recommandé: 6.0)")
            elif ph > 9.0:
                warnings.append(f"pH trop élevé: {ph} (maximum recommandé: 9.0)")
        
        if temperature is not None:
            if temperature < 4:
                warnings.append(f"Température trop basse: {temperature}°C")
            elif temperature > 36:
                warnings.append(f"Température trop élevée: {temperature}°C")
        
        if oxygene_dissous is not None:
            if oxygene_dissous < 3.0:
                warnings.append(f"Oxygène dissous critique: {oxygene_dissous} mg/L (minimum: 3.0)")
            elif oxygene_dissous < 5.0:
                warnings.append(f"Oxygène dissous bas: {oxygene_dissous} mg/L (optimal: >5.0)")
    
    return warnings


def validate_vaccination_date(
    date_prevue: date,
    date_realisee: Optional[date] = None
) -> Tuple[bool, Optional[str]]:
    """Valider les dates de vaccination"""
    if date_prevue < date.today() - timedelta(days=30):
        return False, "La date prévue de vaccination ne peut pas être dans le passé lointain"
    
    if date_realisee and date_realisee < date_prevue - timedelta(days=7):
        return False, "La date réalisée est trop antérieure à la date prévue"
    
    if date_realisee and date_realisee > date.today():
        return False, "La date réalisée ne peut pas être dans le futur"
    
    return True, None


def validate_financial_transaction(
    montant: float,
    quantite: Optional[float] = None,
    prix_unitaire: Optional[float] = None
) -> List[str]:
    """Valider une transaction financière"""
    warnings = []
    
    if montant <= 0:
        warnings.append("Le montant doit être positif")
    
    if quantite is not None and quantite <= 0:
        warnings.append("La quantité doit être positive")
    
    if prix_unitaire is not None and prix_unitaire <= 0:
        warnings.append("Le prix unitaire doit être positif")
    
    if quantite and prix_unitaire:
        expected_montant = quantite * prix_unitaire
        if abs(montant - expected_montant) > 0.01:  # Tolérance 1 centime
            warnings.append(f"Incohérence: montant ({montant}) != quantité * prix unitaire ({expected_montant})")
    
    return warnings