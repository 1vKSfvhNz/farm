# backend/app/core/units.py
from typing import Union, Optional, Dict, Tuple
from enum import Enum


class UnitType(Enum):
    WEIGHT = "weight"
    VOLUME = "volume"
    TEMPERATURE = "temperature"
    LENGTH = "length"
    AREA = "area"
    MONEY = "money"


class UnitConverter:
    """Convertisseur d'unités pour l'application"""
    
    # Facteurs de conversion vers l'unité de base
    CONVERSIONS: Dict[str, Dict[str, float]] = {
        "weight": {
            "kg": 1.0,
            "g": 0.001,
            "lb": 0.453592,
            "ton": 1000.0,
        },
        "volume": {
            "m3": 1.0,
            "l": 0.001,
            "gal_us": 0.00378541,
            "gal_uk": 0.00454609,
        },
        "length": {
            "m": 1.0,
            "cm": 0.01,
            "mm": 0.001,
            "ft": 0.3048,
            "in": 0.0254,
        },
        "area": {
            "m2": 1.0,
            "ha": 10000.0,
            "acre": 4046.86,
        },
        "temperature": {
            "c": "celsius",
            "f": "fahrenheit",
        }
    }
    
    @staticmethod
    def convert_weight(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convertir un poids"""
        if from_unit == to_unit:
            return value
        
        # Convertir vers kg d'abord
        kg_value = value * UnitConverter.CONVERSIONS["weight"].get(from_unit.lower(), 1.0)
        
        # Puis vers l'unité cible
        result = kg_value / UnitConverter.CONVERSIONS["weight"].get(to_unit.lower(), 1.0)
        
        return round(result, 2)
    
    @staticmethod
    def convert_volume(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convertir un volume"""
        if from_unit == to_unit:
            return value
        
        # Convertir vers m3 d'abord
        m3_value = value * UnitConverter.CONVERSIONS["volume"].get(from_unit.lower(), 1.0)
        
        # Puis vers l'unité cible
        result = m3_value / UnitConverter.CONVERSIONS["volume"].get(to_unit.lower(), 1.0)
        
        return round(result, 2)
    
    @staticmethod
    def convert_temperature(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convertir une température"""
        if from_unit == to_unit:
            return value
        
        # Convertir vers Celsius d'abord
        if from_unit.lower() == "f":
            celsius = (value - 32) * 5 / 9
        else:
            celsius = value
        
        # Puis vers l'unité cible
        if to_unit.lower() == "f":
            result = (celsius * 9 / 5) + 32
        else:
            result = celsius
        
        return round(result, 1)
    
    @staticmethod
    def convert_length(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convertir une longueur"""
        if from_unit == to_unit:
            return value
        
        # Convertir vers mètres d'abord
        m_value = value * UnitConverter.CONVERSIONS["length"].get(from_unit.lower(), 1.0)
        
        # Puis vers l'unité cible
        result = m_value / UnitConverter.CONVERSIONS["length"].get(to_unit.lower(), 1.0)
        
        return round(result, 2)
    
    @staticmethod
    def convert_area(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convertir une surface"""
        if from_unit == to_unit:
            return value
        
        # Convertir vers m2 d'abord
        m2_value = value * UnitConverter.CONVERSIONS["area"].get(from_unit.lower(), 1.0)
        
        # Puis vers l'unité cible
        result = m2_value / UnitConverter.CONVERSIONS["area"].get(to_unit.lower(), 1.0)
        
        return round(result, 2)
    
    @staticmethod
    def format_weight(value: float, unit: str = "kg") -> str:
        """Formater un poids pour l'affichage"""
        if unit == "kg":
            if value >= 1000:
                return f"{value/1000:.1f} t"
            elif value >= 1:
                return f"{value:.1f} kg"
            else:
                return f"{value*1000:.0f} g"
        elif unit == "g":
            return f"{value:.0f} g"
        elif unit == "lb":
            return f"{value:.1f} lb"
        return f"{value} {unit}"
    
    @staticmethod
    def format_volume(value: float, unit: str = "m3") -> str:
        """Formater un volume pour l'affichage"""
        if unit == "m3" and value < 1:
            return f"{value*1000:.0f} L"
        elif unit == "l":
            if value >= 1000:
                return f"{value/1000:.1f} m³"
            return f"{value:.0f} L"
        return f"{value} {unit}"
    
    @staticmethod
    def format_temperature(value: float, unit: str = "c") -> str:
        """Formater une température pour l'affichage"""
        symbol = "°C" if unit.lower() == "c" else "°F"
        return f"{value:.0f}{symbol}"


# Fonctions de commodité
def convert_weight(value: float, from_unit: str, to_unit: str) -> float:
    return UnitConverter.convert_weight(value, from_unit, to_unit)

def convert_volume(value: float, from_unit: str, to_unit: str) -> float:
    return UnitConverter.convert_volume(value, from_unit, to_unit)

def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    return UnitConverter.convert_temperature(value, from_unit, to_unit)

def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    return UnitConverter.convert_length(value, from_unit, to_unit)