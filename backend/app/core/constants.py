# backend/app/core/constants.py
from enum import Enum

class RoleEnum(str, Enum):
    ADMIN = "admin"
    VETERINAIRE = "veterinaire"
    RESPONSABLE_ENCLOS = "responsable_enclos"
    TECHNICIEN = "technicien"
    OBSERVATEUR = "observateur"


class EnclosTypeEnum(str, Enum):
    ENCLOS = "enclos"
    BASSIN = "bassin"
    PATURAGE = "pâturage"
    CAGE = "cage"
    BAC = "bac"


class SexeEnum(str, Enum):
    MALE = "male"
    FEMELLE = "femelle"
    HERMAPHRODITE = "hermaphrodite"


class StatutAnimalEnum(str, Enum):
    VIVANT = "vivant"
    VENDU = "vendu"
    DECEDE = "decede"
    TRANSFERE = "transfere"


class EspeceEnum(str, Enum):
    BOVIN = "bovin"
    OVIN = "ovin"
    CAPRIN = "caprin"
    AVICOLE = "avicole"
    PISCICOLE = "piscicole"
    ENTOMOCULTURE = "entomoculture"


# Unités de référence
UNITS = {
    "weight": {
        "base": "kg",
        "available": ["kg", "g", "lb", "ton"],
        "default": "kg",
    },
    "volume": {
        "base": "m3",
        "available": ["m3", "l", "gal_us", "gal_uk"],
        "default": "m3",
    },
    "temperature": {
        "base": "c",
        "available": ["c", "f"],
        "default": "c",
    },
    "length": {
        "base": "m",
        "available": ["m", "cm", "mm", "ft", "in"],
        "default": "m",
    },
    "area": {
        "base": "m2",
        "available": ["m2", "ha", "acre"],
        "default": "m2",
    },
    "money": {
        "base": "eur",
        "available": ["eur", "usd", "gbp", "xaf"],
        "default": "eur",
    },
}


# Seuils de mortalité par espèce (%)
SEUILS_MORTALITE = {
    "bovin": {"normal": 2.0, "alerte_orange": 5.0, "alerte_rouge": 10.0},
    "ovin": {"normal": 3.0, "alerte_orange": 6.0, "alerte_rouge": 12.0},
    "caprin": {"normal": 3.0, "alerte_orange": 6.0, "alerte_rouge": 12.0},
    "avicole": {"normal": 5.0, "alerte_orange": 10.0, "alerte_rouge": 20.0},
    "piscicole": {"normal": 1.0, "alerte_orange": 3.0, "alerte_rouge": 5.0},
    "entomoculture": {"normal": 10.0, "alerte_orange": 20.0, "alerte_rouge": 30.0},
}


# Seuils de conversion alimentaire (FCR)
SEUILS_CONVERSION_ALIMENTAIRE = {
    "bovin_viande": {"excellent": 6.0, "standard": 8.0, "mauvais": 10.0},
    "bovin_lait": {"excellent": 1.2, "standard": 1.5, "mauvais": 1.8},
    "ovin": {"excellent": 5.0, "standard": 7.0, "mauvais": 9.0},
    "caprin": {"excellent": 5.0, "standard": 7.0, "mauvais": 9.0},
    "avicole_viande": {"excellent": 1.8, "standard": 2.2, "mauvais": 2.6},
    "avicole_ponte": {"excellent": 1.8, "standard": 2.2, "mauvais": 2.6},
    "piscicole_tilapia": {"excellent": 1.5, "standard": 1.8, "mauvais": 2.1},
    "piscicole_truite": {"excellent": 1.2, "standard": 1.5, "mauvais": 1.8},
    "entomoculture": {"excellent": 2.0, "standard": 3.0, "mauvais": 4.0},
}


# Seuils qualité eau pour pisciculture
SEUILS_QUALITE_EAU = {
    "tilapia": {
        "oxygen_min": 3.0,
        "oxygen_opt": [5.0, 8.0],
        "ph_min": 6.5,
        "ph_max": 9.0,
        "ph_opt": [7.0, 8.0],
        "temperature_min": 14,
        "temperature_max": 36,
        "temperature_opt": [25, 30],
        "ammoniac_max": 0.1,
        "nitrites_max": 1.0,
    },
    "truite": {
        "oxygen_min": 6.0,
        "oxygen_opt": [8.0, 12.0],
        "ph_min": 6.0,
        "ph_max": 8.5,
        "ph_opt": [6.5, 8.0],
        "temperature_min": 4,
        "temperature_max": 18,
        "temperature_opt": [8, 14],
        "ammoniac_max": 0.02,
        "nitrites_max": 0.2,
    },
    "clarias": {
        "oxygen_min": 4.0,
        "oxygen_opt": [5.0, 7.0],
        "ph_min": 6.0,
        "ph_max": 8.5,
        "ph_opt": [6.5, 8.0],
        "temperature_min": 20,
        "temperature_max": 35,
        "temperature_opt": [25, 30],
        "ammoniac_max": 0.05,
        "nitrites_max": 0.5,
    },
}


# Niveaux de confiance pour les prédictions
CONFIDENCE_LEVELS = {
    "faible": {"min": 0, "max": 40, "label": "Faible", "color": "red"},
    "moyenne": {"min": 40, "max": 70, "label": "Moyenne", "color": "orange"},
    "elevee": {"min": 70, "max": 100, "label": "Élevée", "color": "green"},
}


# Messages d'alerte prédéfinis
ALERT_MESSAGES = {
    "vaccination_retard": "Vaccination en retard pour {animal}",
    "pesee_manquante": "Pesée non effectuée depuis {days} jours pour {animal}",
    "nettoyage_quotidien": "Nettoyage de l'enclos {enclos} non effectué aujourd'hui",
    "mortalite_elevee": "Taux de mortalité élevé ({mortality}%) pour {espece}",
    "eau_oxygene_bas": "Oxygène dissous critique ({value} mg/L) dans le bassin {enclos}",
    "eau_ph_hors_norme": "pH hors norme ({value}) dans le bassin {enclos}",
    "temperature_critique": "Température {value}°C critique pour {espece}",
    "tresorerie_basse": "Trésorerie prévisionnelle basse ({amount} €) dans {days} jours",
    "encombrement_enclos": "Enclos {enclos} à {occupation}% de capacité",
    "compost_maturite": "Compost {compost} proche de la maturité (J+{days})",
}


# Actions suggérées par type d'alerte
ALERT_ACTIONS = {
    "vaccination_retard": "Planifier une vaccination dans les 48 heures",
    "pesee_manquante": "Effectuer une pesée immédiatement",
    "mortalite_elevee": "Consulter un vétérinaire et analyser les causes",
    "eau_oxygene_bas": "Activer l'aération d'urgence",
    "eau_ph_hors_norme": "Corriger le pH (bicarbonate ou acide)",
    "temperature_critique": "Adapter ventilation/ombrage/chauffage",
    "tresorerie_basse": "Réviser les dépenses ou programmer des ventes",
    "encombrement_enclos": "Préparer un transfert vers un autre enclos",
}