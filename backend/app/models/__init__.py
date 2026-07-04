# backend/app/models/__init__.py
from .base import Base
from .user import User, RoleEnum, UserSession, ActionLog
from .enclos import Enclos, EnclosType
from .vaccination import Vaccination, Maladie, Vaccin
from .compost import Compost, CompostType, RetournementCompost
from .animal import Animal
from .bovin import Bovin
from .ovin import Ovin
from .caprin import Caprin
from .avicole import Avicole
from .piscicole import Piscicole
from .entomoculture import EntomocultureLot, EntomocultureCycle
from .pesee import Pesee
from .alimentation import Alimentation, RationAlimentaire
from .naissance import Naissance
from .mortalite import Mortalite
from .accounting import Depense, Recette, CategorieDepenseEnum, CategorieRecetteEnum
from .water_quality import WaterQuality, WaterQualityAlerte
from .video import VideoRecord, Camera
from .bien_etre import BienEtreIndice, BienEtreCritere
from .apiary import Ruche, Reine, RecolteMiel
from .odoni import PiegeOdoni, ComptageOdoni
from .reference import ReferenceCroissance, ReferenceSeuil, ReferenceVaccination, ReferenceNutrition, ReferenceHypothese
from .experimental import ReferenceGenerale, DonneeExperimentale
from .alerts import Alert, AlertRule, AlertHistory, NotificationPreference, AlertNiveauEnum, AlertTypeEnum

__all__ = [
    "Base",
    "User", "RoleEnum", "UserSession", "ActionLog",
    "Enclos", "EnclosType",
    "Vaccination", "Maladie", "Vaccin",
    "Compost", "CompostType", "RetournementCompost",
    "Animal",
    "Bovin", "Ovin", "Caprin", "Avicole", "Piscicole",
    "EntomocultureLot", "EntomocultureCycle",
    "Pesee",
    "Alimentation", "RationAlimentaire",
    "Naissance",
    "Mortalite",
    "Depense", "Recette", "CategorieDepenseEnum", "CategorieRecetteEnum",
    "WaterQuality", "WaterQualityAlerte",
    "VideoRecord", "Camera",
    "BienEtreIndice", "BienEtreCritere",
    "Ruche", "Reine", "RecolteMiel",
    "PiegeOdoni", "ComptageOdoni",
    "ReferenceCroissance", "ReferenceSeuil", "ReferenceVaccination", "ReferenceNutrition", "ReferenceHypothese",
    "ReferenceGenerale", "DonneeExperimentale",
    "Alert", "AlertRule", "AlertHistory", "NotificationPreference",
    "AlertNiveauEnum", "AlertTypeEnum",
]