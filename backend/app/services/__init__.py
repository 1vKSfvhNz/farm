# backend/app/schemas/__init__.py
from ..schemas.auth import LoginRequest, TokenResponse
from ..schemas.user import UserCreate, UserUpdate, UserResponse, UserSessionResponse, ActionLogResponse
from ..schemas.enclos import EnclosCreate, EnclosUpdate, EnclosResponse
from ..schemas.vaccination import MaladieCreate, MaladieResponse, VaccinCreate, VaccinResponse, VaccinationCreate, VaccinationUpdate, VaccinationResponse
from ..schemas.compost import CompostCreate, CompostUpdate, CompostResponse, RetournementCompostCreate, RetournementCompostResponse
from ..schemas.animal import AnimalBase
from ..schemas.bovin import BovinCreate, BovinUpdate, BovinResponse
from ..schemas.ovin import OvinCreate, OvinUpdate, OvinResponse
from ..schemas.caprin import CaprinCreate, CaprinUpdate, CaprinResponse
from ..schemas.avicole import AvicoleCreate, AvicoleUpdate, AvicoleResponse
from ..schemas.piscicole import PiscicoleCreate, PiscicoleUpdate, PiscicoleResponse
from ..schemas.entomoculture import EntomocultureLotCreate, EntomocultureLotUpdate, EntomocultureLotResponse, EntomocultureCycleCreate, EntomocultureCycleResponse
from ..schemas.pesee import PeseeCreate, PeseeUpdate, PeseeResponse
from ..schemas.alimentation import AlimentationCreate, AlimentationUpdate, AlimentationResponse, RationAlimentaireCreate, RationAlimentaireResponse
from ..schemas.naissance import NaissanceCreate, NaissanceUpdate, NaissanceResponse
from ..schemas.accounting import DepenseCreate, DepenseUpdate, DepenseResponse, RecetteCreate, RecetteUpdate, RecetteResponse, AccountSummary
from ..schemas.water_quality import WaterQualityCreate, WaterQualityResponse, WaterQualityAlerteResponse
from ..schemas.media import CameraCreate, CameraUpdate, CameraResponse, VideoRecordCreate, VideoRecordResponse
from ..schemas.bea import BienEtreIndiceCreate, BienEtreIndiceResponse, BienEtreCritereResponse
from ..schemas.predictions import PredictionRequest, PredictionResponse, GrowthPredictionResponse, ProductionPredictionResponse, CashflowPredictionResponse
from ..schemas.alerts import AlertCreate, AlertResponse
from ..schemas.exports import ExportFilter, ExportResponse
from ..schemas.experimental import ReferenceHypothesisCreate, ReferenceHypothesisResponse, ExperimentalModeResponse, ConfidenceResponse
from ..schemas.odoni import PiegeOdoniCreate, PiegeOdoniUpdate, PiegeOdoniResponse, ComptageOdoniCreate, ComptageOdoniUpdate, ComptageOdoniResponse, InfestationLevelResponse, PiegeStatistiquesResponse, AlerteOdoniResponse
from ..schemas.apiary import RucheCreate, RucheUpdate, RucheResponse, RecolteMielCreate, RecolteMielResponse

__all__ = [
    # Auth
    "LoginRequest", "TokenResponse",
    # User
    "UserCreate", "UserUpdate", "UserResponse", "UserSessionResponse", "ActionLogResponse",
    # Enclos
    "EnclosCreate", "EnclosUpdate", "EnclosResponse",
    # Vaccination
    "MaladieCreate", "MaladieResponse", "VaccinCreate", "VaccinResponse",
    "VaccinationCreate", "VaccinationUpdate", "VaccinationResponse",
    # Compost
    "CompostCreate", "CompostUpdate", "CompostResponse",
    "RetournementCompostCreate", "RetournementCompostResponse",
    # Animal
    "AnimalBase",
    # Bovins
    "BovinCreate", "BovinUpdate", "BovinResponse",
    # Ovins
    "OvinCreate", "OvinUpdate", "OvinResponse",
    # Caprins
    "CaprinCreate", "CaprinUpdate", "CaprinResponse",
    # Avicoles
    "AvicoleCreate", "AvicoleUpdate", "AvicoleResponse",
    # Piscicoles
    "PiscicoleCreate", "PiscicoleUpdate", "PiscicoleResponse",
    # Entomoculture
    "EntomocultureLotCreate", "EntomocultureLotUpdate", "EntomocultureLotResponse",
    "EntomocultureCycleCreate", "EntomocultureCycleResponse",
    # Pesée
    "PeseeCreate", "PeseeUpdate", "PeseeResponse",
    # Alimentation
    "AlimentationCreate", "AlimentationUpdate", "AlimentationResponse",
    "RationAlimentaireCreate", "RationAlimentaireResponse",
    # Naissance
    "NaissanceCreate", "NaissanceUpdate", "NaissanceResponse",
    # Comptabilité
    "DepenseCreate", "DepenseUpdate", "DepenseResponse",
    "RecetteCreate", "RecetteUpdate", "RecetteResponse",
    "AccountSummary",
    # Qualité eau
    "WaterQualityCreate", "WaterQualityResponse", "WaterQualityAlerteResponse",
    # Vidéo
    "CameraCreate", "CameraUpdate", "CameraResponse",
    "VideoRecordCreate", "VideoRecordResponse",
    # Bien-être
    "BienEtreIndiceCreate", "BienEtreIndiceResponse", "BienEtreCritereResponse",
    # Prédictions
    "PredictionRequest", "PredictionResponse", "GrowthPredictionResponse",
    "ProductionPredictionResponse", "CashflowPredictionResponse",
    # Alertes
    "AlertCreate", "AlertResponse",
    # Exports
    "ExportFilter", "ExportResponse",
    # Expérimental
    "ReferenceHypothesisCreate", "ReferenceHypothesisResponse",
    "ExperimentalModeResponse", "ConfidenceResponse",
    # Odoni
    "PiegeOdoniCreate", "PiegeOdoniUpdate", "PiegeOdoniResponse",
    "ComptageOdoniCreate", "ComptageOdoniUpdate", "ComptageOdoniResponse",
    "InfestationLevelResponse", "PiegeStatistiquesResponse", "AlerteOdoniResponse",
    # Apiculture
    "RucheCreate", "RucheUpdate", "RucheResponse",
    "RecolteMielCreate", "RecolteMielResponse",
]
