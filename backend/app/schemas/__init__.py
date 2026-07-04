# backend/app/schemas/__init__.py
from .auth import LoginRequest, TokenResponse
from .user import UserCreate, UserUpdate, UserResponse, UserSessionResponse, ActionLogResponse
from .enclos import EnclosCreate, EnclosUpdate, EnclosResponse
from .vaccination import MaladieCreate, MaladieResponse, VaccinCreate, VaccinResponse, VaccinationCreate, VaccinationUpdate, VaccinationResponse
from .compost import CompostCreate, CompostUpdate, CompostResponse, RetournementCompostCreate, RetournementCompostResponse
from .animal import AnimalBase
from .bovin import BovinCreate, BovinUpdate, BovinResponse
from .ovin import OvinCreate, OvinUpdate, OvinResponse
from .apiary import RucheCreate, RucheUpdate, RucheResponse, RecolteMielCreate, RecolteMielResponse, StatutRucheEnum, RucheBase
from .caprin import CaprinCreate, CaprinUpdate, CaprinResponse
from .avicole import AvicoleCreate, AvicoleUpdate, AvicoleResponse
from .piscicole import PiscicoleCreate, PiscicoleUpdate, PiscicoleResponse
from .entomoculture import EntomocultureLotCreate, EntomocultureLotUpdate, EntomocultureLotResponse, EntomocultureCycleCreate, EntomocultureCycleResponse
from .pesee import PeseeCreate, PeseeUpdate, PeseeResponse
from .alimentation import AlimentationCreate, AlimentationUpdate, AlimentationResponse, RationAlimentaireCreate, RationAlimentaireResponse
from .accounting import DepenseCreate, DepenseUpdate, DepenseResponse, RecetteCreate, RecetteUpdate, RecetteResponse
from .water_quality import WaterQualityCreate, WaterQualityResponse, WaterQualityAlerteResponse
from .media import CameraCreate, CameraUpdate, CameraResponse, VideoRecordCreate, VideoRecordResponse
from .bea import BienEtreIndiceCreate, BienEtreIndiceResponse, BienEtreCritereResponse
from .predictions import PredictionRequest, PredictionResponse, GrowthPredictionResponse, ProductionPredictionResponse, CashflowPredictionResponse
from .alerts import AlertCreate, AlertResponse
from .exports import ExportFilter, ExportResponse
from .experimental import ReferenceHypothesisCreate, ReferenceHypothesisResponse, ExperimentalModeResponse, ConfidenceResponse

__all__ = [
    "LoginRequest", "TokenResponse",
    "UserCreate", "UserUpdate", "UserResponse", "UserSessionResponse", "ActionLogResponse",
    "EnclosCreate", "EnclosUpdate", "EnclosResponse",
    "MaladieCreate", "MaladieResponse", "VaccinCreate", "VaccinResponse",
    "VaccinationCreate", "VaccinationUpdate", "VaccinationResponse",
    "CompostCreate", "CompostUpdate", "CompostResponse",
    "RetournementCompostCreate", "RetournementCompostResponse",
    "AnimalBase",
    "BovinCreate", "BovinUpdate", "BovinResponse",
    "OvinCreate", "OvinUpdate", "OvinResponse",
    "CaprinCreate", "CaprinUpdate", "CaprinResponse",
    "AvicoleCreate", "AvicoleUpdate", "AvicoleResponse",
    "PiscicoleCreate", "PiscicoleUpdate", "PiscicoleResponse",
    "EntomocultureLotCreate", "EntomocultureLotUpdate", "EntomocultureLotResponse",
    "EntomocultureCycleCreate", "EntomocultureCycleResponse",
    "PeseeCreate", "PeseeUpdate", "PeseeResponse",
    "AlimentationCreate", "AlimentationUpdate", "AlimentationResponse",
    "RationAlimentaireCreate", "RationAlimentaireResponse",
    "DepenseCreate", "DepenseUpdate", "DepenseResponse",
    "RecetteCreate", "RecetteUpdate", "RecetteResponse",
    "WaterQualityCreate", "WaterQualityResponse", "WaterQualityAlerteResponse",
    "CameraCreate", "CameraUpdate", "CameraResponse",
    "VideoRecordCreate", "VideoRecordResponse",
    "BienEtreIndiceCreate", "BienEtreIndiceResponse", "BienEtreCritereResponse",
    "PredictionRequest", "PredictionResponse", "GrowthPredictionResponse",
    "ProductionPredictionResponse", "CashflowPredictionResponse",
    "AlertCreate", "AlertResponse",
    "ExportFilter", "ExportResponse",
    "ReferenceHypothesisCreate", "ReferenceHypothesisResponse",
    "ExperimentalModeResponse", "ConfidenceResponse",
    "StatutRucheEnum",
    "RucheBase", "RucheCreate", "RucheUpdate", "RucheResponse",
    "RecolteMielBase", "RecolteMielCreate", "RecolteMielUpdate", "RecolteMielResponse",
    "ProductionStatsResponse", "RucheStatsResponse", "ReinesStatsResponse",
    "AlerteApicoleResponse", "DashboardApicoleResponse",
    "InspectionRucheBase", "InspectionRucheCreate", "InspectionRucheResponse",
    "EssaimageBase", "EssaimageCreate", "EssaimageResponse",
]
