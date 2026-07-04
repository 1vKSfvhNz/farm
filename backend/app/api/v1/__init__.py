# backend/app/api/v1/__init__.py
"""
API Version 1 - Routes principales
"""

from .auth import router as auth_router
from .users import router as users_router
from .enclos import router as enclos_router
from .vaccination import router as vaccination_router
from .compost import router as compost_router
from .bovins import router as bovins_router
from .ovins import router as ovins_router
from .caprins import router as caprins_router
from .avicoles import router as avicoles_router
from .piscicoles import router as piscicoles_router
from .entomoculture import router as entomoculture_router
from .accounting import router as accounting_router
from .dashboard import router as dashboard_router
from .predictions import router as predictions_router
from .alerts import router as alerts_router
from .exports import router as exports_router
from .water_quality import router as water_quality_router
from .media import router as video_router
from .weather import router as weather_router
from .bea import router as bea_router
from .blockchain import router as blockchain_router
from .odoni import router as odoni_router
from .apiary import router as apiary_router
from .pesees import router as pesees_router
from .experimental import router as experimental_router

__all__ = [
    "auth_router",
    "users_router",
    "enclos_router",
    "vaccination_router",
    "compost_router",
    "bovins_router",
    "ovins_router",
    "caprins_router",
    "avicoles_router",
    "piscicoles_router",
    "entomoculture_router",
    "accounting_router",
    "dashboard_router",
    "predictions_router",
    "alerts_router",
    "exports_router",
    "water_quality_router",
    "video_router",
    "weather_router",
    "bea_router",
    "blockchain_router",
    "odoni_router",
    "apiary_router",
    "pesees_router",
    "experimental_router",
]