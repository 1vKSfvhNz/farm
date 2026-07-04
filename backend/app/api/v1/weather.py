# backend/app/api/v1/weather.py
"""
Routes météo
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional

from ...services.weather_service import weather_service
from ...api.dependencies.auth import get_current_user
from ...models.user import User
from ...config import settings

router = APIRouter(prefix="/weather", tags=["Météo"])


@router.get("/current")
async def get_current_weather(
    latitude: Optional[float] = Query(None, description="Latitude"),
    longitude: Optional[float] = Query(None, description="Longitude"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la météo actuelle
    """
    lat = latitude or settings.WEATHER_LATITUDE
    lon = longitude or settings.WEATHER_LONGITUDE
    
    if not lat or not lon:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Coordonnées non configurées"
        )
    
    weather = await weather_service.get_current_weather(lat, lon)
    if not weather:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service météo indisponible"
        )
    
    return weather


@router.get("/forecast")
async def get_forecast(
    latitude: Optional[float] = Query(None, description="Latitude"),
    longitude: Optional[float] = Query(None, description="Longitude"),
    days: int = Query(7, description="Nombre de jours", ge=1, le=7),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les prévisions météo
    """
    lat = latitude or settings.WEATHER_LATITUDE
    lon = longitude or settings.WEATHER_LONGITUDE
    
    if not lat or not lon:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Coordonnées non configurées"
        )
    
    forecast = await weather_service.get_forecast(lat, lon, days)
    if not forecast:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service météo indisponible"
        )
    
    return {"forecast": forecast}