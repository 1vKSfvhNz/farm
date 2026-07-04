# backend/app/services/weather_service.py
"""
Service météo - Intégration API externe
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class WeatherService:
    """Service d'intégration météo"""
    
    async def get_current_weather(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[Dict[str, Any]]:
        """Obtenir la météo actuelle"""
        if not settings.WEATHER_API_ENABLED or not settings.WEATHER_API_KEY:
            logger.warning("Weather API not configured")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                if settings.WEATHER_API_PROVIDER == "openweathermap":
                    url = "https://api.openweathermap.org/data/2.5/weather"
                    params = {
                        "lat": latitude,
                        "lon": longitude,
                        "appid": settings.WEATHER_API_KEY,
                        "units": "metric"
                    }
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    return {
                        "temperature": data.get("main", {}).get("temp"),
                        "humidity": data.get("main", {}).get("humidity"),
                        "pressure": data.get("main", {}).get("pressure"),
                        "wind_speed": data.get("wind", {}).get("speed"),
                        "rain": data.get("rain", {}).get("1h", 0),
                        "description": data.get("weather", [{}])[0].get("description")
                    }
            
        except Exception as e:
            logger.error(f"Failed to get weather: {e}")
            return None
    
    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7
    ) -> Optional[list]:
        """Obtenir les prévisions météo"""
        if not settings.WEATHER_API_ENABLED or not settings.WEATHER_API_KEY:
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                if settings.WEATHER_API_PROVIDER == "openweathermap":
                    url = "https://api.openweathermap.org/data/2.5/forecast"
                    params = {
                        "lat": latitude,
                        "lon": longitude,
                        "appid": settings.WEATHER_API_KEY,
                        "units": "metric"
                    }
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    forecasts = []
                    for item in data.get("list", [])[:days * 8]:
                        forecasts.append({
                            "datetime": datetime.fromtimestamp(item.get("dt")),
                            "temperature": item.get("main", {}).get("temp"),
                            "humidity": item.get("main", {}).get("humidity"),
                            "rain": item.get("rain", {}).get("3h", 0)
                        })
                    
                    return forecasts
            
        except Exception as e:
            logger.error(f"Failed to get forecast: {e}")
            return None


weather_service = WeatherService()