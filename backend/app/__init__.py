# backend/app/__init__.py
"""
Application backend Farm Manager
FastAPI + C++ pour hautes performances
"""

__version__ = "1.0.0"
__author__ = "Farm Manager Team"

from .config import settings
from .database import get_db, engine, async_session_maker
from .redis_client import redis_client

__all__ = [
    "settings",
    "engine",
    "get_db",
    "async_session_maker",
    "redis_client",
]