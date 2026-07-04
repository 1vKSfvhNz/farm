# backend/app/api/dependencies/__init__.py
"""
Dépendances FastAPI - Injection de dépendances
"""

from .auth import (
    get_current_user,
    get_current_admin_user,
    get_current_veterinaire_user,
    get_current_responsable_enclos_user,
    get_current_technicien_user,
    get_current_observer_user,
    get_current_user_optional
)
from .pagination import PaginationParams, paginate
from .rate_limit import RateLimiter, rate_limit

__all__ = [
    "get_current_user",
    "get_current_admin_user",
    "get_current_veterinaire_user",
    "get_current_responsable_enclos_user",
    "get_current_technicien_user",
    "get_current_observer_user",
    "get_current_user_optional",
    "PaginationParams",
    "paginate",
    "RateLimiter",
    "rate_limit",
]