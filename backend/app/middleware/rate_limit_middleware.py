# backend/app/middleware/rate_limit_middleware.py
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Tuple, Optional
import time
from collections import defaultdict

from ..redis_client import redis_client
from ..config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware pour limiter le taux de requêtes"""
    
    # Limites par défaut: (requêtes, fenêtre_secondes)
    DEFAULT_LIMITS: Dict[str, Tuple[int, int]] = {
        "default": (100, 60),  # 100 requêtes par minute
        "authenticated": (500, 60),  # 500 requêtes par minute pour utilisateurs authentifiés
        "admin": (1000, 60),  # 1000 requêtes par minute pour admins
    }
    
    # Limites spécifiques par endpoint
    ENDPOINT_LIMITS: Dict[str, Tuple[int, int]] = {
        "/api/v1/auth/login": (5, 60),  # 5 tentatives par minute
        "/api/v1/auth/refresh": (10, 60),
        "/api/v1/exports": (20, 3600),  # 20 exports par heure
        "/api/v1/predictions": (50, 60),  # 50 prédictions par minute
    }
    
    async def dispatch(self, request: Request, call_next):
        # Déterminer l'identifiant du client
        client_id = await self._get_client_id(request)
        
        # Déterminer les limites
        limits = await self._get_limits(request)
        
        # Vérifier le rate limiting
        if not await self._check_rate_limit(client_id, limits):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {limits[0]} requests per {limits[1]} seconds.",
                headers={
                    "X-RateLimit-Limit": str(limits[0]),
                    "X-RateLimit-Window": str(limits[1]),
                }
            )
        
        # Ajouter les headers de rate limit
        response = await call_next(request)
        await self._add_rate_limit_headers(response, client_id, limits)
        
        return response
    
    async def _get_client_id(self, request: Request) -> str:
        """Récupérer l'identifiant unique du client"""
        # Priorité: utilisateur authentifié > IP
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Sinon utiliser l'IP
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def _get_limits(self, request: Request) -> Tuple[int, int]:
        """Déterminer les limites pour cette requête"""
        # Vérifier les limites spécifiques par endpoint
        for endpoint, limits in self.ENDPOINT_LIMITS.items():
            if request.url.path.startswith(endpoint):
                return limits
        
        # Sinon utiliser les limites par rôle
        user_role = getattr(request.state, "user_role", None)
        if user_role == "admin":
            return self.DEFAULT_LIMITS["admin"]
        elif user_role:
            return self.DEFAULT_LIMITS["authenticated"]
        
        return self.DEFAULT_LIMITS["default"]
    
    async def _check_rate_limit(self, client_id: str, limits: Tuple[int, int]) -> bool:
        """Vérifier si le client a dépassé sa limite"""
        max_requests, window_seconds = limits
        
        # Clé Redis
        key = f"rate_limit:{client_id}"
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        try:
            # Nettoyer les anciennes entrées
            await redis_client.zremrangebyscore(key, 0, window_start)
            
            # Compter les requêtes dans la fenêtre
            count = await redis_client.zcard(key)
            
            if count >= max_requests:
                return False
            
            # Ajouter la requête actuelle
            await redis_client.zadd(key, {str(current_time): current_time})
            await redis_client.expire(key, window_seconds)
            
            return True
            
        except Exception as e:
            # En cas d'erreur Redis, on autorise (fail open)
            print(f"Rate limit check failed: {e}")
            return True
    
    async def _add_rate_limit_headers(self, response, client_id: str, limits: Tuple[int, int]):
        """Ajouter les headers de rate limit à la réponse"""
        try:
            max_requests, window_seconds = limits
            key = f"rate_limit:{client_id}"
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            remaining = max_requests - await redis_client.zcard(key)
            reset_at = window_start + window_seconds
            
            response.headers["X-RateLimit-Limit"] = str(max_requests)
            response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
            response.headers["X-RateLimit-Reset"] = str(reset_at)
        except Exception:
            pass