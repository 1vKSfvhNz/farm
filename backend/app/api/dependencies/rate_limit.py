# backend/app/api/dependencies/rate_limit.py
"""
Dépendances pour le rate limiting
"""

from typing import Optional, Callable
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from ...redis_client import redis_client
from ...config import settings


class RateLimiter:
    """
    Limiteur de taux pour les endpoints spécifiques
    
    Usage:
        rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        @router.post("/sensitive")
        async def sensitive_endpoint(rate_limit=Depends(rate_limiter)):
            ...
    """
    
    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
        key_prefix: str = "rate_limit"
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
    
    async def __call__(
        self,
        request: Request,
    ) -> bool:
        """
        Vérifier si la requête respecte les limites de taux
        
        Returns:
            True si la requête est autorisée
        
        Raises:
            HTTPException: Si la limite est dépassée
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True
        
        # Déterminer la clé unique pour le client
        client_id = self._get_client_id(request)
        key = f"{self.key_prefix}:{client_id}"
        
        # Vérifier le rate limit
        is_allowed, remaining = await redis_client.check_rate_limit(
            key,
            self.max_requests,
            self.window_seconds
        )
        
        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Trop de requêtes. Limite: {self.max_requests} par {self.window_seconds} secondes.",
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(self.window_seconds),
                    "Retry-After": str(self.window_seconds),
                }
            )
        
        # Ajouter les headers à la réponse via request.state
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(self.window_seconds),
        }
        
        return True
    
    def _get_client_id(self, request: Request) -> str:
        """Obtenir un identifiant unique pour le client"""
        # Priorité: utilisateur authentifié > IP
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Sinon utiliser l'IP
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour appliquer des limites globales
    (Utilisé comme fallback si les dépendances ne sont pas utilisées)
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.default_limiter = RateLimiter(
            max_requests=settings.RATE_LIMIT_DEFAULT.split("/")[0],
            window_seconds=60
        )
    
    async def dispatch(self, request: Request, call_next):
        # Appliquer le rate limit par défaut pour les routes non protégées
        user_id = getattr(request.state, "user_id", None)
        
        if user_id:
            # Utilisateurs authentifiés
            limiter = RateLimiter(
                max_requests=int(settings.RATE_LIMIT_AUTHENTICATED.split("/")[0]),
                window_seconds=60
            )
        else:
            limiter = self.default_limiter
        
        try:
            await limiter(request)
        except HTTPException as exc:
            return exc
        
        response = await call_next(request)
        
        # Ajouter les headers de rate limit à la réponse
        if hasattr(request.state, "rate_limit_headers"):
            for key, value in request.state.rate_limit_headers.items():
                response.headers[key] = value
        
        return response


def rate_limit(
    max_requests: int = 60,
    window_seconds: int = 60,
    key_prefix: str = "rate_limit"
) -> RateLimiter:
    """
    Factory pour créer un limiteur de taux configurable
    
    Usage:
        @router.post("/endpoint")
        async def endpoint(
            rate_limit: RateLimiter = Depends(rate_limit(max_requests=10, window_seconds=30))
        ):
            ...
    """
    return RateLimiter(
        max_requests=max_requests,
        window_seconds=window_seconds,
        key_prefix=key_prefix
    )