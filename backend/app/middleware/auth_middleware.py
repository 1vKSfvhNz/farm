# backend/app/middleware/auth_middleware.py

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Set
import jwt
from datetime import datetime
import logging

from ..redis_client import redis_client
from ..config import settings

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware d'authentification pour valider les tokens JWT"""
    
    # Routes publiques qui ne nécessitent pas d'authentification
    PUBLIC_PATHS: Set[str] = {
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/health",
        "/api/v1/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/",
    }
    
    async def dispatch(self, request: Request, call_next):
        # Vérifier si la route est publique
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        
        # Vérifier si c'est une route préfixée publique
        if any(request.url.path.startswith(path) for path in self.PUBLIC_PATHS):
            return await call_next(request)
        
        # Extraire le token
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.warning(f"Missing auth header for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme",
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
            )
        
        # Valider le token JWT (sans Redis d'abord)
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": True}  # Vérifier l'expiration
            )
            
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                )
            
            user_id = payload.get("sub")
            username = payload.get("username")
            roles = payload.get("roles", [])
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                )
            
            # OPTIONNEL: Vérifier Redis seulement si disponible
            session_id = payload.get("session_id")
            if session_id and settings.REDIS_ENABLED:
                try:
                    session_key = f"session:{session_id}"
                    session_data = await redis_client.get(session_key)
                    
                    if not session_data:
                        logger.warning(f"Session {session_id} not found in Redis")
                        # En développement, on continue quand même
                        if settings.ENVIRONMENT == "production":
                            raise HTTPException(
                                status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Session expired or invalid",
                            )
                except Exception as e:
                    logger.error(f"Redis error: {e}")
                    # En développement, on ignore les erreurs Redis
                    if settings.ENVIRONMENT == "production":
                        raise
            
            # Ajouter les informations utilisateur à la requête
            request.state.user_id = int(user_id)
            request.state.username = username
            request.state.roles = roles
            request.state.session_id = session_id
            
        except jwt.ExpiredSignatureError:
            logger.warning(f"Expired token for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Auth error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return await call_next(request)