# backend/app/middleware/error_handler.py
import uuid

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any
import traceback

from ..config import settings


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware global pour la gestion des erreurs"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
            
        except HTTPException as exc:
            # Log simple pour les erreurs HTTP
            if exc.status_code >= 500:
                print(f"❌ Server Error: {request.method} {request.url.path} -> {exc.status_code}")
            elif exc.status_code >= 400:
                # Ignorer les 401/403/404 sauf en debug
                if exc.status_code not in [401, 403, 404]:
                    print(f"⚠️ Client Error: {request.method} {request.url.path} -> {exc.status_code}")
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.status_code,
                        "message": exc.detail,
                        "type": "http_exception",
                    }
                },
                headers=exc.headers,
            )
            
        except Exception as exc:
            # Log simple pour les erreurs non gérées
            error_id = str(uuid.uuid4())[:8]
            print(f"💥 Unhandled: {request.method} {request.url.path} -> {type(exc).__name__} [{error_id}]")
            
            content = {
                "error": {
                    "code": 500,
                    "message": "An unexpected error occurred" if settings.ENVIRONMENT != "development" else str(exc),
                    "type": "internal_server_error",
                    "error_id": error_id,
                }
            }
            
            if settings.ENVIRONMENT == "development":
                content["error"]["traceback"] = traceback.format_exc().split("\n")
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=content,
            )


class AppException(Exception):
    """Exception personnalisée pour l'application"""
    def __init__(self, message: str, code: int = 400, details: Dict[str, Any] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(AppException):
    """Exception pour les erreurs de validation"""
    def __init__(self, message: str, field_errors: Dict[str, str] = None):
        super().__init__(message, 422, {"field_errors": field_errors})


class NotFoundException(AppException):
    """Exception pour les ressources non trouvées"""
    def __init__(self, resource_type: str, resource_id: int):
        super().__init__(f"{resource_type} with id {resource_id} not found", 404)


class PermissionDeniedException(AppException):
    """Exception pour les erreurs de permission"""
    def __init__(self, message: str = "Permission denied"):
        super().__init__(message, 403)


class BusinessRuleException(AppException):
    """Exception pour les violations de règles métier"""
    def __init__(self, message: str, rule_name: str = None):
        super().__init__(message, 400, {"rule_name": rule_name})