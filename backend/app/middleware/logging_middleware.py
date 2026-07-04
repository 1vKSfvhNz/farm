# backend/app/middleware/logging_middleware.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time

from ..core.logging import logger
from ..redis_client import redis_client


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour journaliser toutes les requêtes API"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Lire le body pour les requêtes POST/PUT/PATCH
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                body = body.decode("utf-8")[:1000]
            except Exception:
                body = "[Unable to read body]"
        
        # Traiter la requête
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            user_id = getattr(request.state, "user_id", None)
            user_role = getattr(request.state, "user_role", None)
            
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "user_id": user_id,
                "user_role": user_role,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
            }
            
            if body and not any(sensitive in request.url.path for sensitive in ["/auth/login", "/users/password"]):
                log_data["body_preview"] = body[:500]
            
            # Afficher avec plus de détails pour les warnings
            if response.status_code >= 400:
                # Affichage direct dans la console pour debug
                print("\n" + "=" * 60)
                print(f"⚠️ WARNING DETECTED ⚠️")
                print(f"   Method: {request.method}")
                print(f"   Path: {request.url.path}")
                print(f"   Status: {response.status_code}")
                print(f"   Query params: {dict(request.query_params)}")
                if body:
                    print(f"   Body: {body}")
                print(f"   Client IP: {request.client.host if request.client else 'unknown'}")
                print("=" * 60 + "\n")
                
                self.logger.warning(f"API Warning: {request.method} {request.url.path} -> {response.status_code}", extra=log_data)
            elif response.status_code >= 500:
                self.logger.error(f"API Error: {request.method} {request.url.path} -> {response.status_code}", extra=log_data)
            else:
                self.logger.info(f"API Request: {request.method} {request.url.path} -> {response.status_code}", extra=log_data)
            
            await self._store_metric(request.url.path, response.status_code, process_time)
            
            response.headers["X-Process-Time-MS"] = str(round(process_time * 1000, 2))
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print(f"❌ EXCEPTION DETECTED ❌")
            print(f"   Method: {request.method}")
            print(f"   Path: {request.url.path}")
            print(f"   Error: {str(e)}")
            print("=" * 60 + "\n")
            
            self.logger.exception(
                f"Unhandled exception: {request.method} {request.url.path}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else None,
                    "error": str(e),
                    "process_time_ms": round(process_time * 1000, 2),
                }
            )
            raise
    
    async def _store_metric(self, path: str, status_code: int, process_time: float):
        """Stocker les métriques dans Redis"""
        try:
            counter_key = f"metric:counter:{path}:{status_code}"
            await redis_client.incr(counter_key)
            await redis_client.expire(counter_key, 3600)
            
            time_key = f"metric:time:{path}"
            await redis_client.lpush(time_key, process_time)
            await redis_client.ltrim(time_key, 0, 99)
            await redis_client.expire(time_key, 3600)
        except Exception as e:
            pass