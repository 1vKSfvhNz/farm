# backend/app/middleware/diagnostic_middleware.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import json
from typing import Set

class DiagnosticMiddleware(BaseHTTPMiddleware):
    """Middleware de diagnostic - À mettre EN PREMIER"""
    
    # Copiez les routes publiques de AuthMiddleware
    PUBLIC_PATHS: Set[str] = {
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    async def dispatch(self, request: Request, call_next):
        # Vérifier si c'est une route publique
        is_public = (request.url.path in self.PUBLIC_PATHS or 
                    any(request.url.path.startswith(path) for path in self.PUBLIC_PATHS))
        
        if is_public and request.method == "POST":
            print("\n" + "🔍" * 40)
            print(f"📥 DIAGNOSTIC: {request.method} {request.url.path}")
            
            # Lire le body
            body = await request.body()
            if body:
                try:
                    body_str = body.decode('utf-8')
                    print(f"   Body RAW: {body_str}")
                    
                    data = json.loads(body_str)
                    print(f"   JSON parsé: OK")
                    print(f"   - userlogin: '{data.get('userlogin', 'MISSING')}' (len: {len(data.get('userlogin', ''))})")
                    print(f"   - password: '{data.get('password', 'MISSING')}' (len: {len(data.get('password', ''))})")
                    
                    # Vérifier les contraintes Pydantic
                    userlogin = data.get('userlogin', '')
                    password = data.get('password', '')
                    
                    issues = []
                    if len(userlogin) < 8:
                        issues.append(f"userlogin trop court: {len(userlogin)} (min 8)")
                    if len(userlogin) > 100:
                        issues.append(f"userlogin trop long: {len(userlogin)} (max 100)")
                    if len(password) < 6:
                        issues.append(f"password trop court: {len(password)} (min 6)")
                    if len(password) > 8:
                        issues.append(f"password trop long: {len(password)} (max 8)")
                    
                    if issues:
                        print(f"   ❌ VIOLATION DES CONTRAINTES:")
                        for issue in issues:
                            print(f"      - {issue}")
                    else:
                        print(f"   ✅ Contraintes respectées")
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON invalide: {e}")
                except Exception as e:
                    print(f"   ❌ Erreur: {e}")
            
            # Reconstruire le body pour la suite
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive
            
            print("🔍" * 40 + "\n")
        
        # Continuer
        return await call_next(request)