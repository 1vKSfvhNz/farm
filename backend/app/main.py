# backend/app/main.py
"""
Point d'entrée principal de l'application FastAPI
Farm Manager API - Gestion agricole connectée
"""

from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
from datetime import datetime

from fastapi.staticfiles import StaticFiles

from .config import settings
from .lifespan import lifespan  # <-- Import du lifespan séparé
from .api.v1 import (
    auth_router,
    users_router,
    enclos_router,
    vaccination_router,
    compost_router,
    bovins_router,
    ovins_router,
    caprins_router,
    avicoles_router,
    piscicoles_router,
    entomoculture_router,
    accounting_router,
    dashboard_router,
    predictions_router,
    alerts_router,
    exports_router,
    water_quality_router,
    video_router,
    weather_router,
    bea_router,
    blockchain_router,
    odoni_router,
    apiary_router,
    pesees_router,
    experimental_router,
)
from .middleware import (
    AuthMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlerMiddleware
)
from .core.logging import setup_logging

# Configuration du logging
setup_logging()
logger = logging.getLogger(__name__)


# ============ CRÉATION DE L'APPLICATION ============

app = FastAPI(
    title=settings.APP_NAME,
    description="""
    # API Farm Manager
    
    Application de gestion agricole connectée pour le suivi des animaux,
    des cultures, de la comptabilité et des prédictions.
    
    ## Fonctionnalités principales
    
    - **Gestion des animaux**: Bovins, Ovins, Caprins, Avicoles, Piscicoles
    - **Entomoculture**: Gestion des insectes (larves, grillons, etc.)
    - **Suivi sanitaire**: Vaccinations, mortalité, alertes
    - **Comptabilité**: Dépenses, recettes, rentabilité par espèce
    - **Prédictions**: Croissance, production, trésorerie, risques sanitaires
    - **Qualité de l'eau**: Suivi des paramètres pour pisciculture
    - **Bien-être animal**: Indices de confort et alertes
    - **apiary**: Gestion des ruches et production de miel
    - **Lutte contre les nuisibles**: Pièges connectés
    - **Mode expérimental**: Auto-apprentissage des références
    
    ## Technologies
    
    - FastAPI pour le backend
    - PostgreSQL pour la persistence
    - Redis pour le cache et les sessions
    - C++ pour les calculs haute performance
    
    ## Authentification
    
    L'API utilise JWT pour l'authentification.  
    Inclure le token dans le header: `Authorization: Bearer <token>`
    """,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,  # <-- Utilisation du lifespan séparé
)


# ============ MIDDLEWARE ============

# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
    expose_headers=["*"],
    max_age=3600,
)

# 2. Trusted Host (sécurité)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# 3. Logging
app.add_middleware(LoggingMiddleware)

# 4. Rate Limiting
if settings.RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)

# 5. Authentification (en dernier parmi les middlewares généraux)
app.add_middleware(AuthMiddleware)

# 6. Error Handler (dernier middleware pour capturer toutes les erreurs)
# app.add_middleware(ErrorHandlerMiddleware)

# Servir les fichiers uploads
uploads_path = Path(settings.UPLOAD_DIR)
if not uploads_path.exists():
    uploads_path.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")

# ============ ROUTES ============

# API Version 1
API_PREFIX = settings.API_PREFIX

app.include_router(auth_router, prefix=API_PREFIX)
app.include_router(users_router, prefix=API_PREFIX)
app.include_router(enclos_router, prefix=API_PREFIX)
app.include_router(vaccination_router, prefix=API_PREFIX)
app.include_router(compost_router, prefix=API_PREFIX)
app.include_router(bovins_router, prefix=API_PREFIX)
app.include_router(ovins_router, prefix=API_PREFIX)
app.include_router(caprins_router, prefix=API_PREFIX)
app.include_router(avicoles_router, prefix=API_PREFIX)
app.include_router(piscicoles_router, prefix=API_PREFIX)
app.include_router(entomoculture_router, prefix=API_PREFIX)
app.include_router(accounting_router, prefix=API_PREFIX)
app.include_router(dashboard_router, prefix=API_PREFIX)
app.include_router(predictions_router, prefix=API_PREFIX)
app.include_router(alerts_router, prefix=API_PREFIX)
app.include_router(exports_router, prefix=API_PREFIX)
app.include_router(water_quality_router, prefix=API_PREFIX)
app.include_router(video_router, prefix=API_PREFIX)
app.include_router(weather_router, prefix=API_PREFIX)
app.include_router(bea_router, prefix=API_PREFIX)
app.include_router(blockchain_router, prefix=API_PREFIX)
app.include_router(odoni_router, prefix=API_PREFIX)
app.include_router(apiary_router, prefix=API_PREFIX)
app.include_router(pesees_router, prefix=API_PREFIX)
app.include_router(experimental_router, prefix=API_PREFIX)


# ============ ROUTES PUBLIQUES ============

@app.get("/", tags=["Public"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "running",
        "docs": "/docs" if settings.ENVIRONMENT != "production" else None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Public"])
async def health_check():
    """Vérification de l'état de santé de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION
    }


@app.get("/ready", tags=["Public"])
async def readiness_check():
    """Vérification que l'API est prête à recevoir du trafic"""
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat()
    }


# ============ GESTION DES EXCEPTIONS ============

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestionnaire global des exceptions non capturées"""
    logger.error(
        f"Exception non gérée: {type(exc).__name__}: {str(exc)}",
        exc_info=exc,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else None
        }
    )
    
    if settings.ENVIRONMENT == "development":
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": 500,
                    "message": str(exc),
                    "type": type(exc).__name__,
                }
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": 500,
                    "message": "Une erreur interne est survenue. Veuillez réessayer plus tard.",
                }
            }
        )


# ============ INFORMATION SUR LES ENDPOINTS ============

@app.on_event("startup")
async def startup_event():
    """Événement au démarrage - affiche les endpoints disponibles"""
    logger.info("=" * 50)
    logger.info("Endpoints disponibles:")
    logger.info(f"  API Prefix: {API_PREFIX}")
    logger.info(f"  Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"  ReDoc: http://{settings.HOST}:{settings.PORT}/redoc")
    logger.info("=" * 50)
    
    # Compter les routes
    routes_count = len([route for route in app.routes if hasattr(route, "methods")])
    logger.info(f"Total routes chargées: {routes_count}")
    logger.info("=" * 50)


# ============ SI EXÉCUTION DIRECTE ============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1,
        log_level=settings.LOG_LEVEL.lower(),
    )