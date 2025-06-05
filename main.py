from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from lifespan import lifespan, train_all_models_sync, predictors
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Dict, Any
from models import engine, Base, get_db_session
from utils.security import get_current_user

from api import check_permissions, router
import api.users as users
import api.auth as auth

import api.agricole.concombre as concombre
import api.agricole.salade as salade
import api.agricole.oignon as oignon
import api.agricole.mais as mais

import api.elevage.avicole as avicole
import api.elevage.bovin as bovin
import api.elevage.caprin as caprin
import api.elevage.ovin as ovin
import api.elevage.piscicole as piscicole

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Erreur lors de la création des tables de la base de données: {str(e)}")
    raise

# Initialisation de l'app FastAPI
app = FastAPI(
    title="Mon API FastAPI",
    description="Un point d'entrée simple pour FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration du CORS (autoriser certaines origines seulement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplacer par vos origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route principale
@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API FastAPI principale 🚀"}

# Route avec un paramètre dynamique
@app.get("/hello/{name}")
async def greet(name: str):
    return {"message": f"Bonjour, {name}!"}

# Ajouter les routeurs des différents modules
app.include_router(router, prefix="", tags=["Number"])
app.include_router(auth.router, prefix="", tags=["Auth"])
app.include_router(users.router, prefix="", tags=["User"])

app.include_router(avicole.router, prefix="/elevage/avicole", tags=["Avicole"])
app.include_router(bovin.router, prefix="/elevage/bovin", tags=["Bovin"])
app.include_router(caprin.router, prefix="/elevage/caprin", tags=["Caprin"])
app.include_router(ovin.router, prefix="/elevage/ovin", tags=["Ovin"])
app.include_router(piscicole.router, prefix="/elevage/piscicole", tags=["Piscicole"])

# Ajouter les routeurs des modules agricoles
# app.include_router(concombre.router, prefix="/agricole/concombre", tags=["Concombre"])
# app.include_router(salade.router, prefix="/agricole/salade", tags=["Salade"])
# app.include_router(oignon.router, prefix="/agricole/oignon", tags=["Oignon"])
# app.include_router(mais.router, prefix="/agricole/mais", tags=["Mais"])

@app.get("/force-training")
async def force_training(
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Endpoint pour forcer un entraînement immédiat"""
    # Vérification des permissions - seulement admin ou ml_manager
    check_permissions(db, current_user)
    
    try:
        success = await train_all_models_sync()
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Échec de l'entraînement des modèles"
            )
        return {"message": "Entraînement des modèles forcé avec succès"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'entraînement: {str(e)}"
        )

@app.get("/model-status")
async def model_status(
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Retourne l'état des modèles"""
    # Vérification des permissions - lecture seule pour plus de rôles
    check_permissions(db, current_user)
    
    try:
        status: Dict[str, Any] = {}
        for name, predictor in predictors.items():
            status[name] = {
                "last_trained": datetime.now().isoformat(),  # À remplacer par un vrai timestamp
                "status": "ready" if predictor else "not_ready",
                "loaded": predictor is not None,
                "model_type": type(predictor).__name__ if predictor else "N/A"
            }
        return status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération du statut: {str(e)}"
        )
    
# Gestion des erreurs personnalisées
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Données de requête invalides", "errors": exc.errors()},
    )

@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Ressource non trouvée"}
    )