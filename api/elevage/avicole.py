from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import pandas as pd
import os

# Import des dépendances
from models import get_db, add_object_async
from models.elevage.avicole import (
    Volaille, 
    ControlePonte, 
    PerformanceCroissanceAvicole,
)
from utils.security import get_current_user
from schemas.elevage.avicole import (
    VolailleCreate,
    VolailleUpdate,
    ControlePonteCreate,
    ControlePonteUpdate,
    PerformanceCroissanceCreate
)
from api import check_permissions

router = APIRouter(
    prefix="/api/avicole",
    tags=["avicole"],
    responses={404: {"description": "Not found"}},
)

# ==============================================
# Routes pour la gestion des volailles
# ==============================================

@router.post("/volailles/", response_model=VolailleCreate)
async def create_volaille(
    volaille: VolailleCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer une nouvelle volaille"""
    check_permissions(db, current_user)
    
    try:
        db_volaille = Volaille(**volaille.model_dump())
        await add_object_async(db, db_volaille)
        return db_volaille
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la création : {str(e)}"
        )

@router.get("/volailles/", response_model=List[VolailleCreate])
async def read_volailles(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lister toutes les volailles"""
    check_permissions(db, current_user, required_roles=['admin', 'avicole_manager', 'avicole_technicien'])
    
    volailles = db.query(Volaille).offset(skip).limit(limit).all()
    return volailles

@router.get("/volailles/{volaille_id}", response_model=VolailleCreate)
async def read_volaille(
    volaille_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtenir les détails d'une volaille spécifique"""
    check_permissions(db, current_user, required_roles=['admin', 'avicole_manager', 'avicole_technicien'])
    
    volaille = db.query(Volaille).filter(Volaille.id == volaille_id).first()
    if volaille is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Volaille non trouvée"
        )
    return volaille

@router.put("/volailles/{volaille_id}", response_model=VolailleCreate)
async def update_volaille(
    volaille_id: int,
    volaille: VolailleUpdate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mettre à jour une volaille"""
    check_permissions(db, current_user)
    
    db_volaille = db.query(Volaille).filter(Volaille.id == volaille_id).first()
    if db_volaille is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Volaille non trouvée"
        )
    
    try:
        update_data = volaille.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_volaille, key, value)
        
        db.commit()
        db.refresh(db_volaille)
        return db_volaille
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la mise à jour : {str(e)}"
        )

@router.delete("/volailles/{volaille_id}")
async def delete_volaille(
    volaille_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Supprimer une volaille"""
    check_permissions(db, current_user, required_roles=['admin'])
    
    db_volaille = db.query(Volaille).filter(Volaille.id == volaille_id).first()
    if db_volaille is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Volaille non trouvée"
        )
    
    try:
        db.delete(db_volaille)
        db.commit()
        return {"message": "Volaille supprimée avec succès"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la suppression : {str(e)}"
        )

# ==============================================
# Routes pour la gestion des contrôles de ponte
# ==============================================

@router.post("/controles-ponte/", response_model=ControlePonteCreate)
async def create_controle_ponte(
    controle: ControlePonteCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer un nouveau contrôle de ponte"""
    check_permissions(db, current_user)
    
    try:
        db_controle = ControlePonte(**controle.model_dump())
        await add_object_async(db, db_controle)
        return db_controle
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la création : {str(e)}"
        )

@router.get("/controles-ponte/", response_model=List[ControlePonteCreate])
async def read_controles_ponte(
    lot_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lister les contrôles de ponte"""
    check_permissions(db, current_user, required_roles=['admin', 'avicole_manager', 'avicole_technicien'])
    
    query = db.query(ControlePonte)
    if lot_id:
        query = query.filter(ControlePonte.lot_id == lot_id)
    
    controles = query.offset(skip).limit(limit).all()
    return controles

@router.put("/controles-ponte/{controle_id}", response_model=ControlePonteCreate)
async def update_controle_ponte(
    controle_id: int,
    controle: ControlePonteUpdate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mettre à jour un contrôle de ponte"""
    check_permissions(db, current_user)
    
    db_controle = db.query(ControlePonte).filter(ControlePonte.id == controle_id).first()
    if db_controle is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contrôle de ponte non trouvé"
        )
    
    try:
        update_data = controle.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_controle, key, value)
        
        db.commit()
        db.refresh(db_controle)
        return db_controle
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la mise à jour : {str(e)}"
        )

# ==============================================
# Routes pour la gestion des performances de croissance
# ==============================================

@router.post("/performances-croissance/", response_model=PerformanceCroissanceCreate)
async def create_performance_croissance(
    performance: PerformanceCroissanceCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer une nouvelle performance de croissance"""
    check_permissions(db, current_user)
    
    try:
        db_performance = PerformanceCroissanceAvicole(**performance.model_dump())
        await add_object_async(db, db_performance)
        return db_performance
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la création : {str(e)}"
        )

@router.get("/performances-croissance/", response_model=List[PerformanceCroissanceCreate])
async def read_performances_croissance(
    lot_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lister les performances de croissance"""
    check_permissions(db, current_user, required_roles=['admin', 'avicole_manager', 'avicole_technicien'])
    
    query = db.query(PerformanceCroissanceAvicole)
    if lot_id:
        query = query.filter(PerformanceCroissanceAvicole.lot_id == lot_id)
    
    performances = query.offset(skip).limit(limit).all()
    return performances

# ==============================================
# Routes pour les analyses et prédictions
# ==============================================

@router.get("/analyses/alertes")
async def get_alertes_avicoles(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtenir les alertes pour l'élevage avicole"""
    check_permissions(db, current_user)
    
    from machine_learning.analyse.elevage.avicole import AvicoleAnalyzer
    analyzer = AvicoleAnalyzer(db)
    alertes = await analyzer.analyze_farm()
    return alertes

@router.get("/predictions/ponte")
async def predict_ponte(
    volaille_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Prédire les performances de ponte"""
    check_permissions(db, current_user)
    
    from machine_learning.prediction.elevage.avicole import AvicolePredictor
    predictor = AvicolePredictor(db)
    await predictor.prepare_training_data()
    volaille = db.query(Volaille).filter(Volaille.id == volaille_id).first()
    
    if not volaille:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Volaille non trouvée"
        )
    
    prediction = await predictor.predict_ponte(volaille.__dict__)
    return prediction


@router.get("/export/data")
async def export_data(
    file_type: str = Query("csv", description="Type de fichier (csv ou excel)", regex="^(csv|excel)$"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Exporter des données avicoles vers un fichier (CSV ou Excel)"""
    check_permissions(db, current_user, required_roles=['admin'])
    
    try:
        volailles = db.query(Volaille).all()
        df = pd.DataFrame([v.__dict__ for v in volailles])
        
        # Supprimer la colonne '_sa_instance_state' ajoutée par SQLAlchemy
        if '_sa_instance_state' in df.columns:
            df.drop('_sa_instance_state', axis=1, inplace=True)
        
        file_path = f"temp_export_avicole.{'csv' if file_type == 'csv' else 'xlsx'}"
        
        if file_type == "csv":
            df.to_csv(file_path, index=False)
            media_type = 'text/csv'
        else:
            df.to_excel(file_path, index=False)
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        return FileResponse(
            file_path,
            media_type=media_type,
            filename=f"export_avicole_{datetime.now().date()}.{'csv' if file_type == 'csv' else 'xlsx'}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération de l'export : {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)