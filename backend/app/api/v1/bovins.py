# backend/app/api/v1/bovins.py
"""
Routes de gestion des bovins
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from app.models.bovin import Bovin
from app.schemas.animal import AnimalVenteCreate

from ...database import get_db
from ...schemas.bovin import BovinCreate, BovinUpdate, BovinResponse
from ...schemas.pesee import PeseeCreate, PeseeResponse
from ...services.animal_service import animal_service
from ...services.bovin_service import bovin_service
from ...services.pesee_service import pesee_service
from ...api.dependencies.auth import (
    can_read_bovins,
    can_write_bovins
)
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User
from ...models.animal import SexeEnum, StatutAnimalEnum

router = APIRouter(prefix="/bovins", tags=["Bovins"])


@router.get("", response_model=PaginatedResponse[BovinResponse])
async def get_bovins(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    race: Optional[str] = Query(None, description="Filtrer par race"),
    sexe: Optional[str] = Query(None, description="Filtrer par sexe"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    production_type: Optional[str] = Query(None, description="Type de production (lait/viande/reproduction)"),
    statut: Optional[str] = Query(None, description="Filtrer par statut"),
    search: Optional[str] = Query(None, description="Recherche par identification ou race"),
    current_user: User = Depends(can_read_bovins),
):
    """
    Obtenir la liste des bovins avec pagination et filtres
    """
    # Récupérer tous les bovins avec les filtres pour le comptage
    all_bovins = await bovin_service.get_all_bovins_with_filters(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        race=race,
        production_type=production_type,
        enclos_id=enclos_id,
        sexe=sexe,
        statut=statut,
        search=search
    )
    
    total = len(all_bovins)
        
    # Construire les réponses avec le nom de l'enclos
    items = []
    for bovin in all_bovins:
        # Récupérer le dernier poids
        dernier_poids = await animal_service.get_last_weight(db, bovin.id)
        
        # Récupérer le nom de l'enclos
        enclos_name = await animal_service.get_enclos_name_by_id(db, bovin.enclos_id)
        
        # Créer la réponse
        bovin_response = BovinResponse(
            id=bovin.id,
            animal_id=bovin.age_mois,
            identification=bovin.identification,
            type_espece=bovin.type_espece,
            race=bovin.race,
            sexe=bovin.sexe,
            date_naissance=bovin.date_naissance,
            date_arrivee=bovin.date_arrivee,
            provenance=bovin.provenance,
            prix_achat=bovin.prix_achat,
            enclos_id=bovin.enclos_id,
            enclos_name=enclos_name,
            statut=bovin.statut,
            photo_url=bovin.photo_url,
            notes=bovin.notes,
            production_laitiere=bovin.production_laitiere,
            production_viande=bovin.production_viande,
            production_reproduction=bovin.production_reproduction,
            lactation_en_cours=bovin.lactation_en_cours,
            production_lait_quotidienne=bovin.production_lait_quotidienne,
            dernier_poids=dernier_poids,
            age_mois=bovin.age_mois,
            created_at=bovin.created_at,
            updated_at=bovin.updated_at
        )
        items.append(bovin_response)
    
    return PaginatedResponse.create(
        items=items,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )

@router.get("/{bovin_id}", response_model=BovinResponse)
async def get_bovin(
    bovin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_bovins),
):
    """
    Obtenir un bovin par son ID
    """
    bovin = await bovin_service.get_bovin(db, bovin_id)
    if not bovin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bovin non trouvé"
        )
    
    # Récupérer le dernier poids
    dernier_poids = await animal_service.get_last_weight(db, bovin.id)
    
    # Récupérer le nom de l'enclos
    enclos_name = await animal_service.get_enclos_name_by_id(db, bovin.enclos_id)
    
    # Créer la réponse
    return BovinResponse(
        id=bovin.id,
        identification=bovin.identification,
        type_espece=bovin.type_espece,
        race=bovin.race,
        sexe=bovin.sexe,
        date_naissance=bovin.date_naissance,
        date_arrivee=bovin.date_arrivee,
        provenance=bovin.provenance,
        prix_achat=bovin.prix_achat,
        enclos_id=bovin.enclos_id,
        enclos_name=enclos_name,
        statut=bovin.statut,
        photo_url=bovin.photo_url,
        notes=bovin.notes,
        production_laitiere=bovin.production_laitiere,
        production_viande=bovin.production_viande,
        production_reproduction=bovin.production_reproduction,
        lactation_en_cours=bovin.lactation_en_cours,
        production_lait_quotidienne=bovin.production_lait_quotidienne,
        dernier_poids=dernier_poids,
        age_mois=bovin.age_mois,
        created_at=bovin.created_at,
        updated_at=bovin.updated_at
    )


@router.post("/create", response_model=BovinResponse, status_code=status.HTTP_201_CREATED)
async def create_bovin(
    bovin_data: BovinCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_bovins),
):
    """
    Créer un nouveau bovin
    """
    bovin, error = await bovin_service.create_bovin(db, bovin_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Récupérer le dernier poids
    dernier_poids = await animal_service.get_last_weight(db, bovin.id)
    
    # Récupérer le nom de l'enclos
    enclos_name = await animal_service.get_enclos_by_name(db, bovin.enclos_id)
    
    # Créer la réponse avec tous les champs requis
    return BovinResponse(
        id=bovin.id,
        identification=bovin.identification,
        type_espece=bovin.type_espece,
        race=bovin.race,
        sexe=bovin.sexe,
        date_naissance=bovin.date_naissance,
        date_arrivee=bovin.date_arrivee,
        provenance=bovin.provenance,
        prix_achat=bovin.prix_achat,
        enclos_id=bovin.enclos_id,
        enclos_name=enclos_name,
        statut=bovin.statut,
        photo_url=bovin.photo_url,
        notes=bovin.notes,
        production_laitiere=bovin.production_laitiere,
        production_viande=bovin.production_viande,
        production_reproduction=bovin.production_reproduction,
        lactation_en_cours=bovin.lactation_en_cours,
        production_lait_quotidienne=bovin.production_lait_quotidienne,
        dernier_poids=dernier_poids,
        age_mois=bovin.age_mois,
        created_at=bovin.created_at,
        updated_at=bovin.updated_at
    )


@router.put("/update/{bovin_id}", response_model=BovinResponse)
async def update_bovin(
    bovin_id: int,
    bovin_data: BovinUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_bovins),
):
    """
    Mettre à jour un bovin
    """
    bovin, error = await bovin_service.update_bovin(db, bovin_id, bovin_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Récupérer le dernier poids
    dernier_poids = await animal_service.get_last_weight(db, bovin.id)
    
    # Récupérer le nom de l'enclos
    enclos_name = await animal_service.get_enclos_name_by_id(db, bovin.enclos_id)
    
    # Créer la réponse
    return BovinResponse(
        id=bovin.id,
        identification=bovin.identification,
        type_espece=bovin.type_espece,
        race=bovin.race,
        sexe=bovin.sexe,
        date_naissance=bovin.date_naissance,
        date_arrivee=bovin.date_arrivee,
        provenance=bovin.provenance,
        prix_achat=bovin.prix_achat,
        enclos_id=bovin.enclos_id,        
        enclos_name=enclos_name,
        statut=bovin.statut,
        photo_url=bovin.photo_url,
        notes=bovin.notes,
        production_laitiere=bovin.production_laitiere,
        production_viande=bovin.production_viande,
        production_reproduction=bovin.production_reproduction,
        lactation_en_cours=bovin.lactation_en_cours,
        production_lait_quotidienne=bovin.production_lait_quotidienne,
        dernier_poids=dernier_poids,
        age_mois=bovin.age_mois,
        created_at=bovin.created_at,
        updated_at=bovin.updated_at
    )


@router.get("/{bovin_id}/pesees", response_model=List[PeseeResponse])
async def get_bovin_pesees(
    bovin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_bovins),
):
    """
    Obtenir l'historique des pesées d'un bovin
    """
    bovin = await bovin_service.get_bovin(db, bovin_id)
    if not bovin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bovin non trouvé"
        )
    # Utiliser la relation définie dans le modèle Animal
    return bovin.pesees


@router.post("/{bovin_id}/pesees", response_model=PeseeResponse, status_code=status.HTTP_201_CREATED)
async def add_bovin_pesee(
    bovin_id: int,
    pesee_data: PeseeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_bovins),
):
    """
    Ajouter une pesée pour un bovin
    """
    bovin = await bovin_service.get_bovin(db, bovin_id)
    if not bovin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bovin non trouvé"
        )
    
    # S'assurer que l'animal_id est correct
    pesee_data.animal_id = bovin_id
    pesee = await pesee_service.create_pesee(db, pesee_data, current_user.id)
    return pesee


@router.get("/{bovin_id}/croissance")
async def get_bovin_growth(
    bovin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_bovins),
):
    """
    Obtenir la courbe de croissance d'un bovin
    """
    bovin = await bovin_service.get_bovin(db, bovin_id)
    if not bovin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bovin non trouvé"
        )
    
    # Récupérer les pesées pour la courbe de croissance
    progression = await bovin_service.get_weight_progression(db, bovin_id)
    return {"progression": progression, "animal": bovin.identification}


@router.get("/stats/global")
async def get_bovin_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_bovins),
):
    """
    Obtenir des statistiques globales sur les bovins
    """
    stats = await bovin_service.get_bovin_stats(db)
    return stats


@router.get("/lactation/en-cours")
async def get_lactating_cows(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_bovins),
):
    """
    Obtenir les vaches en lactation
    """
    cows = await bovin_service.get_lactating_cows(db)
    # Convertir les objets Bovin en dictionnaires
    cows_data = [
        {
            "id": cow.id,
            "identification": cow.identification,
            "race": cow.race,
            "production_lait_quotidienne": cow.production_lait_quotidienne
        }
        for cow in cows
    ]
    return {"count": len(cows), "cows": cows_data}

# === NOUVELLE ROUTE POUR ENREGISTRER UNE VENTE ===
@router.post("/{bovin_id}/vente", response_model=BovinResponse, status_code=status.HTTP_200_OK)
async def enregistrer_vente_bovin(
    bovin_id: int,
    vente_data: AnimalVenteCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_bovins),
):
    """
    Enregistrer la vente d'un bovin avec le prix, le client et la date
    """
    bovin, error = await animal_service.enregistrer_vente(
        db, bovin_id, 'VENTE', 'bovin', vente_data, current_user.id
    )
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Récupérer les informations complémentaires
    dernier_poids = await animal_service.get_last_weight(db, bovin.id)
    enclos_name = await animal_service.get_enclos_name_by_id(db, bovin.enclos_id)
    
    # Créer la réponse
    return BovinResponse(
        id=bovin.id,
        identification=bovin.identification,
        type_espece=bovin.type_espece,
        race=bovin.race,
        sexe=bovin.sexe,
        date_naissance=bovin.date_naissance,
        date_arrivee=bovin.date_arrivee,
        provenance=bovin.provenance,
        prix_achat=bovin.prix_achat,
        enclos_name=enclos_name,
        statut=bovin.statut,
        photo_url=bovin.photo_url,
        notes=bovin.notes,
        production_laitiere=bovin.production_laitiere,
        production_viande=bovin.production_viande,
        production_reproduction=bovin.production_reproduction,
        lactation_en_cours=bovin.lactation_en_cours,
        production_lait_quotidienne=bovin.production_lait_quotidienne,
        dernier_poids=dernier_poids,
        age_mois=bovin.age_mois,
        created_at=bovin.created_at,
        updated_at=bovin.updated_at,
        # Nouveaux champs
        prix_vente=bovin.prix_vente,
        date_vente=bovin.date_vente,
        client_acheteur=bovin.client_acheteur,
        note_vente=bovin.note_vente
    )


# === NOUVELLE ROUTE POUR LES STATISTIQUES DE VENTE ===
@router.get("/ventes/stats")
async def get_ventes_statistiques(
    date_debut: Optional[date] = Query(None, description="Date de début pour filtrer les ventes"),
    date_fin: Optional[date] = Query(None, description="Date de fin pour filtrer les ventes"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_bovins),
):
    """
    Obtenir des statistiques sur les ventes de bovins
    """
    stats = await bovin_service.get_ventes_stats(db, date_debut, date_fin)
    return stats


# === NOUVELLE ROUTE POUR RÉCUPÉRER LES BOVINS VENDUS ===
@router.get("/ventes/liste", response_model=List[BovinResponse])
async def get_bovins_vendus(
    date_debut: Optional[date] = Query(None, description="Date de début pour filtrer les ventes"),
    date_fin: Optional[date] = Query(None, description="Date de fin pour filtrer les ventes"),
    client: Optional[str] = Query(None, description="Filtrer par client"),
    skip: int = Query(0, ge=0, description="Nombre d'éléments à sauter"),
    limit: int = Query(100, ge=1, le=1000, description="Nombre d'éléments à retourner"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_bovins),
):
    """
    Obtenir la liste des bovins vendus avec filtres
    """
    # Récupérer les bovins vendus
    stmt = select(Bovin).where(Bovin.prix_vente.isnot(None))
    
    if date_debut:
        stmt = stmt.where(Bovin.date_vente >= date_debut)
    if date_fin:
        stmt = stmt.where(Bovin.date_vente <= date_fin)
    if client:
        stmt = stmt.where(Bovin.client_acheteur.ilike(f"%{client}%"))
    
    stmt = stmt.order_by(Bovin.date_vente.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    bovins = result.scalars().all()
    
    # Construire les réponses
    items = []
    for bovin in bovins:
        dernier_poids = await animal_service.get_last_weight(db, bovin.id)
        enclos_name = await animal_service.get_enclos_name_by_id(db, bovin.enclos_id)
        
        bovin_response = BovinResponse(
            id=bovin.id,
            identification=bovin.identification,
            type_espece=bovin.type_espece,
            race=bovin.race,
            sexe=bovin.sexe,
            date_naissance=bovin.date_naissance,
            date_arrivee=bovin.date_arrivee,
            provenance=bovin.provenance,
            prix_achat=bovin.prix_achat,
            enclos_id=bovin.enclos_id,
            enclos_name=enclos_name,
            statut=bovin.statut,
            photo_url=bovin.photo_url,
            notes=bovin.notes,
            production_laitiere=bovin.production_laitiere,
            production_viande=bovin.production_viande,
            production_reproduction=bovin.production_reproduction,
            lactation_en_cours=bovin.lactation_en_cours,
            production_lait_quotidienne=bovin.production_lait_quotidienne,
            dernier_poids=dernier_poids,
            age_mois=bovin.age_mois,
            created_at=bovin.created_at,
            updated_at=bovin.updated_at,
            prix_vente=bovin.prix_vente,
            date_vente=bovin.date_vente,
            client_acheteur=bovin.client_acheteur,
            note_vente=bovin.note_vente
        )
        items.append(bovin_response)
    
    return items