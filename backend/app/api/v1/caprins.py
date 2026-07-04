# backend/app/api/v1/caprins.py
"""
Routes de gestion des caprins
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...schemas.caprin import CaprinCreate, CaprinUpdate, CaprinResponse
from ...schemas.pesee import PeseeCreate, PeseeResponse
from ...schemas.animal import AnimalVenteCreate
from ...models.caprin import Caprin
from ...services.caprin_service import caprin_service
from ...services.animal_service import animal_service
from ...services.pesee_service import pesee_service
from ...api.dependencies.auth import can_read_caprins, can_write_caprins
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.animal import SexeEnum, StatutAnimalEnum
from ...models.user import User

router = APIRouter(prefix="/caprins", tags=["Caprins"])

@router.get("", response_model=PaginatedResponse[CaprinResponse])
async def get_caprins(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    race: Optional[str] = Query(None, description="Filtrer par race"),
    sexe: Optional[str] = Query(None, description="Filtrer par sexe"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    production_type: Optional[str] = Query(None, description="Type de production (viande/reproduction)"),
    statut: Optional[str] = Query(None, description="Filtrer par statut"),
    search: Optional[str] = Query(None, description="Recherche par identification ou race"),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir la liste des caprins avec pagination et filtres
    """
    # Récupérer tous les caprins avec les filtres pour le comptage
    all_caprins = await caprin_service.get_all_caprins_with_filters(
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
    
    total = len([
        caprin for caprin in all_caprins
        if caprin.statut in (StatutAnimalEnum.VIVANT, StatutAnimalEnum.TRANSFERE)
    ])
    
    items = []
    for caprin in all_caprins:
        dernier_poids = await animal_service.get_last_weight(db, caprin.id)
        
        enclos_name = await animal_service.get_enclos_name_by_id(db, caprin.enclos_id)
        
        caprin_response = CaprinResponse(
            id=caprin.id,
            identification=caprin.identification,
            type_espece=caprin.type_espece,
            race=caprin.race,
            sexe=caprin.sexe,
            date_naissance=caprin.date_naissance,
            date_arrivee=caprin.date_arrivee,
            provenance=caprin.provenance,
            prix_achat=caprin.prix_achat,
            enclos_id=caprin.enclos_id,            
            enclos_name=enclos_name,
            statut=caprin.statut,
            photo_url=caprin.photo_url,
            notes=caprin.notes,
            production_viande=caprin.production_viande,
            production_reproduction=caprin.production_reproduction,
            dernier_poids=dernier_poids,
            age_mois=caprin.age_mois,
            created_at=caprin.created_at,
            updated_at=caprin.updated_at,
            # Nouveaux champs de vente
            prix_vente=caprin.prix_vente,
            date_vente=caprin.date_vente,
            client_acheteur=caprin.client_acheteur,
            note_vente=caprin.note_vente
        )
        items.append(caprin_response)
    
    return PaginatedResponse.create(
        items=items,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{caprin_id}", response_model=CaprinResponse)
async def get_caprin(
    caprin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir un caprin par son ID
    """
    caprin = await caprin_service.get_caprin(db, caprin_id)
    if not caprin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Caprin non trouvé"
        )
    
    dernier_poids = await animal_service.get_last_weight(db, caprin.id)
    enclos_name = await animal_service.get_enclos_name_by_id(db, caprin.enclos_id)
    
    return CaprinResponse(
        id=caprin.id,
        identification=caprin.identification,
        type_espece=caprin.type_espece,
        race=caprin.race,
        sexe=caprin.sexe,
        date_naissance=caprin.date_naissance,
        date_arrivee=caprin.date_arrivee,
        provenance=caprin.provenance,
        prix_achat=caprin.prix_achat,
        enclos_id=caprin.enclos_id,        
        enclos_name=enclos_name,
        statut=caprin.statut,
        photo_url=caprin.photo_url,
        notes=caprin.notes,
        production_viande=caprin.production_viande,
        production_reproduction=caprin.production_reproduction,
        dernier_poids=dernier_poids,
        age_mois=caprin.age_mois,
        created_at=caprin.created_at,
        updated_at=caprin.updated_at,
        prix_vente=caprin.prix_vente,
        date_vente=caprin.date_vente,
        client_acheteur=caprin.client_acheteur,
        note_vente=caprin.note_vente
    )


@router.post("/create", response_model=CaprinResponse, status_code=status.HTTP_201_CREATED)
async def create_caprin(
    caprin_data: CaprinCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_caprins),
):
    """
    Créer un nouveau caprin
    """
    caprin, error = await caprin_service.create_caprin(db, caprin_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    dernier_poids = await animal_service.get_last_weight(db, caprin.id)
    enclos_name = await animal_service.get_enclos_name_by_id(db, caprin.enclos_id)
    
    return CaprinResponse(
        id=caprin.id,
        identification=caprin.identification,
        type_espece=caprin.type_espece,
        race=caprin.race,
        sexe=caprin.sexe,
        date_naissance=caprin.date_naissance,
        date_arrivee=caprin.date_arrivee,
        provenance=caprin.provenance,
        prix_achat=caprin.prix_achat,
        enclos_id=caprin.enclos_id,
        enclos_name=enclos_name,
        statut=caprin.statut,
        photo_url=caprin.photo_url,
        notes=caprin.notes,
        production_viande=caprin.production_viande,
        production_reproduction=caprin.production_reproduction,
        dernier_poids=dernier_poids,
        age_mois=caprin.age_mois,
        created_at=caprin.created_at,
        updated_at=caprin.updated_at,
        prix_vente=caprin.prix_vente,
        date_vente=caprin.date_vente,
        client_acheteur=caprin.client_acheteur,
        note_vente=caprin.note_vente
    )


@router.put("/update/{caprin_id}", response_model=CaprinResponse)
async def update_caprin(
    caprin_id: int,
    caprin_data: CaprinUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_caprins),
):
    """
    Mettre à jour un caprin
    """
    caprin, error = await caprin_service.update_caprin(db, caprin_id, caprin_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    dernier_poids = await animal_service.get_last_weight(db, caprin.id)
    enclos_name = await animal_service.get_enclos_name_by_id(db, caprin.enclos_id)
    
    return CaprinResponse(
        id=caprin.id,
        identification=caprin.identification,
        type_espece=caprin.type_espece,
        race=caprin.race,
        sexe=caprin.sexe,
        date_naissance=caprin.date_naissance,
        date_arrivee=caprin.date_arrivee,
        provenance=caprin.provenance,
        prix_achat=caprin.prix_achat,
        enclos_id=caprin.enclos_id,        
        enclos_name=enclos_name,
        statut=caprin.statut,
        photo_url=caprin.photo_url,
        notes=caprin.notes,
        production_viande=caprin.production_viande,
        production_reproduction=caprin.production_reproduction,
        dernier_poids=dernier_poids,
        age_mois=caprin.age_mois,
        created_at=caprin.created_at,
        updated_at=caprin.updated_at,
        prix_vente=caprin.prix_vente,
        date_vente=caprin.date_vente,
        client_acheteur=caprin.client_acheteur,
        note_vente=caprin.note_vente
    )


@router.get("/{caprin_id}/pesees", response_model=List[PeseeResponse])
async def get_caprin_pesees(
    caprin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir l'historique des pesées d'un caprin
    """
    caprin = await caprin_service.get_caprin(db, caprin_id)
    if not caprin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Caprin non trouvé"
        )
    return caprin.pesees


@router.post("/{caprin_id}/pesees", response_model=PeseeResponse, status_code=status.HTTP_201_CREATED)
async def add_caprin_pesee(
    caprin_id: int,
    pesee_data: PeseeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_caprins),
):
    """
    Ajouter une pesée pour un caprin
    """
    caprin = await caprin_service.get_caprin(db, caprin_id)
    if not caprin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Caprin non trouvé"
        )
    
    pesee_data.animal_id = caprin_id
    pesee = await pesee_service.create_pesee(db, pesee_data, current_user.id)
    return pesee


@router.get("/{caprin_id}/croissance")
async def get_caprin_growth(
    caprin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir la courbe de croissance d'un caprin
    """
    caprin = await caprin_service.get_caprin(db, caprin_id)
    if not caprin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Caprin non trouvé"
        )
    
    progression = await caprin_service.get_weight_progression(db, caprin_id)
    return {"progression": progression, "animal": caprin.identification}


@router.get("/reproduction/chevres")
async def get_pregnant_females(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir les chèvres gestantes
    """
    females = await caprin_service.get_pregnant_females(db, enclos_id)
    return {"count": len(females), "females": females}


@router.get("/reproduction/boucs")
async def get_breeding_males(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir les boucs reproducteurs
    """
    males = await caprin_service.get_breeding_males(db, enclos_id)
    return {"count": len(males), "males": males}


@router.get("/jeunes/chevreaux")
async def get_kids(
    age_days_max: int = Query(90, description="Âge maximum en jours"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir les chevreaux
    """
    kids = await caprin_service.get_kids(db, age_days_max, enclos_id)
    return {"count": len(kids), "kids": kids}


@router.post("/reproduction/chevrotage")
async def record_kidding(
    request_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_caprins),
):
    """
    Enregistrer une mise bas (chevrotage)
    """
    doe_id = request_data.get("doe_id")
    buck_id = request_data.get("buck_id")
    kids_data = request_data.get("kids_data", [])
    
    if not doe_id or not kids_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID de la chèvre et données des chevreaux requis"
        )
    
    kids, error = await caprin_service.record_kidding(
        db, doe_id, buck_id, kids_data, current_user.id
    )
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {"message": f"Chevrotage enregistré avec succès", "kids": len(kids)}


@router.get("/stats/global")
async def get_caprin_stats(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir des statistiques globales sur les caprins
    """
    stats = await caprin_service.get_caprin_stats(db, enclos_id)
    return stats


# === NOUVELLE ROUTE POUR ENREGISTRER UNE VENTE ===
@router.post("/{caprin_id}/vente", response_model=CaprinResponse, status_code=status.HTTP_200_OK)
async def enregistrer_vente_caprin(
    caprin_id: int,
    vente_data: AnimalVenteCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_caprins),
):
    """
    Enregistrer la vente d'un caprin avec le prix, le client et la date
    """
    caprin, error = await animal_service.enregistrer_vente(
        db, caprin_id, 'VENTE', 'caprin', vente_data, current_user.id
    )
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Récupérer les informations complémentaires
    dernier_poids = await animal_service.get_last_weight(db, caprin.id)
    enclos = await animal_service.get_enclos_name_by_id(db, caprin.enclos_id)
    enclos_name = enclos.name if enclos else None
    
    # Créer la réponse
    return CaprinResponse(
        id=caprin.id,
        identification=caprin.identification,
        type_espece=caprin.type_espece,
        race=caprin.race,
        sexe=caprin.sexe,
        date_naissance=caprin.date_naissance,
        date_arrivee=caprin.date_arrivee,
        provenance=caprin.provenance,
        prix_achat=caprin.prix_achat,
        enclos_name=enclos_name,
        statut=caprin.statut,
        photo_url=caprin.photo_url,
        notes=caprin.notes,
        production_viande=caprin.production_viande,
        production_reproduction=caprin.production_reproduction,
        dernier_poids=dernier_poids,
        age_mois=caprin.age_mois,
        created_at=caprin.created_at,
        updated_at=caprin.updated_at,
        # Nouveaux champs
        prix_vente=caprin.prix_vente,
        date_vente=caprin.date_vente,
        client_acheteur=caprin.client_acheteur,
        note_vente=caprin.note_vente
    )


# === NOUVELLE ROUTE POUR LES STATISTIQUES DE VENTE ===
@router.get("/ventes/stats")
async def get_ventes_statistiques(
    date_debut: Optional[date] = Query(None, description="Date de début pour filtrer les ventes"),
    date_fin: Optional[date] = Query(None, description="Date de fin pour filtrer les ventes"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir des statistiques sur les ventes de caprins
    """
    stats = await caprin_service.get_ventes_stats(db, date_debut, date_fin)
    return stats


# === NOUVELLE ROUTE POUR RÉCUPÉRER LES CAPRINS VENDUS ===
@router.get("/ventes/liste", response_model=List[CaprinResponse])
async def get_caprins_vendus(
    date_debut: Optional[date] = Query(None, description="Date de début pour filtrer les ventes"),
    date_fin: Optional[date] = Query(None, description="Date de fin pour filtrer les ventes"),
    client: Optional[str] = Query(None, description="Filtrer par client"),
    skip: int = Query(0, ge=0, description="Nombre d'éléments à sauter"),
    limit: int = Query(100, ge=1, le=1000, description="Nombre d'éléments à retourner"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_caprins),
):
    """
    Obtenir la liste des caprins vendus avec filtres
    """
    # Récupérer les caprins vendus
    stmt = select(Caprin).where(Caprin.prix_vente.isnot(None))
    
    if date_debut:
        stmt = stmt.where(Caprin.date_vente >= date_debut)
    if date_fin:
        stmt = stmt.where(Caprin.date_vente <= date_fin)
    if client:
        stmt = stmt.where(Caprin.client_acheteur.ilike(f"%{client}%"))
    
    stmt = stmt.order_by(Caprin.date_vente.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    caprins = result.scalars().all()
    
    # Construire les réponses
    items = []
    for caprin in caprins:
        dernier_poids = await animal_service.get_last_weight(db, caprin.id)
        enclos = await animal_service.get_enclos_name_by_id(db, caprin.enclos_id)
        enclos_name = enclos.name if enclos else None
        
        caprin_response = CaprinResponse(
            id=caprin.id,
            identification=caprin.identification,
            type_espece=caprin.type_espece,
            race=caprin.race,
            sexe=caprin.sexe,
            date_naissance=caprin.date_naissance,
            date_arrivee=caprin.date_arrivee,
            provenance=caprin.provenance,
            prix_achat=caprin.prix_achat,
            enclos_name=enclos_name,
            statut=caprin.statut,
            photo_url=caprin.photo_url,
            notes=caprin.notes,
            production_viande=caprin.production_viande,
            production_reproduction=caprin.production_reproduction,
            dernier_poids=dernier_poids,
            age_mois=caprin.age_mois,
            created_at=caprin.created_at,
            updated_at=caprin.updated_at,
            prix_vente=caprin.prix_vente,
            date_vente=caprin.date_vente,
            client_acheteur=caprin.client_acheteur,
            note_vente=caprin.note_vente
        )
        items.append(caprin_response)
    
    return items