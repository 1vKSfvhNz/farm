# backend/app/api/v1/ovins.py
"""
Routes de gestion des ovins
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from app.models.ovin import Ovin

from ...database import get_db
from ...schemas.ovin import OvinCreate, OvinUpdate, OvinResponse
from ...schemas.pesee import PeseeCreate, PeseeResponse
from ...schemas.animal import AnimalVenteCreate
from ...services.ovin_service import ovin_service
from ...services.animal_service import animal_service
from ...services.pesee_service import pesee_service
from ...api.dependencies.auth import can_read_ovins, can_write_ovins
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User
from ...models.animal import SexeEnum, StatutAnimalEnum

router = APIRouter(prefix="/ovins", tags=["Ovins"])


@router.get("", response_model=PaginatedResponse[OvinResponse])
async def get_ovins(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    race: Optional[str] = Query(None, description="Filtrer par race"),
    sexe: Optional[str] = Query(None, description="Filtrer par sexe"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    production_type: Optional[str] = Query(None, description="Type de production (viande/laine/reproduction)"),
    statut: Optional[str] = Query(None, description="Filtrer par statut"),
    search: Optional[str] = Query(None, description="Recherche par identification ou race"),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir la liste des ovins avec pagination et filtres
    """
    # Récupérer tous les ovins avec les filtres pour le comptage
    all_ovins = await ovin_service.get_all_ovins_with_filters(
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
        ovin for ovin in all_ovins
        if ovin.statut in (StatutAnimalEnum.VIVANT, StatutAnimalEnum.TRANSFERE)
    ])
    
    items = []
    for ovin in all_ovins:
        dernier_poids = await animal_service.get_last_weight(db, ovin.id)
        
        enclos_name = await animal_service.get_enclos_name_by_id(db, ovin.enclos_id)
        
        ovin_response = OvinResponse(
            id=ovin.id,
            identification=ovin.identification,
            type_espece=ovin.type_espece,
            race=ovin.race,
            sexe=ovin.sexe,
            date_naissance=ovin.date_naissance,
            date_arrivee=ovin.date_arrivee,
            provenance=ovin.provenance,
            prix_achat=ovin.prix_achat,
            enclos_id=ovin.enclos_id,
            enclos_name=enclos_name,
            statut=ovin.statut,
            photo_url=ovin.photo_url,
            notes=ovin.notes,
            production_viande=ovin.production_viande,
            production_reproduction=ovin.production_reproduction,
            production_laine=ovin.production_laine,
            qualite_laine=ovin.qualite_laine,
            dernier_poids=dernier_poids,
            age_mois=ovin.age_mois,
            created_at=ovin.created_at,
            updated_at=ovin.updated_at,
            # Nouveaux champs de vente
            prix_vente=ovin.prix_vente,
            date_vente=ovin.date_vente,
            client_acheteur=ovin.client_acheteur,
            note_vente=ovin.note_vente
        )
        items.append(ovin_response)
    
    return PaginatedResponse.create(
        items=items,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{ovin_id}", response_model=OvinResponse)
async def get_ovin(
    ovin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir un ovin par son ID
    """
    ovin = await ovin_service.get_ovin(db, ovin_id)
    if not ovin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ovin non trouvé"
        )
    
    dernier_poids = await animal_service.get_last_weight(db, ovin.id)
    enclos_name = await animal_service.get_enclos_name_by_id(db, ovin.enclos_id)
    
    return OvinResponse(
        id=ovin.id,
        identification=ovin.identification,
        type_espece=ovin.type_espece,
        race=ovin.race,
        sexe=ovin.sexe,
        date_naissance=ovin.date_naissance,
        date_arrivee=ovin.date_arrivee,
        provenance=ovin.provenance,
        prix_achat=ovin.prix_achat,
        enclos_id=ovin.enclos_id,
        enclos_name=enclos_name,
        statut=ovin.statut,
        photo_url=ovin.photo_url,
        notes=ovin.notes,
        production_viande=ovin.production_viande,
        production_reproduction=ovin.production_reproduction,
        production_laine=ovin.production_laine,
        qualite_laine=ovin.qualite_laine,
        dernier_poids=dernier_poids,
        age_mois=ovin.age_mois,
        created_at=ovin.created_at,
        updated_at=ovin.updated_at,
        prix_vente=ovin.prix_vente,
        date_vente=ovin.date_vente,
        client_acheteur=ovin.client_acheteur,
        note_vente=ovin.note_vente
    )


@router.post("/create", response_model=OvinResponse, status_code=status.HTTP_201_CREATED)
async def create_ovin(
    ovin_data: OvinCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_ovins),
):
    """
    Créer un nouvel ovin
    """
    ovin, error = await ovin_service.create_ovin(db, ovin_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    dernier_poids = await animal_service.get_last_weight(db, ovin.id)
    enclos_name = await animal_service.get_enclos_name_by_id(db, ovin.enclos_id)
    
    return OvinResponse(
        id=ovin.id,
        identification=ovin.identification,
        type_espece=ovin.type_espece,
        race=ovin.race,
        sexe=ovin.sexe,
        date_naissance=ovin.date_naissance,
        date_arrivee=ovin.date_arrivee,
        provenance=ovin.provenance,
        prix_achat=ovin.prix_achat,
        enclos_id=ovin.enclos_id,
        enclos_name=enclos_name,
        statut=ovin.statut,
        photo_url=ovin.photo_url,
        notes=ovin.notes,
        production_viande=ovin.production_viande,
        production_reproduction=ovin.production_reproduction,
        production_laine=ovin.production_laine,
        qualite_laine=ovin.qualite_laine,
        dernier_poids=dernier_poids,
        age_mois=ovin.age_mois,
        created_at=ovin.created_at,
        updated_at=ovin.updated_at,
        prix_vente=ovin.prix_vente,
        date_vente=ovin.date_vente,
        client_acheteur=ovin.client_acheteur,
        note_vente=ovin.note_vente
    )


@router.put("/update/{ovin_id}", response_model=OvinResponse)
async def update_ovin(
    ovin_id: int,
    ovin_data: OvinUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_ovins),
):
    """
    Mettre à jour un ovin
    """
    ovin, error = await ovin_service.update_ovin(db, ovin_id, ovin_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    dernier_poids = await animal_service.get_last_weight(db, ovin.id)
    enclos_name = await animal_service.get_enclos_name_by_id(db, ovin.enclos_id)
    
    return OvinResponse(
        id=ovin.id,
        identification=ovin.identification,
        type_espece=ovin.type_espece,
        race=ovin.race,
        sexe=ovin.sexe,
        date_naissance=ovin.date_naissance,
        date_arrivee=ovin.date_arrivee,
        provenance=ovin.provenance,
        prix_achat=ovin.prix_achat,
        enclos_id=ovin.enclos_id,
        enclos_name=enclos_name,
        statut=ovin.statut,
        photo_url=ovin.photo_url,
        notes=ovin.notes,
        production_viande=ovin.production_viande,
        production_reproduction=ovin.production_reproduction,
        production_laine=ovin.production_laine,
        qualite_laine=ovin.qualite_laine,
        dernier_poids=dernier_poids,
        age_mois=ovin.age_mois,
        created_at=ovin.created_at,
        updated_at=ovin.updated_at,
        prix_vente=ovin.prix_vente,
        date_vente=ovin.date_vente,
        client_acheteur=ovin.client_acheteur,
        note_vente=ovin.note_vente
    )


@router.get("/{ovin_id}/pesees", response_model=List[PeseeResponse])
async def get_ovin_pesees(
    ovin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir l'historique des pesées d'un ovin
    """
    ovin = await ovin_service.get_ovin(db, ovin_id)
    if not ovin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ovin non trouvé"
        )
    return ovin.pesees


@router.post("/{ovin_id}/pesees", response_model=PeseeResponse, status_code=status.HTTP_201_CREATED)
async def add_ovin_pesee(
    ovin_id: int,
    pesee_data: PeseeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_ovins),
):
    """
    Ajouter une pesée pour un ovin
    """
    ovin = await ovin_service.get_ovin(db, ovin_id)
    if not ovin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ovin non trouvé"
        )
    
    pesee_data.animal_id = ovin_id
    pesee = await pesee_service.create_pesee(db, pesee_data, current_user.id)
    return pesee


@router.get("/{ovin_id}/croissance")
async def get_ovin_growth(
    ovin_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir la courbe de croissance d'un ovin
    """
    ovin = await ovin_service.get_ovin(db, ovin_id)
    if not ovin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ovin non trouvé"
        )
    
    progression = await ovin_service.get_weight_progression(db, ovin_id)
    return {"progression": progression, "animal": ovin.identification}


@router.get("/reproduction/brebis")
async def get_ewes_for_breeding(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir les brebis aptes à la reproduction
    """
    ewes = await ovin_service.get_ewes_for_breeding(db, enclos_id)
    return {"count": len(ewes), "ewes": ewes}


@router.get("/reproduction/beliers")
async def get_rams(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir les béliers reproducteurs
    """
    rams = await ovin_service.get_rams(db, enclos_id)
    return {"count": len(rams), "rams": rams}


@router.post("/reproduction/agnelage")
async def record_lambing(
    request: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_ovins),
):
    """
    Enregistrer un agnelage
    """
    ewe_id = request.get("ewe_id")
    ram_id = request.get("ram_id")
    lambs_data = request.get("lambs_data", [])
    
    if not ewe_id or not lambs_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID de la brebis et données des agneaux requis"
        )
    
    lambs, error = await ovin_service.record_lambing(
        db, ewe_id, ram_id, lambs_data, current_user.id
    )
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {"message": f"Agnelage enregistré avec succès", "lambs": len(lambs)}


@router.get("/production/laine")
async def get_wool_production(
    year: Optional[int] = Query(None, description="Année"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir les statistiques de production de laine
    """
    stats = await ovin_service.get_wool_production(db, year)
    return stats


@router.get("/stats/global")
async def get_ovin_stats(
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir des statistiques globales sur les ovins
    """
    stats = await ovin_service.get_ovin_stats(db, enclos_id)
    return stats


# === NOUVELLE ROUTE POUR ENREGISTRER UNE VENTE ===
@router.post("/{ovin_id}/vente", response_model=OvinResponse, status_code=status.HTTP_200_OK)
async def enregistrer_vente_ovin(
    ovin_id: int,
    vente_data: AnimalVenteCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_write_ovins),
):
    """
    Enregistrer la vente d'un ovin avec le prix, le client et la date
    """
    ovin, error = await animal_service.enregistrer_vente(
        db, ovin_id, 'VENTE', 'ovin', vente_data, current_user.id
    )
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Récupérer les informations complémentaires
    dernier_poids = await animal_service.get_last_weight(db, ovin.id)
    enclos_name = await animal_service.get_enclos_name_by_id(db, ovin.enclos_id)
    
    # Créer la réponse
    return OvinResponse(
        id=ovin.id,
        identification=ovin.identification,
        type_espece=ovin.type_espece,
        race=ovin.race,
        sexe=ovin.sexe,
        date_naissance=ovin.date_naissance,
        date_arrivee=ovin.date_arrivee,
        provenance=ovin.provenance,
        prix_achat=ovin.prix_achat,
        enclos_id=ovin.enclos_id,
        enclos_name=enclos_name,
        statut=ovin.statut,
        photo_url=ovin.photo_url,
        notes=ovin.notes,
        production_viande=ovin.production_viande,
        production_reproduction=ovin.production_reproduction,
        production_laine=ovin.production_laine,
        qualite_laine=ovin.qualite_laine,
        dernier_poids=dernier_poids,
        age_mois=ovin.age_mois,
        created_at=ovin.created_at,
        updated_at=ovin.updated_at,
        # Nouveaux champs
        prix_vente=ovin.prix_vente,
        date_vente=ovin.date_vente,
        client_acheteur=ovin.client_acheteur,
        note_vente=ovin.note_vente
    )


# === NOUVELLE ROUTE POUR LES STATISTIQUES DE VENTE ===
@router.get("/ventes/stats")
async def get_ventes_statistiques(
    date_debut: Optional[date] = Query(None, description="Date de début pour filtrer les ventes"),
    date_fin: Optional[date] = Query(None, description="Date de fin pour filtrer les ventes"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir des statistiques sur les ventes d'ovins
    """
    stats = await ovin_service.get_ventes_stats(db, date_debut, date_fin)
    return stats


# === NOUVELLE ROUTE POUR RÉCUPÉRER LES OVINS VENDUS ===
@router.get("/ventes/liste", response_model=List[OvinResponse])
async def get_ovins_vendus(
    date_debut: Optional[date] = Query(None, description="Date de début pour filtrer les ventes"),
    date_fin: Optional[date] = Query(None, description="Date de fin pour filtrer les ventes"),
    client: Optional[str] = Query(None, description="Filtrer par client"),
    skip: int = Query(0, ge=0, description="Nombre d'éléments à sauter"),
    limit: int = Query(100, ge=1, le=1000, description="Nombre d'éléments à retourner"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(can_read_ovins),
):
    """
    Obtenir la liste des ovins vendus avec filtres
    """
    # Récupérer les ovins vendus
    stmt = select(Ovin).where(Ovin.prix_vente.isnot(None))
    
    if date_debut:
        stmt = stmt.where(Ovin.date_vente >= date_debut)
    if date_fin:
        stmt = stmt.where(Ovin.date_vente <= date_fin)
    if client:
        stmt = stmt.where(Ovin.client_acheteur.ilike(f"%{client}%"))
    
    stmt = stmt.order_by(Ovin.date_vente.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    ovins = result.scalars().all()
    
    # Construire les réponses
    items = []
    for ovin in ovins:
        dernier_poids = await animal_service.get_last_weight(db, ovin.id)
        enclos_name = await animal_service.get_enclos_name_by_id(db, ovin.enclos_id)
        
        ovin_response = OvinResponse(
            id=ovin.id,
            identification=ovin.identification,
            type_espece=ovin.type_espece,
            race=ovin.race,
            sexe=ovin.sexe,
            date_naissance=ovin.date_naissance,
            date_arrivee=ovin.date_arrivee,
            provenance=ovin.provenance,
            prix_achat=ovin.prix_achat,
            enclos_id=ovin.enclos_id,
            enclos_name=enclos_name,
            statut=ovin.statut,
            photo_url=ovin.photo_url,
            notes=ovin.notes,
            production_viande=ovin.production_viande,
            production_reproduction=ovin.production_reproduction,
            production_laine=ovin.production_laine,
            qualite_laine=ovin.qualite_laine,
            dernier_poids=dernier_poids,
            age_mois=ovin.age_mois,
            created_at=ovin.created_at,
            updated_at=ovin.updated_at,
            prix_vente=ovin.prix_vente,
            date_vente=ovin.date_vente,
            client_acheteur=ovin.client_acheteur,
            note_vente=ovin.note_vente
        )
        items.append(ovin_response)
    
    return items