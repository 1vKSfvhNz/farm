# backend/app/api/v1/users.py
"""
Routes de gestion des utilisateurs
"""

from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...schemas.user import LengthResponse, UserCreate, UserUpdate, UserResponse, UserSessionResponse, ActionLogResponse
from ...services.user_service import user_service
from ...services.auth_service import auth_service
from ...api.dependencies.auth import get_current_admin_user, get_current_user
from ...api.dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User, RoleEnum

router = APIRouter(prefix="/users", tags=["Utilisateurs"])


# ============ ROUTES PUBLIQUES (authentification requise) ============

@router.get("/length", response_model=LengthResponse)
async def get_counts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Obtenir les compteurs pour toutes les entités"""
    counts = await user_service.get_user_counts_by_permissions(db, current_user)
    return LengthResponse(
        users_length=counts.get("users"),
        enclos_length=counts.get("enclos"),
        bovins_length=counts.get("bovin"),
        ovins_length=counts.get("ovin"),
        caprins_length=counts.get("caprin"),
        avicoles_length=counts.get("avicole"),
        piscicoles_length=counts.get("piscicole"),
        ruches_length=counts.get("apiculture"),
        nids_length=counts.get("entomoculture")
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """Obtenir les informations de l'utilisateur courant"""
    return UserResponse.from_orm_with_roles(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mettre à jour son propre profil"""
    user, error = await user_service.update_user(db, current_user.id, user_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return user


@router.get("/me/sessions", response_model=List[UserSessionResponse])
async def get_my_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Obtenir ses propres sessions"""
    sessions = await auth_service.get_user_sessions(db, current_user.id)
    return sessions


@router.get("/me/actions", response_model=List[ActionLogResponse])
async def get_my_actions(
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0, description="Nombre d'éléments à sauter"),
    limit: int = Query(100, ge=1, le=500, description="Nombre maximum d'éléments"),
    current_user: User = Depends(get_current_user),
):
    """Obtenir son propre historique d'actions"""
    actions = await user_service.get_user_actions(db, current_user.id, skip, limit)
    return actions


# ============ ROUTES ADMIN (super admin uniquement) ============

@router.get("", response_model=PaginatedResponse[UserResponse])
async def get_users(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    # Recherche textuelle
    search: Optional[str] = Query(None, description="Recherche par nom, email, username ou matricule"),
    # Filtres de base
    roles: Optional[List[str]] = Query(None, description="Filtrer par rôles"),
    is_active: Optional[bool] = Query(None, description="Filtrer par statut actif"),
    # Filtres employé
    employee_id: Optional[str] = Query(None, description="Filtrer par matricule"),
    department: Optional[str] = Query(None, description="Filtrer par département"),
    employee_status: Optional[List[str]] = Query(None, description="Filtrer par statut employé"),
    employee_type: Optional[List[str]] = Query(None, description="Filtrer par type d'employé"),
    hire_date_from: Optional[date] = Query(None, description="Date d'embauche (min)"),
    hire_date_to: Optional[date] = Query(None, description="Date d'embauche (max)"),
    # Filtres salaire
    salary_min: Optional[float] = Query(None, ge=0, description="Salaire minimum"),
    salary_max: Optional[float] = Query(None, ge=0, description="Salaire maximum"),
    # Filtres dates
    created_from: Optional[datetime] = Query(None, description="Date de création (min)"),
    created_to: Optional[datetime] = Query(None, description="Date de création (max)"),
    # Tri
    order_by: str = Query("created_at", description="Champ de tri"),
    order_direction: str = Query("desc", description="Direction du tri (asc/desc)"),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Obtenir la liste des utilisateurs avec filtres avancés (admin uniquement)
    """
    users, total = await user_service.get_users(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        search=search,
        roles=roles,
        is_active=is_active,
        employee_id=employee_id,
        department=department,
        employee_status=employee_status,
        employee_type=employee_type,
        hire_date_from=hire_date_from,
        hire_date_to=hire_date_to,
        salary_min=salary_min,
        salary_max=salary_max,
        created_from=created_from,
        created_to=created_to,
        order_by=order_by,
        order_direction=order_direction
    )
    
    return PaginatedResponse.create(
        items=users,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """Obtenir un utilisateur par son ID (admin uniquement)"""
    user = await user_service.get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    return user


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """Créer un nouvel utilisateur (admin uniquement)"""
    user, error = await user_service.create_user(db, user_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return user


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """Mettre à jour un utilisateur (admin uniquement)"""
    user, error = await user_service.update_user(db, user_id, user_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return user


@router.get("/{user_id}/sessions", response_model=List[UserSessionResponse])
async def get_user_sessions(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """Obtenir les sessions d'un utilisateur (admin uniquement)"""
    sessions = await auth_service.get_user_sessions(db, user_id)
    return sessions


@router.get("/{user_id}/actions", response_model=List[ActionLogResponse])
async def get_user_actions(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0, description="Nombre d'éléments à sauter"),
    limit: int = Query(100, ge=1, le=500, description="Nombre maximum d'éléments"),
    current_user: User = Depends(get_current_admin_user),
):
    """Obtenir l'historique des actions d'un utilisateur (admin uniquement)"""
    actions = await user_service.get_user_actions(db, user_id, skip, limit)
    return actions


@router.post("/{user_id}/revoke-sessions", status_code=status.HTTP_200_OK)
async def revoke_user_sessions(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """Révoquer toutes les sessions d'un utilisateur (admin uniquement)"""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez pas révoquer vos propres sessions"
        )
    
    success = await auth_service.revoke_all_user_sessions(db, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    return {"message": f"Toutes les sessions de l'utilisateur {user_id} ont été révoquées"}