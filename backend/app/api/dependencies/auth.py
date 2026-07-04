# backend/app/api/dependencies/auth.py
from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select

from ...database import get_db
from ...models.user import User, RoleEnum
from ...services.auth_service import auth_service

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Récupérer l'utilisateur courant AVEC les rôles chargés"""
    token = credentials.credentials
    
    payload = await auth_service.validate_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide",
        )
    
    # CRUCIAL: Charger les rôles avec selectinload
    stmt = select(User).where(
        User.id == int(user_id), 
        User.is_active == True
    ).options(selectinload(User.roles))  # ← AJOUTER CECI
    
    result = await db.execute(stmt)
    user = result.unique().scalar_one_or_none()  # ← unique() important pour les jointures
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur non trouvé",
        )
    
    return user


async def get_current_super_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Vérifier que l'utilisateur est super admin"""
    if not current_user.has_role(RoleEnum.SUPER_ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé au super administrateur",
        )
    return current_user


# ============ RÔLES TRANSVERSES ============

async def get_current_veterinaire_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Vérifier que l'utilisateur a un rôle vétérinaire ou super admin"""
    if not (current_user.has_role(RoleEnum.VETERINAIRE) or 
            current_user.has_role(RoleEnum.SUPER_ADMIN)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé aux vétérinaires",
        )
    return current_user


async def get_current_responsable_enclos_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Vérifier que l'utilisateur est responsable d'enclos ou super admin"""
    if not (current_user.has_role(RoleEnum.RESPONSABLE_ENCLOS) or 
            current_user.has_role(RoleEnum.SUPER_ADMIN)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé aux responsables d'enclos",
        )
    return current_user


async def get_current_technicien_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Vérifier que l'utilisateur a un rôle technicien (n'importe quelle espèce)"""
    technicien_roles = [
        RoleEnum.BOVIN_TECHNICIEN,
        RoleEnum.OVIN_TECHNICIEN, 
        RoleEnum.CAPRIN_TECHNICIEN,
        RoleEnum.AVICOLE_TECHNICIEN,
        RoleEnum.PISCICOLE_TECHNICIEN,
        RoleEnum.APICULTURE_TECHNICIEN,
        RoleEnum.ENTOMOCULTURE_TECHNICIEN,
        RoleEnum.SUPER_ADMIN
    ]
    
    if not current_user.has_any_role(technicien_roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé aux techniciens",
        )
    return current_user


async def get_current_comptable_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Vérifier que l'utilisateur a un rôle comptable ou super admin"""
    if not (current_user.has_role(RoleEnum.COMPTABLE) or 
            current_user.has_role(RoleEnum.SUPER_ADMIN)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé à la comptabilité",
        )
    return current_user


async def get_current_observer_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Tous les utilisateurs authentifiés peuvent observer"""
    return current_user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Récupérer l'utilisateur courant si authentifié, sinon None"""
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = await auth_service.validate_token(token)
    if not payload:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    stmt = select(User).where(User.id == int(user_id), User.is_active == True)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


def require_roles(allowed_roles: List[RoleEnum]):
    """Décorateur pour exiger au moins un des rôles"""
    async def dependency(current_user: User = Depends(get_current_user)):
        if not current_user.has_any_role(allowed_roles):
            role_names = [r.value for r in allowed_roles]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès réservé aux rôles: {', '.join(role_names)}",
            )
        return current_user
    return dependency


def require_espece_permission(espece: str, action: str = "read"):
    """Décorateur pour vérifier les permissions par espèce"""
    async def dependency(current_user: User = Depends(get_current_user)):
        if not current_user.has_permission(espece, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission insuffisante pour {action} sur {espece}",
            )
        return current_user
    return dependency


# ============ PERMISSIONS PAR ESPÈCE ============

# --- Bovins ---
async def can_read_bovins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("bovin", "read"):
        raise HTTPException(status_code=403, detail="Lecture des bovins non autorisée")
    return current_user

async def can_write_bovins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("bovin", "write"):
        raise HTTPException(status_code=403, detail="Modification des bovins non autorisée")
    return current_user

async def can_delete_bovins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("bovin", "delete"):
        raise HTTPException(status_code=403, detail="Suppression des bovins non autorisée")
    return current_user


# --- Ovins ---
async def can_read_ovins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("ovin", "read"):
        raise HTTPException(status_code=403, detail="Lecture des ovins non autorisée")
    return current_user

async def can_write_ovins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("ovin", "write"):
        raise HTTPException(status_code=403, detail="Modification des ovins non autorisée")
    return current_user

async def can_delete_ovins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("ovin", "delete"):
        raise HTTPException(status_code=403, detail="Suppression des ovins non autorisée")
    return current_user


# --- Caprins ---
async def can_read_caprins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("caprin", "read"):
        raise HTTPException(status_code=403, detail="Lecture des caprins non autorisée")
    return current_user

async def can_write_caprins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("caprin", "write"):
        raise HTTPException(status_code=403, detail="Modification des caprins non autorisée")
    return current_user

async def can_delete_caprins(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("caprin", "delete"):
        raise HTTPException(status_code=403, detail="Suppression des caprins non autorisée")
    return current_user


# --- Avicoles ---
async def can_read_avicoles(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("avicole", "read"):
        raise HTTPException(status_code=403, detail="Lecture des avicoles non autorisée")
    return current_user

async def can_write_avicoles(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("avicole", "write"):
        raise HTTPException(status_code=403, detail="Modification des avicoles non autorisée")
    return current_user

async def can_delete_avicoles(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("avicole", "delete"):
        raise HTTPException(status_code=403, detail="Suppression des avicoles non autorisée")
    return current_user


# --- Piscicoles ---
async def can_read_piscicoles(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("piscicole", "read"):
        raise HTTPException(status_code=403, detail="Lecture des piscicoles non autorisée")
    return current_user

async def can_write_piscicoles(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("piscicole", "write"):
        raise HTTPException(status_code=403, detail="Modification des piscicoles non autorisée")
    return current_user

async def can_delete_piscicoles(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("piscicole", "delete"):
        raise HTTPException(status_code=403, detail="Suppression des piscicoles non autorisée")
    return current_user


# --- Apiculture (AJOUTÉ) ---
async def can_read_apiculture(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("apiculture", "read"):
        raise HTTPException(status_code=403, detail="Lecture de l'apiculture non autorisée")
    return current_user

async def can_write_apiculture(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("apiculture", "write"):
        raise HTTPException(status_code=403, detail="Modification de l'apiculture non autorisée")
    return current_user

async def can_delete_apiculture(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("apiculture", "delete"):
        raise HTTPException(status_code=403, detail="Suppression de l'apiculture non autorisée")
    return current_user


# --- Entomoculture (AJOUTÉ) ---
async def can_read_entomoculture(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("entomoculture", "read"):
        raise HTTPException(status_code=403, detail="Lecture de l'entomoculture non autorisée")
    return current_user

async def can_write_entomoculture(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("entomoculture", "write"):
        raise HTTPException(status_code=403, detail="Modification de l'entomoculture non autorisée")
    return current_user

async def can_delete_entomoculture(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.has_permission("entomoculture", "delete"):
        raise HTTPException(status_code=403, detail="Suppression de l'entomoculture non autorisée")
    return current_user


# ============ PERMISSIONS SPÉCIALES ============

async def can_access_accounting(current_user: User = Depends(get_current_user)) -> User:
    """Accès à la comptabilité (rôle comptable ou super admin)"""
    if not (current_user.has_role(RoleEnum.COMPTABLE) or 
            current_user.has_role(RoleEnum.SUPER_ADMIN)):
        raise HTTPException(status_code=403, detail="Accès comptabilité non autorisé")
    return current_user


async def can_access_veterinaire(current_user: User = Depends(get_current_user)) -> User:
    """Accès aux fonctions vétérinaires (rôle vétérinaire ou super admin)"""
    if not (current_user.has_role(RoleEnum.VETERINAIRE) or 
            current_user.has_role(RoleEnum.SUPER_ADMIN)):
        raise HTTPException(status_code=403, detail="Accès vétérinaire non autorisé")
    return current_user


async def can_access_vision_globale(current_user: User = Depends(get_current_user)) -> User:
    """Accès à la vision globale (voir toutes les espèces)"""
    if not (current_user.has_role(RoleEnum.VISION_GLOBALE) or 
            current_user.has_role(RoleEnum.SUPER_ADMIN)):
        raise HTTPException(status_code=403, detail="Vision globale non autorisée")
    return current_user


# ============ COMPATIBILITÉ AVEC L'ANCIEN CODE ============
get_current_admin_user = get_current_super_admin