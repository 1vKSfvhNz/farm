# backend/app/api/v1/auth.py
"""
Routes d'authentification
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...schemas.auth import ChangePasswordRequest, ForgotPasswordRequest, LoginRequest, RefreshTokenRequest, RefreshTokenResponse, ResetPasswordRequest, TokenResponse
from ...services.auth_service import auth_service
from ...api.dependencies.auth import get_current_user
from ...models.user import User

router = APIRouter(prefix="/auth", tags=["Authentification"])

@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authentifier un utilisateur et retourner un token JWT
    """
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    token, error = await auth_service.login(
        db, login_data, client_ip, user_agent
    )
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error
        )
    
    return token

@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_access_token(
    request: Request,
    refresh_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Rafraîchir le token d'accès
    """
    client_ip = request.client.host if request.client else None
    
    result, error = await auth_service.refresh_token(
        db, refresh_data.refresh_token, client_ip
    )
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error
        )
    
    return result


@router.post("/revoke")
async def revoke_token(
    request: Request,
    refresh_data: RefreshTokenRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Révoquer un refresh token
    """
    await auth_service.revoke_refresh_token(db, refresh_data.refresh_token, current_user.id)
    
    return {"message": "Token révoqué avec succès"}


@router.post("/revoke-all")
async def revoke_all_tokens(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Révoquer tous les tokens de l'utilisateur
    """
    session_id = request.state.session_id
    count = await auth_service.revoke_all_user_tokens(db, current_user.id, session_id)
    
    return {"message": f"{count} tokens révoqués"}

@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Déconnecter l'utilisateur courant
    """
    client_ip = request.client.host if request.client else None
    session_id = request.state.session_id
    
    await auth_service.logout(db, current_user.id, session_id, client_ip)
    
    return {"message": "Déconnecté avec succès"}


@router.post("/logout-all")
async def logout_all_devices(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Déconnecter l'utilisateur de tous les appareils
    """
    session_id = request.state.session_id
    count = await auth_service.logout_all_devices(db, current_user.id, session_id)
    
    return {"message": f"Déconnecté de {count} autres appareils"}


@router.get("/me")  # ← Retirer response_model temporairement
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les informations de l'utilisateur courant
    """
    # Maintenant current_user.roles est chargé (grâce à selectinload)
    # Construire la réponse manuellement
    return {
        "id": current_user.id,
        "email": current_user.email,
        "phone": current_user.phone,
        "username": current_user.username,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "roles": [role.name.value if hasattr(role.name, 'value') else str(role.name) 
                  for role in current_user.roles]
    }

@router.post("/forgot-password")
async def forgot_password(
    request: Request,
    forgot_data: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Demander la réinitialisation du mot de passe
    """
    client_ip = request.client.host if request.client else None
    
    success, message = await auth_service.forgot_password(
        db, forgot_data.email, client_ip
    )
    
    return {"message": message}


@router.post("/reset-password")
async def reset_password(
    request: Request,
    reset_data: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Réinitialiser le mot de passe avec le token reçu par email
    """
    client_ip = request.client.host if request.client else None
    
    success, message = await auth_service.reset_password(
        db, reset_data.token, reset_data.new_password, client_ip
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return {"message": message}


@router.get("/verify-reset-token/{token}")
async def verify_reset_token(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Vérifier si un token de réinitialisation est valide
    """
    is_valid, user_id = await auth_service.verify_reset_token(db, token)
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token invalide ou expiré"
        )
    
    return {"valid": True, "user_id": user_id}


@router.post("/change-password")
async def change_password(
    request: Request,
    password_data: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Changer le mot de passe (utilisateur connecté)
    """
    client_ip = request.client.host if request.client else None
    
    success, message = await auth_service.change_password(
        db, current_user.id, password_data.old_password, password_data.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return {"message": message}