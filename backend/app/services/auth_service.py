# backend/app/services/auth_service.py
"""
Service d'authentification - Login, logout, sessions, JWT
"""

import logging
from datetime import datetime, timedelta
import secrets
from typing import Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..models.user import User, UserSession, ActionLog
from ..schemas.auth import LoginRequest, TokenResponse, RefreshTokenResponse
from ..core.security import (
    hash_password, verify_password, create_access_token, create_refresh_token, decode_refresh_token,
    generate_session_id, decode_token
)
from ..redis_client import redis_client
from ..config import settings

logger = logging.getLogger(__name__)


class AuthService:
    """Service d'authentification et gestion des sessions"""
    
    async def login(
        self,
        db: AsyncSession,
        login_data: LoginRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[Optional[TokenResponse], Optional[str]]:
        """
        Authentifier un utilisateur et créer une session
        
        Returns:
            (TokenResponse, error_message)
        """
        # Rechercher l'utilisateur par username, email OU phone
        userlogin = login_data.userlogin.strip()
        stmt = select(User).where(
            (User.username == userlogin) | 
            (User.email == userlogin) | 
            (User.phone == userlogin),
            User.is_active == True           # ← CORRIGER la position
        ).options(selectinload(User.roles))  # Charger les rôles en même temps
        result = await db.execute(stmt)
        user: User = result.scalar_one_or_none()
        
        print("=" * 80)
        print(f"🔐 TENTATIVE DE LOGIN:")
        print(f"   userlogin: {userlogin}")
        print(f"   User trouvé: {user.username if user else 'None'}")
        if user:
            print(f"   ID: {user.id}")
            print(f"   Email: {user.email}")
            print(f"   Phone: {user.phone}")
            print(f"   Rôles: {[role.name.value if hasattr(role.name, 'value') else role.name for role in user.roles]}")
        print("=" * 80)
        
        if not user:
            return None, "Nom d'utilisateur, email ou mot de passe incorrect"
        
        # Vérifier le mot de passe
        if not verify_password(login_data.password, user.hashed_password):
            # Journaliser la tentative échouée
            await self._log_action(db, user.id, "LOGIN_FAILED", details={"ip": ip_address})
            await db.commit()
            return None, "Nom d'utilisateur, email ou mot de passe incorrect"
        
        # Vérifier le rôle actif (simplifié - permettre tous les rôles existants)
        if not user.roles:
            return None, "Compte sans rôle valide"
        
        # Créer la session
        session_id = generate_session_id()
        expires_at = datetime.now() + timedelta(hours=settings.SESSION_EXPIRE_HOURS)
        
        # Stocker en base de données
        user_session = UserSession(
            user_id=user.id,
            token=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            is_valid=True
        )
        db.add(user_session)
        
        # Stocker dans Redis
        await redis_client.create_session(
            session_id=session_id,
            user_id=user.id,
            user_roles=[role.name for role in user.roles] if user.roles else [],
            ttl=settings.SESSION_EXPIRE_HOURS * 3600
        )
        
        # Créer les tokens
        token_data = {
            "sub": str(user.id),
            "session_id": session_id,
            "username": user.username,
            "roles": [role.name for role in user.roles] if user.roles else []
        }
        
        access_token = create_access_token(
            data=token_data,
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        refresh_token = create_refresh_token(
            data={"sub": str(user.id), "session_id": session_id},
            expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )
        
        # Stocker le refresh token dans Redis
        await redis_client.set(
            f"refresh_token:{refresh_token}",
            {"user_id": user.id, "session_id": session_id},
            ttl=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600
        )
        
        # Journaliser le login
        await self._log_action(db, user.id, "LOGIN", details={"ip": ip_address, "user_agent": user_agent})
        await db.commit()
        
        logger.info(f"User {user.username} logged in from {ip_address}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_expires_in=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
            user_id=user.id,
            email=user.email,
            username=user.username,
            phone=user.phone,
            roles=[role.name for role in user.roles] if user.roles else []
        ), None

    
    async def refresh_token(
        self,
        db: AsyncSession,
        refresh_token: str,
        ip_address: Optional[str] = None
    ) -> Tuple[Optional[RefreshTokenResponse], Optional[str]]:
        """Rafraîchir le token d'accès"""
        
        # Décoder le refresh token
        payload = decode_refresh_token(refresh_token)
        if not payload:
            return None, "Token de rafraîchissement invalide"
        
        user_id = payload.get("sub")
        session_id = payload.get("session_id")
        
        if not user_id or not session_id:
            return None, "Token de rafraîchissement invalide"
        
        # Convertir user_id en entier
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            return None, "Token de rafraîchissement invalide"
        
        # Vérifier si le refresh token existe encore dans Redis
        stored = await redis_client.get(f"refresh_token:{refresh_token}")
        if not stored:
            return None, "Token de rafraîchissement expiré ou révoqué"
        
        # Vérifier l'utilisateur
        stmt = select(User).where(
            User.id == user_id_int, 
            User.is_active == True
        ).options(selectinload(User.roles))
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return None, "Utilisateur non trouvé"
        
        # Vérifier la session - CORRECTION ICI : user_id doit être un entier
        stmt = select(UserSession).where(
            UserSession.token == session_id,
            UserSession.user_id == user_id_int,  # ← Utiliser user_id_int, pas la string
            UserSession.is_valid == True
        )
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session:
            return None, "Session invalide"
        
        # Créer un nouveau refresh token (rotation)
        new_refresh_token = create_refresh_token(
            data={"sub": str(user.id), "session_id": session_id},
            expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )
        
        # Stocker le nouveau refresh token
        await redis_client.set(
            f"refresh_token:{new_refresh_token}",
            {"user_id": user.id, "session_id": session_id},
            ttl=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600
        )
        
        # Supprimer l'ancien refresh token
        await redis_client.delete(f"refresh_token:{refresh_token}")
        
        # Créer un nouveau token d'accès
        token_data = {
            "sub": str(user.id),
            "session_id": session_id,
            "username": user.username,
            "roles": [role.name for role in user.roles] if user.roles else []
        }
        
        new_access_token = create_access_token(
            data=token_data,
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        # Journaliser
        await self._log_action(db, user.id, "REFRESH_TOKEN", details={"ip": ip_address})
        await db.commit()
        
        return RefreshTokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_expires_in=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600
        ), None
    
    
    async def revoke_refresh_token(
        self,
        db: AsyncSession,
        refresh_token: str,
        user_id: int
    ) -> bool:
        """Révoquer un refresh token"""
        await redis_client.delete(f"refresh_token:{refresh_token}")
        
        await self._log_action(db, user_id, "REVOKE_REFRESH_TOKEN")
        await db.commit()
        
        return True
    
    
    async def revoke_all_user_tokens(
        self,
        db: AsyncSession,
        user_id: int,
        current_session_id: Optional[str] = None
    ) -> int:
        """Révoquer tous les tokens d'un utilisateur"""
        count = 0
        
        # Supprimer toutes les sessions
        stmt = select(UserSession).where(
            UserSession.user_id == user_id,
            UserSession.is_valid == True
        )
        if current_session_id:
            stmt = stmt.where(UserSession.token != current_session_id)
        
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        for session in sessions:
            session.is_valid = False
            session.logout_at = datetime.now()
            await redis_client.delete(f"session:{session.token}")
            count += 1
        
        # Supprimer tous les refresh tokens de l'utilisateur
        keys = await redis_client.keys("refresh_token:*")
        for key in keys:
            data = await redis_client.get_json(key)
            if data and data.get("user_id") == user_id:
                await redis_client.delete(key)
                count += 1
        
        await db.commit()
        
        return count
    
    async def logout(
        self,
        db: AsyncSession,
        user_id: int,
        session_id: str,
        ip_address: Optional[str] = None
    ) -> bool:
        """Déconnecter un utilisateur"""
        # Invalider la session en base
        stmt = select(UserSession).where(
            UserSession.token == session_id,
            UserSession.user_id == user_id,
            UserSession.is_valid == True
        )
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if session:
            session.is_valid = False
            session.logout_at = datetime.now()
        
        # Supprimer de Redis
        await redis_client.delete(f"session:{session_id}")
        
        # Journaliser le logout
        await self._log_action(db, user_id, "LOGOUT", details={"ip": ip_address})
        await db.commit()
        
        logger.info(f"User {user_id} logged out")
        return True
    
    async def logout_all_devices(
        self,
        db: AsyncSession,
        user_id: int,
        current_session_id: str
    ) -> int:
        """Déconnecter l'utilisateur de tous les appareils sauf celui-ci"""
        # Invalider toutes les sessions sauf la courante
        stmt = select(UserSession).where(
            UserSession.user_id == user_id,
            UserSession.is_valid == True,
            UserSession.token != current_session_id
        )
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        count = 0
        for session in sessions:
            session.is_valid = False
            session.logout_at = datetime.now()
            await redis_client.delete(f"session:{session.token}")
            count += 1
        
        await db.commit()
        logger.info(f"User {user_id} logged out from {count} other devices")
        return count

    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Valider un token JWT et retourner les données"""
        payload = decode_token(token)  # Utiliser decode_token au lieu de decode_access_token
        if not payload:
            return None
        
        # Vérifier que la session existe encore
        session_id = payload.get("session_id")
        if session_id:
            session_data = await redis_client.get_session(session_id)
            if not session_data:
                return None
        
        return payload
        
    async def change_password(
        self,
        db: AsyncSession,
        user_id: int,
        old_password: str,
        new_password: str
    ) -> Tuple[bool, str]:
        """Changer le mot de passe d'un utilisateur"""
        # Récupérer l'utilisateur
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return False, "Utilisateur non trouvé"
        
        # Vérifier l'ancien mot de passe
        if not verify_password(old_password, user.hashed_password):
            return False, "Ancien mot de passe incorrect"
        
        # Valider la complexité du nouveau mot de passe
        if len(new_password) < settings.PASSWORD_MIN_LENGTH:
            return False, f"Le mot de passe doit contenir au moins {settings.PASSWORD_MIN_LENGTH} caractères"
        
        # Changer le mot de passe
        user.hashed_password = hash_password(new_password)
        
        # Invalider toutes les sessions SAUF la courante (forcer reconnexion ailleurs)
        stmt = select(UserSession).where(
            UserSession.user_id == user_id, 
            UserSession.is_valid == True
        )
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        for session in sessions:
            session.is_valid = False
            await redis_client.delete(f"session:{session.token}")
        
        await self._log_action(db, user_id, "PASSWORD_CHANGE")
        await db.commit()
        
        logger.info(f"User {user_id} changed password")
        return True, "Mot de passe changé avec succès"
    
    async def auto_logout_at_midnight(self, db: AsyncSession) -> int:
        """
        Déconnexion automatique de tous les appareils à minuit
        Exécuté par un job planifié
        """
        # Invalider toutes les sessions
        stmt = select(UserSession).where(UserSession.is_valid == True)
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        count = 0
        for session in sessions:
            session.is_valid = False
            session.logout_at = datetime.now()
            await redis_client.delete(f"session:{session.token}")
            count += 1
        
        await db.commit()
        logger.info(f"Auto-logout at midnight: {count} sessions invalidated")
        return count
    
    async def get_user_sessions(
        self,
        db: AsyncSession,
        user_id: int
    ) -> list:
        """Obtenir toutes les sessions d'un utilisateur"""
        stmt = select(UserSession).where(UserSession.user_id == user_id).order_by(UserSession.created_at.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def _log_action(
        self,
        db: AsyncSession,
        user_id: int,
        action: str,
        entity_type: str = None,
        entity_id: int = None,
        details: dict = None
    ) -> None:
        """Journaliser une action utilisateur"""
        log = ActionLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details
        )
        db.add(log)

    async def forgot_password(
        self,
        db: AsyncSession,
        user_id: int
    ) -> list:
        """Obtenir toutes les sessions d'un utilisateur"""
        stmt = select(UserSession).where(UserSession.user_id == user_id).order_by(UserSession.created_at.desc())
        result = await db.execute(stmt)
        return result.scalars().all()

    async def forgot_password(
        self,
        db: AsyncSession,
        email: str,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Générer un token de réinitialisation de mot de passe
        Returns:
            (success, message)
        """
        # Rechercher l'utilisateur par email
        stmt = select(User).where(User.email == email, User.is_active == True)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            # Pour des raisons de sécurité, on ne révèle pas si l'email existe
            return True, "Si un compte existe avec cet email, vous recevrez un lien de réinitialisation"
        
        # Générer un token sécurisé
        token = secrets.token_urlsafe(32)
        
        # Stocker le token dans Redis avec expiration (1 heure)
        reset_key = f"password_reset:{token}"
        await redis_client.set(
            reset_key,
            {
                "user_id": user.id,
                "email": user.email,
                "ip_address": ip_address
            },
            ttl=3600  # 1 heure
        )
        
        # Journaliser la demande
        await self._log_action(db, user.id, "PASSWORD_RESET_REQUESTED", details={"ip": ip_address})
        await db.commit()
        
        # Envoyer l'email (à implémenter)
        # await send_password_reset_email(user.email, token)
        
        logger.info(f"Password reset requested for user {user.email} from {ip_address}")
        
        return True, "Si un compte existe avec cet email, vous recevrez un lien de réinitialisation"
    
    async def reset_password(
        self,
        db: AsyncSession,
        token: str,
        new_password: str,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Réinitialiser le mot de passe avec un token
        Returns:
            (success, message)
        """
        # Vérifier le token dans Redis
        reset_key = f"password_reset:{token}"
        reset_data = await redis_client.get_json(reset_key)
        
        if not reset_data:
            return False, "Token invalide ou expiré"
        
        user_id = reset_data.get("user_id")
        if not user_id:
            return False, "Token invalide"
        
        # Récupérer l'utilisateur
        stmt = select(User).where(User.id == user_id, User.is_active == True)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return False, "Utilisateur non trouvé"
        
        # Valider la complexité du nouveau mot de passe
        if len(new_password) < settings.PASSWORD_MIN_LENGTH:
            return False, f"Le mot de passe doit contenir au moins {settings.PASSWORD_MIN_LENGTH} caractères"
        
        # Changer le mot de passe
        user.hashed_password = hash_password(new_password)
        
        # Invalider toutes les sessions (forcer reconnexion)
        stmt = select(UserSession).where(
            UserSession.user_id == user_id, 
            UserSession.is_valid == True
        )
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        for session in sessions:
            session.is_valid = False
            session.logout_at = datetime.now()
            await redis_client.delete(f"session:{session.token}")
        
        # Supprimer tous les refresh tokens de l'utilisateur
        keys = await redis_client.keys("refresh_token:*")
        for key in keys:
            data = await redis_client.get_json(key)
            if data and data.get("user_id") == user_id:
                await redis_client.delete(key)
        
        # Supprimer le token de réinitialisation
        await redis_client.delete(reset_key)
        
        # Journaliser la réinitialisation
        await self._log_action(db, user_id, "PASSWORD_RESET_COMPLETED", details={"ip": ip_address})
        await db.commit()
        
        logger.info(f"Password reset completed for user {user.email} from {ip_address}")
        
        return True, "Mot de passe réinitialisé avec succès"
    
    async def verify_reset_token(
        self,
        db: AsyncSession,
        token: str
    ) -> Tuple[bool, Optional[int]]:
        """
        Vérifier si un token de réinitialisation est valide
        Returns:
            (is_valid, user_id)
        """
        reset_key = f"password_reset:{token}"
        reset_data = await redis_client.get_json(reset_key)
        
        if not reset_data:
            return False, None
        
        user_id = reset_data.get("user_id")
        if not user_id:
            return False, None
        
        # Vérifier que l'utilisateur existe toujours
        stmt = select(User).where(User.id == user_id, User.is_active == True)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return False, None
        
        return True, user_id

auth_service = AuthService()