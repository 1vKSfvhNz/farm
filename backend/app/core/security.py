# backend/app/core/security.py
import random
import string

from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
import secrets
import uuid

from ..config import settings

# Contexte pour le hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_password(length: int = 6, numeric_only: bool = True) -> str:
    """
    Générer un mot de passe aléatoire
    
    Args:
        length: Longueur du mot de passe (défaut: 6)
        numeric_only: Si True, génère uniquement des chiffres (défaut: True)
    
    Returns:
        str: Mot de passe généré
    """
    if numeric_only:
        # Générer un code à 6 chiffres
        return ''.join(str(random.randint(0, 9)) for _ in range(length))
    else:
        # Générer un mot de passe alphanumérique
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_verification_code(length: int = 6) -> str:
    """
    Générer un code de vérification à 6 chiffres
    """
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


def hash_password(password: str) -> str:
    """Hacher un mot de passe"""
    if len(password) > 72:
        password = password[:72]
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifier un mot de passe"""
    if len(plain_password) > 72:
        plain_password = plain_password[:72]
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Créer un token JWT d'accès"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Créer un token JWT de rafraîchissement"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Décoder un token JWT"""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.PyJWTError:
        return None


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Décoder un token d'accès JWT"""
    payload = decode_token(token)
    if payload and payload.get("type") == "access":
        return payload
    return None


def decode_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """Décoder un token de rafraîchissement JWT"""
    payload = decode_token(token)
    if payload and payload.get("type") == "refresh":
        return payload
    return None


def generate_session_id() -> str:
    """Générer un ID de session unique"""
    return str(uuid.uuid4())


def generate_reset_token() -> str:
    """Générer un token de réinitialisation de mot de passe"""
    return secrets.token_urlsafe(32)


def get_password_strength(password: str) -> Dict[str, any]:
    """Évaluer la force d'un mot de passe"""
    score = 0
    feedback = []
    
    if len(password) >= 8:
        score += 1
    else:
        feedback.append("Au moins 8 caractères")
    
    if any(c.isupper() for c in password):
        score += 1
    else:
        feedback.append("Au moins une majuscule")
    
    if any(c.islower() for c in password):
        score += 1
    else:
        feedback.append("Au moins une minuscule")
    
    if any(c.isdigit() for c in password):
        score += 1
    else:
        feedback.append("Au moins un chiffre")
    
    if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        score += 1
    else:
        feedback.append("Au moins un caractère spécial")
    
    strength_levels = ["Très faible", "Faible", "Moyen", "Fort", "Très fort"]
    
    return {
        "score": score,
        "max_score": 5,
        "strength": strength_levels[score] if score < 5 else strength_levels[4],
        "feedback": feedback,
    }