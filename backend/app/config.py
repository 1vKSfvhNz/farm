# backend/app/config.py
"""
Configuration de l'application
Gestion des variables d'environnement et des paramètres
"""

import os
import ast
from pathlib import Path
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, field_validator


class Settings(BaseSettings):
    """Configuration principale de l'application"""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ============ APPLICATION ============
    APP_NAME: str = "Farm Manager API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    SMTP_HOST: str = Field(default="", env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USER: str = Field(default="", env="SMTP_USER")
    SMTP_PASSWORD: str = Field(default="", env="SMTP_PASSWORD")
    SMTP_FROM_EMAIL: str = Field(default="", env="SMTP_FROM_EMAIL")
    SMTP_USE_TLS: bool = Field(default=True, env="SMTP_USE_TLS")

    # Configuration des uploads - Utiliser /tmp en Docker
    @property
    def UPLOAD_DIR(self) -> Path:
        """Retourne un dossier accessible en écriture"""
        # Variable d'environnement prioritaire
        if os.getenv("UPLOAD_DIR"):
            return Path(os.getenv("UPLOAD_DIR"))
        
        # En Docker, utiliser /tmp
        if os.path.exists('/.dockerenv'):
            upload_dir = Path("/tmp/farm_manager_uploads")
        else:
            # En développement local
            upload_dir = Path(__file__).parent.parent / "uploads"
        
        # Créer le dossier s'il n'existe pas
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Changer les permissions si possible
        try:
            os.chmod(upload_dir, 0o777)
        except:
            pass
            
        return upload_dir
    
    UPLOAD_MAX_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    PHOTO_QUALITY: int = 85
    PHOTO_MAX_WIDTH: int = 800
    PHOTO_MAX_HEIGHT: int = 800
    
    @property
    def UPLOAD_URL_PREFIX(self) -> str:
        return "/uploads"
    
    # ============ API ============
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = Field(default=4, ge=1, le=16)

    BACKEND_URL: str = Field(default="http://localhost:8000")
    FRONTEND_URL: str = Field(default="http://localhost:5173")
    
    # CORS - Support des chaînes JSON ou listes
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Origines autorisées pour CORS"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Hôtes autorisés"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    )
    CORS_ALLOW_HEADERS: List[str] = Field(
        default=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"]
    )
    
    @field_validator('CORS_ORIGINS', 'ALLOWED_HOSTS', 'CORS_ALLOW_METHODS', 'CORS_ALLOW_HEADERS', mode='before')
    @classmethod
    def parse_list(cls, value):
        """Convertir une chaîne JSON ou une liste en liste Python"""
        if isinstance(value, str):
            try:
                # Essayer de parser comme JSON
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                # Sinon, séparer par des virgules
                return [item.strip() for item in value.split(',') if item.strip()]
        return value
    
    # ============ DATABASE ============
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "farm_user"
    POSTGRES_PASSWORD: str = "wxcvbnjklm"
    POSTGRES_DB: str = "farm_manager"
    
    @property
    def DATABASE_URL(self) -> str:
        """URL de connexion PostgreSQL asynchrone"""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """URL synchrone pour Alembic"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # ============ REDIS ============
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_ENABLED: bool = Field(default=False)  # Permet de désactiver Redis
    
    @property
    def REDIS_URL(self) -> str:
        """URL de connexion Redis"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # ============ JWT AUTHENTIFICATION ============
    JWT_SECRET_KEY: str = Field(
        default="your-super-secret-jwt-key-change-this-in-production-minimum-32-characters",
        min_length=32
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15  # 15 minutes
    ACCESS_TOKEN_EXPIRE_DAYS: int = 1  # 1 day
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7 

    
    # ============ CELERY ============
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")
    CELERY_TASK_ALWAYS_EAGER: bool = Field(default=False)
    
    # ============ STORAGE ============
    STORAGE_TYPE: str = Field(default="local", pattern="^(local|s3|gcs)$")
    STORAGE_PATH: str = "./storage"
    STORAGE_VIDEO_PATH: str = "./storage/videos"
    STORAGE_EXPORT_PATH: str = "./storage/exports"
    
    # AWS S3 (optionnel)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "eu-west-1"
    AWS_S3_BUCKET: Optional[str] = None
    
    # ============ VIDEO & IA ============
    VIDEO_AI_ENABLED: bool = Field(default=False)
    VIDEO_MODEL_PATH: Optional[str] = "./models/yolov8n.onnx"
    VIDEO_USE_GPU: bool = Field(default=False)
    VIDEO_FPS_TARGET: int = Field(default=15, ge=1, le=60)
    VIDEO_MAX_DURATION_SECONDS: int = Field(default=300, ge=10, le=3600)
    VIDEO_RETENTION_DAYS: int = Field(default=30, ge=1, le=365)
    
    # ============ NOTIFICATIONS ============
    SMTP_HOST: Optional[str] = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: str = "noreply@farm-manager.com"
    
    SMS_ENABLED: bool = False
    SMS_PROVIDER: str = "twilio"
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # ============ WEATHER API ============
    WEATHER_API_ENABLED: bool = False
    WEATHER_API_PROVIDER: str = "openweathermap"
    WEATHER_API_KEY: Optional[str] = None
    WEATHER_LATITUDE: float = 48.8566
    WEATHER_LONGITUDE: float = 2.3522
    
    # ============ RATE LIMITING ============
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_AUTHENTICATED: str = "500/minute"
    RATE_LIMIT_ADMIN: str = "1000/minute"
    
    # ============ BATCH JOBS ============
    BATCH_PREDICTIONS_HOUR: int = Field(default=2, ge=0, le=23)
    BATCH_REPORTS_HOUR: int = Field(default=6, ge=0, le=23)
    BATCH_CLEANUP_HOUR: int = Field(default=3, ge=0, le=23)
    
    # ============ EXPERIMENTAL MODE ============
    EXPERIMENTAL_MODE_ENABLED: bool = True
    REFERENCE_LEARNING_DAYS: int = Field(default=30, ge=7, le=365)
    MIN_PESEE_COUNT_FOR_PREDICTION: int = Field(default=5, ge=1, le=50)
    
    # ============ SECURITY ============
    PASSWORD_MIN_LENGTH: int = 8
    SESSION_EXPIRE_HOURS: int = 24
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15
    
    # ============ BLOCKCHAIN ============
    BLOCKCHAIN_ENABLED: bool = False
    BLOCKCHAIN_NETWORK: str = "sepolia"
    BLOCKCHAIN_RPC_URL: Optional[str] = None
    BLOCKCHAIN_CONTRACT_ADDRESS: Optional[str] = None
    BLOCKCHAIN_PRIVATE_KEY: Optional[str] = None
    
    def validate_environment(self) -> None:
        """Valider la configuration selon l'environnement"""
        if self.ENVIRONMENT == "production":
            assert self.JWT_SECRET_KEY not in [
                "your-secret-key-change-in-production",
                "your-super-secret-jwt-key-change-this-in-production-minimum-32-characters"
            ], "JWT_SECRET_KEY must be changed in production"
            assert self.POSTGRES_PASSWORD not in ["farm_manager_password", "wxcvbnjklm"], \
                "POSTGRES_PASSWORD must be changed in production"
            assert not self.DEBUG, "DEBUG must be False in production"


@lru_cache()
def get_settings() -> Settings:
    """Obtenir les settings (cached)"""
    settings = Settings()
    settings.validate_environment()
    return settings


# Instance globale
settings = get_settings()