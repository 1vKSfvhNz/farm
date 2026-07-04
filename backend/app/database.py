# backend/app/database.py
"""
Configuration de la base de données PostgreSQL
SQLAlchemy asynchrone avec connexion pool
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy import text
from sqlalchemy.pool import NullPool

from .config import settings
from .models.base import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gestionnaire de connexions database"""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._async_session_maker: Optional[async_sessionmaker] = None
        self._initialized = False
    
    def init(self) -> None:
        """Initialiser les connexions à la base de données"""
        # Configuration du pool
        pool_config = {}
        if settings.ENVIRONMENT == "production":
            pool_config = {
                "pool_size": 20,
                "max_overflow": 40,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": True,
            }
        else:
            pool_config = {"poolclass": NullPool}
        
        # Créer l'engine asynchrone
        self._engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,  # Désactiver les logs SQL
            future=True,
            **pool_config
        )
        
        # Créer le session maker
        self._async_session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        self._initialized = True
        logger.info(f"Database initialized: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
    
    async def close(self) -> None:
        """Fermer toutes les connexions"""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
        self._initialized = False
    
    @property
    def engine(self) -> AsyncEngine:
        """Obtenir l'engine"""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self._engine
    
    @property
    def async_session_maker(self) -> async_sessionmaker:
        """Obtenir le session maker"""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self._async_session_maker
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Obtenir une session pour les endpoints FastAPI"""
        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Database session error: {e}")
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """Contexte de transaction"""
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()
    
    async def check_health(self) -> bool:
        """Vérifier la santé de la connexion database"""
        if not self._initialized:
            return False
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Instance globale
db_manager = DatabaseManager()


# Fonctions de commodité pour les endpoints FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dépendance FastAPI pour obtenir une session database.
    Utilisation: async def my_endpoint(db: AsyncSession = Depends(get_db))
    """
    async for session in db_manager.get_session():
        yield session


def init_db() -> None:
    """Initialiser la base de données (appel au démarrage)"""
    db_manager.init()


async def close_db() -> None:
    """Fermer la base de données (appel à l'arrêt)"""
    await db_manager.close()


async def create_tables() -> None:
    """Créer toutes les tables (développement uniquement)"""
    if settings.ENVIRONMENT != "production" and db_manager.is_initialized:
        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
    elif settings.ENVIRONMENT == "production":
        logger.warning("Table creation skipped in production. Use migrations instead.")


async def drop_tables() -> None:
    """Supprimer toutes les tables (développement uniquement)"""
    if settings.ENVIRONMENT != "production" and db_manager.is_initialized:
        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")
    else:
        logger.warning("Table dropping is not allowed in production")


# ============ EXPORTS POUR LE FICHIER __init__.py ============
def update_exports():
    """Mettre à jour les exports globaux après initialisation"""
    global engine, async_session_maker
    if db_manager.is_initialized:
        engine = db_manager.engine
        async_session_maker = db_manager.async_session_maker
    else:
        engine = None
        async_session_maker = None


engine = None
async_session_maker = None


def get_engine():
    """Obtenir l'engine après initialisation"""
    return db_manager.engine if db_manager.is_initialized else None


def get_async_session_maker():
    """Obtenir le session maker après initialisation"""
    return db_manager.async_session_maker if db_manager.is_initialized else None