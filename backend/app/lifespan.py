# backend/app/lifespan.py
"""
Gestion du cycle de vie de l'application FastAPI
Démarrage et arrêt des connexions
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .config import settings
from .database import init_db, close_db, create_tables, db_manager, update_exports
from .redis_client import init_redis, close_redis
from .models.user import User, Role, RoleEnum
from .core.security import hash_password

logger = logging.getLogger(__name__)


async def create_admin_user_if_empty():
    """
    Crée un utilisateur admin par défaut si aucun utilisateur n'existe
    """
    try:
        async with db_manager.async_session_maker() as db:
            # Vérifier s'il y a des utilisateurs AVEC chargement explicite des rôles
            stmt = select(User).options(selectinload(User.roles))
            result = await db.execute(stmt)
            users = result.scalars().all()
            
            if len(users) == 0:
                logger.info("=" * 50)
                logger.info("📝 Aucun utilisateur trouvé. Création de l'utilisateur admin par défaut...")
                logger.info("=" * 50)
                
                # Récupérer ou créer les rôles
                roles_to_create = [
                    RoleEnum.SUPER_ADMIN,
                    RoleEnum.BOVIN_ADMIN, RoleEnum.BOVIN_TECHNICIEN, RoleEnum.BOVIN_OBSERVATEUR,
                    RoleEnum.OVIN_ADMIN, RoleEnum.OVIN_TECHNICIEN, RoleEnum.OVIN_OBSERVATEUR,
                    RoleEnum.CAPRIN_ADMIN, RoleEnum.CAPRIN_TECHNICIEN, RoleEnum.CAPRIN_OBSERVATEUR,
                    RoleEnum.AVICOLE_ADMIN, RoleEnum.AVICOLE_TECHNICIEN, RoleEnum.AVICOLE_OBSERVATEUR,
                    RoleEnum.PISCICOLE_ADMIN, RoleEnum.PISCICOLE_TECHNICIEN, RoleEnum.PISCICOLE_OBSERVATEUR,
                    RoleEnum.APICULTURE_ADMIN, RoleEnum.APICULTURE_TECHNICIEN, RoleEnum.APICULTURE_OBSERVATEUR,
                    RoleEnum.ENTOMOCULTURE_ADMIN, RoleEnum.ENTOMOCULTURE_TECHNICIEN, RoleEnum.ENTOMOCULTURE_OBSERVATEUR,
                    RoleEnum.VETERINAIRE, RoleEnum.RESPONSABLE_ENCLOS, RoleEnum.COMPTABLE, RoleEnum.VISION_GLOBALE
                ]
                
                all_roles = []
                for role_enum in roles_to_create:
                    stmt = select(Role).where(Role.name == role_enum)
                    result = await db.execute(stmt)
                    role = result.scalar_one_or_none()
                    if not role:
                        role = Role(name=role_enum, description=f"Rôle: {role_enum.value}")
                        db.add(role)
                        await db.flush()
                    all_roles.append(role)
                
                # Créer l'utilisateur admin
                admin_user = User(
                    email="ozias@farm-manager.bf",
                    phone="+22661506121",
                    username="Ozias",
                    hashed_password=hash_password("12345678"),
                    full_name="Ozias Administrateur",
                    is_active=True
                )
                db.add(admin_user)
                await db.flush()
                
                # Ajouter tous les rôles à l'utilisateur - AVEC await pour éviter le lazy loading
                # Récupérer l'utilisateur avec ses rôles chargés
                stmt = select(User).where(User.id == admin_user.id).options(selectinload(User.roles))
                result = await db.execute(stmt)
                admin_user_with_roles = result.scalar_one()
                
                for role in all_roles:
                    admin_user_with_roles.roles.append(role)
                
                await db.commit()
                
                logger.info("=" * 50)
                logger.info("✅ UTILISATEUR ADMIN CRÉÉ AVEC SUCCÈS !")
                logger.info("=" * 50)
                logger.info(f"   👤 Nom d'utilisateur: Ozias")
                logger.info(f"   📞 Téléphone: +22661506121")
                logger.info(f"   🔑 Mot de passe: 12345678")
                logger.info(f"   ⭐ Rôles: {len(all_roles)} rôles attribués")
                logger.info("=" * 50)
                logger.info("🌐 Connectez-vous avec:")
                logger.info("   URL: http://localhost:8000/docs")
                logger.info("   Identifiant: Ozias ou +22661506121")
                logger.info("   Mot de passe: 12345678")
                logger.info("=" * 50)
            else:
                logger.info(f"✅ {len(users)} utilisateur(s) existant(s) dans la base de données.")
                for user in users:
                    role_names = [role.name.value if hasattr(role.name, 'value') else str(role.name) for role in user.roles]
                    logger.info(f"   - {user.username} (rôles: {role_names})")
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'utilisateur admin: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application
    Démarrage et arrêt des connexions
    """
    # ============ DÉMARRAGE ============
    logger.info("=" * 50)
    logger.info(f"Démarrage de {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environnement: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info("=" * 50)
    
    # Initialisation des connexions
    try:
        logger.info("Initialisation de la base de données...")
        init_db()
        logger.info("Base de données initialisée avec succès")
        
        # Mettre à jour les exports globaux
        update_exports()
        
        # Création des tables en développement uniquement
        if settings.ENVIRONMENT == "development":
            logger.info("Création des tables (mode développement)...")
            await create_tables()
            logger.info("Tables créées avec succès")
        
        # Créer un utilisateur admin si la base est vide
        logger.info("Vérification des utilisateurs...")
        await create_admin_user_if_empty()
        
        logger.info("Initialisation de Redis...")
        await init_redis()
        logger.info("Redis initialisé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield  # L'application tourne ici
    
    # ============ ARRÊT ============
    logger.info("=" * 50)
    logger.info("Arrêt de l'application...")
    logger.info("=" * 50)
    
    try:
        logger.info("Fermeture de Redis...")
        await close_redis()
        logger.info("Redis fermé")
        
        logger.info("Fermeture de la base de données...")
        await close_db()
        logger.info("Base de données fermée")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {e}")