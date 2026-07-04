# backend/app/services/user_service.py
"""
Service de gestion des utilisateurs
"""

import logging
from typing import Dict, Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.core.constants import StatutAnimalEnum
from app.core.emails import send_user_updated_email, send_welcome_email
from app.models.animal import Animal
from app.models.apiary import Ruche
from app.models.avicole import Avicole
from app.models.enclos import Enclos
from app.models.entomoculture import EntomocultureLot
from app.models.piscicole import Piscicole
from app.services.id_service import log_action

from ..models.user import Role, User, UserSession, ActionLog, RoleEnum
from ..schemas.user import *
from ..core.security import hash_password, generate_password
from ..redis_client import redis_client

logger = logging.getLogger(__name__)


class UserService:
    """Service de gestion des utilisateurs"""
    
    async def create_user(
        self,
        db: AsyncSession,
        user_data: UserCreate,
        created_by: int
    ) -> Tuple[Optional[UserResponse], Optional[str]]:
        """Créer un nouvel utilisateur"""
        # Vérifier si l'email existe déjà
        stmt = select(User).where(User.email == user_data.email)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            return None, "Cet email est déjà utilisé"
        
        # Vérifier si le matricule existe déjà
        if user_data.employee_id:
            stmt = select(User).where(User.employee_id == user_data.employee_id)
            result = await db.execute(stmt)
            if result.scalar_one_or_none():
                return None, "Ce matricule est déjà utilisé"

        # Générer un mot de passe à 6 chiffres
        password = generate_password(length=6, numeric_only=True)
        
        # Créer l'utilisateur
        user = User(
            email=user_data.email,
            phone=user_data.phone or "",
            username=user_data.username,
            hashed_password=hash_password(password),
            full_name=user_data.full_name,
            is_active=user_data.is_active,
            # Champs employé
            employee_id=user_data.employee_id,
            department=user_data.department,
            hire_date=user_data.hire_date,
            base_salary=user_data.base_salary,
            salary_currency=user_data.salary_currency or "XOF",
            salary_frequency=user_data.salary_frequency or "monthly",
            bonus=user_data.bonus or 0.0,
            employee_status=user_data.employee_status or EmployeeStatusEnum.ACTIF,
            employee_type=user_data.employee_type or EmployeeTypeEnum.PERMANENT,
            bank_name=user_data.bank_name,
            bank_account=user_data.bank_account,
            rib=user_data.rib,
            national_id=user_data.national_id,
            social_security_number=user_data.social_security_number,
            tax_id=user_data.tax_id,
            emergency_contact_name=user_data.emergency_contact_name,
            emergency_contact_phone=user_data.emergency_contact_phone,
            observations=user_data.observations
        )
        db.add(user)
        await db.flush()
        
        # Ajouter les rôles
        if user_data.roles:
            role_stmt = select(Role).where(Role.name.in_(user_data.roles))
            role_result = await db.execute(role_stmt)
            roles = role_result.scalars().all()
            user.roles = roles
        
        # Journaliser
        await log_action(db, created_by, "CREATE_USER", "user", user.id, {"created": user_data.model_dump()})
        await db.commit()
        
        # Envoyer l'email de bienvenue avec le mot de passe à 6 chiffres
        await send_welcome_email(
            to_email=user.email,
            full_name=user.full_name,
            username=user.username,
            password=password,  # Code à 6 chiffres
            employee_id=user.employee_id
        )
        
        logger.info(f"User created: {user.username} by {created_by}")
        return self._to_response(user), None

    async def update_user(
        self,
        db: AsyncSession,
        user_id: int,
        user_data: UserUpdate,
        updated_by: int
    ) -> Tuple[Optional[UserResponse], Optional[str]]:
        """Mettre à jour un utilisateur"""
        stmt = select(User).options(selectinload(User.roles)).where(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return None, "Utilisateur non trouvé"
        
        # Préparer les données avant modification
        old_data = {
            "email": user.email,
            "full_name": user.full_name,
            "employee_id": user.employee_id,
            "department": user.department,
            "base_salary": user.base_salary,
            "employee_status": user.employee_status.value if user.employee_status else None,
            "employee_type": user.employee_type.value if user.employee_type else None,
            "is_active": user.is_active,
            "roles": user.role_list
        }
        
        update_data = {}
        password_changed = False
        new_password = None
        
        # Mettre à jour les champs de base
        if user_data.email is not None:
            stmt = select(User).where(User.email == user_data.email, User.id != user_id)
            result = await db.execute(stmt)
            if result.scalar_one_or_none():
                return None, "Cet email est déjà utilisé"
            user.email = user_data.email
            update_data["email"] = user_data.email
        
        if user_data.phone is not None:
            user.phone = user_data.phone
            update_data["phone"] = user_data.phone
        
        if user_data.username is not None:
            user.username = user_data.username
            update_data["username"] = user_data.username
        
        if user_data.full_name is not None:
            user.full_name = user_data.full_name
            update_data["full_name"] = user_data.full_name
        
        if user_data.is_active is not None:
            user.is_active = user_data.is_active
            update_data["is_active"] = "Actif" if user_data.is_active else "Inactif"
        
        # Mettre à jour les champs employé
        employee_fields = [
            'employee_id', 'department', 'hire_date',
            'base_salary', 'salary_currency', 'salary_frequency', 'bonus',
            'employee_status', 'employee_type',
            'bank_name', 'bank_account', 'rib',
            'national_id', 'social_security_number', 'tax_id',
            'emergency_contact_name', 'emergency_contact_phone', 'observations'
        ]
        
        # Labels pour les champs
        field_labels = {
            "email": "📧 Email",
            "phone": "📱 Téléphone",
            "username": "👤 Nom d'utilisateur",
            "full_name": "📝 Nom complet",
            "is_active": "🔓 Statut",
            "employee_id": "📋 Matricule",
            "department": "🏢 Département",
            "hire_date": "📅 Date d'embauche",
            "base_salary": "💰 Salaire de base",
            "salary_currency": "💱 Devise",
            "salary_frequency": "📊 Fréquence",
            "bonus": "🎯 Prime",
            "employee_status": "📌 Statut employé",
            "employee_type": "🏷️ Type d'employé",
            "bank_name": "🏦 Banque",
            "bank_account": "🔢 Compte bancaire",
            "rib": "📄 RIB",
            "national_id": "🪪 CNI/Passeport",
            "social_security_number": "🛡️ Sécurité sociale",
            "tax_id": "📋 Numéro fiscal",
            "emergency_contact_name": "👤 Contact urgence",
            "emergency_contact_phone": "📱 Tél. urgence",
            "observations": "📝 Observations"
        }
        
        for field in employee_fields:
            value = getattr(user_data, field, None)
            if value is not None:
                setattr(user, field, value)
                label = field_labels.get(field, field.replace('_', ' ').title())
                # Formater la valeur pour l'affichage
                if hasattr(value, 'value'):
                    formatted_value = value.value
                elif isinstance(value, datetime):
                    formatted_value = value.strftime('%d/%m/%Y')
                elif isinstance(value, float):
                    formatted_value = f"{value:,.0f} FCFA" if field == 'base_salary' else str(value)
                else:
                    formatted_value = str(value)
                update_data[label] = formatted_value
        
        # Gestion du mot de passe (si fourni)
        if user_data.password:
            user.hashed_password = hash_password(user_data.password)
            password_changed = True
            new_password = user_data.password
            update_data["🔑 Mot de passe"] = "Modifié"
        
        # Mettre à jour les rôles (si fournis)
        if user_data.roles is not None:
            # Supprimer les rôles existants
            user.roles = []
            # Ajouter les nouveaux rôles
            if user_data.roles:
                role_stmt = select(Role).where(Role.name.in_(user_data.roles))
                role_result = await db.execute(role_stmt)
                new_roles = role_result.scalars().all()
                user.roles = new_roles
            update_data["🎯 Rôles"] = ", ".join(user_data.roles) if user_data.roles else "Aucun"
        
        # Journaliser
        await log_action(
            db, 
            updated_by, 
            "UPDATE_USER", 
            "user", 
            user_id, 
            {"old": old_data, "new": update_data}
        )
        await db.commit()
        
        # Rafraîchir l'utilisateur
        stmt_refresh = select(User).options(selectinload(User.roles)).where(User.id == user_id)
        result_refresh = await db.execute(stmt_refresh)
        refreshed_user = result_refresh.scalar_one_or_none()
        
        # Envoyer l'email de notification
        if update_data:
            # Si le mot de passe a été changé, envoyer un email séparé ou inclure dans l'update
            if password_changed:
                await send_user_updated_email(
                    to_email=user.email,
                    full_name=user.full_name,
                    changes=update_data,
                    new_password=new_password,
                    is_active=user.is_active
                )
            else:
                await send_user_updated_email(
                    to_email=user.email,
                    full_name=user.full_name,
                    changes=update_data,
                    new_password=None,
                    is_active=user.is_active
                )
        
        logger.info(f"User updated: {user.username} by {updated_by}")
        return self._to_response(refreshed_user or user), None


    async def get_user(
        self,
        db: AsyncSession,
        user_id: int,
        include_deleted: bool = False
    ) -> Optional[UserResponse]:
        """Obtenir un utilisateur par son ID"""
        stmt = select(User).where(User.id == user_id)
        if not include_deleted:
            stmt = stmt.where(User.deleted_at.is_(None))
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        return self._to_response(user) if user else None
    
    async def get_user_by_username(
        self,
        db: AsyncSession,
        username: str
    ) -> Optional[User]:
        """Obtenir un utilisateur par son nom d'utilisateur"""
        stmt = select(User).where(User.username == username, User.deleted_at.is_(None))
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_users(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        # Paramètres de recherche
        search: Optional[str] = None,  # Recherche textuelle (nom, email, username, matricule)
        roles: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        # Filtres employé
        employee_id: Optional[str] = None,
        department: Optional[str] = None,
        employee_status: Optional[List[str]] = None,
        employee_type: Optional[List[str]] = None,
        hire_date_from: Optional[date] = None,
        hire_date_to: Optional[date] = None,
        # Filtres salaire
        salary_min: Optional[float] = None,
        salary_max: Optional[float] = None,
        # Filtres dates
        created_from: Optional[datetime] = None,
        created_to: Optional[datetime] = None,
        updated_from: Optional[datetime] = None,
        updated_to: Optional[datetime] = None,
        # Tri
        order_by: str = "created_at",
        order_direction: str = "desc"
    ) -> Tuple[List[UserResponse], int]:
        """
        Obtenir la liste des utilisateurs avec filtres avancés
        
        Args:
            skip: Nombre d'éléments à sauter
            limit: Nombre maximum d'éléments
            search: Recherche textuelle (nom, email, username, matricule)
            roles: Filtrer par rôles
            is_active: Filtrer par statut actif
            employee_id: Filtrer par matricule
            department: Filtrer par département
            employee_status: Filtrer par statut employé (liste)
            employee_type: Filtrer par type d'employé (liste)
            hire_date_from: Date d'embauche minimale
            hire_date_to: Date d'embauche maximale
            salary_min: Salaire minimum
            salary_max: Salaire maximum
            created_from: Date de création minimale
            created_to: Date de création maximale
            updated_from: Date de mise à jour minimale
            updated_to: Date de mise à jour maximale
            order_by: Champ de tri
            order_direction: Direction du tri (asc/desc)
        
        Returns:
            Tuple[List[UserResponse], int]: (Liste des utilisateurs, Total)
        """
        stmt = select(User).options(selectinload(User.roles))
        
        # 1. Recherche textuelle
        if search and search.strip():
            search_term = f"%{search.strip()}%"
            stmt = stmt.where(
                or_(
                    User.full_name.ilike(search_term),
                    User.email.ilike(search_term),
                    User.username.ilike(search_term),
                    User.employee_id.ilike(search_term),
                    User.phone.ilike(search_term)
                )
            )
        
        # 2. Filtres de base
        if roles:
            stmt = stmt.join(User.roles).where(Role.name.in_(roles))
        
        if is_active is not None:
            stmt = stmt.where(User.is_active == is_active)
        
        # 3. Filtres employé
        if employee_id:
            stmt = stmt.where(User.employee_id.ilike(f"%{employee_id}%"))
        
        if department:
            stmt = stmt.where(User.department.ilike(f"%{department}%"))
        
        if employee_status:
            stmt = stmt.where(User.employee_status.in_(employee_status))
        
        if employee_type:
            stmt = stmt.where(User.employee_type.in_(employee_type))
        
        if hire_date_from:
            stmt = stmt.where(User.hire_date >= hire_date_from)
        
        if hire_date_to:
            stmt = stmt.where(User.hire_date <= hire_date_to)
        
        # 4. Filtres salaire
        if salary_min is not None:
            stmt = stmt.where(User.base_salary >= salary_min)
        
        if salary_max is not None:
            stmt = stmt.where(User.base_salary <= salary_max)
        
        # 5. Filtres dates
        if created_from:
            stmt = stmt.where(User.created_at >= created_from)
        
        if created_to:
            stmt = stmt.where(User.created_at <= created_to)
        
        if updated_from:
            stmt = stmt.where(User.updated_at >= updated_from)
        
        if updated_to:
            stmt = stmt.where(User.updated_at <= updated_to)
        
        # 6. Exclure les utilisateurs supprimés (soft delete)
        stmt = stmt.where(User.deleted_at.is_(None))
        
        # 7. Compter le total AVANT le tri et la pagination
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await db.execute(count_stmt)
        total = count_result.scalar() or 0
        
        # 8. Tri
        order_column = getattr(User, order_by, User.created_at)
        if order_direction.lower() == "asc":
            stmt = stmt.order_by(order_column.asc())
        else:
            stmt = stmt.order_by(order_column.desc())
        
        # 9. Pagination
        stmt = stmt.offset(skip).limit(limit)
        
        # 10. Exécution
        result = await db.execute(stmt)
        users = result.scalars().all()
        
        # 11. Conversion en réponses
        user_responses = [self._to_response(user) for user in users]
        
        return user_responses, total
            
    def _sanitize_for_json(self, obj):
        """Convertir les objets non sérialisables en JSON"""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        if hasattr(obj, 'value'):  # Enum
            return obj.value
        if hasattr(obj, '__dict__'):
            return {k: self._sanitize_for_json(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        return str(obj)
    
    async def get_user_actions(
        self,
        db: AsyncSession,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[ActionLog]:
        """Obtenir l'historique des actions d'un utilisateur"""
        stmt = select(ActionLog).where(ActionLog.user_id == user_id)
        stmt = stmt.offset(skip).limit(limit).order_by(ActionLog.created_at.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    def _to_response(self, user: User) -> UserResponse:
        """Convertir un modèle User en UserResponse"""
        return UserResponse.from_orm_with_roles(user)
    
    async def get_user_counts_by_permissions(
        self,
        db: AsyncSession,
        user: User
    ) -> Dict[str, Optional[int]]:
        """
        Obtenir les compteurs pour toutes les entités en fonction des privilèges de l'utilisateur
        
        Args:
            db: Session de base de données
            user: Utilisateur courant
            
        Returns:
            Dictionnaire contenant les compteurs pour chaque entité
        """        
        # Définir les permissions pour chaque entité
        permissions_config = {
            "users": {
                "model": User,
                "permission": None,
            },
            "enclos": {
                "model": Enclos,
                "permission": None,
                "extra_roles": [RoleEnum.RESPONSABLE_ENCLOS]
            },
            "bovin": {
                "model": Animal,
                "type_espece": "bovin",
                "permission": "bovin"
            },
            "ovin": {
                "model": Animal,
                "type_espece": "ovin",
                "permission": "ovin"
            },
            "caprin": {
                "model": Animal,
                "type_espece": "caprin",
                "permission": "caprin"
            },
            "avicole": {
                "model": Avicole,
                "permission": "avicole"
            },
            "piscicole": {
                "model": Piscicole,
                "permission": "piscicole"
            },
            "apiculture": {
                "model": Ruche,
                "permission": "apiculture"
            },
            "entomoculture": {
                "model": EntomocultureLot,
                "permission": "entomoculture"
            }
        }
        
        is_super_admin = user.has_role(RoleEnum.SUPER_ADMIN)
        has_vision_globale = user.has_role(RoleEnum.VISION_GLOBALE)
        
        # Liste des rôles techniciens pour l'accès aux enclos
        technicien_roles = [
            RoleEnum.BOVIN_TECHNICIEN,
            RoleEnum.OVIN_TECHNICIEN,
            RoleEnum.CAPRIN_TECHNICIEN,
            RoleEnum.AVICOLE_TECHNICIEN,
            RoleEnum.PISCICOLE_TECHNICIEN,
            RoleEnum.APICULTURE_TECHNICIEN,
            RoleEnum.ENTOMOCULTURE_TECHNICIEN,
        ]
        
        result_counts = {}
        
        for key, config in permissions_config.items():
            model = config["model"]
            permission = config.get("permission")
            extra_roles = config.get("extra_roles", [])
            type_espece = config.get("type_espece")
            
            # Vérifier si l'utilisateur peut accéder à cette entité
            can_access = False
            
            if is_super_admin or has_vision_globale:
                can_access = True
            elif permission and user.has_permission(permission, "read"):
                can_access = True
            elif extra_roles and user.has_any_role(extra_roles):
                can_access = True
            elif key == "enclos" and user.has_any_role(technicien_roles):
                can_access = True
            
            if can_access:
                # Compter les entrées
                stmt = select(func.count(model.id))
                
                # Pour les animaux (bovin, ovin, caprin), filtrer par type_espece et statut
                if type_espece:
                    stmt = stmt.where(
                        model.type_espece == type_espece,
                        model.statut.in_([StatutAnimalEnum.VIVANT, StatutAnimalEnum.TRANSFERE])
                    )
                
                result = await db.execute(stmt)
                count = result.scalar() or 0
                result_counts[key] = count
            else:
                result_counts[key] = None
        
        return result_counts


user_service = UserService()