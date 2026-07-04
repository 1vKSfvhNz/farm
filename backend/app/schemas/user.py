# backend/app/schemas/user.py
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from typing import Optional, List
from datetime import datetime, date
from enum import Enum

from app.models.user import RoleEnum, EmployeeStatusEnum, EmployeeTypeEnum, User


class EmployeeStatus(str, Enum):
    ACTIF = "actif"
    CONGE = "conge"
    MALADIE = "maladie"
    SUSPENDU = "suspendu"
    LICENCIE = "licencie"
    RETRAITE = "retraite"
    STAGIAIRE = "stagiaire"


class EmployeeType(str, Enum):
    PERMANENT = "permanent"
    STAGIAIRE = "stagiaire"
    CONTRACTUEL = "contractuel"
    SAISONNIER = "saisonnier"
    CONSULTANT = "consultant"


class LengthResponse(BaseModel):
    users_length: Optional[int] = 0
    enclos_length: Optional[int] = 0
    bovins_length: Optional[int] = 0
    ovins_length: Optional[int] = 0
    caprins_length: Optional[int] = 0
    avicoles_length: Optional[int] = 0
    piscicoles_length: Optional[int] = 0
    ruches_length: Optional[int] = 0
    nids_length: Optional[int] = 0


# ===== CHAMPS EMPLOYÉ =====
class EmployeeInfoBase(BaseModel):
    """Informations employé de base"""
    employee_id: Optional[str] = Field(None, max_length=50, description="Matricule")
    position: Optional[str] = Field(None, max_length=100, description="Poste")
    department: Optional[str] = Field(None, max_length=100, description="Département")
    hire_date: Optional[date] = Field(None, description="Date d'embauche")
    
    # Salaire
    base_salary: Optional[float] = Field(None, ge=0, description="Salaire de base")
    salary_currency: str = Field("XOF", max_length=3, description="Devise")
    salary_frequency: str = Field("monthly", max_length=20, description="Fréquence")
    bonus: Optional[float] = Field(0.0, ge=0, description="Prime")
    
    # Statut
    employee_status: Optional[EmployeeStatus] = EmployeeStatus.ACTIF
    employee_type: Optional[EmployeeType] = EmployeeType.PERMANENT
    
    # Informations bancaires
    bank_name: Optional[str] = Field(None, max_length=100)
    bank_account: Optional[str] = Field(None, max_length=50)
    rib: Optional[str] = Field(None, max_length=50)
    
    # Informations administratives
    national_id: Optional[str] = Field(None, max_length=50)
    social_security_number: Optional[str] = Field(None, max_length=50)
    tax_id: Optional[str] = Field(None, max_length=50)
    
    # Contacts d'urgence
    emergency_contact_name: Optional[str] = Field(None, max_length=100)
    emergency_contact_phone: Optional[str] = Field(None, max_length=32)
    
    observations: Optional[str] = Field(None)


class UserBase(BaseModel):
    email: EmailStr
    phone: Optional[str] = Field(None, min_length=8, max_length=32)
    username: str = Field(..., min_length=3, max_length=64)
    full_name: str = Field(..., min_length=1, max_length=128)
    is_active: bool = True
    roles: List[str] = []


class UserCreate(UserBase, EmployeeInfoBase):
    """Création d'un utilisateur avec informations employé"""
    # Les champs EmployeeInfoBase sont hérités et optionnels
    pass

class UserUpdate(BaseModel):
    """Mise à jour d'un utilisateur"""
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, min_length=8, max_length=32)
    username: Optional[str] = Field(None, min_length=3, max_length=64)
    full_name: Optional[str] = Field(None, min_length=1, max_length=128)
    is_active: Optional[bool] = None
    roles: Optional[List[str]] = None
    password: Optional[str] = Field(None, min_length=6, max_length=128, description="Mot de passe")
    
    # Champs employé (optionnels)
    employee_id: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    hire_date: Optional[date] = None
    base_salary: Optional[float] = Field(None, ge=0)
    salary_currency: Optional[str] = Field(None, max_length=3)
    salary_frequency: Optional[str] = Field(None, max_length=20)
    bonus: Optional[float] = Field(None, ge=0)
    employee_status: Optional[EmployeeStatus] = None
    employee_type: Optional[EmployeeType] = None
    bank_name: Optional[str] = Field(None, max_length=100)
    bank_account: Optional[str] = Field(None, max_length=50)
    rib: Optional[str] = Field(None, max_length=50)
    national_id: Optional[str] = Field(None, max_length=50)
    social_security_number: Optional[str] = Field(None, max_length=50)
    tax_id: Optional[str] = Field(None, max_length=50)
    emergency_contact_name: Optional[str] = Field(None, max_length=100)
    emergency_contact_phone: Optional[str] = Field(None, max_length=32)
    observations: Optional[str] = None


class UserResponse(UserBase, EmployeeInfoBase):
    """Réponse complète utilisateur"""
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
    
    @classmethod
    def from_orm_with_roles(cls, user: "User") -> "UserResponse":
        """Méthode utilitaire pour créer une réponse sans lazy loading"""
        
        # Récupérer et filtrer les rôles
        roles = cls._filter_roles(user)
        
        return cls(
            id=user.id,
            email=user.email,
            phone=user.phone or "",
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active if hasattr(user, 'is_active') else True,
            roles=roles,
            created_at=user.created_at if hasattr(user, 'created_at') else datetime.now(),
            updated_at=user.updated_at if hasattr(user, 'updated_at') else datetime.now(),
            # Champs employé
            employee_id=getattr(user, 'employee_id', None),
            department=getattr(user, 'department', None),
            hire_date=getattr(user, 'hire_date', None),
            base_salary=getattr(user, 'base_salary', None),
            salary_currency=getattr(user, 'salary_currency', "XOF"),
            salary_frequency=getattr(user, 'salary_frequency', "monthly"),
            bonus=getattr(user, 'bonus', 0.0),
            employee_status=getattr(user, 'employee_status', None),
            employee_type=getattr(user, 'employee_type', None),
            bank_name=getattr(user, 'bank_name', None),
            bank_account=getattr(user, 'bank_account', None),
            rib=getattr(user, 'rib', None),
            national_id=getattr(user, 'national_id', None),
            social_security_number=getattr(user, 'social_security_number', None),
            tax_id=getattr(user, 'tax_id', None),
            emergency_contact_name=getattr(user, 'emergency_contact_name', None),
            emergency_contact_phone=getattr(user, 'emergency_contact_phone', None),
            observations=getattr(user, 'observations', None)
        )
    
    @classmethod
    def _filter_roles(cls, user: "User") -> List[str]:
        """
        Filtre les rôles pour ne retourner que ceux qui sont pertinents.
        - Si SUPER_ADMIN est présent, on ne retourne que SUPER_ADMIN
        - Sinon, on retourne :
            - Le rôle admin de la catégorie la plus élevée
            - Les rôles transverses (Vétérinaire, Responsable Enclos, Comptable, Vision Globale)
        """
        if not hasattr(user, 'roles') or not user.roles:
            return []
        
        try:
            # Extraire les noms des rôles
            all_roles = []
            for role in user.roles:
                if hasattr(role.name, 'value'):
                    all_roles.append(role.name.value)
                else:
                    all_roles.append(str(role.name))
            
            # Rôles transverses (toujours inclus si présents)
            transverse_roles = [
                "veterinaire",
                "responsable_enclos", 
                "responsable_account",
                "vision_globale"
            ]
            
            # Vérifier si SUPER_ADMIN est présent
            if "super_admin" in all_roles:
                # Ne retourner que SUPER_ADMIN (les transverses sont redondants)
                return ["super_admin"]
            
            # Liste des rôles admin par catégorie (ordre de priorité)
            admin_roles = [
                "bovin_admin",
                "ovin_admin", 
                "caprin_admin",
                "avicole_admin",
                "piscicole_admin",
                "apiculture_admin",
                "entomoculture_admin"
            ]
            
            # Récupérer les rôles transverses présents
            present_transverse = [role for role in transverse_roles if role in all_roles]
            
            # Trouver le premier rôle admin présent (le plus prioritaire)
            admin_role = None
            for role in admin_roles:
                if role in all_roles:
                    admin_role = role
                    break
            
            # Construire la liste des rôles à retourner
            result = []
            
            # Ajouter le rôle admin s'il existe
            if admin_role:
                result.append(admin_role)
            
            # Ajouter tous les rôles transverses présents
            result.extend(present_transverse)
            
            # Si aucun admin trouvé, retourner tous les rôles (pour les techniciens, observateurs, etc.)
            if not result:
                return all_roles
            
            return result
            
        except Exception:
            return []

class UserSessionResponse(BaseModel):
    id: int
    user_id: int
    ip_address: Optional[str]
    user_agent: Optional[str]
    device_info: Optional[dict]
    created_at: datetime
    expires_at: datetime
    is_valid: bool
    logout_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


class ActionLogResponse(BaseModel):
    id: int
    user_id: int
    action: str
    entity_type: Optional[str]
    entity_id: Optional[int]
    details: Optional[dict]
    ip_address: Optional[str]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)