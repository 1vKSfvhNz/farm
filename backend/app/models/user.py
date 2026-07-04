# backend/app/models/user.py
from typing import List, Optional
from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey, Text, Boolean, JSON, Table, Float, Date
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from .base import Base, TimestampMixin, SoftDeleteMixin


class RoleEnum(str, enum.Enum):
    # Rôles administratifs
    SUPER_ADMIN = "super_admin"
    
    # Rôles par type d'élevage - Bovins
    BOVIN_ADMIN = "bovin_admin"
    BOVIN_TECHNICIEN = "bovin_technicien"
    BOVIN_OBSERVATEUR = "bovin_observateur"
    
    # Rôles par type d'élevage - Ovins
    OVIN_ADMIN = "ovin_admin"
    OVIN_TECHNICIEN = "ovin_technicien"
    OVIN_OBSERVATEUR = "ovin_observateur"
    
    # Rôles par type d'élevage - Caprins
    CAPRIN_ADMIN = "caprin_admin"
    CAPRIN_TECHNICIEN = "caprin_technicien"
    CAPRIN_OBSERVATEUR = "caprin_observateur"
    
    # Rôles par type d'élevage - Avicoles
    AVICOLE_ADMIN = "avicole_admin"
    AVICOLE_TECHNICIEN = "avicole_technicien"
    AVICOLE_OBSERVATEUR = "avicole_observateur"
    
    # Rôles par type d'élevage - Piscicoles
    PISCICOLE_ADMIN = "piscicole_admin"
    PISCICOLE_TECHNICIEN = "piscicole_technicien"
    PISCICOLE_OBSERVATEUR = "piscicole_observateur"
    
    # Rôles par type d'élevage - Apiculture
    APICULTURE_ADMIN = "apiculture_admin"
    APICULTURE_TECHNICIEN = "apiculture_technicien"
    APICULTURE_OBSERVATEUR = "apiculture_observateur"
    
    # Rôles par type d'élevage - Entomoculture
    ENTOMOCULTURE_ADMIN = "entomoculture_admin"
    ENTOMOCULTURE_TECHNICIEN = "entomoculture_technicien"
    ENTOMOCULTURE_OBSERVATEUR = "entomoculture_observateur"
    
    # Rôles transverses
    VETERINAIRE = "veterinaire"
    RESPONSABLE_ENCLOS = "responsable_enclos"
    COMPTABLE = "responsable_account"
    VISION_GLOBALE = "vision_globale"


class EmployeeStatusEnum(str, enum.Enum):
    """Statut d'emploi d'un employé"""
    ACTIF = "actif"
    CONGE = "conge"
    MALADIE = "maladie"
    SUSPENDU = "suspendu"
    LICENCIE = "licencie"
    RETRAITE = "retraite"
    STAGIAIRE = "stagiaire"


class EmployeeTypeEnum(str, enum.Enum):
    """Type d'employé"""
    PERMANENT = "permanent"
    STAGIAIRE = "stagiaire"
    CONTRACTUEL = "contractuel"
    SAISONNIER = "saisonnier"
    CONSULTANT = "consultant"


# Table d'association pour les rôles multiples
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)
)


class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"
    
    # Informations de connexion
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    phone = Column(String(32), unique=True, nullable=False, index=True)
    username = Column(String(32), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(64), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # ===== CHAMPS EMPLOYÉ =====
    # Informations professionnelles
    employee_id = Column(String(50), unique=True, nullable=True, index=True)  # Matricule
    department = Column(String(100), nullable=True)  # Département/Service
    hire_date = Column(Date, nullable=True)  # Date d'embauche
    
    # Salaire et rémunération
    base_salary = Column(Float, nullable=True)  # Salaire de base
    salary_currency = Column(String(3), default="XOF")  # Devise (FCFA)
    salary_frequency = Column(String(20), default="monthly")  # Mensuel, horaire, etc.
    bonus = Column(Float, default=0.0)  # Prime/Bonus
    
    # Statut et type
    employee_status = Column(Enum(EmployeeStatusEnum), default=EmployeeStatusEnum.ACTIF)
    employee_type = Column(Enum(EmployeeTypeEnum), default=EmployeeTypeEnum.PERMANENT)
    
    # Informations bancaires
    bank_name = Column(String(100), nullable=True)
    bank_account = Column(String(50), nullable=True)
    rib = Column(String(50), nullable=True)  # RIB/IBAN
    
    # Informations administratives
    national_id = Column(String(50), nullable=True)  # CNI/Passeport
    social_security_number = Column(String(50), nullable=True)  # Numéro de sécurité sociale
    tax_id = Column(String(50), nullable=True)  # Numéro fiscal
    
    # Autres
    emergency_contact_name = Column(String(100), nullable=True)
    emergency_contact_phone = Column(String(32), nullable=True)
    observations = Column(Text, nullable=True)
    
    # Relations
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    actions = relationship("ActionLog", back_populates="user", cascade="all, delete-orphan")
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    
    @property
    def role_list(self) -> List[str]:
        """Retourne la liste des noms de rôles de l'utilisateur"""
        roles = []
        for role in self.roles:
            if hasattr(role.name, 'value'):
                roles.append(role.name.value)
            else:
                roles.append(str(role.name))
        return roles
    
    def has_role(self, role: RoleEnum) -> bool:
        """Vérifie si l'utilisateur a un rôle spécifique"""
        return any(r.name == role for r in self.roles)
    
    def has_any_role(self, roles: List[RoleEnum]) -> bool:
        """Vérifie si l'utilisateur a au moins un des rôles"""
        return any(self.has_role(r) for r in roles)
    
    def has_permission(self, espece: str, action: str = "read") -> bool:
        """
        Vérifie les permissions pour une espèce et une action
        """
        if self.has_role(RoleEnum.SUPER_ADMIN):
            return True
        
        role_map = {
            "bovin": (RoleEnum.BOVIN_ADMIN, RoleEnum.BOVIN_TECHNICIEN, RoleEnum.BOVIN_OBSERVATEUR),
            "ovin": (RoleEnum.OVIN_ADMIN, RoleEnum.OVIN_TECHNICIEN, RoleEnum.OVIN_OBSERVATEUR),
            "caprin": (RoleEnum.CAPRIN_ADMIN, RoleEnum.CAPRIN_TECHNICIEN, RoleEnum.CAPRIN_OBSERVATEUR),
            "avicole": (RoleEnum.AVICOLE_ADMIN, RoleEnum.AVICOLE_TECHNICIEN, RoleEnum.AVICOLE_OBSERVATEUR),
            "piscicole": (RoleEnum.PISCICOLE_ADMIN, RoleEnum.PISCICOLE_TECHNICIEN, RoleEnum.PISCICOLE_OBSERVATEUR),
            "apiculture": (RoleEnum.APICULTURE_ADMIN, RoleEnum.APICULTURE_TECHNICIEN, RoleEnum.APICULTURE_OBSERVATEUR),
            "entomoculture": (RoleEnum.ENTOMOCULTURE_ADMIN, RoleEnum.ENTOMOCULTURE_TECHNICIEN, RoleEnum.ENTOMOCULTURE_OBSERVATEUR),
        }
        
        roles = role_map.get(espece, ())
        
        if action == "read":
            return self.has_any_role(list(roles))
        elif action == "write":
            return self.has_any_role([roles[0], roles[1]]) if len(roles) > 1 else self.has_any_role([roles[0]])
        elif action == "delete":
            return self.has_any_role([roles[0]]) if roles else False
        
        return False
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, employee_id={self.employee_id})>"


class Role(Base, TimestampMixin):
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Enum(RoleEnum), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    
    users = relationship("User", secondary=user_roles, back_populates="roles")
    
    def __repr__(self):
        role_name = self.name.value if hasattr(self.name, 'value') else str(self.name)
        return f"<Role(id={self.id}, name={role_name})>"


class UserSession(Base, TimestampMixin):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token = Column(String(500), unique=True, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_info = Column(JSON, nullable=True)
    expires_at = Column(DateTime, nullable=False)
    is_valid = Column(Boolean, default=True)
    logout_at = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, valid={self.is_valid})>"


class ActionLog(Base, TimestampMixin):
    __tablename__ = "action_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=True)
    entity_id = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    user = relationship("User", back_populates="actions")
    
    def __repr__(self):
        return f"<ActionLog(id={self.id}, user_id={self.user_id}, action={self.action})>"