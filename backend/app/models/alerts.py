# backend/app/models/alert.py
"""
Modèles pour les alertes système
"""

from sqlalchemy import Column, JSON, Integer, String, DateTime, ForeignKey, Text, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from .base import Base, TimestampMixin


class AlertNiveauEnum(str, enum.Enum):
    """Niveau de sévérité d'une alerte"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertTypeEnum(str, enum.Enum):
    """Type d'alerte"""
    VACCINATION = "vaccination"
    PESEE = "pesee"
    NETTOYAGE = "nettoyage"
    MORTALITE = "mortalite"
    WATER_QUALITY = "water_quality"
    BEA = "bea"
    COMPTABLE = "comptable"
    STOCK = "stock"
    TEMPERATURE = "temperature"
    ODONI = "odoni"
    REPRODUCTION = "reproduction"
    COMPOST = "compost"
    SURPOPULATION = "surpopulation"
    SANTE = "sante"


class Alert(Base, TimestampMixin):
    """
    Alerte système pour les notifications
    """
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Type et niveau
    type = Column(SQLEnum(AlertTypeEnum), nullable=False)
    niveau = Column(SQLEnum(AlertNiveauEnum), nullable=False, default=AlertNiveauEnum.INFO)
    
    # Message
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # Contexte
    espece = Column(String(50), nullable=True)
    animal_id = Column(Integer, ForeignKey("animaux.id"), nullable=True)
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=True)
    compost_id = Column(Integer, ForeignKey("composts.id"), nullable=True)
    utilisateur_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Dates
    date_alerte = Column(DateTime, nullable=False, default=func.now())
    date_limite = Column(DateTime, nullable=True)
    date_lue = Column(DateTime, nullable=True)
    date_traitement = Column(DateTime, nullable=True)
    
    # Statut
    est_lue = Column(Boolean, default=False)
    est_traitee = Column(Boolean, default=False)
    
    # Métadonnées
    action_suggestee = Column(Text, nullable=True)
    resolution_note = Column(Text, nullable=True)
    source = Column(String(100), nullable=True)  # automatique, manuel, api
    
    # Utilisateur qui a traité l'alerte
    utilisateur_traitement_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relations
    animal = relationship("Animal", foreign_keys=[animal_id])
    enclos = relationship("Enclos", foreign_keys=[enclos_id])
    compost = relationship("Compost", foreign_keys=[compost_id])
    utilisateur = relationship("User", foreign_keys=[utilisateur_id])
    utilisateur_traitement = relationship("User", foreign_keys=[utilisateur_traitement_id])
    
    def __repr__(self):
        return f"<Alert(id={self.id}, type={self.type}, niveau={self.niveau}, est_traitee={self.est_traitee})>"
    
    def mark_as_read(self):
        """Marquer l'alerte comme lue"""
        self.est_lue = True
        self.date_lue = datetime.now()
    
    def mark_as_resolved(self, user_id: int, note: str = None):
        """Marquer l'alerte comme résolue"""
        self.est_traitee = True
        self.date_traitement = datetime.now()
        self.utilisateur_traitement_id = user_id
        if note:
            self.resolution_note = note


class AlertRule(Base, TimestampMixin):
    """
    Règle de génération d'alerte
    """
    __tablename__ = "alert_rules"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Type et niveau
    type = Column(SQLEnum(AlertTypeEnum), nullable=False)
    niveau = Column(SQLEnum(AlertNiveauEnum), nullable=False, default=AlertNiveauEnum.WARNING)
    
    # Paramètres
    parametre = Column(String(100), nullable=False)  # temperature, mortalite, etc.
    seuil_min = Column(Integer, nullable=True)
    seuil_max = Column(Integer, nullable=True)
    fenetre_jours = Column(Integer, default=1)
    
    # Condition
    condition = Column(String(255), nullable=True)  # expression SQL ou logique
    
    # Action
    message_template = Column(Text, nullable=False)
    action_suggestee = Column(Text, nullable=True)
    
    # Statut
    is_active = Column(Boolean, default=True)
    
    # Espèce concernée (optionnel)
    espece = Column(String(50), nullable=True)
    
    def __repr__(self):
        return f"<AlertRule(id={self.id}, type={self.type}, parametre={self.parametre})>"


class AlertHistory(Base, TimestampMixin):
    """
    Historique des alertes envoyées (pour éviter les doublons)
    """
    __tablename__ = "alert_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    canal = Column(String(50), nullable=False)  # email, sms, websocket, push
    
    destinataire = Column(String(255), nullable=False)  # email, phone, user_id
    statut = Column(String(50), nullable=False)  # sent, failed, pending
    
    erreur_message = Column(Text, nullable=True)
    date_envoi = Column(DateTime, nullable=False, default=func.now())
    
    # Relation
    alert = relationship("Alert", foreign_keys=[alert_id])
    
    def __repr__(self):
        return f"<AlertHistory(id={self.id}, alert_id={self.alert_id}, canal={self.canal})>"


class NotificationPreference(Base, TimestampMixin):
    """
    Préférences de notification par utilisateur
    """
    __tablename__ = "notification_preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    utilisateur_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    
    # Canaux activés
    email_enabled = Column(Boolean, default=True)
    sms_enabled = Column(Boolean, default=False)
    websocket_enabled = Column(Boolean, default=True)
    push_enabled = Column(Boolean, default=False)
    
    # Types d'alertes à recevoir
    alert_types = Column(JSON, nullable=True)  # {"vaccination": true, "mortalite": false, ...}
    
    # Niveau minimum
    min_niveau = Column(SQLEnum(AlertNiveauEnum), default=AlertNiveauEnum.INFO)
    
    # Périodes silencieuses
    quiet_hours_start = Column(String(5), nullable=True)  # "22:00"
    quiet_hours_end = Column(String(5), nullable=True)    # "08:00"
    
    # Relation
    utilisateur = relationship("User", foreign_keys=[utilisateur_id])
    
    def __repr__(self):
        return f"<NotificationPreference(id={self.id}, utilisateur_id={self.utilisateur_id})>"