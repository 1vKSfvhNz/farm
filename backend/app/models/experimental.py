# backend/app/models/experimental.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text, Boolean, JSON, DateTime
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class ReferenceGenerale(Base, TimestampMixin):
    """Références générales stockées pour l'auto-apprentissage"""
    __tablename__ = "references_generales"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    espece = Column(String(50), nullable=False)
    race = Column(String(100), nullable=True)
    type_reference = Column(String(50), nullable=False)  # croissance, mortalite, production, conversion
    donnees = Column(JSON, nullable=False)  # {age: poids_moyen, ...} ou {mois: taux_mortalite}
    confiance = Column(Float, default=0.0)  # 0-1
    nombre_donnees = Column(Integer, default=0)
    date_derniere_mise_a_jour = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<ReferenceGenerale(id={self.id}, espece={self.espece}, type={self.type_reference})>"


class DonneeExperimentale(Base, TimestampMixin):
    """Données marquées comme expérimentales (essais)"""
    __tablename__ = "donnees_experimentales"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    utilisateur_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    entite_type = Column(String(50), nullable=False)  # pesee, alimentation, mortalite
    entite_id = Column(Integer, nullable=False)  # ID dans la table correspondante
    est_essai = Column(Boolean, default=True)
    est_production = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    date_essai = Column(Date, nullable=False)
    
    # Relation
    utilisateur = relationship("User")
    
    def __repr__(self):
        return f"<DonneeExperimentale(id={self.id}, entite_type={self.entite_type}, est_essai={self.est_essai})>"