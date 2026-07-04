# backend/app/models/reference.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, Date, Boolean, JSON
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class ReferenceCroissance(Base, TimestampMixin):
    __tablename__ = "references_croissance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    espece = Column(String(50), nullable=False)  # bovin, ovin, etc.
    race = Column(String(100), nullable=False)
    age_jours = Column(Integer, nullable=False)
    poids_min = Column(Float, nullable=False)  # kg
    poids_moyen = Column(Float, nullable=False)
    poids_max = Column(Float, nullable=False)
    source = Column(String(255), nullable=True)  # standard, importe, apprise
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<ReferenceCroissance(id={self.id}, espece={self.espece}, race={self.race}, age={self.age_jours})>"


class ReferenceSeuil(Base, TimestampMixin):
    __tablename__ = "references_seuils"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    espece = Column(String(50), nullable=False)
    parametre = Column(String(100), nullable=False)  # oxygene_dissous, pH, temperature, mortalite
    unite = Column(String(20), nullable=True)
    seuil_min = Column(Float, nullable=True)
    seuil_max = Column(Float, nullable=True)
    seuil_optimal_min = Column(Float, nullable=True)
    seuil_optimal_max = Column(Float, nullable=True)
    niveau_alerte = Column(String(20), nullable=True)  # warning, critical
    source = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<ReferenceSeuil(id={self.id}, espece={self.espece}, parametre={self.parametre})>"


class ReferenceVaccination(Base, TimestampMixin):
    __tablename__ = "references_vaccination"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    espece = Column(String(50), nullable=False)
    maladie = Column(String(100), nullable=False)
    age_recommande_jours = Column(Integer, nullable=True)
    rappel_mois = Column(Integer, nullable=True)
    saison_recommandee = Column(String(50), nullable=True)
    vaccin_nom = Column(String(100), nullable=True)
    source = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<ReferenceVaccination(id={self.id}, espece={self.espece}, maladie={self.maladie})>"


class ReferenceNutrition(Base, TimestampMixin):
    __tablename__ = "references_nutrition"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    espece = Column(String(50), nullable=False)
    categorie = Column(String(50), nullable=False)  # veau, adulte, lactant
    proteines_pourcent = Column(Float, nullable=True)
    energie_kcal = Column(Float, nullable=True)
    calcium_g = Column(Float, nullable=True)
    phosphore_g = Column(Float, nullable=True)
    source = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<ReferenceNutrition(id={self.id}, espece={self.espece}, categorie={self.categorie})>"


class ReferenceHypothese(Base, TimestampMixin):
    __tablename__ = "references_hypotheses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    utilisateur_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    espece = Column(String(50), nullable=False)
    race = Column(String(100), nullable=True)
    parametre = Column(String(100), nullable=False)  # croissance, mortalite, production
    valeur_estimee = Column(Float, nullable=False)
    unite = Column(String(20), nullable=True)
    date_creation = Column(Date, nullable=False)
    validee = Column(Boolean, default=False)
    date_validation = Column(Date, nullable=True)
    
    def __repr__(self):
        return f"<ReferenceHypothese(id={self.id}, espece={self.espece}, parametre={self.parametre})>"