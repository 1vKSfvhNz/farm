# backend/app/models/water_quality.py
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, Text, Boolean, String
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class WaterQuality(Base, TimestampMixin):
    __tablename__ = "water_qualities"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Paramètres de qualité d'eau
    ph = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)  # °C
    oxygene_dissous = Column(Float, nullable=True)  # mg/L
    ammoniac = Column(Float, nullable=True)  # mg/L NH3
    nitrites = Column(Float, nullable=True)  # mg/L NO2
    nitrates = Column(Float, nullable=True)  # mg/L NO3
    conductivite = Column(Float, nullable=True)  # µS/cm
    turbidite = Column(Float, nullable=True)  # NTU
    
    # Métadonnées
    source = Column(String(50), nullable=True)  # capteur_automatique, mesure_manuelle
    alerte_generee = Column(Boolean, default=False)
    
    # Relations - CORRECTION: back_populates="water_quality"
    enclos = relationship("Enclos", back_populates="water_qualities")
    alertes = relationship("WaterQualityAlerte", back_populates="water_quality", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<WaterQuality(id={self.id}, enclos_id={self.enclos_id}, timestamp={self.timestamp})>"


class WaterQualityAlerte(Base, TimestampMixin):
    __tablename__ = "water_quality_alertes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    water_quality_id = Column(Integer, ForeignKey("water_qualities.id"), nullable=False)
    parametre = Column(String(50), nullable=False)
    valeur = Column(Float, nullable=False)
    seuil = Column(Float, nullable=False)
    niveau = Column(String(20), nullable=False)  # warning, critical
    message = Column(Text, nullable=True)
    traitee = Column(Boolean, default=False)
    
    # Relations - CORRECTION: back_populates="alertes" et water_quality
    water_quality = relationship("WaterQuality", back_populates="alertes")
    
    def __repr__(self):
        return f"<WaterQualityAlerte(id={self.id}, parametre={self.parametre}, niveau={self.niveau})>"