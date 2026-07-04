# backend/app/models/enclos.py
from sqlalchemy import Column, Integer, String, Float, Enum, Text
from sqlalchemy.orm import relationship
import enum
from .base import Base, TimestampMixin, SoftDeleteMixin

class EnclosType(str, enum.Enum):
    ENCLOS = "enclos"
    BASSIN = "bassin"
    PATURAGE = "pâturage"
    CAGE = "cage"
    BAC = "bac"

    
class Enclos(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "enclos"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, unique=True)
    type = Column(Enum(EnclosType), nullable=False)
    longueur = Column(Float, nullable=False)  # mètres
    largeur = Column(Float, nullable=False)  # mètres
    hauteur = Column(Float, nullable=True)   # mètres ou cm selon type
    localisation_gps = Column(String(64), nullable=True)
    zone = Column(String(32), nullable=True)
    description = Column(Text, nullable=True)
    
    # Relations
    animaux = relationship("Animal", back_populates="enclos")
    water_qualities = relationship("WaterQuality", back_populates="enclos")
    cameras = relationship("Camera", back_populates="enclos")
    avicoles = relationship("Avicole", back_populates="enclos") 

    # Propriétés simples (pas de requête base de données)
    @property
    def surface(self) -> float:
        """Calcule la surface de l'enclos"""
        return self.longueur * self.largeur
    
    @property
    def volume(self) -> float:
        """Calcule le volume de l'enclos (si hauteur définie)"""
        if self.hauteur:
            return self.surface * self.hauteur
        return 0.0
    
    def __repr__(self):
        return f"<Enclos(id={self.id}, name={self.name}, type={self.type})>"