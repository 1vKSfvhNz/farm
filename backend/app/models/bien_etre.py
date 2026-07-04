# backend/app/models/bien_etre.py
from sqlalchemy import Column, Integer, Float, Date, ForeignKey, Text, String
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class BienEtreIndice(Base, TimestampMixin):
    __tablename__ = "bien_etre_indices"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    indice_global = Column(Float, nullable=False)  # 0-100%
    
    # Sous-indices
    indice_proprete = Column(Float, nullable=True)
    indice_acces_eau = Column(Float, nullable=True)
    indice_densite = Column(Float, nullable=True)
    indice_comportement = Column(Float, nullable=True)
    
    notes = Column(Text, nullable=True)
    
    # Relations
    enclos = relationship("Enclos")
    
    def __repr__(self):
        return f"<BienEtreIndice(id={self.id}, enclos_id={self.enclos_id}, indice={self.indice_global})>"


class BienEtreCritere(Base, TimestampMixin):
    __tablename__ = "bien_etre_criteres"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    ponderation = Column(Float, nullable=False, default=1.0)  # Poids dans le calcul
    seuil_alerte = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<BienEtreCritere(id={self.id}, nom={self.nom})>"