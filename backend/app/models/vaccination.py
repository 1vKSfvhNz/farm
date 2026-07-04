# backend/app/models/vaccination.py
from sqlalchemy import Column, Integer, Boolean, String, Date, ForeignKey, Text, Float
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class Maladie(Base, TimestampMixin):
    __tablename__ = "maladies"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(100), nullable=False, unique=True)
    especes_concernees = Column(String(255), nullable=False)  # liste séparée par des virgules
    description = Column(Text, nullable=True)
    vaccin_disponible = Column(String(100), nullable=True)
    
    def __repr__(self):
        return f"<Maladie(id={self.id}, nom={self.nom})>"


class Vaccin(Base, TimestampMixin):
    __tablename__ = "vaccins"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(100), nullable=False)
    fabricant = Column(String(100), nullable=True)
    lot = Column(String(50), nullable=True)
    maladie_id = Column(Integer, ForeignKey("maladies.id"), nullable=False)
    
    # Relation
    maladie = relationship("Maladie")
    
    def __repr__(self):
        return f"<Vaccin(id={self.id}, nom={self.nom}, lot={self.lot})>"


class Vaccination(Base, TimestampMixin):
    __tablename__ = "vaccinations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    animal_id = Column(Integer, ForeignKey("animaux.id"), nullable=False, index=True)
    maladie_id = Column(Integer, ForeignKey("maladies.id"), nullable=False)
    vaccin_id = Column(Integer, ForeignKey("vaccins.id"), nullable=True)
    date_prevue = Column(Date, nullable=False)
    date_realisee = Column(Date, nullable=True)
    dose = Column(String(50), nullable=True)  # mL, unités
    rappel_necessaire = Column(Boolean, default=False)
    date_prochain_rappel = Column(Date, nullable=True)
    veterinaire_responsable = Column(String(100), nullable=True)
    cout = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relations
    animal = relationship("Animal", back_populates="vaccinations")
    maladie = relationship("Maladie")
    vaccin = relationship("Vaccin")
    
    @property
    def est_a_jour(self) -> bool:
        if not self.rappel_necessaire:
            return self.date_realisee is not None
        if self.date_prochain_rappel:
            from datetime import date
            return date.today() <= self.date_prochain_rappel
        return False
    
    def __repr__(self):
        return f"<Vaccination(id={self.id}, animal_id={self.animal_id}, maladie={self.maladie_id})>"