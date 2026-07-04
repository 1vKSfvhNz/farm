# backend/app/models/odon.py
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin, SoftDeleteMixin


class PiegeOdoni(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "pieges_odoni"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    emplacement = Column(String(255), nullable=True)
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=True)
    compost_id = Column(Integer, ForeignKey("composts.id"), nullable=True)
    date_installation = Column(Date, nullable=False)
    type_piege = Column(String(50), nullable=True)  # collant, lumineux, etc.
    is_active = Column(Boolean, default=True)
    
    # Relations
    comptages = relationship("ComptageOdoni", back_populates="piege", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<PiegeOdoni(id={self.id}, name={self.name})>"


class ComptageOdoni(Base, TimestampMixin):
    __tablename__ = "comptages_odoni"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    piege_id = Column(Integer, ForeignKey("pieges_odoni.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    nombre = Column(Integer, nullable=False)
    espece = Column(String(100), nullable=True)  # mouche, moustique, etc.
    methode = Column(String(50), nullable=True)  # capteur_ir, comptage_manuel, camera
    
    # Relations
    piege = relationship("PiegeOdoni", back_populates="comptages")
    
    def __repr__(self):
        return f"<ComptageOdoni(id={self.id}, piege_id={self.piege_id}, nombre={self.nombre})>"