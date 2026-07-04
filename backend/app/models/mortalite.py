# backend/app/models/mortalite.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class Mortalite(Base, TimestampMixin):
    __tablename__ = "mortalites"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    animal_id = Column(Integer, ForeignKey("animaux.id"), nullable=False, index=True)
    date_mort = Column(Date, nullable=False)
    cause = Column(String(255), nullable=True)
    autopsie_realisee = Column(Boolean, default=False)
    poids_cadavre = Column(Float, nullable=True)  # kg
    notes = Column(Text, nullable=True)
    
    # Relation
    animal = relationship("Animal", back_populates="mortalite")
    
    def __repr__(self):
        return f"<Mortalite(id={self.id}, animal_id={self.animal_id}, date={self.date_mort})>"