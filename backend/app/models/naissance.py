# backend/app/models/naissance.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class Naissance(Base, TimestampMixin):
    __tablename__ = "naissances"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    mere_id = Column(Integer, ForeignKey("animaux.id"), nullable=False, index=True)
    pere_bovin_id = Column(Integer, ForeignKey("bovins.id"), nullable=True)
    pere_ovin_id = Column(Integer, ForeignKey("ovins.id"), nullable=True)
    pere_caprin_id = Column(Integer, ForeignKey("caprins.id"), nullable=True)
    animal_ne_id = Column(Integer, ForeignKey("animaux.id"), nullable=False, index=True)
    date_naissance = Column(Date, nullable=False)
    poids_naissance = Column(Float, nullable=True)  # kg
    sexe = Column(String(10), nullable=False)
    complications = Column(Boolean, default=False)
    notes = Column(String(255), nullable=True)
    
    # Relations
    mere = relationship("Animal", foreign_keys=[mere_id], back_populates="naissances")
    pere_bovin = relationship("Bovin", foreign_keys=[pere_bovin_id])
    pere_ovin = relationship("Ovin", foreign_keys=[pere_ovin_id])
    pere_caprin = relationship("Caprin", foreign_keys=[pere_caprin_id])
    animal_ne = relationship("Animal", foreign_keys=[animal_ne_id])
    
    def __repr__(self):
        return f"<Naissance(id={self.id}, mere_id={self.mere_id}, date={self.date_naissance})>"