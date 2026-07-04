# backend/app/models/pesee.py
from sqlalchemy import Column, Integer, Float, Date, ForeignKey, Text, String
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class Pesee(Base, TimestampMixin):
    __tablename__ = "pesees"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    animal_id = Column(Integer, ForeignKey("animaux.id"), nullable=True, index=True)
    lot_entomo_id = Column(Integer, ForeignKey("entomoculture_lots.id"), nullable=True, index=True)
    lot_avicole_id = Column(Integer, ForeignKey("avicoles.id"), nullable=True, index=True)  # ← Maintenant avicoles.id existe
    date_pesee = Column(Date, nullable=False, index=True)
    poids = Column(Float, nullable=False)
    methode = Column(String(50), nullable=True)
    video_record_id = Column(Integer, ForeignKey("video_records.id"), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relations
    animal = relationship("Animal", foreign_keys=[animal_id], back_populates="pesees")
    lot_entomo = relationship("EntomocultureLot", foreign_keys=[lot_entomo_id], back_populates="pesees")
    lot_avicole = relationship("Avicole", foreign_keys=[lot_avicole_id], back_populates="pesees")
    video = relationship("VideoRecord", foreign_keys=[video_record_id], back_populates="pesee")
    
    def __repr__(self):
        return f"<Pesee(id={self.id}, lot_avicole_id={self.lot_avicole_id}, poids={self.poids})>"