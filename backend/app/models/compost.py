# backend/app/models/compost.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
import enum
from .base import Base, TimestampMixin, SoftDeleteMixin


class CompostType(str, enum.Enum):
    DECRETS_VERTS = "déchets verts"
    FUMIER = "fumier"
    MIXTE = "mixte"


class Compost(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "composts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    type = Column(Enum(CompostType), nullable=False)
    date_demarrage = Column(Date, nullable=False)
    volume_initial = Column(Float, nullable=False)  # m³
    volume_final = Column(Float, nullable=True)  # m³
    date_maturite_estimee = Column(Date, nullable=True)
    date_maturite_reelle = Column(Date, nullable=True)
    utilisation_finale = Column(String(255), nullable=True)  # épandage, vente
    notes = Column(Text, nullable=True)
    
    # Relations
    retournements = relationship("RetournementCompost", back_populates="compost", cascade="all, delete-orphan")
    
    @property
    def temperature_moyenne(self) -> float:
        # À implémenter avec les capteurs
        pass
    
    @property
    def humidite_moyenne(self) -> float:
        # À implémenter avec les capteurs
        pass
    
    def __repr__(self):
        return f"<Compost(id={self.id}, name={self.name}, type={self.type})>"


class RetournementCompost(Base, TimestampMixin):
    __tablename__ = "retournements_compost"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    compost_id = Column(Integer, ForeignKey("composts.id"), nullable=False)
    date_retournement = Column(Date, nullable=False)
    responsable = Column(String(100), nullable=True)
    temperature_avant = Column(Float, nullable=True)  # °C
    temperature_apres = Column(Float, nullable=True)  # °C
    humidite_avant = Column(Float, nullable=True)  # %
    humidite_apres = Column(Float, nullable=True)  # %
    notes = Column(Text, nullable=True)
    
    # Relation
    compost = relationship("Compost", back_populates="retournements")
    
    def __repr__(self):
        return f"<RetournementCompost(id={self.id}, compost_id={self.compost_id}, date={self.date_retournement})>"