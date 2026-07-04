# backend/app/models/apiary.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text, Boolean, Enum
from sqlalchemy.orm import relationship
import enum
from .base import Base, TimestampMixin, SoftDeleteMixin


class StatutRucheEnum(str, enum.Enum):
    ACTIVE = "active"
    ORPHELINE = "orpheline"
    EN_ESSIMAGE = "en_essaimage"
    MORTE = "morte"


class Ruche(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "ruches"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    identification = Column(String(50), unique=True, nullable=False)
    emplacement = Column(String(255), nullable=True)
    date_installation = Column(Date, nullable=False)
    race = Column(String(100), nullable=True)
    statut = Column(Enum(StatutRucheEnum), default=StatutRucheEnum.ACTIVE)
    nombre_cadres = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relations
    reines = relationship("Reine", back_populates="ruche", cascade="all, delete-orphan")
    recoltes = relationship("RecolteMiel", back_populates="ruche", cascade="all, delete-orphan")

    @classmethod
    def get_ruche_stats(cls, session):
        """Retourne les statistiques pour les ruches"""
        from sqlalchemy import func
        
        total_ruches = session.query(func.count(cls.id)).scalar() or 0
        
        stats = {
            "total_ruches": total_ruches,
            "actives": session.query(func.count(cls.id)).filter(cls.statut == StatutRucheEnum.ACTIVE).scalar() or 0,
            "orphelines": session.query(func.count(cls.id)).filter(cls.statut == StatutRucheEnum.ORPHELINE).scalar() or 0,
            "en_essaimage": session.query(func.count(cls.id)).filter(cls.statut == StatutRucheEnum.EN_ESSIMAGE).scalar() or 0,
            "mortes": session.query(func.count(cls.id)).filter(cls.statut == StatutRucheEnum.MORTE).scalar() or 0,
            "total_reines": session.query(func.count(Reine.id)).filter(Reine.ruche_id == cls.id).scalar() or 0,
            "total_recoltes": session.query(func.count(RecolteMiel.id)).filter(RecolteMiel.ruche_id == cls.id).scalar() or 0,
            "poids_miel_total": session.query(func.sum(RecolteMiel.poids_kg)).filter(RecolteMiel.ruche_id == cls.id).scalar() or 0,
            "poids_miel_moyen": session.query(func.avg(RecolteMiel.poids_kg)).filter(RecolteMiel.ruche_id == cls.id).scalar() or 0,
        }
        
        return stats
    
    def __repr__(self):
        return f"<Ruche(id={self.id}, identification={self.identification})>"


class Reine(Base, TimestampMixin):
    __tablename__ = "reines"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ruche_id = Column(Integer, ForeignKey("ruches.id"), nullable=False)
    annee_naissance = Column(Integer, nullable=False)
    origine = Column(String(255), nullable=True)
    date_remplacement = Column(Date, nullable=True)
    est_marquee = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    
    # Relations
    ruche = relationship("Ruche", back_populates="reines")
    
    def __repr__(self):
        return f"<Reine(id={self.id}, ruche_id={self.ruche_id}, annee={self.annee_naissance})>"


class RecolteMiel(Base, TimestampMixin):
    __tablename__ = "recoltes_miel"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ruche_id = Column(Integer, ForeignKey("ruches.id"), nullable=False)
    date_recolte = Column(Date, nullable=False)
    poids_kg = Column(Float, nullable=False)
    qualite = Column(String(50), nullable=True)  # toutes fleurs, miellat, etc.
    taux_eau = Column(Float, nullable=True)  # %
    notes = Column(Text, nullable=True)
    
    # Relations
    ruche = relationship("Ruche", back_populates="recoltes")
    
    def __repr__(self):
        return f"<RecolteMiel(id={self.id}, ruche_id={self.ruche_id}, poids={self.poids_kg})>"