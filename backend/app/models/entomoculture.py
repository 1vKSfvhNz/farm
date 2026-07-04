# backend/app/models/entomoculture.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
import enum
from .base import Base, TimestampMixin, SoftDeleteMixin


class StadeInsecteEnum(str, enum.Enum):
    OEUF = "oeuf"
    LARVE = "larve"
    PREPUPE = "prepupe"
    PUPE = "pupe"
    ADULTE = "adulte"


class TypeProductionInsecteEnum(str, enum.Enum):
    LARVES = "larves"
    REPRODUCTION = "reproduction"
    OEUFS = "oeufs"


class EntomocultureLot(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "entomoculture_lots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    identification = Column(String(100), unique=True, nullable=False, index=True)
    espece = Column(String(100), nullable=False)  # Hermetia illucens, Tenebrio molitor, etc.
    stade_actuel = Column(Enum(StadeInsecteEnum), nullable=False)
    date_arrivee = Column(Date, nullable=False)
    provenance = Column(String(255), nullable=True)
    prix_achat = Column(Float, nullable=True)  # €/kg ou € total
    poids_initial = Column(Float, nullable=True)  # kg
    quantite_estimative = Column(Integer, nullable=True)  # nombre estimé
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=True)
    type_production = Column(Enum(TypeProductionInsecteEnum), nullable=False)
    notes = Column(Text, nullable=True)
    
    # Relations
    cycles = relationship("EntomocultureCycle", back_populates="lot", cascade="all, delete-orphan")
    pesees = relationship("Pesee", back_populates="lot_entomo")

    @classmethod
    def get_entomoculture_stats(cls, session):
        """Retourne les statistiques pour l'entomoculture"""
        from sqlalchemy import func
        
        total_lots = session.query(func.count(cls.id)).scalar() or 0
        
        stats = {
            "total_lots": total_lots,
            "en_cours": session.query(func.count(cls.id)).filter(cls.stade_actuel != StadeInsecteEnum.ADULTE).scalar() or 0,
            "termines": session.query(func.count(cls.id)).filter(cls.stade_actuel == StadeInsecteEnum.ADULTE).scalar() or 0,
            "total_cycles": session.query(func.count(EntomocultureCycle.id)).scalar() or 0,
            "production_totale_grammes": session.query(func.sum(EntomocultureCycle.production_grammes)).scalar() or 0,
            "taux_mortalite_moyen": session.query(func.avg(EntomocultureCycle.taux_mortalite)).scalar() or 0,
            "especes_distinctes": session.query(func.count(cls.espece.distinct())).scalar() or 0,
        }
        
        return stats
    
    @property
    def taux_mortalite(self) -> float:
        """Taux de mortalité estimé (%)"""
        cycles_fermes = [c for c in self.cycles if c.date_fin]
        if cycles_fermes:
            dernier_cycle = max(cycles_fermes, key=lambda c: c.date_fin)
            return dernier_cycle.taux_mortalite
        return None
    
    def __repr__(self):
        return f"<EntomocultureLot(id={self.id}, identification={self.identification}, espece={self.espece})>"


class EntomocultureCycle(Base, TimestampMixin):
    __tablename__ = "entomoculture_cycles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    lot_id = Column(Integer, ForeignKey("entomoculture_lots.id"), nullable=False)
    date_debut = Column(Date, nullable=False)
    date_fin = Column(Date, nullable=True)
    stade_debut = Column(Enum(StadeInsecteEnum), nullable=False)
    stade_fin = Column(Enum(StadeInsecteEnum), nullable=True)
    production_grammes = Column(Float, nullable=True)  # grammes produits
    taux_mortalite = Column(Float, nullable=True)  # %
    substrat_utilise = Column(String(255), nullable=True)
    
    # Relations
    lot = relationship("EntomocultureLot", back_populates="cycles")

    @classmethod
    def get_cycle_stats(cls, session, lot_id=None):
        """Retourne les statistiques des cycles pour un lot spécifique ou global"""
        from sqlalchemy import func
        
        query = session.query(cls)
        if lot_id:
            query = query.filter(cls.lot_id == lot_id)
        
        stats = {
            "total_cycles": query.count(),
            "cycles_termines": query.filter(cls.date_fin.isnot(None)).count(),
            "duree_moyenne_jours": query.filter(cls.date_fin.isnot(None)).select_from(cls).with_entities(
                func.avg(func.date_part('day', cls.date_fin - cls.date_debut))
            ).scalar() or 0,
            "production_moyenne_grammes": query.filter(cls.production_grammes.isnot(None)).select_from(cls).with_entities(
                func.avg(cls.production_grammes)
            ).scalar() or 0,
        }
        
        return stats
    
    def __repr__(self):
        return f"<EntomocultureCycle(id={self.id}, lot_id={self.lot_id})>"