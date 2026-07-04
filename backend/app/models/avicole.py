# backend/app/models/avicole.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text, Enum, Boolean
from sqlalchemy.orm import relationship
import enum
from .base import Base, TimestampMixin, SoftDeleteMixin


class TypeProductionAvicoleEnum(str, enum.Enum):
    VIANDE = "viande"
    PONTE = "ponte"
    REPRODUCTION = "reproduction"
    MIXTE = "mixte"


class StatutLotAvicoleEnum(str, enum.Enum):
    ACTIF = "actif"
    VENDU = "vendu"
    ABATTU = "abattu"
    DECEDE = "decede"


class Avicole(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "avicoles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    identification = Column(String(100), unique=True, nullable=False, index=True)
    espece = Column(String(100), nullable=False)  # Poulet, Poule, Pintade, Canard, Dinde, etc.
    race = Column(String(100), nullable=False)
    type_production = Column(Enum(TypeProductionAvicoleEnum), nullable=False)
    
    # Informations sur le lot
    quantite_initial = Column(Integer, nullable=False, default=0)  # Nombre d'individus au départ
    quantite_actuelle = Column(Integer, nullable=False, default=0)  # Nombre actuel d'individus
    
    # Suivi des entrées/sorties
    date_arrivee = Column(Date, nullable=False)
    provenance = Column(String(255), nullable=True)
    prix_achat_unitaire = Column(Float, nullable=True)  # € par individu
    prix_achat_total = Column(Float, nullable=True)  # € total
    
    # Suivi de la production
    production_viande = Column(Boolean, default=False)
    production_reproduction = Column(Boolean, default=False)
    production_ponte = Column(Boolean, default=False)
    
    # Suivi ponte (pour pondeuses)
    oeufs_pondus_total = Column(Integer, default=0)  # Nombre total d'œufs pondus par le lot
    oeufs_pondus_jour = Column(Integer, default=0)  # Nombre d'œufs pondus aujourd'hui
    poids_oeufs_total = Column(Float, default=0)  # kg
    poids_oeufs_moyen = Column(Float, nullable=True)  # Poids moyen par œuf en grammes
    
    # Suivi mortalité
    mortalite_total = Column(Integer, default=0)  # Nombre total de morts
    taux_mortalite = Column(Float, default=0)  # % de mortalité
    
    # Suivi des abattages/ventes
    vendus_total = Column(Integer, default=0)  # Nombre total vendus
    abattus_total = Column(Integer, default=0)  # Nombre total abattus
    date_derniere_vente = Column(Date, nullable=True)
    
    # Suivi des poids (pour la production de viande)
    poids_moyen_initial = Column(Float, nullable=True)  # kg
    poids_moyen_actuel = Column(Float, nullable=True)  # kg
    poids_total_viande = Column(Float, default=0)  # kg de viande produite
    
    # Emplacement
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=False)
    
    # Statut
    statut = Column(Enum(StatutLotAvicoleEnum), default=StatutLotAvicoleEnum.ACTIF)
    date_fermeture = Column(Date, nullable=True)  # Date de fin du lot (vente totale, abattage total)
    
    # Métadonnées
    notes = Column(Text, nullable=True)
    photo_url = Column(Text, nullable=True)
    
    # Relations
    enclos = relationship("Enclos", back_populates="avicoles")
    pesees = relationship("Pesee", back_populates="lot_avicole")
    mortalites = relationship("AvicoleMortalite", back_populates="lot", cascade="all, delete-orphan")
    productions = relationship("AvicoleProduction", back_populates="lot", cascade="all, delete-orphan")
    
    @property
    def taux_mortalite_calcule(self) -> float:
        """Calcule le taux de mortalité actuel"""
        if self.quantite_initial > 0:
            return (self.mortalite_total / self.quantite_initial) * 100
        return 0
    
    @property
    def taux_survie(self) -> float:
        """Calcule le taux de survie actuel"""
        if self.quantite_initial > 0:
            return ((self.quantite_actuelle) / self.quantite_initial) * 100
        return 0
    
    @property
    def rendement_viande(self) -> float:
        """Rendement estimé en viande (%)"""
        if self.quantite_abattus > 0 and self.poids_total_viande > 0:
            # Calcul basé sur le poids moyen à l'abattage
            poids_abattage_estime = self.poids_moyen_abattage
            if poids_abattage_estime > 0:
                return (self.poids_total_viande / (self.quantite_abattus * poids_abattage_estime)) * 100
        return 0
    
    @property
    def production_oeufs_par_jour(self) -> float:
        """Production moyenne d'œufs par jour par individu"""
        if self.quantite_actuelle > 0 and self.production_ponte:
            return self.oeufs_pondus_jour / self.quantite_actuelle
        return 0
    
    @classmethod
    def get_avicole_stats(cls, session):
        """Retourne les statistiques pour les avicoles"""
        from sqlalchemy import func
        
        total_lots = session.query(func.count(cls.id)).scalar() or 0
        
        stats = {
            "total_lots": total_lots,
            "total_individus": session.query(func.sum(cls.quantite_actuelle)).scalar() or 0,
            "lots_actifs": session.query(func.count(cls.id)).filter(cls.statut == StatutLotAvicoleEnum.ACTIF).scalar() or 0,
            "production_viande": session.query(func.count(cls.id)).filter(cls.production_viande == True).scalar() or 0,
            "production_reproduction": session.query(func.count(cls.id)).filter(cls.production_reproduction == True).scalar() or 0,
            "production_ponte": session.query(func.count(cls.id)).filter(cls.production_ponte == True).scalar() or 0,
            "total_oeufs_pondus": session.query(func.sum(cls.oeufs_pondus_total)).scalar() or 0,
            "total_poids_oeufs_kg": session.query(func.sum(cls.poids_oeufs_total)).scalar() or 0,
            "total_viande_kg": session.query(func.sum(cls.poids_total_viande)).scalar() or 0,
            "mortalite_totale": session.query(func.sum(cls.mortalite_total)).scalar() or 0,
            "taux_mortalite_moyen": session.query(func.avg(cls.taux_mortalite)).scalar() or 0,
        }
        
        return stats
    
    def __repr__(self):
        return f"<Avicole(id={self.id}, identification={self.identification}, espece={self.espece}, quantite={self.quantite_actuelle})>"


class AvicoleMortalite(Base, TimestampMixin):
    """Enregistrement quotidien des mortalités pour les lots avicoles"""
    __tablename__ = "avicoles_mortalites"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    lot_id = Column(Integer, ForeignKey("avicoles.id"), nullable=False)
    date = Column(Date, nullable=False)
    nombre_morts = Column(Integer, nullable=False)
    cause = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relations
    lot = relationship("Avicole", back_populates="mortalites")
    
    def __repr__(self):
        return f"<AvicoleMortalite(id={self.id}, lot_id={self.lot_id}, date={self.date}, nombre={self.nombre_morts})>"


class AvicoleProduction(Base, TimestampMixin):
    """Enregistrement quotidien de la production (œufs) pour les lots avicoles"""
    __tablename__ = "avicoles_productions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    lot_id = Column(Integer, ForeignKey("avicoles.id"), nullable=False)
    date = Column(Date, nullable=False)
    nombre_oeufs = Column(Integer, default=0)
    poids_oeufs_kg = Column(Float, default=0)  # Poids total des œufs en kg
    nombre_morts = Column(Integer, default=0)  # Mortalité du jour
    notes = Column(Text, nullable=True)
    
    # Relations
    lot = relationship("Avicole", back_populates="productions")
    
    @property
    def poids_moyen_oeuf_grammes(self) -> float:
        """Poids moyen d'un œuf en grammes"""
        if self.nombre_oeufs > 0:
            return (self.poids_oeufs_kg / self.nombre_oeufs) * 1000
        return 0
    
    def __repr__(self):
        return f"<AvicoleProduction(id={self.id}, lot_id={self.lot_id}, date={self.date}, oeufs={self.nombre_oeufs})>"