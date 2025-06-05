from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text, Enum as SqlEnum
from sqlalchemy.orm import relationship
from models import Base
from models.elevage import TypeElevage
from enums.elevage.avicole import TypeVolailleEnum, TypeProductionAvicoleEnum, SystemeElevageAvicoleEnum

class Volaille(Base):
    __tablename__ = 'volailles'
    
    id = Column(Integer, ForeignKey('animaux.id'), primary_key=True)
    type_volaille = Column(SqlEnum(TypeVolailleEnum), nullable=False)
    type_production = Column(SqlEnum(TypeProductionAvicoleEnum), nullable=False)
    systeme_elevage = Column(SqlEnum(SystemeElevageAvicoleEnum))
    souche = Column(String(100))  # Souche génétique
    date_mise_en_place = Column(Date)
    date_reforme = Column(Date)
    nombre_oeufs_cumules = Column(Integer)  # Pour les pondeuses
    poids_vif = Column(Float)  # Pour les poulets de chair
    
    __mapper_args__ = {
        'polymorphic_identity': TypeElevage.AVICOLE
    }

class ControlePonte(Base):
    __tablename__ = 'controles_ponte'
    
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey('lots.id'), nullable=False)
    date_controle = Column(Date, nullable=False)
    nombre_oeufs = Column(Integer)
    poids_moyen_oeuf = Column(Float)  # g
    taux_ponte = Column(Float)  # %
    taux_casses = Column(Float)  # %
    taux_sales = Column(Float)  # %
    notes = Column(Text)
    
    lot = relationship("Lot")

class PerformanceCroissanceAvicole(Base):
    __tablename__ = 'performances_croissance_avicole'
    
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey('lots.id'), nullable=False)
    date_controle = Column(Date, nullable=False)
    poids_moyen = Column(Float)  # g
    gain_moyen_journalier = Column(Float)  # g/jour
    consommation_aliment = Column(Float)  # kg
    indice_consommation = Column(Float)  # kg aliment/kg poids vif
    taux_mortalite = Column(Float)  # %
    uniformite = Column(Float)  # %
    
    lot = relationship("Lot")

