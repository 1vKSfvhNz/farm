from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, ForeignKey, Enum as SqlEnum
from sqlalchemy.orm import relationship
from models import Base
from enums import QualiteEauEnum
from enums.elevage import TypeElevage
from enums.elevage.piscicole import EspecePoissonEnum, TypeAlimentPoissonEnum, TypeMilieuPiscicoleEnum, TypeElevagePiscicoleEnum

class Poisson(Base):
    __tablename__ = 'poissons'
    
    id = Column(Integer, ForeignKey('animaux.id'), primary_key=True)
    espece = Column(SqlEnum(EspecePoissonEnum), nullable=False)
    bassin_id = Column(Integer, ForeignKey('bassins_piscicoles.id'))
    date_ensemencement = Column(Date)
    origine = Column(String(100))  # Ecloserie, pêche, etc.
    poids_ensemencement = Column(Float)  # g
    taille_ensemencement = Column(Float)  # cm
    alimentation_type = Column(SqlEnum(TypeAlimentPoissonEnum))
    
    __mapper_args__ = {
        'polymorphic_identity': TypeElevage.PISCICOLE
    }

class BassinPiscicole(Base):
    __tablename__ = 'bassins_piscicoles'
    
    id = Column(Integer, primary_key=True)
    nom = Column(String(100), nullable=False)
    type_milieu = Column(SqlEnum(TypeMilieuPiscicoleEnum), nullable=False)
    type_elevage = Column(SqlEnum(TypeElevagePiscicoleEnum), nullable=False)
    volume = Column(Float)  # m3
    superficie = Column(Float)  # m2
    profondeur_moyenne = Column(Float)  # m
    capacite_max = Column(Integer)  # nombre de poissons
    date_mise_en_service = Column(Date)
    systeme_filtration = Column(String(200))
    systeme_aeration = Column(String(200))
    notes = Column(Text)
    
    poissons = relationship("Poisson", back_populates="bassin")
    controles_eau = relationship("ControleEau", back_populates="bassin")

class ControleEau(Base):
    __tablename__ = 'controles_eau'
    
    id = Column(Integer, primary_key=True)
    bassin_id = Column(Integer, ForeignKey('bassins_piscicoles.id'), nullable=False)
    date_controle = Column(DateTime, nullable=False)
    temperature = Column(Float)  # °C
    ph = Column(Float)
    oxygene_dissous = Column(Float)  # mg/l
    ammoniac = Column(Float)  # mg/l
    nitrites = Column(Float)  # mg/l
    nitrates = Column(Float)  # mg/l
    salinite = Column(Float)  # Pour eau saumâtre/marine (en ppt)
    turbidite = Column(Float)  # NTU
    qualite_eau = Column(SqlEnum(QualiteEauEnum))
    notes = Column(Text)
    
    bassin = relationship("BassinPiscicole", back_populates="controles_eau")

class RecoltePoisson(Base):
    __tablename__ = 'recoltes_poissons'
    
    id = Column(Integer, primary_key=True)
    bassin_id = Column(Integer, ForeignKey('bassins_piscicoles.id'), nullable=False)
    date_recolte = Column(Date, nullable=False)
    nombre_poissons = Column(Integer)
    poids_total = Column(Float)  # kg
    poids_moyen = Column(Float)  # g/poisson
    taux_survie = Column(Float)  # %
    destination = Column(String(100))  # Vente, transformation, etc.
    notes = Column(Text)
    
    bassin = relationship("BassinPiscicole")