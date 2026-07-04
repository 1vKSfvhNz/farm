# backend/app/models/animal.py
from datetime import date

from sqlalchemy import Column, Integer, String, Date, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
import enum
from .base import Base, TimestampMixin, SoftDeleteMixin


class SexeEnum(str, enum.Enum):
    MALE = "male"
    FEMELLE = "femelle"
    HERMAPHRODITE = "hermaphrodite"


class StatutAnimalEnum(str, enum.Enum):
    VIVANT = "vivant"
    VENDU = "vendu"
    DECEDE = "decede"
    TRANSFERE = "transfere"


class Animal(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "animaux"
    __mapper_args__ = {
        "polymorphic_identity": "animal",
        "polymorphic_on": "type_espece"
    }
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    type_espece = Column(String(32), nullable=False)  # bovin, ovin, caprin, avicole, piscicole
    identification = Column(String(32), unique=True, nullable=False, index=True)  # Boucle, puce, etc.
    race = Column(String(32), nullable=False)
    sexe = Column(Enum(SexeEnum), nullable=False)
    date_naissance = Column(Date, nullable=True)
    date_arrivee = Column(Date, nullable=False)
    provenance = Column(String(32), nullable=True)  # Élevage d'origine
    prix_achat = Column(Integer, nullable=True)  # €
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=False)
    statut = Column(Enum(StatutAnimalEnum), default=StatutAnimalEnum.VIVANT)
    type_production = Column(String(50), nullable=True)  # lait, viande, reproduction, ponte, laine
    photo_url = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    # === NOUVEAUX CHAMPS POUR LA VENTE ===
    prix_vente = Column(Integer, nullable=True)  # Prix de vente en FCFA
    date_vente = Column(Date, nullable=True)   # Date de vente
    client_acheteur = Column(String(32), nullable=True)  # Nom du client/acheteur
    note_vente = Column(Text, nullable=True)   # Notes supplémentaires sur la vente
    
    # Relations
    enclos = relationship("Enclos", back_populates="animaux")
    pesees = relationship("Pesee", back_populates="animal", cascade="all, delete-orphan")
    alimentations = relationship("Alimentation", back_populates="animal", cascade="all, delete-orphan")
    vaccinations = relationship("Vaccination", back_populates="animal", cascade="all, delete-orphan")
    naissances = relationship("Naissance", foreign_keys="Naissance.mere_id", back_populates="mere")
    mortalite = relationship("Mortalite", back_populates="animal", uselist=False, cascade="all, delete-orphan")
    videos = relationship("VideoRecord", back_populates="animal")
    
    @property
    def age_mois(self):
        if not self.date_naissance:
            return None

        today = date.today()

        return (
            (today.year - self.date_naissance.year) * 12
            + (today.month - self.date_naissance.month)
        )
        
    def __repr__(self):
        return f"<Animal(id={self.id}, identification={self.identification}, espece={self.type_espece})>"