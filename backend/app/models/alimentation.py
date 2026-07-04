# backend/app/models/alimentation.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class Alimentation(Base, TimestampMixin):
    __tablename__ = "alimentations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    animal_id = Column(Integer, ForeignKey("animaux.id"), nullable=True, index=True)
    lot_entomo_id = Column(Integer, ForeignKey("entomoculture_lots.id"), nullable=True, index=True)
    date = Column(Date, nullable=False, index=True)
    poids_nourriture = Column(Float, nullable=False)  # kg ou grammes
    type_nourriture = Column(String(100), nullable=False)  # granulés, fourrage, concentré, etc.
    composition = Column(String(255), nullable=True)  # protéines%, lipides%, etc.
    cout = Column(Float, nullable=True)  # €
    
    # Relations
    animal = relationship("Animal", back_populates="alimentations")
    lot_entomo = relationship("EntomocultureLot")
    
    def __repr__(self):
        return f"<Alimentation(id={self.id}, animal_id={self.animal_id}, date={self.date})>"


class RationAlimentaire(Base, TimestampMixin):
    __tablename__ = "rations_alimentaires"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    espece = Column(String(50), nullable=False)
    race = Column(String(100), nullable=True)
    categorie = Column(String(50), nullable=False)  # veau, adulte, lactant, etc.
    proteines_pourcent = Column(Float, nullable=True)
    energie_kcal = Column(Float, nullable=True)
    calcium_g = Column(Float, nullable=True)
    phosphore_g = Column(Float, nullable=True)
    quantite_recommandee_kg = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<RationAlimentaire(id={self.id}, espece={self.espece}, categorie={self.categorie})>"