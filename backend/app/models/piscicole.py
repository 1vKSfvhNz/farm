# backend/app/models/piscicole.py
from sqlalchemy import Column, Float, Boolean, ForeignKey, Integer, func
from .animal import Animal
from .base import Base


class Piscicole(Animal):
    __tablename__ = "piscicoles"
    __mapper_args__ = {"polymorphic_identity": "piscicole"}
    
    id = Column(Integer, ForeignKey("animaux.id"), primary_key=True)
    
    # Spécificités piscicoles
    production_viande = Column(Boolean, default=False)
    production_reproduction = Column(Boolean, default=False)
    taille_moyenne = Column(Float, nullable=True)  # cm

    @classmethod
    def get_piscicole_stats(cls, session):
        """Retourne les statistiques pour les piscicoles"""
        total_piscicoles = session.query(func.count(cls.id)).scalar() or 0
        
        stats = {
            "total_piscicoles": total_piscicoles,
            "production_viande": session.query(func.count(cls.id)).filter(cls.production_viande == True).scalar() or 0,
            "production_reproduction": session.query(func.count(cls.id)).filter(cls.production_reproduction == True).scalar() or 0,
            "taille_moyenne_globale": session.query(func.avg(cls.taille_moyenne)).scalar() or 0,
            "taille_min": session.query(func.min(cls.taille_moyenne)).scalar() or 0,
            "taille_max": session.query(func.max(cls.taille_moyenne)).scalar() or 0,
        }
        
        return stats
    
    
    def __repr__(self):
        return f"<Piscicole(id={self.id}, identification={self.identification})>"