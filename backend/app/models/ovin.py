# backend/app/models/ovin.py
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, func
from .animal import Animal
from .base import Base


class Ovin(Animal):
    __tablename__ = "ovins"
    __mapper_args__ = {"polymorphic_identity": "ovin"}
    
    id = Column(Integer, ForeignKey("animaux.id"), primary_key=True)
    
    # Spécificités ovins
    production_viande = Column(Boolean, default=False)
    production_reproduction = Column(Boolean, default=False)
    production_laine = Column(Boolean, default=False)
    qualite_laine = Column(String(50), nullable=True)  # fine, moyenne, grossière
    
    @classmethod
    def get_ovin_stats(cls, session):
        """Retourne les statistiques pour les ovins"""
        total_ovins = session.query(func.count(cls.id)).scalar() or 0
        
        stats = {
            "total_ovins": total_ovins,
            "production_viande": session.query(func.count(cls.id)).filter(cls.production_viande == True).scalar() or 0,
            "production_reproduction": session.query(func.count(cls.id)).filter(cls.production_reproduction == True).scalar() or 0,
            "production_laine": session.query(func.count(cls.id)).filter(cls.production_laine == True).scalar() or 0,

            # === NOUVEAU ===
            "total_ventes": session.query(func.count(cls.id)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
            "montant_total_ventes": session.query(func.sum(cls.prix_vente)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
            "prix_vente_moyen": session.query(func.avg(cls.prix_vente)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
        }
        
        return stats
    
    def __repr__(self):
        return f"<Ovin(id={self.id}, identification={self.identification})>"