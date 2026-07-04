# backend/app/models/caprin.py
from sqlalchemy import Column, Boolean, ForeignKey, Integer, func
from .animal import Animal
from .base import Base


class Caprin(Animal):
    __tablename__ = "caprins"
    __mapper_args__ = {"polymorphic_identity": "caprin"}
    
    id = Column(Integer, ForeignKey("animaux.id"), primary_key=True)
    
    # Spécificités caprins
    production_viande = Column(Boolean, default=False)
    production_reproduction = Column(Boolean, default=False)

    @classmethod
    def get_caprin_stats(cls, session):
        """Retourne les statistiques pour les caprins"""
        total_caprins = session.query(func.count(cls.id)).scalar() or 0
        
        stats = {
            "total_caprins": total_caprins,
            "production_viande": session.query(func.count(cls.id)).filter(cls.production_viande == True).scalar() or 0,
            "production_reproduction": session.query(func.count(cls.id)).filter(cls.production_reproduction == True).scalar() or 0,

            # === NOUVEAU ===
            "total_ventes": session.query(func.count(cls.id)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
            "montant_total_ventes": session.query(func.sum(cls.prix_vente)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
            "prix_vente_moyen": session.query(func.avg(cls.prix_vente)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
        }
        
        return stats
    
    def __repr__(self):
        return f"<Caprin(id={self.id}, identification={self.identification})>"