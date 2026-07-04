# backend/app/models/bovin.py
from sqlalchemy import Column, Float, Boolean, ForeignKey, Integer
from sqlalchemy.orm import relationship
from .animal import Animal

class Bovin(Animal):
    __tablename__ = "bovins"
    __mapper_args__ = {"polymorphic_identity": "bovin"}
    
    id = Column(Integer, ForeignKey("animaux.id"), primary_key=True)
    
    # Spécificités bovins
    production_laitiere = Column(Boolean, default=False)
    production_viande = Column(Boolean, default=False)
    production_reproduction = Column(Boolean, default=False)
    
    # Pour les vaches laitières
    lactation_en_cours = Column(Boolean, default=False)
    production_lait_quotidienne = Column(Float, nullable=True)  # litres/jour
    
    # Relations
    naissances = relationship("Naissance", foreign_keys="Naissance.pere_bovin_id")
    
    @classmethod
    def get_bovin_stats(cls, session):
        """Retourne les statistiques pour les bovins"""
        from sqlalchemy import func
        
        total_bovins = session.query(func.count(cls.id)).scalar() or 0
        
        stats = {
            "total_bovins": total_bovins,
            "production_laitiere": session.query(func.count(cls.id)).filter(cls.production_laitiere == True).scalar() or 0,
            "production_viande": session.query(func.count(cls.id)).filter(cls.production_viande == True).scalar() or 0,
            "production_reproduction": session.query(func.count(cls.id)).filter(cls.production_reproduction == True).scalar() or 0,
            "lactation_en_cours": session.query(func.count(cls.id)).filter(cls.lactation_en_cours == True).scalar() or 0,
            "production_lait_moyenne": session.query(func.avg(cls.production_lait_quotidienne)).filter(cls.production_lait_quotidienne.isnot(None)).scalar() or 0,
            # === NOUVEAU ===
            "total_ventes": session.query(func.count(cls.id)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
            "montant_total_ventes": session.query(func.sum(cls.prix_vente)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
            "prix_vente_moyen": session.query(func.avg(cls.prix_vente)).filter(cls.prix_vente.isnot(None)).scalar() or 0,
        }
        
        return stats
        
    def __repr__(self):
        return f"<Bovin(id={self.id}, identification={self.identification})>"    