# backend/app/models/accounting.py
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Text, Enum, func
from sqlalchemy.orm import relationship
import enum
from .base import Base, TimestampMixin


class CategorieDepenseEnum(str, enum.Enum):
    ACHAT_ANIMAUX = "achat_animaux"
    ACHAT_OEUFS = "achat_oeufs"
    ALIMENTATION = "alimentation"
    VACCINS_SOINS = "vaccins_soins"
    EQUIPEMENT = "equipement"
    PERSONNEL = "personnel"
    EAU_ELECTRICITE = "eau_electricite"
    ENTRETIEN = "entretien"
    COMPOSTAGE = "compostage"
    TRANSPORT = "transport"
    FRAIS_DIVERS = "frais_divers"


class CategorieRecetteEnum(str, enum.Enum):
    VENTE_ANIMAUX_VIVANTS = "vente_animaux_vivants"
    VENTE_VIANDE = "vente_viande"
    VENTE_LAIT = "vente_lait"
    VENTE_LAINE = "vente_laine"
    VENTE_OEUFS = "vente_oeufs"
    VENTE_LARVES = "vente_larves"
    VENTE_COMPOST = "vente_compost"
    VENTE_FUMIER = "vente_fumier"
    SUBVENTIONS = "subventions"
    AUTRES = "autres"


class Depense(Base, TimestampMixin):
    __tablename__ = "depenses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    categorie = Column(Enum(CategorieDepenseEnum), nullable=False)
    montant = Column(Float, nullable=False)
    date = Column(Date, nullable=False)
    description = Column(Text, nullable=True)
    fournisseur = Column(String(255), nullable=True)
    quantite = Column(Float, nullable=True)
    prix_unitaire = Column(Float, nullable=True)
    animal_id = Column(Integer, ForeignKey("animaux.id"), nullable=True)
    lot_entomo_id = Column(Integer, ForeignKey("entomoculture_lots.id"), nullable=True)
    piece_jointe_url = Column(Text, nullable=True)
    
    # Relations
    animal = relationship("Animal")
    lot_entomo = relationship("EntomocultureLot")
    
    @classmethod
    def get_depense_stats(cls, session, annee=None, mois=None):
        """Retourne les statistiques des dépenses"""
        query = session.query(cls)
        
        if annee:
            query = query.filter(func.extract('year', cls.date) == annee)
        if mois:
            query = query.filter(func.extract('month', cls.date) == mois)
        
        stats = {
            "total_depenses": query.filter().count(),
            "montant_total": query.filter().with_entities(func.sum(cls.montant)).scalar() or 0,
            "montant_moyen": query.filter().with_entities(func.avg(cls.montant)).scalar() or 0,
            "montant_min": query.filter().with_entities(func.min(cls.montant)).scalar() or 0,
            "montant_max": query.filter().with_entities(func.max(cls.montant)).scalar() or 0,
        }
        
        # Statistiques par catégorie
        par_categorie = {}
        for cat in CategorieDepenseEnum:
            total = query.filter(cls.categorie == cat).with_entities(func.sum(cls.montant)).scalar() or 0
            count = query.filter(cls.categorie == cat).count()
            if total > 0 or count > 0:
                par_categorie[cat.value] = {
                    "total": total,
                    "count": count,
                    "moyenne": total / count if count > 0 else 0
                }
        stats["par_categorie"] = par_categorie
        
        # Top fournisseurs
        top_fournisseurs = session.query(
            cls.fournisseur,
            func.sum(cls.montant).label('total')
        ).filter(cls.fournisseur.isnot(None)).group_by(cls.fournisseur).order_by(func.sum(cls.montant).desc()).limit(5).all()
        
        stats["top_fournisseurs"] = [
            {"fournisseur": f, "total": t} for f, t in top_fournisseurs
        ]
        
        return stats
    
    @classmethod
    def get_depenses_mensuelles(cls, session, annee):
        """Retourne les dépenses mensuelles pour une année"""
        resultats = {}
        for mois in range(1, 13):
            total = session.query(func.sum(cls.montant)).filter(
                func.extract('year', cls.date) == annee,
                func.extract('month', cls.date) == mois
            ).scalar() or 0
            resultats[mois] = total
        return resultats
    
    def __repr__(self):
        return f"<Depense(id={self.id}, categorie={self.categorie}, montant={self.montant}, date={self.date})>"


class Recette(Base, TimestampMixin):
    __tablename__ = "recettes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    categorie = Column(Enum(CategorieRecetteEnum), nullable=False)
    montant = Column(Float, nullable=False)
    date = Column(Date, nullable=False)
    description = Column(Text, nullable=True)
    client = Column(String(255), nullable=True)
    quantite = Column(Float, nullable=True)
    prix_unitaire = Column(Float, nullable=True)
    animal_id = Column(Integer, ForeignKey("animaux.id"), nullable=True)
    lot_entomo_id = Column(Integer, ForeignKey("entomoculture_lots.id"), nullable=True)
    
    # Relations
    animal = relationship("Animal")
    lot_entomo = relationship("EntomocultureLot")
    
    @classmethod
    def get_recette_stats(cls, session, annee=None, mois=None):
        """Retourne les statistiques des recettes"""
        query = session.query(cls)
        
        if annee:
            query = query.filter(func.extract('year', cls.date) == annee)
        if mois:
            query = query.filter(func.extract('month', cls.date) == mois)
        
        stats = {
            "total_recettes": query.filter().count(),
            "montant_total": query.filter().with_entities(func.sum(cls.montant)).scalar() or 0,
            "montant_moyen": query.filter().with_entities(func.avg(cls.montant)).scalar() or 0,
            "montant_min": query.filter().with_entities(func.min(cls.montant)).scalar() or 0,
            "montant_max": query.filter().with_entities(func.max(cls.montant)).scalar() or 0,
        }
        
        # Statistiques par catégorie
        par_categorie = {}
        for cat in CategorieRecetteEnum:
            total = query.filter(cls.categorie == cat).with_entities(func.sum(cls.montant)).scalar() or 0
            count = query.filter(cls.categorie == cat).count()
            if total > 0 or count > 0:
                par_categorie[cat.value] = {
                    "total": total,
                    "count": count,
                    "moyenne": total / count if count > 0 else 0
                }
        stats["par_categorie"] = par_categorie
        
        # Top clients
        top_clients = session.query(
            cls.client,
            func.sum(cls.montant).label('total')
        ).filter(cls.client.isnot(None)).group_by(cls.client).order_by(func.sum(cls.montant).desc()).limit(5).all()
        
        stats["top_clients"] = [
            {"client": c, "total": t} for c, t in top_clients
        ]
        
        return stats
    
    @classmethod
    def get_recettes_mensuelles(cls, session, annee):
        """Retourne les recettes mensuelles pour une année"""
        resultats = {}
        for mois in range(1, 13):
            total = session.query(func.sum(cls.montant)).filter(
                func.extract('year', cls.date) == annee,
                func.extract('month', cls.date) == mois
            ).scalar() or 0
            resultats[mois] = total
        return resultats
    
    def __repr__(self):
        return f"<Recette(id={self.id}, categorie={self.categorie}, montant={self.montant}, date={self.date})>"


# Fonction utilitaire pour obtenir les statistiques financières globales
def get_financial_stats(session, annee=None, mois=None):
    """Retourne les statistiques financières globales (recettes, dépenses, bénéfices)"""
    depense_stats = Depense.get_depense_stats(session, annee, mois)
    recette_stats = Recette.get_recette_stats(session, annee, mois)
    
    total_depenses = depense_stats["montant_total"]
    total_recettes = recette_stats["montant_total"]
    
    return {
        "depenses": depense_stats,
        "recettes": recette_stats,
        "benefice_total": total_recettes - total_depenses,
        "ratio_benefice": (total_recettes / total_depenses) if total_depenses > 0 else 0,
        "marge_brute": ((total_recettes - total_depenses) / total_recettes * 100) if total_recettes > 0 else 0,
        "periode": {
            "annee": annee,
            "mois": mois
        }
    }