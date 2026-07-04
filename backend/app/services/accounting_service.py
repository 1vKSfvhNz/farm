# backend/app/services/accounting_service.py
"""
Service de comptabilité - Dépenses et recettes
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..models.accounting import Depense, Recette, CategorieDepenseEnum, CategorieRecetteEnum
from ..schemas.accounting import *

logger = logging.getLogger(__name__)


class AccountingService:
    """Service de gestion comptable"""
    
    async def create_depense(
        self,
        db: AsyncSession,
        depense_data: DepenseCreate,
        created_by: int
    ) -> Depense:
        """Créer une dépense"""
        depense = Depense(
            categorie=depense_data.categorie,
            montant=depense_data.montant,
            date=depense_data.date,
            description=depense_data.description,
            fournisseur=depense_data.fournisseur,
            quantite=depense_data.quantite,
            prix_unitaire=depense_data.prix_unitaire,
            animal_id=depense_data.animal_id,
            lot_entomo_id=depense_data.lot_entomo_id,
            piece_jointe_url=depense_data.piece_jointe_url
        )
        db.add(depense)
        await db.commit()
        
        logger.info(f"Depense created: {depense.categorie.value} - {depense.montant}€")
        return depense
    
    async def create_recette(
        self,
        db: AsyncSession,
        recette_data: RecetteCreate,
        created_by: int
    ) -> Recette:
        """Créer une recette"""
        recette = Recette(
            categorie=recette_data.categorie,
            montant=recette_data.montant,
            date=recette_data.date,
            description=recette_data.description,
            client=recette_data.client,
            quantite=recette_data.quantite,
            prix_unitaire=recette_data.prix_unitaire,
            animal_id=recette_data.animal_id,
            lot_entomo_id=recette_data.lot_entomo_id
        )
        db.add(recette)
        await db.commit()
        
        logger.info(f"Recette created: {recette.categorie.value} - {recette.montant}€")
        return recette
    
    async def get_summary(
        self,
        db: AsyncSession,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> AccountSummary:
        """Obtenir le résumé comptable"""
        if not start_date:
            start_date = date.today().replace(day=1)
        if not end_date:
            end_date = date.today()
        
        # Dépenses
        stmt = select(Depense).where(
            Depense.date >= start_date,
            Depense.date <= end_date
        )
        result = await db.execute(stmt)
        depenses = result.scalars().all()
        
        # Recettes
        stmt = select(Recette).where(
            Recette.date >= start_date,
            Recette.date <= end_date
        )
        result = await db.execute(stmt)
        recettes = result.scalars().all()
        
        total_depenses = sum(d.montant for d in depenses)
        total_recettes = sum(r.montant for r in recettes)
        benefice = total_recettes - total_depenses
        
        # Par catégorie
        depenses_par_categorie = {}
        for d in depenses:
            key = d.categorie.value
            depenses_par_categorie[key] = depenses_par_categorie.get(key, 0) + d.montant
        
        recettes_par_categorie = {}
        for r in recettes:
            key = r.categorie.value
            recettes_par_categorie[key] = recettes_par_categorie.get(key, 0) + r.montant
        
        # Trésorerie prévisionnelle (simplifiée)
        tresorerie_previsionnelle = await self._get_cashflow_forecast(db, start_date, end_date)
        
        marge_brute_pourcent = (benefice / total_recettes * 100) if total_recettes > 0 else 0
        
        return AccountSummary(
            total_depenses=round(total_depenses, 2),
            total_recettes=round(total_recettes, 2),
            benefice=round(benefice, 2),
            marge_brute_pourcent=round(marge_brute_pourcent, 1),
            depenses_par_categorie=depenses_par_categorie,
            recettes_par_categorie=recettes_par_categorie,
            tresorerie_previsionnelle=tresorerie_previsionnelle
        )
    
    async def get_profitability_by_species(
        self,
        db: AsyncSession,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Dict[str, float]]:
        """Obtenir la rentabilité par espèce"""
        result = {}
        
        # À implémenter avec les liens vers les animaux
        species_list = ["bovin", "ovin", "caprin", "avicole", "piscicole", "entomoculture"]
        
        for species in species_list:
            # Dépenses liées à l'espèce
            stmt = select(Depense).where(
                Depense.date >= start_date if start_date else True,
                Depense.date <= end_date if end_date else True
            )
            # Filtrer par animal_id lié à l'espèce - implémentation simplifiée
            result[species] = {
                "depenses": 0,
                "recettes": 0,
                "benefice": 0,
                "marge": 0
            }
        
        return result
    
    async def _get_cashflow_forecast(
        self,
        db: AsyncSession,
        start_date: date,
        end_date: date
    ) -> Dict[str, float]:
        """Prévision de trésorerie"""
        return {
            "current": 0,
            "30days": 0,
            "60days": 0,
            "90days": 0
        }

    async def get_depenses_by_category(
        self,
        db: AsyncSession,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, float]:
        """Obtenir les dépenses par catégorie"""
        if not start_date:
            start_date = date.today().replace(day=1)
        if not end_date:
            end_date = date.today()
        
        stmt = select(
            Depense.categorie,
            func.sum(Depense.montant).label('total')
        ).where(
            Depense.date >= start_date,
            Depense.date <= end_date
        ).group_by(Depense.categorie)
        
        result = await db.execute(stmt)
        return {row[0].value: float(row[1]) for row in result}
    
    async def get_recettes_by_category(
        self,
        db: AsyncSession,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, float]:
        """Obtenir les recettes par catégorie"""
        if not start_date:
            start_date = date.today().replace(day=1)
        if not end_date:
            end_date = date.today()
        
        stmt = select(
            Recette.categorie,
            func.sum(Recette.montant).label('total')
        ).where(
            Recette.date >= start_date,
            Recette.date <= end_date
        ).group_by(Recette.categorie)
        
        result = await db.execute(stmt)
        return {row[0].value: float(row[1]) for row in result}
    
    async def get_monthly_summary(
        self,
        db: AsyncSession,
        year: int
    ) -> Dict[str, Any]:
        """Obtenir le résumé mensuel pour une année"""
        monthly_data = {}
        
        for month in range(1, 13):
            start_date = date(year, month, 1)
            if month == 12:
                end_date = date(year, month, 31)
            else:
                end_date = date(year, month + 1, 1) - timedelta(days=1)
            
            summary = await self.get_summary(db, start_date, end_date)
            monthly_data[month] = {
                "depenses": summary.total_depenses,
                "recettes": summary.total_recettes,
                "benefice": summary.benefice,
                "marge": summary.marge_brute_pourcent
            }
        
        return monthly_data
    
    async def get_depense(
        self,
        db: AsyncSession,
        depense_id: int
    ) -> Optional[Depense]:
        """Obtenir une dépense par son ID"""
        stmt = select(Depense).where(Depense.id == depense_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_recette(
        self,
        db: AsyncSession,
        recette_id: int
    ) -> Optional[Recette]:
        """Obtenir une recette par son ID"""
        stmt = select(Recette).where(Recette.id == recette_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update_depense(
        self,
        db: AsyncSession,
        depense_id: int,
        depense_data: DepenseUpdate,
        updated_by: int
    ) -> Tuple[Optional[Depense], Optional[str]]:
        """Mettre à jour une dépense"""
        depense = await self.get_depense(db, depense_id)
        if not depense:
            return None, "Dépense non trouvée"
        
        if depense_data.categorie is not None:
            depense.categorie = depense_data.categorie
        if depense_data.montant is not None:
            depense.montant = depense_data.montant
        if depense_data.date is not None:
            depense.date = depense_data.date
        if depense_data.description is not None:
            depense.description = depense_data.description
        if depense_data.fournisseur is not None:
            depense.fournisseur = depense_data.fournisseur
        if depense_data.quantite is not None:
            depense.quantite = depense_data.quantite
        if depense_data.prix_unitaire is not None:
            depense.prix_unitaire = depense_data.prix_unitaire
        
        await db.commit()
        logger.info(f"Depense updated: {depense_id} by {updated_by}")
        return depense, None
    
    async def update_recette(
        self,
        db: AsyncSession,
        recette_id: int,
        recette_data: RecetteUpdate,
        updated_by: int
    ) -> Tuple[Optional[Recette], Optional[str]]:
        """Mettre à jour une recette"""
        recette = await self.get_recette(db, recette_id)
        if not recette:
            return None, "Recette non trouvée"
        
        if recette_data.categorie is not None:
            recette.categorie = recette_data.categorie
        if recette_data.montant is not None:
            recette.montant = recette_data.montant
        if recette_data.date is not None:
            recette.date = recette_data.date
        if recette_data.description is not None:
            recette.description = recette_data.description
        if recette_data.client is not None:
            recette.client = recette_data.client
        if recette_data.quantite is not None:
            recette.quantite = recette_data.quantite
        if recette_data.prix_unitaire is not None:
            recette.prix_unitaire = recette_data.prix_unitaire
        
        await db.commit()
        logger.info(f"Recette updated: {recette_id} by {updated_by}")
        return recette, None
    

accounting_service = AccountingService()