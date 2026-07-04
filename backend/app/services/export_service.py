# backend/app/services/export_service.py
"""
Service d'export de données - CSV, PDF
"""

import logging
import csv
import io
from typing import List, Dict, Any, Optional
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..schemas.exports import *
from ..config import settings

logger = logging.getLogger(__name__)


class ExportService:
    """Service d'export de données"""
    
    async def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filename: str
    ) -> bytes:
        """Exporter des données au format CSV"""
        if not data:
            return b""
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue().encode('utf-8')
    
    async def export_animals(
        self,
        db: AsyncSession,
        espece: Optional[str] = None,
        enclos_id: Optional[int] = None
    ) -> bytes:
        """Exporter la liste des animaux"""
        from ..models.animal import Animal
        
        stmt = select(Animal).where(Animal.deleted_at.is_(None))
        if espece:
            stmt = stmt.where(Animal.type_espece == espece)
        if enclos_id:
            stmt = stmt.where(Animal.enclos_id == enclos_id)
        
        result = await db.execute(stmt)
        animals = result.scalars().all()
        
        data = []
        for a in animals:
            data.append({
                "id": a.id,
                "identification": a.identification,
                "espece": a.type_espece,
                "race": a.race,
                "sexe": a.sexe.value if a.sexe else "",
                "date_naissance": a.date_naissance,
                "enclos_id": a.enclos_id,
                "statut": a.statut.value if a.statut else ""
            })
        
        return await self.export_to_csv(data, f"animaux_{date.today()}.csv")
    
    async def export_financial(
        self,
        db: AsyncSession,
        start_date: date,
        end_date: date
    ) -> bytes:
        """Exporter les données financières"""
        from ..models.accounting import Depense, Recette
        
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
        
        data = []
        for d in depenses:
            data.append({
                "type": "DEPENSE",
                "date": d.date,
                "categorie": d.categorie.value,
                "montant": d.montant,
                "description": d.description,
                "fournisseur": d.fournisseur
            })
        
        for r in recettes:
            data.append({
                "type": "RECETTE",
                "date": r.date,
                "categorie": r.categorie.value,
                "montant": r.montant,
                "description": r.description,
                "client": r.client
            })
        
        return await self.export_to_csv(data, f"financial_{start_date}_{end_date}.csv")

    async def export_animals_by_species(
        self,
        db: AsyncSession,
        espece: str
    ) -> bytes:
        """Exporter les animaux d'une espèce spécifique"""
        from ..models.animal import Animal
        
        stmt = select(Animal).where(
            Animal.type_espece == espece,
            Animal.deleted_at.is_(None)
        )
        result = await db.execute(stmt)
        animals = result.scalars().all()
        
        data = []
        for a in animals:
            data.append({
                "id": a.id,
                "identification": a.identification,
                "race": a.race,
                "sexe": a.sexe.value if a.sexe else "",
                "date_naissance": a.date_naissance,
                "age_jours": a.age_jours,
                "dernier_poids_kg": a.dernier_poids,
                "enclos_id": a.enclos_id,
                "statut": a.statut.value if a.statut else "",
                "type_production": a.type_production,
                "date_arrivee": a.date_arrivee,
                "provenance": a.provenance,
                "prix_achat": a.prix_achat
            })
        
        return await self.export_to_csv(data, f"{espece}_{date.today()}.csv")
    
    async def export_compost_data(
        self,
        db: AsyncSession,
        compost_id: Optional[int] = None
    ) -> bytes:
        """Exporter les données de compost"""
        from ..models.compost import Compost, RetournementCompost
        
        if compost_id:
            stmt = select(Compost).where(Compost.id == compost_id)
            result = await db.execute(stmt)
            composts = [result.scalar_one_or_none()]
        else:
            stmt = select(Compost).where(Compost.deleted_at.is_(None))
            result = await db.execute(stmt)
            composts = result.scalars().all()
        
        data = []
        for compost in composts:
            if not compost:
                continue
            
            # Récupérer les retournements
            stmt = select(RetournementCompost).where(
                RetournementCompost.compost_id == compost.id
            ).order_by(RetournementCompost.date_retournement)
            result = await db.execute(stmt)
            turnings = result.scalars().all()
            
            compost_data = {
                "compost_id": compost.id,
                "nom": compost.name,
                "type": compost.type.value if compost.type else "",
                "date_demarrage": compost.date_demarrage,
                "volume_initial_m3": compost.volume_initial,
                "volume_final_m3": compost.volume_final,
                "date_maturite_estimee": compost.date_maturite_estimee,
                "date_maturite_reelle": compost.date_maturite_reelle,
                "nombre_retournements": len(turnings),
                "notes": compost.notes
            }
            data.append(compost_data)
            
            # Ajouter les retournements comme lignes séparées
            for turning in turnings:
                data.append({
                    "compost_id": compost.id,
                    "type": "retournement",
                    "date": turning.date_retournement,
                    "responsable": turning.responsable,
                    "temperature_avant": turning.temperature_avant,
                    "temperature_apres": turning.temperature_apres,
                    "humidite_avant": turning.humidite_avant,
                    "humidite_apres": turning.humidite_apres,
                    "notes": turning.notes
                })
        
        return await self.export_to_csv(data, f"compost_{date.today()}.csv")
    
    async def export_financial_detailed(
        self,
        db: AsyncSession,
        start_date: date,
        end_date: date
    ) -> bytes:
        """Exporter les données financières détaillées"""
        from ..models.accounting import Depense, Recette
        
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
        
        data = []
        
        # Ajouter un résumé
        total_depenses = sum(d.montant for d in depenses)
        total_recettes = sum(r.montant for r in recettes)
        
        data.append({
            "type": "RESUME",
            "date_debut": start_date,
            "date_fin": end_date,
            "total_depenses": total_depenses,
            "total_recettes": total_recettes,
            "benefice": total_recettes - total_depenses
        })
        
        # Dépenses détaillées
        for d in depenses:
            data.append({
                "type": "DEPENSE",
                "date": d.date,
                "categorie": d.categorie.value,
                "montant": d.montant,
                "description": d.description,
                "fournisseur": d.fournisseur,
                "quantite": d.quantite,
                "prix_unitaire": d.prix_unitaire
            })
        
        # Recettes détaillées
        for r in recettes:
            data.append({
                "type": "RECETTE",
                "date": r.date,
                "categorie": r.categorie.value,
                "montant": r.montant,
                "description": r.description,
                "client": r.client,
                "quantite": r.quantite,
                "prix_unitaire": r.prix_unitaire
            })
        
        return await self.export_to_csv(data, f"financial_detailed_{start_date}_{end_date}.csv")
    
export_service = ExportService()