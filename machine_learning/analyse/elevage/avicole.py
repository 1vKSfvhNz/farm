# avicole_analysis.py
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
from dataclasses import dataclass

# Import des modèles existants
from models.elevage.avicole import (
    Volaille, 
    ControlePonte, 
    PerformanceCroissanceAvicole,
    TypeProductionAvicoleEnum,
)
from models.elevage import Lot, StatutAnimalEnum
from machine_learning.prediction.elevage.avicole import AvicolePredictor
from enums import AlertSeverity

@dataclass
class AvicoleAlert:
    title: str
    description: str
    severity: AlertSeverity
    recommendation: str
    related_data: Optional[Dict] = None

class AvicoleAnalyzer:
    def __init__(self, async_db_session: AsyncSession):
        self.async_db_session = async_db_session
        self.predictor = AvicolePredictor(async_db_session)
    
    async def analyze_farm(self) -> List[AvicoleAlert]:
        """Analyse complète de l'élevage et retourne toutes les alertes"""
        alerts = []
        
        # Charger les données nécessaires
        volailles, lots, controles_ponte, performances = await self._load_data()
        
        # Analyse de la ponte
        alerts.extend(await self._analyze_ponte(volailles, controles_ponte))
        
        # Analyse de la croissance
        alerts.extend(await self._analyze_croissance(volailles, performances))
        
        # Analyse des conditions d'élevage
        alerts.extend(await self._analyze_conditions(volailles, lots))
        
        # Analyse de la santé globale
        alerts.extend(await self._analyze_sante(volailles))
        
        # Analyse économique
        alerts.extend(await self._analyze_economique(volailles))
        
        return alerts
    
    async def _load_data(self):
        """Charge les données nécessaires depuis la base de données"""
        async with self.async_db_session() as session:
            # Charger les volailles
            volailles_result = await session.execute(Volaille.select())
            volailles = [v[0] for v in volailles_result]
            
            # Charger les lots
            lots_result = await session.execute(Lot.select())
            lots = [l[0] for l in lots_result]
            
            # Charger les contrôles de ponte
            controles_ponte_result = await session.execute(ControlePonte.select())
            controles_ponte = [c[0] for c in controles_ponte_result]
            
            # Charger les performances de croissance
            performances_result = await session.execute(PerformanceCroissanceAvicole.select())
            performances = [p[0] for p in performances_result]
            
        return volailles, lots, controles_ponte, performances
    
    async def _analyze_ponte(self, volailles: List[Volaille], controles: List[ControlePonte]) -> List[AvicoleAlert]:
        """Analyse les performances de ponte"""
        alerts = []
        
        # Grouper les contrôles par lot
        controles_df = pd.DataFrame([c.__dict__ for c in controles])
        if controles_df.empty:
            return alerts
            
        # Calculer les moyennes par lot
        stats = controles_df.groupby('lot_id').agg({
            'taux_ponte': ['mean', 'std'],
            'nombre_oeufs': ['mean', 'sum'],
            'taux_casses': 'mean',
            'taux_sales': 'mean'
        })
        
        # Identifier les problèmes
        for lot_id, row in stats.iterrows():
            # Alerte pour taux de ponte faible
            if row[('taux_ponte', 'mean')] < 60:  # Seuil à ajuster
                volaille = next((v for v in volailles if v.lot_id == lot_id), None)
                alerts.append(AvicoleAlert(
                    title=f"Taux de ponte faible dans le lot {lot_id}",
                    description=f"Le taux de ponte moyen est de {row[('taux_ponte', 'mean')]:.1f}%, ce qui est en dessous des normes attendues.",
                    severity=AlertSeverity.HIGH,
                    recommendation="Vérifier l'alimentation, la santé des volailles et les conditions d'élevage. Consulter un spécialiste si le problème persiste.",
                    related_data={
                        'lot_id': lot_id,
                        'taux_ponte': row[('taux_ponte', 'mean')],
                        'type_volaille': volaille.type_volaille.value if volaille else "Inconnu"
                    }
                ))
            
            # Alerte pour taux de casses élevé
            if row[('taux_casses', 'mean')] > 5:  # Seuil à ajuster
                alerts.append(AvicoleAlert(
                    title=f"Taux de casses élevé dans le lot {lot_id}",
                    description=f"Le taux de casses moyen est de {row[('taux_casses', 'mean')]:.1f}%, ce qui est au-dessus des normes acceptables.",
                    severity=AlertSeverity.MEDIUM,
                    recommendation="Inspecter les systèmes de collecte et de manipulation des œufs. Former le personnel à la manipulation appropriée.",
                    related_data={
                        'lot_id': lot_id,
                        'taux_casses': row[('taux_casses', 'mean')]
                    }
                ))
        
        return alerts
    
    async def _analyze_croissance(self, volailles: List[Volaille], performances: List[PerformanceCroissanceAvicole]) -> List[AvicoleAlert]:
        """Analyse les performances de croissance"""
        alerts = []
        
        performances_df = pd.DataFrame([p.__dict__ for p in performances])
        if performances_df.empty:
            return alerts
            
        # Calculer les indicateurs clés
        stats = performances_df.groupby('lot_id').agg({
            'poids_moyen': ['mean', 'std'],
            'gain_moyen_journalier': 'mean',
            'indice_consommation': 'mean',
            'taux_mortalite': 'mean'
        })
        
        for lot_id, row in stats.iterrows():
            volaille = next((v for v in volailles if v.lot_id == lot_id), None)
            if not volaille:
                continue
                
            # Normes différentes selon le type de production
            if volaille.type_production == TypeProductionAvicoleEnum.VIANDE:
                # Alerte pour gain journalier insuffisant
                if row[('gain_moyen_journalier', 'mean')] < 45:  # g/jour pour poulets de chair
                    alerts.append(AvicoleAlert(
                        title=f"Croissance lente dans le lot {lot_id}",
                        description=f"Le gain moyen journalier est de {row[('gain_moyen_journalier', 'mean')]:.1f}g/jour, en dessous des attentes.",
                        severity=AlertSeverity.MEDIUM,
                        recommendation="Vérifier la qualité et la quantité d'aliment, les conditions d'élevage et la santé des volailles.",
                        related_data={
                            'lot_id': lot_id,
                            'gain_journalier': row[('gain_moyen_journalier', 'mean')],
                            'type_production': volaille.type_production.value
                        }
                    ))
                
                # Alerte pour indice de consommation élevé
                if row[('indice_consommation', 'mean')] > 1.8:  # kg aliment/kg poids vif
                    alerts.append(AvicoleAlert(
                        title=f"Efficacité alimentaire faible dans le lot {lot_id}",
                        description=f"L'indice de consommation est de {row[('indice_consommation', 'mean')]:.2f}, indiquant une conversion alimentaire inefficace.",
                        severity=AlertSeverity.HIGH,
                        recommendation="Réévaluer la formulation des rations, vérifier la qualité des aliments et l'état de santé du lot.",
                        related_data={
                            'lot_id': lot_id,
                            'indice_consommation': row[('indice_consommation', 'mean')]
                        }
                    ))
            
            # Alerte pour mortalité élevée
            if row[('taux_mortalite', 'mean')] > 3:  # %
                alerts.append(AvicoleAlert(
                    title=f"Mortalité élevée dans le lot {lot_id}",
                    description=f"Le taux de mortalité est de {row[('taux_mortalite', 'mean')]:.1f}%, ce qui est anormalement élevé.",
                    severity=AlertSeverity.CRITICAL,
                    recommendation="Examiner immédiatement les causes possibles (maladies, stress, conditions d'élevage) et consulter un vétérinaire.",
                    related_data={
                        'lot_id': lot_id,
                        'taux_mortalite': row[('taux_mortalite', 'mean')]
                    }
                ))
        
        return alerts
    
    async def _analyze_conditions(self, volailles: List[Volaille], lots: List[Lot]) -> List[AvicoleAlert]:
        """Analyse les conditions d'élevage"""
        alerts = []
        
        # Vérifier la densité dans les lots
        for lot in lots:
            nb_animaux = len([v for v in volailles if v.lot_id == lot.id])
            if lot.capacite_max and nb_animaux > lot.capacite_max * 1.1:  # 10% au-dessus de la capacité
                alerts.append(AvicoleAlert(
                    title=f"Surpopulation dans le lot {lot.nom}",
                    description=f"Le lot contient {nb_animaux} volailles pour une capacité maximale recommandée de {lot.capacite_max}.",
                    severity=AlertSeverity.HIGH,
                    recommendation="Réduire la densité pour éviter le stress et les problèmes de santé. Penser à diviser le lot ou à augmenter l'espace disponible.",
                    related_data={
                        'lot_id': lot.id,
                        'nom_lot': lot.nom,
                        'nb_animaux': nb_animaux,
                        'capacite_max': lot.capacite_max
                    }
                ))
        
        return alerts
    
    async def _analyze_sante(self, volailles: List[Volaille]) -> List[AvicoleAlert]:
        """Analyse l'état de santé général"""
        alerts = []
        
        # Vérifier les animaux malades ou morts
        for volaille in volailles:
            animal = volaille.animal  # Relation SQLAlchemy
            if animal.statut == StatutAnimalEnum.MORT:
                days_since_death = (datetime.now() - animal.date_deces).days if animal.date_deces else 0
                if days_since_death < 7:  # Décès récent (moins d'une semaine)
                    alerts.append(AvicoleAlert(
                        title=f"Décès récent de la volaille {animal.numero_identification}",
                        description=f"La volaille est décédée il y a {days_since_death} jours. Cause: {animal.cause_deces or 'non spécifiée'}",
                        severity=AlertSeverity.CRITICAL,
                        recommendation="Analyser les causes du décès et vérifier si d'autres animaux présentent des symptômes similaires.",
                        related_data={
                            'animal_id': animal.id,
                            'numero_identification': animal.numero_identification,
                            'cause_deces': animal.cause_deces,
                            'date_deces': animal.date_deces.isoformat() if animal.date_deces else None
                        }
                    ))
        
        return alerts
    
    async def _analyze_economique(self, volailles: List[Volaille]) -> List[AvicoleAlert]:
        """Analyse les aspects économiques"""
        alerts = []
        
        # Identifier les volailles en production depuis trop longtemps avec faible productivité
        for volaille in volailles:
            animal = volaille.animal
            if (volaille.type_production == TypeProductionAvicoleEnum.PONTE and 
                animal.statut == StatutAnimalEnum.PRODUCTIF and
                animal.date_mise_en_production):
                
                days_in_production = (datetime.now() - animal.date_mise_en_production).days
                if days_in_production > 400 and volaille.nombre_oeufs_cumules < 250:  # Seuils à ajuster
                    alerts.append(AvicoleAlert(
                        title=f"Productivité faible pour la volaille {animal.numero_identification}",
                        description=f"La volaille est en production depuis {days_in_production} jours avec seulement {volaille.nombre_oeufs_cumules} œufs cumulés.",
                        severity=AlertSeverity.MEDIUM,
                        recommendation="Envisager la réforme de cette volaille si la productivité ne s'améliore pas.",
                        related_data={
                            'animal_id': animal.id,
                            'numero_identification': animal.numero_identification,
                            'jours_en_production': days_in_production,
                            'oeufs_cumules': volaille.nombre_oeufs_cumules
                        }
                    ))
        
        return alerts