import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import datetime
from typing import Dict, Any, Optional, List, Union
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from dataclasses import asdict

# Import des dépendances de la base de données
from models import DatabaseLoader
from models.elevage import Lot, Animal
from models.elevage.avicole import Volaille, ControlePonte, PerformanceCroissanceAvicole
from machine_learning.base import ModelPerformance

from pathlib import Path
# Configuration des chemins - définie au niveau module
PARENT_DIR = Path(__file__).parent.parent
MODELS_DIR = PARENT_DIR / "ml_files"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

class AvicolePredictor:
    def __init__(self, db_session: Optional[Union[AsyncSession, Session]] = None, 
                 model_path: Optional[str] = None):
        """
        Initialise le prédicteur avicole.
        
        Args:
            db_session: Session SQLAlchemy (synchrone ou asynchrone)
            model_path: Chemin vers un modèle pré-entraîné à charger
        """
        self.loader = DatabaseLoader(db_session)
        self.db_session = db_session
        self.is_async = isinstance(db_session, AsyncSession) if db_session else False
        self.label_encoders = {}
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.ponte_data = None
        self.croissance_data = None
        
        if model_path:
            self.load_model(model_path)
        else:
            self.ponte_rate_model = None
            self.oeufs_count_model = None
            self.poids_model = None
            self.gain_model = None
    
    async def prepare_training_data_async(self):
        """Prépare les données d'entraînement à partir de la base de données (mode asynchrone)."""
        if not self.db_session or not self.is_async:
            raise ValueError("Async DB session must be set to prepare training data")
            
        # Charger les données depuis la base de manière asynchrone
        volailles = await self.loader.execute_query(select(Volaille))
        animaux = await self.loader.execute_query(select(Animal))
        lots = await self.loader.execute_query(select(Lot))
        controles_ponte = await self.loader.execute_query(select(ControlePonte))
        performances = await self.loader.execute_query(select(PerformanceCroissanceAvicole))
        
        # Convertir les résultats en DataFrames
        volailles = pd.DataFrame([dict(v) for v in volailles])
        animaux = pd.DataFrame([dict(a) for a in animaux])
        lots = pd.DataFrame([dict(l) for l in lots])
        controles_ponte = pd.DataFrame([dict(c) for c in controles_ponte])
        performances = pd.DataFrame([dict(p) for p in performances])
        
        # Fusionner et préparer les données
        data = pd.merge(volailles, animaux, left_on='id', right_on='id')
        data = pd.merge(data, lots, left_on='lot_id', right_on='id', suffixes=('', '_lot'))
        
        # Préparer les données spécifiques
        self._prepare_data(data, controles_ponte, performances)
    
    def prepare_training_data_sync(self):
        """Prépare les données d'entraînement à partir de la base de données (mode synchrone)."""
        if not self.db_session or self.is_async:
            raise ValueError("Sync DB session must be set to prepare training data")
            
        # Charger les données depuis la base de manière synchrone
        volailles = self.loader.execute_query("SELECT * FROM volailles")
        animaux = self.loader.execute_query("SELECT * FROM animaux")
        lots = self.loader.execute_query("SELECT * FROM lots")
        controles_ponte = self.loader.execute_query("SELECT * FROM controles_ponte")
        performances = self.loader.execute_query("SELECT * FROM performances_croissance_avicole")
        
        # Fusionner et préparer les données
        data = pd.merge(volailles, animaux, left_on='id', right_on='id')
        data = pd.merge(data, lots, left_on='lot_id', right_on='id', suffixes=('', '_lot'))
        
        # Préparer les données spécifiques
        self._prepare_data(data, controles_ponte, performances)
        
    async def prepare_training_data(self):
        """Prépare les données d'entraînement (choisit automatiquement le mode)."""
        if self.is_async:
            await self.prepare_training_data_async()
        else:
            self.prepare_training_data_sync()
    
    def _prepare_data(self, data: pd.DataFrame, controles_ponte: pd.DataFrame, performances: pd.DataFrame):
        """Prépare les données pour l'entraînement."""
        # Encoder les variables catégorielles
        categorical_cols = ['type_volaille', 'type_production', 'systeme_elevage', 'sexe', 'statut']
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        # Calculer l'âge des volailles
        data['age'] = (datetime.datetime.now() - pd.to_datetime(data['date_naissance'])).dt.days
        
        # Préparer les datasets spécifiques
        self.ponte_data = self._prepare_ponte_data(data, controles_ponte)
        self.croissance_data = self._prepare_croissance_data(data, performances)
    
    def _prepare_ponte_data(self, data: pd.DataFrame, controles_ponte: pd.DataFrame) -> pd.DataFrame:
        """Prépare les données pour le modèle de ponte."""
        # Implémentation de la logique de préparation des données de ponte
        merged = pd.merge(data, controles_ponte, left_on='id', right_on='lot_id')
        # Ajouter ici les transformations spécifiques
        return merged.dropna()
    
    def _prepare_croissance_data(self, data: pd.DataFrame, performances: pd.DataFrame) -> pd.DataFrame:
        """Prépare les données pour le modèle de croissance."""
        # Implémentation de la logique de préparation des données de croissance
        merged = pd.merge(data, performances, left_on='id', right_on='lot_id')
        # Ajouter ici les transformations spécifiques
        return merged.dropna()
    
    async def train_models(self):
        """Entraîne tous les modèles et enregistre leurs performances."""
        if self.ponte_data is None or self.croissance_data is None:
            await self.prepare_training_data()
            
        await self.train_ponte_model()
        await self.train_croissance_model()
    
    async def train_ponte_model(self):
        """Entraîne un modèle pour prédire la ponte."""
        X = self.ponte_data.drop(['taux_ponte', 'nombre_oeufs'], axis=1)
        y_ponte = self.ponte_data['taux_ponte']
        y_oeufs = self.ponte_data['nombre_oeufs']
        
        X_train, X_test, y_train_ponte, y_test_ponte = train_test_split(
            X, y_ponte, test_size=0.2, random_state=42)
        _, _, y_train_oeufs, y_test_oeufs = train_test_split(
            X, y_oeufs, test_size=0.2, random_state=42)
        
        self.ponte_rate_model = GradientBoostingRegressor(n_estimators=100)
        self.oeufs_count_model = RandomForestRegressor(n_estimators=100)
        
        self.ponte_rate_model.fit(X_train, y_train_ponte)
        self.oeufs_count_model.fit(X_train, y_train_oeufs)
        
        self._evaluate_and_store_performance(
            "PonteRateModel", self.ponte_rate_model, X_test, y_test_ponte)
        self._evaluate_and_store_performance(
            "OeufsCountModel", self.oeufs_count_model, X_test, y_test_oeufs)
    
    async def train_croissance_model(self):
        """Entraîne un modèle pour prédire la croissance."""
        X = self.croissance_data.drop(['poids_moyen', 'gain_moyen_journalier'], axis=1)
        y_poids = self.croissance_data['poids_moyen']
        y_gain = self.croissance_data['gain_moyen_journalier']
        
        X_train, X_test, y_train_poids, y_test_poids = train_test_split(
            X, y_poids, test_size=0.2, random_state=42)
        _, _, y_train_gain, y_test_gain = train_test_split(
            X, y_gain, test_size=0.2, random_state=42)
        
        self.poids_model = GradientBoostingRegressor(n_estimators=100)
        self.gain_model = RandomForestRegressor(n_estimators=100)
        
        self.poids_model.fit(X_train, y_train_poids)
        self.gain_model.fit(X_train, y_train_gain)
        
        self._evaluate_and_store_performance(
            "PoidsModel", self.poids_model, X_test, y_test_poids)
        self._evaluate_and_store_performance(
            "GainModel", self.gain_model, X_test, y_test_gain)
    
    def _evaluate_and_store_performance(self, model_name: str, model, X_test, y_test):
        """Évalue un modèle et stocke ses performances."""
        y_pred = model.predict(X_test)
        
        performance = ModelPerformance(
            model_name=model_name,
            mse=mean_squared_error(y_test, y_pred),
            mae=mean_absolute_error(y_test, y_pred),
            r2=r2_score(y_test, y_pred),
            cv_score=np.mean(cross_val_score(model, X_test, y_test, cv=5))
        )
        
        self.model_performances[model_name] = performance
    
    def get_model_performances(self) -> List[Dict]:
        """Retourne les performances de tous les modèles."""
        return [asdict(perf) for perf in self.model_performances.values()]
    
    def get_best_model(self, metric: str = 'r2') -> Dict:
        """
        Retourne le meilleur modèle selon la métrique spécifiée.
        
        Args:
            metric: 'r2', 'mae', 'mse' ou 'cv_score'
        """
        if not self.model_performances:
            return {}
            
        reverse = metric in ['r2', 'cv_score']
        best_model = max(self.model_performances.values(), 
                        key=lambda x: getattr(x, metric)) if reverse else \
                   min(self.model_performances.values(), 
                        key=lambda x: getattr(x, metric))
        
        return asdict(best_model)
    
    async def predict_ponte_async(self, volaille_data: Dict[str, Any]) -> Dict[str, float]:
        """Prédit le taux de ponte et le nombre d'œufs (mode asynchrone)."""
        return await self._predict_wrapper(self._predict_ponte, volaille_data)
    
    def predict_ponte_sync(self, volaille_data: Dict[str, Any]) -> Dict[str, float]:
        """Prédit le taux de ponte et le nombre d'œufs (mode synchrone)."""
        return self._predict_ponte(volaille_data)
    
    async def predict_croissance_async(self, volaille_data: Dict[str, Any]) -> Dict[str, float]:
        """Prédit la croissance (mode asynchrone)."""
        return await self._predict_wrapper(self._predict_croissance, volaille_data)
    
    def predict_croissance_sync(self, volaille_data: Dict[str, Any]) -> Dict[str, float]:
        """Prédit la croissance (mode synchrone)."""
        return self._predict_croissance(volaille_data)
    
    async def _predict_wrapper(self, predict_func, *args, **kwargs):
        """Wrapper pour exécuter des prédictions en mode asynchrone."""
        # Cette méthode permet de gérer les opérations asynchrones si nécessaire
        return predict_func(*args, **kwargs)
    
    def _predict_ponte(self, volaille_data: Dict[str, Any]) -> Dict[str, float]:
        """Implémentation synchrone de la prédiction de ponte."""
        if not self.ponte_rate_model or not self.oeufs_count_model:
            raise ValueError("Models not trained. Call train_ponte_model first.")
            
        input_data = self._prepare_input(volaille_data, 'ponte')
        
        return {
            'taux_ponte': float(self.ponte_rate_model.predict(input_data)[0]),
            'nombre_oeufs': float(self.oeufs_count_model.predict(input_data)[0])
        }
    
    def _predict_croissance(self, volaille_data: Dict[str, Any]) -> Dict[str, float]:
        """Implémentation synchrone de la prédiction de croissance."""
        if not self.poids_model or not self.gain_model:
            raise ValueError("Models not trained. Call train_croissance_model first.")
            
        input_data = self._prepare_input(volaille_data, 'croissance')
        
        return {
            'poids_moyen': float(self.poids_model.predict(input_data)[0]),
            'gain_moyen_journalier': float(self.gain_model.predict(input_data)[0])
        }
    
    def _prepare_input(self, data: Dict[str, Any], model_type: str) -> pd.DataFrame:
        """Prépare les données d'entrée pour la prédiction."""
        df = pd.DataFrame([data])
        
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        
        if model_type == 'ponte' and 'date_mise_en_production' in df.columns:
            df['jours_en_production'] = (
                pd.to_datetime(df['date_controle']) - 
                pd.to_datetime(df['date_mise_en_production'])
            ).dt.days
        elif model_type == 'croissance' and 'date_mise_en_place' in df.columns:
            df['jours_en_elevage'] = (
                pd.to_datetime(df['date_controle']) - 
                pd.to_datetime(df['date_mise_en_place'])
            ).dt.days
        
        return df
    
    def save_model(self, filename: str = "avicole_model.joblib"):
        """
        Sauvegarde les modèles dans le dossier 'files' du répertoire précédent.
        
        Args:
            filename: Nom du fichier de sauvegarde (par défaut: 'avicole_model.joblib')
        """                
        # Chemin complet du fichier
        filepath = MODELS_DIR / filename
        
        # Sauvegarde des modèles
        dump({
            'models': {
                'ponte_rate': self.ponte_rate_model,
                'oeufs_count': self.oeufs_count_model,
                'poids': self.poids_model,
                'gain': self.gain_model
            },
            'encoders': self.label_encoders
        }, filepath)
        
        return str(filepath)  # Retourne le chemin complet pour information

    def load_model(self, filename: str = "avicole_model.joblib"):
        """
        Charge les modèles depuis le dossier 'files' du répertoire précédent.
        
        Args:
            filename: Nom du fichier à charger (par défaut: 'avicole_model.joblib')
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        filepath = MODELS_DIR / filename
        
        # Vérification que le fichier existe
        if not filepath.exists():
            raise FileNotFoundError(f"Le fichier de modèle {filepath} n'existe pas")
        
        # Chargement des données
        data = load(filepath)
        models = data['models']
        
        # Attribution des modèles
        self.ponte_rate_model = models['ponte_rate']
        self.oeufs_count_model = models['oeufs_count']
        self.poids_model = models['poids']
        self.gain_model = models['gain']
        self.label_encoders = data['encoders']
        
        return self  # Pour permettre le chaînage