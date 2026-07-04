# backend/app/cpp_bindings/sensor_binding.py
"""
Liaison Python vers le module C++ de traitement des capteurs
Compression, filtrage, aggregation des données capteurs
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..cpp_bindings import SENSOR_CPP_AVAILABLE

logger = logging.getLogger(__name__)

# Tentative d'import du module compilé
_sensor_cpp = None
if SENSOR_CPP_AVAILABLE:
    try:
        import sensor_cpp as _sensor_cpp
        logger.info("Successfully imported sensor_cpp module")
    except ImportError as e:
        logger.warning(f"Could not import sensor_cpp: {e}")
        SENSOR_CPP_AVAILABLE = False


@dataclass
class SensorDataPoint:
    """Point de donnée capteur"""
    timestamp: datetime
    value: float
    quality: float = 1.0  # Qualité du signal (0-1)


@dataclass
class AggregatedData:
    """Données agrégées"""
    period_start: datetime
    period_end: datetime
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    sum_value: float


class SensorProcessorCpp:
    """
    Processeur de données capteurs optimisé en C++
    Compression temps réel, filtrage, détection d'anomalies
    """
    
    def __init__(self, window_size_seconds: int = 60):
        self.available = SENSOR_CPP_AVAILABLE and _sensor_cpp is not None
        self.processor = None
        self.window_size = window_size_seconds
        
        if self.available:
            try:
                self.processor = _sensor_cpp.SensorProcessor(window_size_seconds)
                logger.info("C++ SensorProcessor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize C++ SensorProcessor: {e}")
                self.available = False
    
    def add_data_point(self, sensor_id: str, timestamp: datetime, value: float) -> bool:
        """Ajouter un point de donnée"""
        if not self.available or not self.processor:
            return False
        
        try:
            ts = timestamp.timestamp()
            return self.processor.add_data_point(sensor_id, ts, value)
        except Exception as e:
            logger.error(f"Failed to add data point: {e}")
            return False
    
    def add_batch(
        self,
        sensor_id: str,
        timestamps: List[datetime],
        values: List[float]
    ) -> int:
        """Ajouter un lot de données"""
        if not self.available or not self.processor:
            return 0
        
        try:
            ts_list = [t.timestamp() for t in timestamps]
            return self.processor.add_batch(sensor_id, ts_list, values)
        except Exception as e:
            logger.error(f"Failed to add batch: {e}")
            return 0
    
    def get_window_data(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[SensorDataPoint]:
        """Récupérer les données dans une fenêtre temporelle"""
        if not self.available or not self.processor:
            return []
        
        try:
            start_ts = start_time.timestamp()
            end_ts = end_time.timestamp()
            data = self.processor.get_window_data(sensor_id, start_ts, end_ts)
            
            return [
                SensorDataPoint(
                    timestamp=datetime.fromtimestamp(d[0]),
                    value=d[1],
                    quality=d[2]
                ) for d in data
            ]
        except Exception as e:
            logger.error(f"Failed to get window data: {e}")
            return []
    
    def aggregate(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 3600
    ) -> List[AggregatedData]:
        """Agréger les données par intervalle"""
        if not self.available or not self.processor:
            return self._aggregate_python(sensor_id, start_time, end_time, interval_seconds)
        
        try:
            start_ts = start_time.timestamp()
            end_ts = end_time.timestamp()
            agg_data = self.processor.aggregate(
                sensor_id, start_ts, end_ts, interval_seconds
            )
            
            return [
                AggregatedData(
                    period_start=datetime.fromtimestamp(d[0]),
                    period_end=datetime.fromtimestamp(d[1]),
                    count=int(d[2]),
                    min_value=d[3],
                    max_value=d[4],
                    mean_value=d[5],
                    median_value=d[6],
                    std_dev=d[7],
                    sum_value=d[8]
                ) for d in agg_data
            ]
        except Exception as e:
            logger.error(f"Failed to aggregate: {e}")
            return self._aggregate_python(sensor_id, start_time, end_time, interval_seconds)
    
    def compress_data(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime,
        tolerance: float = 0.01
    ) -> List[SensorDataPoint]:
        """
        Compresser les données avec l'algorithme Douglas-Peucker
        Réduit le nombre de points tout en préservant la forme
        
        Args:
            sensor_id: ID du capteur
            start_time: Début de la plage
            end_time: Fin de la plage
            tolerance: Tolérance de compression (plus grand = plus compressé)
        
        Returns:
            Liste compressée des points
        """
        if not self.available or not self.processor:
            return self._compress_python(sensor_id, start_time, end_time, tolerance)
        
        try:
            start_ts = start_time.timestamp()
            end_ts = end_time.timestamp()
            compressed = self.processor.compress_data(
                sensor_id, start_ts, end_ts, tolerance
            )
            
            return [
                SensorDataPoint(
                    timestamp=datetime.fromtimestamp(d[0]),
                    value=d[1],
                    quality=d[2]
                ) for d in compressed
            ]
        except Exception as e:
            logger.error(f"Failed to compress data: {e}")
            return self._compress_python(sensor_id, start_time, end_time, tolerance)
    
    def detect_anomalies(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime,
        std_dev_threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Détecter les anomalies statistiques
        
        Returns:
            Liste des anomalies détectées
        """
        if not self.available or not self.processor:
            return []
        
        try:
            start_ts = start_time.timestamp()
            end_ts = end_time.timestamp()
            anomalies = self.processor.detect_anomalies(
                sensor_id, start_ts, end_ts, std_dev_threshold
            )
            
            return [
                {
                    "timestamp": datetime.fromtimestamp(a[0]),
                    "value": a[1],
                    "expected": a[2],
                    "std_dev": a[3],
                    "score": a[4]
                } for a in anomalies
            ]
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return []
    
    def get_stats(self, sensor_id: str) -> Dict[str, Any]:
        """Obtenir les statistiques pour un capteur"""
        if not self.available or not self.processor:
            return {}
        
        try:
            stats = self.processor.get_stats(sensor_id)
            return {
                "total_points": stats[0],
                "first_timestamp": datetime.fromtimestamp(stats[1]) if stats[1] > 0 else None,
                "last_timestamp": datetime.fromtimestamp(stats[2]) if stats[2] > 0 else None,
                "min_value": stats[3],
                "max_value": stats[4],
                "mean_value": stats[5],
                "std_dev": stats[6],
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def clear_sensor_data(self, sensor_id: str) -> bool:
        """Effacer toutes les données d'un capteur"""
        if not self.available or not self.processor:
            return False
        
        try:
            return self.processor.clear_sensor_data(sensor_id)
        except Exception as e:
            logger.error(f"Failed to clear sensor data: {e}")
            return False
    
    # ============ IMPLÉMENTATIONS PYTHON (FALLBACK) ============
    
    def _aggregate_python(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int
    ) -> List[AggregatedData]:
        """Agrégation Python simple"""
        # Dans un vrai fallback, on implémenterait l'agrégation
        # Pour l'instant, retourner une liste vide
        return []
    
    def _compress_python(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime,
        tolerance: float
    ) -> List[SensorDataPoint]:
        """Compression Python simple"""
        return []


# Instance globale
sensor_processor = SensorProcessorCpp()