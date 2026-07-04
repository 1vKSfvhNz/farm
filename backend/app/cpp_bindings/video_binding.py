# backend/app/cpp_bindings/video_binding.py
"""
Liaison Python vers le module C++ de traitement vidéo avec IA
Détection d'objets, tracking, analyse comportementale
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..cpp_bindings import VIDEO_CPP_AVAILABLE

logger = logging.getLogger(__name__)

# Tentative d'import du module compilé
_video_cpp = None
if VIDEO_CPP_AVAILABLE:
    try:
        import video_detector_cpp as _video_cpp
        logger.info("Successfully imported video_detector_cpp module")
    except ImportError as e:
        logger.warning(f"Could not import video_detector_cpp: {e}")
        VIDEO_CPP_AVAILABLE = False


class DetectionClass(str, Enum):
    """Classes d'objets détectables"""
    ANIMAL = "animal"
    PERSON = "person"
    VEHICLE = "vehicle"
    ANOMALY = "anomaly"


@dataclass
class BoundingBox:
    """Boîte englobante pour la détection"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    @property
    def center_x(self) -> int:
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.y + self.height // 2
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Detection:
    """Résultat de détection"""
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None


@dataclass 
class TrackedObject:
    """Objet suivi dans le temps"""
    track_id: int
    class_name: str
    positions: List[Tuple[int, int]]  # Historique des positions
    frames_seen: int
    first_seen_frame: int
    last_seen_frame: int
    is_active: bool = True


@dataclass
class VideoAnalysisResult:
    """Résultat complet d'analyse vidéo"""
    frame_number: int
    timestamp: float
    detections: List[Detection]
    tracked_objects: List[TrackedObject]
    anomalies: List[Dict[str, Any]]
    processing_time_ms: float


class VideoAnalyzerCpp:
    """
    Analyseur vidéo utilisant le module C++ avec optimisation CUDA
    Détection YOLOv8 + Tracking BoT-SORT
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        self.available = VIDEO_CPP_AVAILABLE and _video_cpp is not None
        self.use_gpu = use_gpu and self.available
        self.analyzer = None
        
        if self.available:
            try:
                self.analyzer = _video_cpp.VideoAnalyzer(
                    model_path or "",
                    use_gpu,
                    confidence_threshold,
                    iou_threshold
                )
                logger.info(f"C++ VideoAnalyzer initialized (GPU: {use_gpu})")
            except Exception as e:
                logger.error(f"Failed to initialize C++ VideoAnalyzer: {e}")
                self.available = False
    
    def analyze_frame(
        self,
        frame_data: bytes,
        width: int,
        height: int,
        channels: int = 3
    ) -> Optional[VideoAnalysisResult]:
        """
        Analyser une frame vidéo
        
        Args:
            frame_data: Données brutes de l'image (RGB)
            width: Largeur de l'image
            height: Hauteur de l'image
            channels: Nombre de canaux (3 pour RGB)
        
        Returns:
            Résultat d'analyse ou None si erreur
        """
        if not self.available or not self.analyzer:
            return self._analyze_frame_python(frame_data, width, height)
        
        try:
            result = self.analyzer.analyze_frame(frame_data, width, height, channels)
            return self._parse_cpp_result(result)
        except Exception as e:
            logger.error(f"C++ frame analysis failed: {e}")
            return self._analyze_frame_python(frame_data, width, height)
    
    def process_video_stream(
        self,
        stream_url: str,
        callback=None,
        max_frames: int = -1
    ) -> List[VideoAnalysisResult]:
        """
        Traiter un flux vidéo en continu
        
        Args:
            stream_url: URL RTSP ou chemin fichier
            callback: Fonction callback pour résultats temps réel
            max_frames: Nombre max de frames (-1 = infini)
        
        Returns:
            Liste des résultats d'analyse
        """
        results = []
        
        if not self.available or not self.analyzer:
            logger.warning("C++ video processing not available")
            return results
        
        try:
            # Appel C++ avec callback
            cpp_callback = None
            if callback:
                # Wrapper pour le callback Python
                def wrapper(frame_num, timestamp, detections_json):
                    callback(frame_num, timestamp, detections_json)
                cpp_callback = wrapper
            
            raw_results = self.analyzer.process_stream(
                stream_url, cpp_callback, max_frames
            )
            
            for r in raw_results:
                results.append(self._parse_cpp_result(r))
            
        except Exception as e:
            logger.error(f"C++ video stream processing failed: {e}")
        
        return results
    
    def set_roi(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Définir la région d'intérêt (ROI)
        Optimise les performances en ne traitant qu'une zone
        
        Returns:
            True si succès
        """
        if not self.available or not self.analyzer:
            return False
        
        try:
            return self.analyzer.set_roi(x, y, width, height)
        except Exception as e:
            logger.error(f"Failed to set ROI: {e}")
            return False
    
    def clear_roi(self) -> bool:
        """Effacer la ROI"""
        if not self.available or not self.analyzer:
            return False
        
        try:
            return self.analyzer.clear_roi()
        except Exception as e:
            logger.error(f"Failed to clear ROI: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Obtenir les statistiques de performance"""
        if not self.available or not self.analyzer:
            return {"fps": 0, "avg_inference_ms": 0, "gpu_memory_mb": 0}
        
        try:
            stats = self.analyzer.get_performance_stats()
            return {
                "fps": stats[0],
                "avg_inference_ms": stats[1],
                "gpu_memory_mb": stats[2],
            }
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {"fps": 0, "avg_inference_ms": 0, "gpu_memory_mb": 0}
    
    def release(self):
        """Libérer les ressources C++"""
        if self.available and self.analyzer:
            try:
                self.analyzer.release()
            except Exception as e:
                logger.error(f"Failed to release video analyzer: {e}")
    
    # ============ IMPLÉMENTATION PYTHON (FALLBACK) ============
    
    def _analyze_frame_python(
        self,
        frame_data: bytes,
        width: int,
        height: int
    ) -> Optional[VideoAnalysisResult]:
        """Fallback Python simple (sans détection IA)"""
        import time
        
        return VideoAnalysisResult(
            frame_number=0,
            timestamp=time.time(),
            detections=[],
            tracked_objects=[],
            anomalies=[],
            processing_time_ms=0
        )
    
    def _parse_cpp_result(self, raw_result) -> VideoAnalysisResult:
        """Parser le résultat C++ vers Python"""
        # Cette méthode sera implémentée selon la structure retournée par C++
        # Format attendu: (frame_num, timestamp, detections, tracked, anomalies, time_ms)
        
        detections = []
        tracked_objects = []
        anomalies = []
        
        if isinstance(raw_result, (list, tuple)) and len(raw_result) >= 6:
            frame_num, timestamp, dets, tracks, anoms, proc_time = raw_result[:6]
            
            for d in dets:
                detections.append(Detection(
                    class_name=d[0],
                    confidence=d[1],
                    bbox=BoundingBox(d[2], d[3], d[4], d[5], d[1])
                ))
            
            for t in tracks:
                tracked_objects.append(TrackedObject(
                    track_id=t[0],
                    class_name=t[1],
                    positions=t[2],
                    frames_seen=t[3],
                    first_seen_frame=t[4],
                    last_seen_frame=t[5]
                ))
            
            anomalies = anoms
        
        return VideoAnalysisResult(
            frame_number=raw_result[0] if isinstance(raw_result, (list, tuple)) else 0,
            timestamp=raw_result[1] if isinstance(raw_result, (list, tuple)) else 0,
            detections=detections,
            tracked_objects=tracked_objects,
            anomalies=anomalies,
            processing_time_ms=raw_result[5] if isinstance(raw_result, (list, tuple)) and len(raw_result) > 5 else 0
        )


# Instance globale (à initialiser avec les bons paramètres)
video_analyzer = None

def get_video_analyzer(
    model_path: Optional[str] = None,
    use_gpu: bool = True
) -> VideoAnalyzerCpp:
    """Obtenir l'instance globale de l'analyseur vidéo"""
    global video_analyzer
    if video_analyzer is None:
        video_analyzer = VideoAnalyzerCpp(model_path, use_gpu)
    return video_analyzer