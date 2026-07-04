# backend/app/cpp_bindings/__init__.py
"""
Liaisons C++ pour les calculs hautes performances
Ces modules sont optionnels - si les bibliothèques compilées ne sont pas disponibles,
un fallback Python est utilisé automatiquement.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Variables globales pour indiquer si les modules C++ sont disponibles
GROWTH_CPP_AVAILABLE = False
VIDEO_CPP_AVAILABLE = False
SENSOR_CPP_AVAILABLE = False

# Tentative d'import des modules compilés
try:
    from . import growth_binding
    GROWTH_CPP_AVAILABLE = True
    logger.info("C++ growth module loaded successfully")
except ImportError as e:
    logger.warning(f"C++ growth module not available: {e}. Using Python fallback.")

try:
    from . import video_binding
    VIDEO_CPP_AVAILABLE = True
    logger.info("C++ video module loaded successfully")
except ImportError as e:
    logger.warning(f"C++ video module not available: {e}. Using Python fallback.")

try:
    from . import sensor_binding
    SENSOR_CPP_AVAILABLE = True
    logger.info("C++ sensor module loaded successfully")
except ImportError as e:
    logger.warning(f"C++ sensor module not available: {e}. Using Python fallback.")

__all__ = [
    "GROWTH_CPP_AVAILABLE",
    "VIDEO_CPP_AVAILABLE", 
    "SENSOR_CPP_AVAILABLE",
    "growth_binding",
    "video_binding",
    "sensor_binding",
]