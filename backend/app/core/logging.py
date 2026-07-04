# backend/app/core/logging.py
import logging
import sys
from datetime import datetime
import json

from ..config import settings


class JSONFormatter(logging.Formatter):
    """Formateur JSON pour les logs structurés"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Formateur minimal pour la console"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Format minimal : seulement le message
        return f"{record.getMessage()}"


def setup_logging() -> None:
    """Configurer le logging pour l'application"""
    
    # Désactiver TOUS les logs par défaut
    logging.root.setLevel(logging.ERROR)
    
    # Supprimer tous les handlers existants
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Créer un handler console avec niveau ERROR uniquement
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(ConsoleFormatter())
    logging.root.addHandler(console_handler)
    
    # Désactiver complètement SQLAlchemy
    logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.dialects").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.orm").setLevel(logging.ERROR)
    
    # Désactiver Uvicorn
    # logging.getLogger("uvicorn").setLevel(logging.ERROR)
    # logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
    # logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    
    # Désactiver httpx
    logging.getLogger("httpx").setLevel(logging.ERROR)
    
    # Désactiver Redis
    logging.getLogger("redis").setLevel(logging.ERROR)
    
    # Désactiver passlib
    logging.getLogger("passlib").setLevel(logging.ERROR)


# Logger principal (niveau ERROR uniquement)
logger = logging.getLogger("farm_manager")
logger.setLevel(logging.ERROR)