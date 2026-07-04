# backend/app/models/video.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


class Camera(Base, TimestampMixin):
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    url = Column(String(500), nullable=False)  # RTSP ou autre
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    roi_config = Column(Text, nullable=True)  # JSON des régions d'intérêt
    fps = Column(Integer, default=15)
    
    # Relations
    enclos = relationship("Enclos", back_populates="cameras")
    videos = relationship("VideoRecord", back_populates="camera")
    
    def __repr__(self):
        return f"<Camera(id={self.id}, name={self.name})>"


class VideoRecord(Base, TimestampMixin):
    __tablename__ = "video_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    animal_id = Column(Integer, ForeignKey("animaux.id"), nullable=True)
    enclos_id = Column(Integer, ForeignKey("enclos.id"), nullable=True)
    
    # Métadonnées vidéo
    debut = Column(DateTime, nullable=False)
    fin = Column(DateTime, nullable=True)
    duree_secondes = Column(Float, nullable=True)
    file_path = Column(Text, nullable=False)
    file_size_mb = Column(Float, nullable=True)
    
    # Type d'action déclencheuse
    action_type = Column(String(50), nullable=True)  # pesee, nettoyage, soin, naissance, mortalite, compost
    action_id = Column(Integer, nullable=True)  # ID de l'action associée
    utilisateur_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Traitement IA
    ia_anomalie_detectee = Column(Boolean, default=False)
    ia_anomalie_type = Column(String(100), nullable=True)
    ia_confiance = Column(Float, nullable=True)
    
    # Relations
    camera = relationship("Camera", back_populates="videos")
    animal = relationship("Animal", back_populates="videos")
    pesee = relationship("Pesee", back_populates="video", uselist=False)
    utilisateur = relationship("User")
    
    def __repr__(self):
        return f"<VideoRecord(id={self.id}, camera_id={self.camera_id}, debut={self.debut})>"