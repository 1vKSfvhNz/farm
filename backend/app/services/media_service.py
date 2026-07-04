# backend/app/services/media_service.py
"""
Service de gestion des médias (photos et vidéos)
"""

import os
import uuid
import base64
import io
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from PIL import Image
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..config import settings
from ..models.video import Camera, VideoRecord
from ..models.animal import Animal
from ..schemas.media import *

logger = logging.getLogger(__name__)


class MediaService:
    """Service unifié de gestion des médias (photos et vidéos)"""
    
    def __init__(self):
        # Configuration des photos
        self.upload_dir = settings.UPLOAD_DIR
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        self.max_size = settings.UPLOAD_MAX_SIZE
        self.quality = settings.PHOTO_QUALITY
        self.max_width = settings.PHOTO_MAX_WIDTH
        self.max_height = settings.PHOTO_MAX_HEIGHT
        
        # Créer les dossiers nécessaires
        self._ensure_directories()
    
    # ==================== PARTIE PHOTOS ====================
    
    def _ensure_directories(self):
        """Crée les dossiers nécessaires pour les uploads"""
        species_dirs = ["bovins", "ovins", "caprins", "volailles", "poissons", "temp"]
        
        # Utiliser un dossier temporaire si permission denied
        try:
            # Tenter de créer dans le dossier configuré
            if not self.upload_dir.exists():
                self.upload_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fallback: utiliser /tmp
            import tempfile
            temp_base = Path(tempfile.gettempdir()) / "farm_manager_uploads"
            self.upload_dir = temp_base
            logger.warning(f"Permission denied pour le dossier configuré, utilisation de: {self.upload_dir}")
            self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer les sous-dossiers
        for dir_name in species_dirs:
            dir_path = self.upload_dir / dir_name
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logger.error(f"Impossible de créer {dir_path}")
                pass
    
    def _get_extension_from_mime(self, mime_type: str) -> str:
        """Convertit le type MIME en extension"""
        mime_to_ext = {
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp'
        }
        return mime_to_ext.get(mime_type, '.jpg')
    
    def _get_extension_from_filename(self, filename: str) -> str:
        """Extrait l'extension d'un nom de fichier"""
        return Path(filename).suffix.lower()
    
    def _optimize_image(self, image_data: bytes, extension: str) -> bytes:
        """Optimise l'image (redimensionnement et compression)"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Convertir RGBA en RGB si nécessaire
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionner si nécessaire
            if image.width > self.max_width or image.height > self.max_height:
                image.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
            
            # Sauvegarder l'image optimisée
            output = io.BytesIO()
            save_format = 'JPEG' if extension in ['.jpg', '.jpeg'] else 'PNG'
            image.save(output, format=save_format, quality=self.quality, optimize=True)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation de l'image: {e}")
            return image_data
    
    def save_base64_photo(self, photo_base64: str, espece: str, animal_id: int) -> Optional[str]:
        """
        Sauvegarde une photo encodée en base64
        Retourne l'URL relative
        """
        try:
            # Décoder le base64
            if ',' in photo_base64:
                header, encoded = photo_base64.split(',', 1)
                mime_type = header.split(':')[1].split(';')[0]
                extension = self._get_extension_from_mime(mime_type)
            else:
                encoded = photo_base64
                extension = '.jpg'
            
            # Décoder les données
            image_data = base64.b64decode(encoded)
            
            # Vérifier la taille
            if len(image_data) > self.max_size:
                raise ValueError(f"Photo trop grande: {len(image_data)} > {self.max_size}")
            
            # Optimiser l'image
            optimized_data = self._optimize_image(image_data, extension)
            
            # Générer un nom de fichier unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{espece}_{animal_id}_{timestamp}_{unique_id}{extension}"
            
            # Chemin du dossier spécifique
            species_dir = self.upload_dir / espece
            species_dir.mkdir(exist_ok=True)
            
            # Sauvegarder le fichier
            filepath = species_dir / filename
            with open(filepath, 'wb') as f:
                f.write(optimized_data)
            
            # Retourner l'URL relative
            return f"/uploads/{espece}/{filename}"
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la photo: {e}")
            return None
    
    async def save_upload_file(self, file: UploadFile, espece: str, animal_id: int) -> Optional[str]:
        """
        Sauvegarde un fichier uploadé directement
        """
        try:
            # Vérifier l'extension
            extension = self._get_extension_from_filename(file.filename)
            if extension not in self.allowed_extensions:
                raise ValueError(f"Extension non autorisée: {extension}")
            
            # Lire le contenu
            content = await file.read()
            
            # Vérifier la taille
            if len(content) > self.max_size:
                raise ValueError(f"Fichier trop grand: {len(content)} > {self.max_size}")
            
            # Optimiser l'image
            optimized_data = self._optimize_image(content, extension)
            
            # Générer un nom de fichier unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{espece}_{animal_id}_{timestamp}_{unique_id}{extension}"
            
            # Sauvegarder
            species_dir = self.upload_dir / espece
            species_dir.mkdir(exist_ok=True)
            
            filepath = species_dir / filename
            with open(filepath, 'wb') as f:
                f.write(optimized_data)
            
            return f"/uploads/{espece}/{filename}"
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du fichier: {e}")
            return None
    
    def delete_photo(self, photo_url: str) -> bool:
        """Supprime une photo du système de fichiers"""
        if not photo_url:
            return False
        
        try:
            # Extraire le chemin relatif
            if photo_url.startswith('/uploads/'):
                relative_path = photo_url[9:]  # Enlever '/uploads/'
            else:
                relative_path = photo_url
            
            filepath = self.upload_dir / relative_path
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Photo supprimée: {filepath}")
                return True
            else:
                logger.warning(f"Photo non trouvée: {filepath}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la photo: {e}")
        
        return False
    
    def update_photo(self, photo_base64: str, espece: str, animal_id: int, old_photo_url: Optional[str]) -> Optional[str]:
        """Met à jour une photo (supprime l'ancienne, sauvegarde la nouvelle)"""
        # Supprimer l'ancienne photo
        if old_photo_url:
            self.delete_photo(old_photo_url)
        
        # Sauvegarder la nouvelle
        return self.save_base64_photo(photo_base64, espece, animal_id)
    
    async def get_animal_photo(self, db: AsyncSession, animal_id: int) -> Optional[Dict[str, Any]]:
        """Récupère la photo d'un animal avec ses métadonnées"""
        stmt = select(Animal).where(Animal.id == animal_id)
        result = await db.execute(stmt)
        animal = result.scalar_one_or_none()
        
        if animal and animal.photo_url:
            return {
                "url": animal.photo_url,
                "animal_id": animal.id,
                "identification": animal.identification,
                "photo_url": animal.photo_url
            }
        return None
    
    # ==================== PARTIE VIDÉOS ====================
    
    async def get_camera_stream_url(
        self,
        db: AsyncSession,
        camera_id: int
    ) -> Optional[str]:
        """Obtenir l'URL du flux vidéo d'une caméra"""
        stmt = select(Camera).where(
            Camera.id == camera_id,
            Camera.is_active == True
        )
        result = await db.execute(stmt)
        camera = result.scalar_one_or_none()
        
        return camera.stream_url if camera else None
    
    async def get_all_active_cameras(
        self,
        db: AsyncSession,
        enclos_id: Optional[int] = None
    ) -> List[Camera]:
        """Obtenir toutes les caméras actives"""
        stmt = select(Camera).where(Camera.is_active == True)
        
        if enclos_id:
            stmt = stmt.where(Camera.enclos_id == enclos_id)
        
        stmt = stmt.order_by(Camera.name)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_videos_for_animal(
        self,
        db: AsyncSession,
        animal_id: int,
        limit: int = 10,
        skip: int = 0
    ) -> List[VideoRecord]:
        """Obtenir les vidéos associées à un animal"""
        stmt = select(VideoRecord).where(
            VideoRecord.animal_id == animal_id
        ).order_by(VideoRecord.debut.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_videos_for_enclos(
        self,
        db: AsyncSession,
        enclos_id: int,
        limit: int = 50,
        skip: int = 0
    ) -> List[VideoRecord]:
        """Obtenir les vidéos associées à un enclos"""
        stmt = select(VideoRecord).where(
            VideoRecord.enclos_id == enclos_id
        ).order_by(VideoRecord.debut.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_videos_by_date_range(
        self,
        db: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        animal_id: Optional[int] = None,
        enclos_id: Optional[int] = None
    ) -> List[VideoRecord]:
        """Obtenir les vidéos dans une plage de dates"""
        stmt = select(VideoRecord).where(
            and_(
                VideoRecord.debut >= start_date,
                VideoRecord.fin <= end_date
            )
        )
        
        if animal_id:
            stmt = stmt.where(VideoRecord.animal_id == animal_id)
        if enclos_id:
            stmt = stmt.where(VideoRecord.enclos_id == enclos_id)
        
        stmt = stmt.order_by(VideoRecord.debut.desc())
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def link_video_to_action(
        self,
        db: AsyncSession,
        video_id: int,
        action_type: str,
        action_id: int
    ) -> bool:
        """Lier une vidéo à une action (alimentation, vaccination, etc.)"""
        stmt = select(VideoRecord).where(VideoRecord.id == video_id)
        result = await db.execute(stmt)
        video = result.scalar_one_or_none()
        
        if video:
            video.action_type = action_type
            video.action_id = action_id
            await db.commit()
            logger.info(f"Vidéo {video_id} liée à {action_type}/{action_id}")
            return True
        
        return False
    
    async def create_video_record(
        self,
        db: AsyncSession,
        video_data: VideoRecordCreate,
        created_by: int
    ) -> VideoRecord:
        """Créer un enregistrement vidéo"""
        video = VideoRecord(
            **video_data.dict(),
            created_by=created_by,
            updated_by=created_by
        )
        db.add(video)
        await db.commit()
        await db.refresh(video)
        
        logger.info(f"Enregistrement vidéo créé: {video.id}")
        return video
    
    async def update_camera_status(
        self,
        db: AsyncSession,
        camera_id: int,
        is_active: bool
    ) -> Optional[Camera]:
        """Mettre à jour le statut d'une caméra"""
        stmt = select(Camera).where(Camera.id == camera_id)
        result = await db.execute(stmt)
        camera = result.scalar_one_or_none()
        
        if camera:
            camera.is_active = is_active
            await db.commit()
            logger.info(f"Caméra {camera_id} status: {is_active}")
            return camera
        
        return None
    
    async def get_camera_by_enclos(
        self,
        db: AsyncSession,
        enclos_id: int
    ) -> List[Camera]:
        """Obtenir les caméras d'un enclos"""
        stmt = select(Camera).where(
            Camera.enclos_id == enclos_id,
            Camera.is_active == True
        )
        result = await db.execute(stmt)
        return result.scalars().all()
    
    # ==================== STATISTIQUES MÉDIAS ====================
    
    async def get_media_stats(
        self,
        db: AsyncSession,
        animal_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Obtenir des statistiques sur les médias"""
        stats = {
            "photos": {"total": 0, "by_species": {}},
            "videos": {"total": 0, "by_action_type": {}, "recent": []},
            "cameras": {"total": 0, "active": 0}
        }
        
        # Comptage des photos
        if animal_id:
            stmt = select(Animal).where(Animal.id == animal_id)
            result = await db.execute(stmt)
            animal = result.scalar_one_or_none()
            if animal and animal.photo_url:
                stats["photos"]["total"] = 1
        
        # Comptage des vidéos
        video_stmt = select(VideoRecord)
        if animal_id:
            video_stmt = video_stmt.where(VideoRecord.animal_id == animal_id)
        
        result = await db.execute(video_stmt)
        videos = result.scalars().all()
        stats["videos"]["total"] = len(videos)
        
        # Vidéos récentes
        recent_videos = videos[:5] if len(videos) > 5 else videos
        stats["videos"]["recent"] = [
            {
                "id": v.id,
                "debut": v.debut.isoformat(),
                "action_type": v.action_type
            }
            for v in recent_videos
        ]
        
        # Caméras
        camera_stmt = select(Camera)
        result = await db.execute(camera_stmt)
        cameras = result.scalars().all()
        stats["cameras"]["total"] = len(cameras)
        stats["cameras"]["active"] = len([c for c in cameras if c.is_active])
        
        return stats
    
    async def cleanup_orphaned_photos(self, db: AsyncSession) -> int:
        """
        Nettoie les photos orphelines (non liées à un animal)
        Retourne le nombre de photos supprimées
        """
        deleted_count = 0
        
        # Parcourir tous les dossiers d'espèces
        for species_dir in self.upload_dir.iterdir():
            if not species_dir.is_dir() or species_dir.name == "temp":
                continue
            
            for photo_path in species_dir.glob("*.jpg"):
                # Extraire l'ID animal du nom de fichier
                # Format: bovins_123_20241215_12345678.jpg
                parts = photo_path.stem.split('_')
                if len(parts) >= 2:
                    try:
                        animal_id = int(parts[1])
                        # Vérifier si l'animal existe
                        stmt = select(Animal).where(Animal.id == animal_id)
                        result = await db.execute(stmt)
                        animal = result.scalar_one_or_none()
                        
                        if not animal:
                            # Photo orpheline, la supprimer
                            photo_path.unlink()
                            deleted_count += 1
                            logger.info(f"Photo orpheline supprimée: {photo_path}")
                    except ValueError:
                        continue
        
        return deleted_count


# Instance globale
media_service = MediaService()