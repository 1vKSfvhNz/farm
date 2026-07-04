# backend/app/api/v1/media.py
"""
Routes de gestion media
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...schemas.media import CameraCreate, CameraUpdate, CameraResponse, VideoRecordResponse
from ...services import media_service
from ..dependencies.auth import get_current_user, get_current_admin_user
from ..dependencies.pagination import PaginationParams, get_pagination_params, PaginatedResponse
from ...models.user import User

router = APIRouter(prefix="/video", tags=["Vidéo"])


@router.get("/cameras", response_model=PaginatedResponse[CameraResponse])
async def get_cameras(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    is_active: Optional[bool] = Query(None, description="Filtrer par statut actif"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir la liste des caméras
    """
    cameras = await media_service.get_cameras(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        enclos_id=enclos_id,
        is_active=is_active
    )
    
    return PaginatedResponse.create(
        items=cameras,
        total=len(cameras),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/cameras/{camera_id}/stream")
async def get_camera_stream(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le flux vidéo d'une caméra
    """
    stream_url = await media_service.get_camera_stream_url(db, camera_id)
    if not stream_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Caméra non trouvée ou flux indisponible"
        )
    
    # Rediriger vers le flux ou retourner l'URL
    return {"stream_url": stream_url}


@router.get("/records", response_model=PaginatedResponse[VideoRecordResponse])
async def get_video_records(
    db: AsyncSession = Depends(get_db),
    pagination: PaginationParams = Depends(get_pagination_params),
    animal_id: Optional[int] = Query(None, description="Filtrer par animal"),
    enclos_id: Optional[int] = Query(None, description="Filtrer par enclos"),
    action_type: Optional[str] = Query(None, description="Filtrer par type d'action"),
    start_date: Optional[str] = Query(None, description="Date de début"),
    end_date: Optional[str] = Query(None, description="Date de fin"),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir l'historique des enregistrements vidéo
    """
    records = await media_service.get_video_records(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        animal_id=animal_id,
        enclos_id=enclos_id,
        action_type=action_type,
        start_date=start_date,
        end_date=end_date
    )
    
    return PaginatedResponse.create(
        items=records,
        total=len(records),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/records/{record_id}")
async def get_video_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Télécharger ou visionner un enregistrement vidéo
    """
    record = await media_service.get_video_record(db, record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enregistrement non trouvé"
        )
    
    return record


@router.get("/records/animal/{animal_id}", response_model=List[VideoRecordResponse])
async def get_videos_for_animal(
    animal_id: int,
    limit: int = Query(10, description="Nombre maximum d'enregistrements"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir les vidéos associées à un animal
    """
    videos = await media_service.get_videos_for_animal(db, animal_id, limit)
    return videos


@router.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Ajouter une nouvelle caméra (admin uniquement)
    """
    camera = await media_service.create_camera(db, camera_data, current_user.id)
    return camera


@router.put("/cameras/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int,
    camera_data: CameraUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Mettre à jour une caméra (admin uniquement)
    """
    camera, error = await media_service.update_camera(db, camera_id, camera_data, current_user.id)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    return camera


@router.delete("/cameras/{camera_id}")
async def delete_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Supprimer une caméra (admin uniquement)
    """
    success = await media_service.delete_camera(db, camera_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Caméra non trouvée"
        )
    return {"message": "Caméra supprimée avec succès"}