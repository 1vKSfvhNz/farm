# backend/app/api/dependencies/pagination.py
"""
Dépendances pour la pagination des résultats
"""

from typing import TypeVar, Generic, List, Optional, Any
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession


T = TypeVar("T")


class PaginationParams(BaseModel):
    """Paramètres de pagination pour les requêtes"""
    skip: int = Field(default=0, ge=0, description="Nombre d'éléments à sauter")
    limit: int = Field(default=100, ge=1, le=1000, description="Nombre d'éléments par page")
    page: Optional[int] = Field(default=None, ge=1, description="Numéro de page (alternative à skip)")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Si page est fourni, calculer skip à partir de page
        if self.page and self.page > 0:
            self.skip = (self.page - 1) * self.limit
    
    @property
    def offset(self) -> int:
        return self.skip
    
    @property
    def page_number(self) -> int:
        """Retourner le numéro de page actuel"""
        if self.limit > 0:
            return (self.skip // self.limit) + 1
        return 1
    
    @property
    def total_pages(self, total_count: int) -> int:
        """Calculer le nombre total de pages"""
        if self.limit > 0:
            return (total_count + self.limit - 1) // self.limit
        return 1


class PaginatedResponse(BaseModel, Generic[T]):
    """Réponse paginée générique"""
    items: List[T]
    total: int
    skip: int
    limit: int
    page: int
    total_pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        skip: int,
        limit: int
    ) -> "PaginatedResponse[T]":
        """Créer une réponse paginée"""
        page = (skip // limit) + 1 if limit > 0 else 1
        total_pages = (total + limit - 1) // limit if limit > 0 else 1
        
        return cls(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            page=page,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )


async def paginate(
    db: AsyncSession,
    stmt,
    pagination: PaginationParams,
    count_stmt = None
) -> PaginatedResponse:
    """
    Exécuter une requête paginée
    
    Args:
        db: Session database
        stmt: Requête SQLAlchemy pour récupérer les données
        pagination: Paramètres de pagination
        count_stmt: Requête SQLAlchemy pour compter le total (optionnel)
    
    Returns:
        Réponse paginée
    """
    # Compter le total
    if count_stmt is None:
        # Créer une requête de comptage à partir de stmt
        from sqlalchemy import select, func
        count_stmt = select(func.count()).select_from(stmt.subquery())
    
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0
    
    # Appliquer la pagination
    paginated_stmt = stmt.offset(pagination.skip).limit(pagination.limit)
    result = await db.execute(paginated_stmt)
    items = result.scalars().all()
    
    return PaginatedResponse.create(
        items=items,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


def get_pagination_params(
    skip: int = 0,
    limit: int = 100,
    page: Optional[int] = None
) -> PaginationParams:
    """
    Dépendance FastAPI pour obtenir les paramètres de pagination
    
    Usage:
        @router.get("/items")
        async def get_items(pagination: PaginationParams = Depends(get_pagination_params)):
            ...
    """
    return PaginationParams(skip=skip, limit=limit, page=page)