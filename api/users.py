from datetime import timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_, and_, func
from sqlalchemy.orm import Session
from typing import Optional

from models import get_db
from models.user import User
from schemas.users import *
from utils.security import get_current_user
from utils.config import get_error_key

router = APIRouter()

@router.get("/user_list", response_model=UsersResponse)
async def user_list(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    q: Optional[str] = Query(None, alias="q"),  # Paramètre de recherche
    page: int = Query(1, alias="page"),  # Page par défaut 1
    limit: int = Query(10, alias="limit"),  # Limite par défaut
    sort: Optional[str] = Query(None, alias="sort"),  # Champ de tri
    order: Optional[str] = Query("asc", alias="order")  # Ordre de tri
):
    user = db.query(User).filter(User.email == current_user['email']).first()
    if user.role != 'admin':
        raise HTTPException(status_code=403, detail=get_error_key("users", "list", "no_permission"))
    
    query = db.query(User)

    if q:
        search_terms = q.lower().split()
        search_filters = []
        # Recherche dans plusieurs colonnes
        for term in search_terms:
            term_filter = or_(
                func.lower(User.username).contains(term),
                func.lower(User.email).contains(term),
                func.lower(User.role).contains(term),
            )
            search_filters.append(term_filter)
        
        # Combiner tous les termes avec AND (tous les termes doivent être présents)
        query = query.filter(and_(*search_filters))
    
        # Tri des résultats
    if sort is not None:
        if hasattr(User, sort):  # Vérifier que l'attribut existe
            if order == "desc":
                query = query.order_by(getattr(User, sort).desc())
            else:
                query = query.order_by(getattr(User, sort).asc())
        else:
            # Valeur par défaut si l'attribut n'existe pas
            query = query.order_by(User.created_at.desc())
    else:
        # Tri par défaut si aucun tri n'est spécifié
        query = query.order_by(User.created_at.desc())

    # Compter le nombre total d'items pour la pagination
    total_items = query.count()
    total_pages = (total_items + limit - 1) // limit  # Calcul du nombre total de pages

    # Pagination
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit)

    users = query.all()
    db.commit()
    # Retourner les produits avec les informations de pagination
    return {
        "users": users,
        "pagination": {
            "currentPage": page,
            "totalPages": total_pages,
            "totalItems": total_items,
            "itemsPerPage": limit
        }
    }

@router.get("/user_data")
async def user_data(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == current_user['email']).first()
    if not user:
        raise HTTPException(status_code=404, detail=get_error_key("users", "not_found"))
    user.last_login = datetime.now(timezone.utc)

    return {
        'username': user.username,
        'email': user.email,
        'phone': user.phone,
        'role': user.role,
    }

@router.post("/user_lang/{lang}")
async def user_lang(
    lang: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == current_user['email']).first()
    if not user:
        raise HTTPException(status_code=404, detail=get_error_key("users", "not_found"))

    user.lang = lang
    db.commit()
    return {}