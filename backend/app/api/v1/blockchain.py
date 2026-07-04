# backend/app/api/v1/blockchain.py
"""
Routes pour la traçabilité blockchain (option premium)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from ...database import get_db
from ...services.blockchain_service import blockchain_service
from ...api.dependencies.auth import get_current_user, get_current_admin_user
from ...models.user import User

router = APIRouter(prefix="/blockchain", tags=["Blockchain (Premium)"])


@router.get("/status")
async def get_blockchain_status(
    current_user: User = Depends(get_current_user),
):
    """
    Obtenir le statut du service blockchain
    """
    return {
        "enabled": blockchain_service.enabled,
        "network": blockchain_service.network if blockchain_service.enabled else None,
        "message": "Fonctionnalité premium - Contactez l'administrateur pour activer" if not blockchain_service.enabled else "Blockchain activée"
    }


@router.post("/record/vaccination")
async def record_vaccination_on_blockchain(
    animal_id: int,
    vaccination_id: int,
    maladie: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Enregistrer une vaccination sur la blockchain (admin uniquement - premium)
    """
    if not blockchain_service.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fonctionnalité premium non activée"
        )
    
    from datetime import datetime
    tx_hash = await blockchain_service.record_vaccination(
        animal_id=animal_id,
        vaccination_id=vaccination_id,
        maladie=maladie,
        date_vaccination=datetime.now()
    )
    
    if not tx_hash:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de l'enregistrement sur la blockchain"
        )
    
    return {
        "success": True,
        "transaction_hash": tx_hash,
        "message": "Vaccination enregistrée sur la blockchain"
    }


@router.post("/record/sale")
async def record_sale_on_blockchain(
    animal_id: int,
    buyer: str,
    price: float,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Enregistrer une vente sur la blockchain (admin uniquement - premium)
    """
    if not blockchain_service.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fonctionnalité premium non activée"
        )
    
    from datetime import datetime
    tx_hash = await blockchain_service.record_sale(
        animal_id=animal_id,
        buyer=buyer,
        price=price,
        date_sale=datetime.now()
    )
    
    if not tx_hash:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de l'enregistrement sur la blockchain"
        )
    
    return {
        "success": True,
        "transaction_hash": tx_hash,
        "message": "Vente enregistrée sur la blockchain"
    }


@router.get("/verify/animal/{animal_id}")
async def verify_animal_history(
    animal_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Vérifier l'historique d'un animal sur la blockchain (premium)
    """
    if not blockchain_service.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fonctionnalité premium non activée"
        )
    
    result = await blockchain_service.verify_animal_history(animal_id)
    
    return {
        "animal_id": animal_id,
        "verified": result.get("verified", False),
        "records": result.get("records", []),
        "message": result.get("message", "")
    }


@router.get("/certificate/animal/{animal_id}")
async def generate_blockchain_certificate(
    animal_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Générer un certificat de traçabilité blockchain pour un animal (premium)
    """
    if not blockchain_service.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fonctionnalité premium non activée"
        )
    
    certificate = await blockchain_service.generate_certificate(animal_id)
    
    if not certificate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Animal non trouvé ou aucune donnée blockchain"
        )
    
    return certificate


@router.get("/transactions/latest")
async def get_latest_transactions(
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Obtenir les dernières transactions blockchain (admin uniquement - premium)
    """
    if not blockchain_service.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fonctionnalité premium non activée"
        )
    
    transactions = await blockchain_service.get_latest_transactions(limit)
    
    return {
        "transactions": transactions,
        "count": len(transactions)
    }


@router.get("/transaction/{tx_hash}")
async def get_transaction_details(
    tx_hash: str,
    current_user: User = Depends(get_current_admin_user),
):
    """
    Obtenir les détails d'une transaction blockchain (admin uniquement - premium)
    """
    if not blockchain_service.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fonctionnalité premium non activée"
        )
    
    details = await blockchain_service.get_transaction_details(tx_hash)
    
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction non trouvée"
        )
    
    return details