from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.logging import logger
from app.models.animal import Animal

async def generate_identification(
    db: AsyncSession,
    prefix: str
) -> str:
    """
    Génère un identifiant unique en récupérant le dernier ID utilisé.
    """
    prefix_upper = prefix.upper()
    logger.info(f"🔍 generate_identification called with prefix={prefix}, prefix_upper={prefix_upper}")
    
    try:
        # Récupérer le dernier ID utilisé pour ce préfixe
        stmt = select(Animal.identification).where(
            Animal.identification.startswith(f"{prefix_upper}-")
        ).order_by(Animal.id.desc()).limit(1)
        
        result = await db.execute(stmt)
        last_id = result.scalar_one_or_none()
        
        logger.info(f"📊 Last ID found: {last_id}")
        
        if last_id:
            # Extraire le numéro et incrémenter
            try:
                # Format: BOV-000001
                parts = last_id.split('-')
                logger.info(f"📊 Split result: {parts}")
                if len(parts) >= 2:
                    last_number = int(parts[1])
                    next_number = last_number + 1
                else:
                    next_number = 1
            except (IndexError, ValueError) as e:
                logger.error(f"❌ Error parsing ID: {e}")
                next_number = 1
        else:
            next_number = 1
        
        new_id = f"{prefix_upper}-{str(next_number).zfill(6)}"
        logger.info(f"✨ Generated ID: {new_id}")
        
        # Sécurité : vérifier que l'ID n'existe pas
        stmt = select(Animal).where(Animal.identification == new_id)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            logger.warning(f"⚠️ ID collision for {new_id}, retrying...")
            return await generate_identification(db, prefix)
        
        logger.info(f"✅ Final ID: {new_id}")
        return new_id
        
    except Exception as e:
        logger.error(f"❌ Exception in generate_identification: {e}", exc_info=True)
        # Fallback: utiliser timestamp
        import time
        fallback_id = f"{prefix_upper}-{int(time.time())}"
        logger.info(f"🔄 Using fallback ID: {fallback_id}")
        return fallback_id


async def log_action(
    db: AsyncSession,
    user_id: int,
    action: str,
    entity_type: str = None,
    entity_id: int = None,
    details: dict = None
) -> None:
    """Journaliser une action"""
    from ..models.user import ActionLog
    log = ActionLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details
    )
    db.add(log)