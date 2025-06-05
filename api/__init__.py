from fastapi import HTTPException, status, APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Union, List
from datetime import datetime
from os import makedirs
import logging

from models import get_db_session
from models.user import User
from models.elevage.bovin import Bovin
from models.elevage.caprin import Caprin
from models.elevage.ovin import Ovin
from schemas.elevage import AnimalNumberResponse
from utils.security import get_current_user

router = APIRouter(
    prefix="/api",
    tags=[""],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ['jpg', 'png', 'jpeg']
UPLOAD_IMAGE_DIR_Bovin = "uploads/images/bovins"
UPLOAD_IMAGE_DIR_Caprin = "uploads/images/caprins"
UPLOAD_IMAGE_DIR_Ovin = "uploads/images/ovins"

# Création des dossiers s'ils n'existent pas
makedirs(UPLOAD_IMAGE_DIR_Bovin, exist_ok=True)
makedirs(UPLOAD_IMAGE_DIR_Caprin, exist_ok=True)
makedirs(UPLOAD_IMAGE_DIR_Ovin, exist_ok=True)

@router.post("/generer-numero", response_model=AnimalNumberResponse)
def generer_numero_animal_valid(
    espece: str = "vache",  # peut être "vache", "chevre", "mouton"
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
) -> str:
    """
    Génère un identifiant unique pour un animal selon le format :
    - VYYNNN pour vache
    - CYYNNN pour chèvre
    - MYYNNN pour mouton

    :param numero_ordre: Numéro d'ordre de l'animal (ex: 1 → 00001)
    :param espece: Espèce de l'animal ('vache', 'chevre', 'mouton')
    :return: Identifiant de type 'V24001', 'C24001', etc.
    """
    check_permissions(db, current_user)

    prefix_map = {
        "vache": "V",
        "chevre": "C",
        "mouton": "M"
    }

    if espece == 'vache':
        numero_ordre = db.query(Bovin).count()
    elif espece == 'chevre':
        numero_ordre = db.query(Caprin).count()
    elif espece == 'mouton':
        numero_ordre = db.query(Ovin).count()

    prefix = prefix_map.get(espece.lower())
    if not prefix:
        raise ValueError(f"Espèce inconnue: {espece}")

    annee = datetime.now().year
    suffix_annee = str(annee)[-2:]
    numero_str = f"{numero_ordre:05d}"  # 5 chiffres pour éviter collision

    return f"{prefix}{suffix_annee}{numero_str}"


def check_permissions(
    db: Session,
    current_user: dict,
    required_roles: Union[str, List[str]] = ['admin', 'avicole_manager'],
    user_email_field: str = 'email'
) -> None:
    """
    Vérifie si l'utilisateur courant a les permissions requises
    
    Args:
        db: Session SQLAlchemy
        current_user: Dictionnaire contenant les infos de l'utilisateur courant
        required_roles: Rôle(s) requis (peut être une string ou une liste)
        user_email_field: Champ email dans current_user
        
    Raises:
        HTTPException 403 si permission refusée
    """
    # Convertir required_roles en liste si c'est une string
    if isinstance(required_roles, str):
        required_roles = [required_roles]
    
    user = db.query(User).filter(User.email == current_user[user_email_field]).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Utilisateur non trouvé"
        )
    
    if user.role not in required_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission refusée"
        )
    
