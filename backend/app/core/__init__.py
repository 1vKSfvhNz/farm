# backend/app/core/__init__.py
from .security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    generate_session_id,
)
from .logging import logger, setup_logging
from .cache import CacheManager, get_cache
from .validators import (
    validate_date_range,
    validate_weight_range,
    validate_pesee_frequency,
    validate_animal_age,
)
from .units import (
    convert_weight,
    convert_volume,
    convert_temperature,
    convert_length,
    UnitConverter,
)
from .constants import (
    RoleEnum,
    EnclosTypeEnum,
    SexeEnum,
    StatutAnimalEnum,
    EspeceEnum,
    UNITS,
    SEUILS_MORTALITE,
    SEUILS_CONVERSION_ALIMENTAIRE,
)

__all__ = [
    "hash_password",
    "verify_password", 
    "create_access_token",
    "decode_access_token",
    "generate_session_id",
    "logger",
    "setup_logging",
    "CacheManager",
    "get_cache",
    "validate_date_range",
    "validate_weight_range",
    "validate_pesee_frequency",
    "validate_animal_age",
    "convert_weight",
    "convert_volume",
    "convert_temperature",
    "convert_length",
    "UnitConverter",
    "RoleEnum",
    "EnclosTypeEnum",
    "SexeEnum",
    "StatutAnimalEnum",
    "EspeceEnum",
    "UNITS",
    "SEUILS_MORTALITE",
    "SEUILS_CONVERSION_ALIMENTAIRE",
]