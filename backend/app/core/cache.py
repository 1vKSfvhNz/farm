# backend/app/core/cache.py
from typing import Any, Optional, Callable, Awaitable
import json
from functools import wraps

from ..redis_client import redis_client
from ..config import settings


class CacheManager:
    """Gestionnaire de cache Redis"""
    
    DEFAULT_TTL = 300  # 5 minutes
    
    @staticmethod
    async def get(key: str) -> Optional[Any]:
        """Récupérer une valeur du cache"""
        try:
            data = await redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    @staticmethod
    async def set(
        key: str,
        value: Any,
        ttl: int = DEFAULT_TTL
    ) -> bool:
        """Stocker une valeur dans le cache"""
        try:
            await redis_client.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    @staticmethod
    async def delete(key: str) -> bool:
        """Supprimer une valeur du cache"""
        try:
            await redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    @staticmethod
    async def delete_pattern(pattern: str) -> int:
        """Supprimer toutes les clés correspondant à un pattern"""
        try:
            keys = await redis_client.keys(pattern)
            if keys:
                return await redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache delete pattern error: {e}")
            return 0
    
    @staticmethod
    async def exists(key: str) -> bool:
        """Vérifier si une clé existe"""
        try:
            return await redis_client.exists(key) > 0
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False
    
    @staticmethod
    async def increment(key: str, amount: int = 1) -> Optional[int]:
        """Incrémenter une valeur"""
        try:
            return await redis_client.incrby(key, amount)
        except Exception as e:
            print(f"Cache increment error: {e}")
            return None
    
    @staticmethod
    async def get_or_set(
        key: str,
        fetcher: Callable[[], Awaitable[Any]],
        ttl: int = DEFAULT_TTL
    ) -> Optional[Any]:
        """Récupérer du cache ou exécuter la fonction fetcher"""
        cached = await CacheManager.get(key)
        if cached is not None:
            return cached
        
        data = await fetcher()
        if data is not None:
            await CacheManager.set(key, data, ttl)
        
        return data


def cached(ttl: int = CacheManager.DEFAULT_TTL, key_prefix: str = None):
    """Décorateur pour mettre en cache les résultats de fonction"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Générer la clé de cache
            if key_prefix:
                cache_key = f"{key_prefix}:{args}:{kwargs}"
            else:
                cache_key = f"{func.__module__}:{func.__name__}:{args}:{kwargs}"
            
            # Nettoyer la clé (enlever les espaces, etc.)
            cache_key = cache_key.replace(" ", "_").replace("'", "")
            
            return await CacheManager.get_or_set(cache_key, lambda: func(*args, **kwargs), ttl)
        return wrapper
    return decorator


get_cache = CacheManager()