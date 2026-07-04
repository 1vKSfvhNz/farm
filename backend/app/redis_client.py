# backend/app/redis_client.py
"""
Client Redis pour cache, sessions, rate limiting, et queues
"""

import json
import logging
from typing import Optional, Any, List, Dict, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from .config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Client Redis asynchrone pour l'application
    Gère: cache, sessions utilisateur, rate limiting, queues Celery
    """
    
    def __init__(self):
        self._client: Optional[Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
    
    async def connect(self) -> None:
        """Établir la connexion à Redis"""
        try:
            self._client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            
            # Tester la connexion
            await self._client.ping()
            
            # Créer le client PubSub pour les alertes temps réel
            self._pubsub = self._client.pubsub()
            
            logger.info(f"Redis connected: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Fermer la connexion Redis"""
        if self._pubsub:
            await self._pubsub.close()
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")
    
    @property
    def client(self) -> Redis:
        """Obtenir le client Redis"""
        if self._client is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client
    
    # ============ OPÉRATIONS DE BASE ============
    
    async def get(self, key: str) -> Optional[str]:
        """Récupérer une valeur"""
        return await self.client.get(key)
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Récupérer une valeur JSON"""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False
    ) -> bool:
        """
        Définir une valeur
        
        Args:
            key: Clé Redis
            value: Valeur (string ou JSON-serializable)
            ttl: Durée de vie en secondes
            nx: Si True, set seulement si la clé n'existe pas
        """
        if not isinstance(value, str):
            value = json.dumps(value, default=str)
        
        if nx:
            result = await self.client.setnx(key, value)
            if result and ttl:
                await self.client.expire(key, ttl)
            return bool(result)
        else:
            if ttl:
                await self.client.setex(key, ttl, value)
            else:
                await self.client.set(key, value)
            return True
    
    async def delete(self, key: str) -> int:
        """Supprimer une clé"""
        return await self.client.delete(key)
    
    async def delete_pattern(self, pattern: str) -> int:
        """Supprimer toutes les clés correspondant à un pattern"""
        keys = await self.client.keys(pattern)
        if keys:
            return await self.client.delete(*keys)
        return 0
    
    async def exists(self, key: str) -> bool:
        """Vérifier si une clé existe"""
        return await self.client.exists(key) > 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Définir la durée de vie d'une clé"""
        return await self.client.expire(key, ttl)
    
    async def ttl(self, key: str) -> int:
        """Obtenir le TTL restant"""
        return await self.client.ttl(key)
    
    # ============ OPÉRATIONS AVEC COMPTEURS ============
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """Incrémenter un compteur"""
        return await self.client.incrby(key, amount)
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """Décrémenter un compteur"""
        return await self.client.decrby(key, amount)
    
    async def get_counter(self, key: str) -> int:
        """Obtenir la valeur d'un compteur"""
        val = await self.get(key)
        return int(val) if val else 0
    
    # ============ OPÉRATIONS AVEC LISTES ============
    
    async def lpush(self, key: str, *values) -> int:
        """Ajouter à gauche de la liste"""
        return await self.client.lpush(key, *values)
    
    async def rpush(self, key: str, *values) -> int:
        """Ajouter à droite de la liste"""
        return await self.client.rpush(key, *values)
    
    async def lpop(self, key: str) -> Optional[str]:
        """Retirer de la gauche"""
        return await self.client.lpop(key)
    
    async def rpop(self, key: str) -> Optional[str]:
        """Retirer de la droite"""
        return await self.client.rpop(key)
    
    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Obtenir une plage de la liste"""
        return await self.client.lrange(key, start, end)
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Tronquer la liste"""
        return await self.client.ltrim(key, start, end)
    
    async def llen(self, key: str) -> int:
        """Longueur de la liste"""
        return await self.client.llen(key)
    
    # ============ OPÉRATIONS AVEC SETS ============
    
    async def sadd(self, key: str, *members) -> int:
        """Ajouter des membres à un set"""
        return await self.client.sadd(key, *members)
    
    async def srem(self, key: str, *members) -> int:
        """Retirer des membres d'un set"""
        return await self.client.srem(key, *members)
    
    async def sismember(self, key: str, member: str) -> bool:
        """Vérifier l'appartenance à un set"""
        return await self.client.sismember(key, member)
    
    async def smembers(self, key: str) -> List[str]:
        """Obtenir tous les membres du set"""
        return await self.client.smembers(key)
    
    async def scard(self, key: str) -> int:
        """Nombre de membres du set"""
        return await self.client.scard(key)
    
    # ============ OPÉRATIONS AVEC SORTED SETS ============
    
    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Ajouter des membres à un sorted set avec score"""
        return await self.client.zadd(key, mapping)
    
    async def zrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
        withscores: bool = False
    ) -> List[Union[str, tuple]]:
        """Obtenir les membres par score"""
        return await self.client.zrangebyscore(key, min_score, max_score, withscores=withscores)
    
    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Supprimer les membres par score"""
        return await self.client.zremrangebyscore(key, min_score, max_score)
    
    async def zcard(self, key: str) -> int:
        """Nombre de membres du sorted set"""
        return await self.client.zcard(key)
    
    # ============ PUB/SUB ============
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publier un message sur un canal"""
        if not isinstance(message, str):
            message = json.dumps(message, default=str)
        return await self.client.publish(channel, message)
    
    async def subscribe(self, channels: List[str]):
        """S'abonner à des canaux"""
        if self._pubsub:
            await self._pubsub.subscribe(*channels)
    
    async def unsubscribe(self, channels: List[str]):
        """Se désabonner des canaux"""
        if self._pubsub:
            await self._pubsub.unsubscribe(*channels)
    
    async def get_pubsub_message(self, timeout: float = 1.0) -> Optional[Dict]:
        """Obtenir le prochain message PubSub"""
        if self._pubsub:
            return await self._pubsub.get_message(timeout=timeout, ignore_subscribe_messages=True)
        return None
    
    # ============ GESTION DES SESSIONS ============
    
    async def create_session(
        self,
        session_id: str,
        user_id: int,
        user_roles: List[str] = None,
        ttl: int = None
    ) -> bool:
        """Créer une session utilisateur"""
        session_data = {
            "user_id": user_id,
            "user_roles": user_roles,
            "created_at": datetime.now().isoformat(),
        }
        ttl = ttl or settings.SESSION_EXPIRE_HOURS * 3600
        return await self.set(f"session:{session_id}", session_data, ttl)
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Obtenir une session utilisateur"""
        return await self.get_json(f"session:{session_id}")
    
    async def delete_session(self, session_id: str) -> bool:
        """Supprimer une session utilisateur"""
        await self.delete(f"session:{session_id}")
        return True
    
    async def delete_all_user_sessions(self, user_id: int) -> int:
        """Supprimer toutes les sessions d'un utilisateur"""
        return await self.delete_pattern(f"session:*")
    
    # ============ RATE LIMITING ============
    
    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Vérifier le rate limiting
        Returns: (est_dans_limites, requêtes_restantes)
        """
        current_time = datetime.now().timestamp()
        window_start = current_time - window_seconds
        
        # Nettoyer les anciennes requêtes
        await self.zremrangebyscore(key, 0, window_start)
        
        # Compter les requêtes
        count = await self.zcard(key)
        
        if count >= max_requests:
            return False, 0
        
        # Ajouter la requête
        await self.zadd(key, {str(current_time): current_time})
        await self.expire(key, window_seconds)
        
        remaining = max_requests - count - 1
        return True, remaining
    
    # ============ CACHE ============
    
    async def cache_get_or_set(
        self,
        key: str,
        fetch_func,
        ttl: int = 300
    ) -> Any:
        """Récupérer du cache ou exécuter la fonction"""
        cached = await self.get_json(key)
        if cached is not None:
            return cached
        
        data = await fetch_func()
        if data:
            await self.set(key, data, ttl)
        return data
    
    # ============ QUEUES (pour Celery) ============
    
    async def enqueue(self, queue_name: str, task: Dict) -> int:
        """Ajouter une tâche à une queue"""
        return await self.lpush(f"queue:{queue_name}", json.dumps(task, default=str))
    
    async def dequeue(self, queue_name: str) -> Optional[Dict]:
        """Récupérer une tâche de la queue"""
        task = await self.rpop(f"queue:{queue_name}")
        if task:
            return json.loads(task)
        return None
    
    async def queue_length(self, queue_name: str) -> int:
        """Longueur de la queue"""
        return await self.llen(f"queue:{queue_name}")
    
    # ============ HEALTH CHECK ============
    
    async def health_check(self) -> bool:
        """Vérifier la santé de Redis"""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


# Instance globale
redis_client = RedisClient()


async def init_redis() -> None:
    """Initialiser Redis (appel au démarrage)"""
    await redis_client.connect()


async def close_redis() -> None:
    """Fermer Redis (appel à l'arrêt)"""
    await redis_client.disconnect()