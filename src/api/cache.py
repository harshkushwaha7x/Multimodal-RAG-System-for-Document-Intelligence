"""
Caching Module for RAG System.
Implements Redis-based caching for embeddings and responses.
"""

import hashlib
import json
import pickle
from typing import Any, Optional, List, Union
from datetime import timedelta

from ..utils import get_logger

logger = get_logger(__name__)


class CacheBackend:
    """Base cache backend interface."""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        raise NotImplementedError
    
    def clear(self) -> bool:
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """
    Simple in-memory cache for development.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self._cache = {}
        self._expiry = {}
        logger.info("In-memory cache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        import time
        
        if key in self._expiry:
            if time.time() > self._expiry[key]:
                self.delete(key)
                return None
        
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        import time
        
        # Evict if at max size
        if len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            self.delete(oldest)
        
        self._cache[key] = value
        if ttl:
            self._expiry[key] = time.time() + ttl
        
        return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        self._cache.pop(key, None)
        self._expiry.pop(key, None)
        return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cache
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        self._cache.clear()
        self._expiry.clear()
        return True


class RedisCache(CacheBackend):
    """
    Redis-based cache for production.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "rag:"
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix
        """
        self.prefix = prefix
        self._redis = None
        
        try:
            import redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False
            )
            self._redis.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self._redis = None
    
    def _key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self._redis:
            return None
        
        try:
            data = self._redis.get(self._key(key))
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self._redis:
            return False
        
        try:
            data = pickle.dumps(value)
            if ttl:
                self._redis.setex(self._key(key), ttl, data)
            else:
                self._redis.set(self._key(key), data)
            return True
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self._redis:
            return False
        
        try:
            self._redis.delete(self._key(key))
            return True
        except Exception as e:
            logger.debug(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self._redis:
            return False
        
        try:
            return self._redis.exists(self._key(key)) > 0
        except Exception:
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        if not self._redis:
            return False
        
        try:
            keys = self._redis.keys(f"{self.prefix}*")
            if keys:
                self._redis.delete(*keys)
            return True
        except Exception as e:
            logger.debug(f"Cache clear error: {e}")
            return False


class RAGCache:
    """
    High-level caching for RAG operations.
    Caches embeddings, search results, and LLM responses.
    """
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        """
        Initialize RAG cache.
        
        Args:
            backend: Cache backend to use
        """
        if backend:
            self.backend = backend
        else:
            # Try Redis first, fallback to in-memory
            redis_cache = RedisCache()
            if redis_cache._redis:
                self.backend = redis_cache
            else:
                self.backend = InMemoryCache()
        
        self.embedding_ttl = 86400  # 24 hours
        self.search_ttl = 3600  # 1 hour
        self.response_ttl = 1800  # 30 minutes
    
    def _hash_key(self, *args) -> str:
        """Generate cache key from arguments."""
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding."""
        key = f"emb:{self._hash_key(text, model)}"
        return self.backend.get(key)
    
    def set_embedding(self, text: str, model: str, embedding: List[float]) -> bool:
        """Cache an embedding."""
        key = f"emb:{self._hash_key(text, model)}"
        return self.backend.set(key, embedding, self.embedding_ttl)
    
    def get_search_results(self, query: str, top_k: int) -> Optional[Any]:
        """Get cached search results."""
        key = f"search:{self._hash_key(query, top_k)}"
        return self.backend.get(key)
    
    def set_search_results(self, query: str, top_k: int, results: Any) -> bool:
        """Cache search results."""
        key = f"search:{self._hash_key(query, top_k)}"
        return self.backend.set(key, results, self.search_ttl)
    
    def get_response(self, query: str, context_hash: str) -> Optional[str]:
        """Get cached LLM response."""
        key = f"resp:{self._hash_key(query, context_hash)}"
        return self.backend.get(key)
    
    def set_response(self, query: str, context_hash: str, response: str) -> bool:
        """Cache LLM response."""
        key = f"resp:{self._hash_key(query, context_hash)}"
        return self.backend.set(key, response, self.response_ttl)
    
    def invalidate_all(self) -> bool:
        """Invalidate all cached data."""
        return self.backend.clear()


# Global instance
_cache: Optional[RAGCache] = None


def get_cache() -> RAGCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = RAGCache()
    return _cache
