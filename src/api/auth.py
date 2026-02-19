"""
Authentication Module for RAG API.
Implements JWT-based authentication with rate limiting.
"""

import os
import time
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass
from functools import wraps

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class User:
    """User model."""
    user_id: str
    username: str
    email: str
    api_key: str
    created_at: datetime
    is_active: bool = True
    role: str = "user"  # user, admin


class JWTAuth:
    """
    JWT-based authentication handler.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30
    ):
        """
        Initialize JWT authentication.
        
        Args:
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm
            access_token_expire_minutes: Token expiration time
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        
        # In-memory user store (replace with database in production)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create a default admin user."""
        admin_key = os.getenv("ADMIN_API_KEY", "rag-admin-key-12345")
        admin = User(
            user_id="admin",
            username="admin",
            email="admin@localhost",
            api_key=admin_key,
            created_at=datetime.utcnow(),
            role="admin"
        )
        self.users["admin"] = admin
        self.api_keys[admin_key] = "admin"
        logger.info("Default admin user created")
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token.
        
        Args:
            user_id: User identifier
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        try:
            import jwt
        except ImportError:
            logger.error("PyJWT not installed. Install with: pip install PyJWT")
            raise
        
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[str]:
        """
        Verify a JWT token and return user_id.
        
        Args:
            token: JWT token string
            
        Returns:
            User ID if valid, None otherwise
        """
        try:
            import jwt
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload.get("sub")
        except Exception as e:
            logger.debug(f"Token verification failed: {e}")
            return None
    
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """
        Verify an API key and return the user.
        
        Args:
            api_key: API key string
            
        Returns:
            User if valid, None otherwise
        """
        user_id = self.api_keys.get(api_key)
        if user_id:
            return self.users.get(user_id)
        return None
    
    def create_user(self, username: str, email: str, role: str = "user") -> User:
        """
        Create a new user with API key.
        
        Args:
            username: Username
            email: Email address
            role: User role
            
        Returns:
            Created user
        """
        user_id = secrets.token_hex(8)
        api_key = secrets.token_urlsafe(32)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            api_key=api_key,
            created_at=datetime.utcnow(),
            role=role
        )
        
        self.users[user_id] = user
        self.api_keys[api_key] = user_id
        
        logger.info(f"Created user: {username}")
        return user


class RateLimiter:
    """
    Simple in-memory rate limiter.
    Uses sliding window algorithm.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Track requests: user_id -> list of timestamps
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, user_id: str) -> bool:
        """
        Check if a request is allowed for the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests
        minute_ago = now - 60
        hour_ago = now - 3600
        
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if ts > hour_ago
        ]
        
        # Check limits
        recent_minute = sum(1 for ts in self.requests[user_id] if ts > minute_ago)
        recent_hour = len(self.requests[user_id])
        
        if recent_minute >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded (minute) for {user_id}")
            return False
        
        if recent_hour >= self.requests_per_hour:
            logger.warning(f"Rate limit exceeded (hour) for {user_id}")
            return False
        
        # Record request
        self.requests[user_id].append(now)
        return True
    
    def get_remaining(self, user_id: str) -> Dict[str, int]:
        """
        Get remaining requests for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with remaining requests
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        requests = self.requests.get(user_id, [])
        recent_minute = sum(1 for ts in requests if ts > minute_ago)
        recent_hour = sum(1 for ts in requests if ts > hour_ago)
        
        return {
            "minute_remaining": max(0, self.requests_per_minute - recent_minute),
            "hour_remaining": max(0, self.requests_per_hour - recent_hour)
        }


# Global instances
_auth: Optional[JWTAuth] = None
_rate_limiter: Optional[RateLimiter] = None


def get_auth() -> JWTAuth:
    """Get global auth instance."""
    global _auth
    if _auth is None:
        _auth = JWTAuth()
    return _auth


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
