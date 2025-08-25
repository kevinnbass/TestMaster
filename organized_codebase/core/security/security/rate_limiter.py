"""
Agency-Swarm Derived Rate Limiting Security Module
Enhanced rate limiting system for API security based on FastAPI patterns
"""

import time
import redis
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Lock
from .error_handler import RateLimitError, security_error_handler


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    requests_per_window: int
    window_seconds: int
    burst_allowance: int = 0
    cooldown_seconds: int = 60
    blocked_duration_seconds: int = 300


@dataclass
class RateLimitStatus:
    """Current rate limit status for a client"""
    client_id: str
    requests_made: int
    window_start: datetime
    is_blocked: bool = False
    block_expires: Optional[datetime] = None
    last_request: datetime = field(default_factory=datetime.utcnow)


class InMemoryRateLimiter:
    """In-memory rate limiter for development/testing"""
    
    def __init__(self):
        self.client_stats: Dict[str, RateLimitStatus] = {}
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
    
    def check_rate_limit(self, client_id: str, rule: RateLimitRule) -> Tuple[bool, RateLimitStatus]:
        """Check if request is within rate limits"""
        with self.lock:
            now = datetime.utcnow()
            
            if client_id not in self.client_stats:
                self.client_stats[client_id] = RateLimitStatus(
                    client_id=client_id,
                    requests_made=0,
                    window_start=now
                )
            
            status = self.client_stats[client_id]
            
            # Check if client is currently blocked
            if status.is_blocked and status.block_expires and now < status.block_expires:
                return False, status
            
            # Reset block if expired
            if status.is_blocked and status.block_expires and now >= status.block_expires:
                status.is_blocked = False
                status.block_expires = None
                status.requests_made = 0
                status.window_start = now
            
            # Check if we need to reset the window
            window_age = (now - status.window_start).total_seconds()
            if window_age >= rule.window_seconds:
                status.requests_made = 0
                status.window_start = now
            
            # Check rate limit
            if status.requests_made >= rule.requests_per_window:
                # Block the client
                status.is_blocked = True
                status.block_expires = now + timedelta(seconds=rule.blocked_duration_seconds)
                
                self.logger.warning(
                    f"Rate limit exceeded for client {client_id}. "
                    f"Blocked until {status.block_expires}"
                )
                return False, status
            
            # Allow request and increment counter
            status.requests_made += 1
            status.last_request = now
            return True, status
    
    def get_stats(self) -> Dict[str, Dict[str, any]]:
        """Get rate limiting statistics"""
        with self.lock:
            return {
                client_id: {
                    'requests_made': status.requests_made,
                    'window_start': status.window_start.isoformat(),
                    'is_blocked': status.is_blocked,
                    'block_expires': status.block_expires.isoformat() if status.block_expires else None,
                    'last_request': status.last_request.isoformat()
                }
                for client_id, status in self.client_stats.items()
            }


class RedisRateLimiter:
    """Redis-backed distributed rate limiter"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
    
    def check_rate_limit(self, client_id: str, rule: RateLimitRule) -> Tuple[bool, RateLimitStatus]:
        """Check rate limit using Redis sliding window"""
        try:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=rule.window_seconds)
            
            # Keys for Redis
            requests_key = f"rate_limit:{rule.name}:{client_id}:requests"
            block_key = f"rate_limit:{rule.name}:{client_id}:blocked"
            
            # Check if client is blocked
            block_until = self.redis.get(block_key)
            if block_until:
                block_expires = datetime.fromisoformat(block_until.decode())
                if now < block_expires:
                    return False, RateLimitStatus(
                        client_id=client_id,
                        requests_made=0,
                        window_start=window_start,
                        is_blocked=True,
                        block_expires=block_expires
                    )
                else:
                    # Block expired, clean up
                    self.redis.delete(block_key)
            
            # Use Redis sorted set for sliding window
            pipe = self.redis.pipeline()
            
            # Remove old requests outside the window
            pipe.zremrangebyscore(requests_key, 0, window_start.timestamp())
            
            # Count current requests
            pipe.zcard(requests_key)
            
            # Add current request timestamp
            pipe.zadd(requests_key, {str(now.timestamp()): now.timestamp()})
            
            # Set expiration for the key
            pipe.expire(requests_key, rule.window_seconds + 60)
            
            results = pipe.execute()
            current_requests = results[1]
            
            if current_requests >= rule.requests_per_window:
                # Block the client
                block_expires = now + timedelta(seconds=rule.blocked_duration_seconds)
                self.redis.set(block_key, block_expires.isoformat(), ex=rule.blocked_duration_seconds)
                
                # Remove the request we just added since it's blocked
                self.redis.zrem(requests_key, str(now.timestamp()))
                
                self.logger.warning(
                    f"Rate limit exceeded for client {client_id}. "
                    f"Blocked until {block_expires}"
                )
                
                return False, RateLimitStatus(
                    client_id=client_id,
                    requests_made=current_requests,
                    window_start=window_start,
                    is_blocked=True,
                    block_expires=block_expires
                )
            
            return True, RateLimitStatus(
                client_id=client_id,
                requests_made=current_requests + 1,
                window_start=window_start,
                is_blocked=False
            )
            
        except Exception as e:
            self.logger.error(f"Redis rate limiter error: {e}")
            # Fall back to allowing request if Redis fails
            return True, RateLimitStatus(
                client_id=client_id,
                requests_made=0,
                window_start=now - timedelta(seconds=rule.window_seconds),
                is_blocked=False
            )


class RateLimitManager:
    """Central rate limiting management system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.logger = logging.getLogger(__name__)
        
        # Use Redis if available, otherwise fall back to in-memory
        if redis_client:
            self.limiter = RedisRateLimiter(redis_client)
            self.logger.info("Using Redis-backed rate limiter")
        else:
            self.limiter = InMemoryRateLimiter()
            self.logger.info("Using in-memory rate limiter")
        
        # Default rate limiting rules
        self.rules: Dict[str, RateLimitRule] = {
            'api_general': RateLimitRule(
                name='api_general',
                requests_per_window=100,
                window_seconds=60,
                burst_allowance=10,
                blocked_duration_seconds=300
            ),
            'api_auth': RateLimitRule(
                name='api_auth',
                requests_per_window=5,
                window_seconds=60,
                burst_allowance=2,
                blocked_duration_seconds=900  # 15 minutes for auth failures
            ),
            'api_upload': RateLimitRule(
                name='api_upload',
                requests_per_window=10,
                window_seconds=60,
                burst_allowance=2,
                blocked_duration_seconds=600
            ),
            'websocket': RateLimitRule(
                name='websocket',
                requests_per_window=1000,
                window_seconds=60,
                burst_allowance=50,
                blocked_duration_seconds=120
            )
        }
    
    def check_limit(self, client_id: str, rule_name: str = 'api_general', 
                   context: Dict[str, any] = None) -> bool:
        """Check if client is within rate limits"""
        try:
            if rule_name not in self.rules:
                self.logger.warning(f"Unknown rate limit rule: {rule_name}")
                return True  # Allow if rule not found
            
            rule = self.rules[rule_name]
            allowed, status = self.limiter.check_rate_limit(client_id, rule)
            
            if not allowed:
                error = RateLimitError(
                    f"Rate limit exceeded for rule '{rule_name}'. "
                    f"Limit: {rule.requests_per_window} requests per {rule.window_seconds} seconds.",
                    details={
                        'client_id': client_id,
                        'rule_name': rule_name,
                        'requests_made': status.requests_made,
                        'block_expires': status.block_expires.isoformat() if status.block_expires else None,
                        'context': context
                    }
                )
                security_error_handler.handle_error(error, context)
                raise error
            
            return True
            
        except RateLimitError:
            raise
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error to avoid breaking the system
    
    def add_rule(self, rule: RateLimitRule):
        """Add or update a rate limiting rule"""
        self.rules[rule.name] = rule
        self.logger.info(f"Added rate limit rule: {rule.name}")
    
    def get_client_status(self, client_id: str, rule_name: str = 'api_general') -> Optional[RateLimitStatus]:
        """Get current status for a client"""
        if rule_name not in self.rules:
            return None
        
        rule = self.rules[rule_name]
        _, status = self.limiter.check_rate_limit(client_id, rule)
        return status
    
    def reset_client_limits(self, client_id: str):
        """Reset rate limits for a specific client"""
        # This implementation depends on the limiter type
        if isinstance(self.limiter, InMemoryRateLimiter):
            with self.limiter.lock:
                if client_id in self.limiter.client_stats:
                    del self.limiter.client_stats[client_id]
                    self.logger.info(f"Reset rate limits for client: {client_id}")


def get_client_identifier(request_context: Dict[str, any]) -> str:
    """Extract client identifier from request context"""
    # Try different identification methods in order of preference
    
    # 1. Use authenticated user ID if available
    if 'user_id' in request_context:
        return f"user:{request_context['user_id']}"
    
    # 2. Use API key if available
    if 'api_key' in request_context:
        import hashlib
        key_hash = hashlib.sha256(request_context['api_key'].encode()).hexdigest()[:16]
        return f"apikey:{key_hash}"
    
    # 3. Use IP address
    if 'ip_address' in request_context:
        return f"ip:{request_context['ip_address']}"
    
    # 4. Fallback to generic identifier
    return "anonymous"


# Global rate limiter instance
rate_limit_manager = RateLimitManager()


def check_rate_limit(client_id: str, rule_name: str = 'api_general', 
                    context: Dict[str, any] = None) -> bool:
    """Convenience function for rate limit checking"""
    return rate_limit_manager.check_limit(client_id, rule_name, context)