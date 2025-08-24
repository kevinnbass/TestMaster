"""
API Rate Limiting - Advanced rate limiting with multiple algorithms and policies

This module provides comprehensive rate limiting capabilities including:
- Multiple rate limiting algorithms (Token bucket, Sliding window, Fixed window)
- Per-user, per-endpoint, and global rate limits
- Dynamic rate limit adjustment
- Burst handling and quotas
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
import math

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.api_models import RateLimitType, RateLimitStatus, APIUser

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Rate limit scope levels"""
    GLOBAL = "global"
    USER = "user"
    ENDPOINT = "endpoint"
    USER_ENDPOINT = "user_endpoint"
    IP_ADDRESS = "ip_address"


@dataclass
class RateLimitRule:
    """Rate limiting rule definition"""
    name: str
    scope: RateLimitScope
    algorithm: RateLimitAlgorithm
    limit: int
    window_size: int  # seconds
    burst_limit: Optional[int] = None
    priority: int = 0
    enabled: bool = True
    
    # Advanced options
    grace_period: int = 0  # seconds
    penalty_multiplier: float = 1.0
    quota_reset_time: Optional[str] = None  # "daily", "weekly", "monthly"
    
    def __post_init__(self):
        """Validate rule configuration"""
        if self.burst_limit is None:
            self.burst_limit = self.limit
        if self.burst_limit < self.limit:
            self.burst_limit = self.limit


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed
        """
        now = time.time()
        self._refill(now)
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self, now: float):
        """Refill tokens based on time elapsed"""
        if now > self.last_refill:
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now


@dataclass
class SlidingWindow:
    """Sliding window for rate limiting"""
    window_size: int  # seconds
    max_requests: int
    requests: deque = field(default_factory=deque)
    
    def add_request(self, timestamp: float) -> bool:
        """
        Add request to sliding window
        
        Args:
            timestamp: Request timestamp
            
        Returns:
            True if request is allowed
        """
        # Clean old requests
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        
        # Check if we can add new request
        if len(self.requests) < self.max_requests:
            self.requests.append(timestamp)
            return True
        
        return False
    
    def get_count(self, timestamp: float) -> int:
        """Get current request count in window"""
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        return len(self.requests)


class RateLimiter:
    """
    Advanced rate limiting engine with multiple algorithms and policies
    
    Supports complex rate limiting scenarios with burst handling,
    dynamic limits, and comprehensive monitoring.
    """
    
    def __init__(self):
        """Initialize rate limiter"""
        self.rules: Dict[str, RateLimitRule] = {}
        self.buckets: Dict[str, TokenBucket] = {}
        self.windows: Dict[str, SlidingWindow] = {}
        self.fixed_windows: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Global tracking
        self.global_stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'rules_triggered': defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Default rules
        self._setup_default_rules()
        
        logger.info("Rate Limiter initialized")
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules"""
        default_rules = [
            RateLimitRule(
                name="global_per_second",
                scope=RateLimitScope.GLOBAL,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                limit=1000,
                window_size=1,
                burst_limit=1500,
                priority=1
            ),
            RateLimitRule(
                name="user_per_minute",
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                limit=100,
                window_size=60,
                priority=2
            ),
            RateLimitRule(
                name="user_per_hour",
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                limit=1000,
                window_size=3600,
                priority=3
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: RateLimitRule):
        """
        Add rate limiting rule
        
        Args:
            rule: Rate limiting rule to add
        """
        with self._lock:
            self.rules[rule.name] = rule
            logger.info(f"Added rate limit rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove rate limiting rule
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed
        """
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                # Clean up associated data structures
                keys_to_remove = [k for k in self.buckets.keys() if k.startswith(rule_name)]
                for key in keys_to_remove:
                    del self.buckets[key]
                logger.info(f"Removed rate limit rule: {rule_name}")
                return True
            return False
    
    def check_rate_limit(
        self,
        user_id: Optional[str],
        endpoint: str,
        ip_address: Optional[str] = None,
        custom_limits: Optional[Dict[str, Tuple[int, RateLimitType]]] = None
    ) -> Tuple[bool, List[RateLimitStatus]]:
        """
        Check if request should be rate limited
        
        Args:
            user_id: User identifier
            endpoint: API endpoint
            ip_address: Client IP address
            custom_limits: Custom rate limits for this request
            
        Returns:
            Tuple of (allowed, list of rate limit statuses)
        """
        with self._lock:
            self.global_stats['total_requests'] += 1
            now = time.time()
            statuses = []
            allowed = True
            
            # Check each applicable rule
            applicable_rules = self._get_applicable_rules(user_id, endpoint, ip_address)
            
            for rule in applicable_rules:
                if not rule.enabled:
                    continue
                
                status = self._check_rule(rule, user_id, endpoint, ip_address, now)
                statuses.append(status)
                
                if not status.can_proceed():
                    allowed = False
                    self.global_stats['blocked_requests'] += 1
                    self.global_stats['rules_triggered'][rule.name] += 1
                    logger.debug(f"Rate limit triggered: {rule.name} for {user_id or ip_address}")
            
            # Check custom limits if provided
            if custom_limits:
                for limit_name, (limit, limit_type) in custom_limits.items():
                    status = self._check_custom_limit(
                        limit_name, limit, limit_type, user_id, endpoint, now
                    )
                    statuses.append(status)
                    if not status.can_proceed():
                        allowed = False
            
            return allowed, statuses
    
    def _get_applicable_rules(
        self,
        user_id: Optional[str],
        endpoint: str,
        ip_address: Optional[str]
    ) -> List[RateLimitRule]:
        """Get rules applicable to this request"""
        applicable = []
        
        for rule in self.rules.values():
            if rule.scope == RateLimitScope.GLOBAL:
                applicable.append(rule)
            elif rule.scope == RateLimitScope.USER and user_id:
                applicable.append(rule)
            elif rule.scope == RateLimitScope.ENDPOINT:
                applicable.append(rule)
            elif rule.scope == RateLimitScope.USER_ENDPOINT and user_id:
                applicable.append(rule)
            elif rule.scope == RateLimitScope.IP_ADDRESS and ip_address:
                applicable.append(rule)
        
        # Sort by priority
        applicable.sort(key=lambda r: r.priority)
        return applicable
    
    def _check_rule(
        self,
        rule: RateLimitRule,
        user_id: Optional[str],
        endpoint: str,
        ip_address: Optional[str],
        now: float
    ) -> RateLimitStatus:
        """Check specific rate limiting rule"""
        key = self._generate_key(rule, user_id, endpoint, ip_address)
        
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._check_token_bucket(rule, key, now)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return self._check_sliding_window(rule, key, now)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return self._check_fixed_window(rule, key, now)
        else:
            # Default to sliding window
            return self._check_sliding_window(rule, key, now)
    
    def _check_token_bucket(
        self,
        rule: RateLimitRule,
        key: str,
        now: float
    ) -> RateLimitStatus:
        """Check token bucket rate limit"""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(
                capacity=rule.burst_limit,
                tokens=rule.burst_limit,
                refill_rate=rule.limit / rule.window_size,
                last_refill=now
            )
        
        bucket = self.buckets[key]
        can_proceed = bucket.consume(1)
        
        remaining = int(bucket.tokens)
        reset_time = datetime.fromtimestamp(
            now + (rule.burst_limit - bucket.tokens) / bucket.refill_rate
        )
        
        return RateLimitStatus(
            user_id=key.split(':')[1] if ':' in key else "",
            endpoint=rule.name,
            limit=rule.limit,
            limit_type=RateLimitType.PER_SECOND,  # Convert window_size to appropriate type
            current_count=rule.burst_limit - remaining,
            remaining_requests=remaining,
            reset_time=reset_time,
            is_limited=not can_proceed
        )
    
    def _check_sliding_window(
        self,
        rule: RateLimitRule,
        key: str,
        now: float
    ) -> RateLimitStatus:
        """Check sliding window rate limit"""
        if key not in self.windows:
            self.windows[key] = SlidingWindow(
                window_size=rule.window_size,
                max_requests=rule.limit
            )
        
        window = self.windows[key]
        can_proceed = window.add_request(now)
        
        current_count = window.get_count(now)
        remaining = max(0, rule.limit - current_count)
        reset_time = datetime.fromtimestamp(now + rule.window_size)
        
        return RateLimitStatus(
            user_id=key.split(':')[1] if ':' in key else "",
            endpoint=rule.name,
            limit=rule.limit,
            limit_type=self._window_size_to_limit_type(rule.window_size),
            current_count=current_count,
            remaining_requests=remaining,
            reset_time=reset_time,
            is_limited=not can_proceed
        )
    
    def _check_fixed_window(
        self,
        rule: RateLimitRule,
        key: str,
        now: float
    ) -> RateLimitStatus:
        """Check fixed window rate limit"""
        window_start = now - (now % rule.window_size)
        window_key = f"{key}:{window_start}"
        
        if window_key not in self.fixed_windows:
            self.fixed_windows[window_key] = []
        
        requests = self.fixed_windows[window_key]
        
        # Clean old windows
        cutoff = now - rule.window_size * 2  # Keep last 2 windows
        expired_keys = [k for k in self.fixed_windows.keys() 
                       if float(k.split(':')[-1]) < cutoff]
        for k in expired_keys:
            del self.fixed_windows[k]
        
        can_proceed = len(requests) < rule.limit
        if can_proceed:
            requests.append(now)
        
        current_count = len(requests)
        remaining = max(0, rule.limit - current_count)
        reset_time = datetime.fromtimestamp(window_start + rule.window_size)
        
        return RateLimitStatus(
            user_id=key.split(':')[1] if ':' in key else "",
            endpoint=rule.name,
            limit=rule.limit,
            limit_type=self._window_size_to_limit_type(rule.window_size),
            current_count=current_count,
            remaining_requests=remaining,
            reset_time=reset_time,
            is_limited=not can_proceed
        )
    
    def _check_custom_limit(
        self,
        limit_name: str,
        limit: int,
        limit_type: RateLimitType,
        user_id: Optional[str],
        endpoint: str,
        now: float
    ) -> RateLimitStatus:
        """Check custom rate limit"""
        window_size = self._limit_type_to_window_size(limit_type)
        
        # Create temporary rule for custom limit
        temp_rule = RateLimitRule(
            name=f"custom_{limit_name}",
            scope=RateLimitScope.USER_ENDPOINT,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            limit=limit,
            window_size=window_size
        )
        
        key = f"custom:{user_id}:{endpoint}:{limit_name}"
        return self._check_sliding_window(temp_rule, key, now)
    
    def _generate_key(
        self,
        rule: RateLimitRule,
        user_id: Optional[str],
        endpoint: str,
        ip_address: Optional[str]
    ) -> str:
        """Generate unique key for rate limit tracking"""
        if rule.scope == RateLimitScope.GLOBAL:
            return f"{rule.name}:global"
        elif rule.scope == RateLimitScope.USER:
            return f"{rule.name}:user:{user_id or 'anonymous'}"
        elif rule.scope == RateLimitScope.ENDPOINT:
            return f"{rule.name}:endpoint:{endpoint}"
        elif rule.scope == RateLimitScope.USER_ENDPOINT:
            return f"{rule.name}:user_endpoint:{user_id or 'anonymous'}:{endpoint}"
        elif rule.scope == RateLimitScope.IP_ADDRESS:
            return f"{rule.name}:ip:{ip_address or 'unknown'}"
        else:
            return f"{rule.name}:unknown"
    
    def _window_size_to_limit_type(self, window_size: int) -> RateLimitType:
        """Convert window size to RateLimitType"""
        if window_size <= 1:
            return RateLimitType.PER_SECOND
        elif window_size <= 60:
            return RateLimitType.PER_MINUTE
        elif window_size <= 3600:
            return RateLimitType.PER_HOUR
        else:
            return RateLimitType.PER_DAY
    
    def _limit_type_to_window_size(self, limit_type: RateLimitType) -> int:
        """Convert RateLimitType to window size"""
        mapping = {
            RateLimitType.PER_SECOND: 1,
            RateLimitType.PER_MINUTE: 60,
            RateLimitType.PER_HOUR: 3600,
            RateLimitType.PER_DAY: 86400,
            RateLimitType.PER_WEEK: 604800
        }
        return mapping.get(limit_type, 60)
    
    def get_remaining_requests(
        self,
        user_id: Optional[str],
        endpoint: str,
        rule_name: str
    ) -> int:
        """
        Get remaining requests for specific rule
        
        Args:
            user_id: User identifier
            endpoint: API endpoint
            rule_name: Name of rate limiting rule
            
        Returns:
            Number of remaining requests
        """
        rule = self.rules.get(rule_name)
        if not rule:
            return 0
        
        key = self._generate_key(rule, user_id, endpoint, None)
        now = time.time()
        
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            if key in self.buckets:
                bucket = self.buckets[key]
                bucket._refill(now)
                return int(bucket.tokens)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            if key in self.windows:
                window = self.windows[key]
                current_count = window.get_count(now)
                return max(0, rule.limit - current_count)
        
        return rule.limit
    
    def reset_user_limits(self, user_id: str):
        """
        Reset all rate limits for a user
        
        Args:
            user_id: User identifier
        """
        with self._lock:
            # Remove user-specific buckets and windows
            keys_to_remove = [
                k for k in self.buckets.keys() 
                if f":user:{user_id}" in k or f":user_endpoint:{user_id}" in k
            ]
            for key in keys_to_remove:
                del self.buckets[key]
            
            keys_to_remove = [
                k for k in self.windows.keys() 
                if f":user:{user_id}" in k or f":user_endpoint:{user_id}" in k
            ]
            for key in keys_to_remove:
                del self.windows[key]
            
            logger.info(f"Reset rate limits for user: {user_id}")
    
    def adjust_rule_limit(self, rule_name: str, new_limit: int) -> bool:
        """
        Dynamically adjust rate limit
        
        Args:
            rule_name: Name of rule to adjust
            new_limit: New limit value
            
        Returns:
            True if successful
        """
        with self._lock:
            if rule_name in self.rules:
                old_limit = self.rules[rule_name].limit
                self.rules[rule_name].limit = new_limit
                logger.info(f"Adjusted rate limit for {rule_name}: {old_limit} -> {new_limit}")
                return True
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        with self._lock:
            stats = {
                "global_stats": dict(self.global_stats),
                "rules_count": len(self.rules),
                "active_buckets": len(self.buckets),
                "active_windows": len(self.windows),
                "rules_triggered": dict(self.global_stats['rules_triggered'])
            }
            
            if self.global_stats['total_requests'] > 0:
                stats["block_rate"] = (
                    self.global_stats['blocked_requests'] / 
                    self.global_stats['total_requests']
                )
            else:
                stats["block_rate"] = 0.0
            
            return stats
    
    def cleanup_expired_data(self):
        """Clean up expired rate limiting data"""
        with self._lock:
            now = time.time()
            cutoff = now - 3600  # 1 hour ago
            
            # Clean old fixed windows
            expired_windows = []
            for window_key in self.fixed_windows.keys():
                try:
                    window_time = float(window_key.split(':')[-1])
                    if window_time < cutoff:
                        expired_windows.append(window_key)
                except (ValueError, IndexError):
                    expired_windows.append(window_key)
            
            for key in expired_windows:
                del self.fixed_windows[key]
            
            if expired_windows:
                logger.info(f"Cleaned up {len(expired_windows)} expired rate limit windows")


# Factory function
def create_rate_limiter() -> RateLimiter:
    """
    Create and configure rate limiter
    
    Returns:
        Configured rate limiter instance
    """
    return RateLimiter()