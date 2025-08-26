"""
API Rate Limiting Manager

This module implements rate limiting functionality for the API gateway
to prevent abuse and ensure fair usage across users.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from .api_types import APIUser, RateLimit, RateLimitType


class RateLimiter:
    """Manages rate limiting for API requests"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting storage
        self.user_rate_limits: Dict[str, Dict[str, RateLimit]] = defaultdict(dict)
        self.ip_rate_limits: Dict[str, Dict[str, RateLimit]] = defaultdict(dict)
        self.endpoint_rate_limits: Dict[str, Dict[str, RateLimit]] = defaultdict(dict)
        
        # Request tracking
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Default rate limits
        self.default_limits = {
            RateLimitType.PER_SECOND: 10,
            RateLimitType.PER_MINUTE: 100,
            RateLimitType.PER_HOUR: 1000,
            RateLimitType.PER_DAY: 10000
        }
        
        self.logger.info("RateLimiter initialized")
    
    def check_rate_limit(self, user_id: Optional[str], ip_address: str,
                        endpoint: str, custom_limit: Optional[Tuple[int, RateLimitType]] = None) -> Tuple[bool, Optional[RateLimit]]:
        """Check if request is within rate limits"""
        try:
            # Check custom endpoint limit first
            if custom_limit:
                limit_key = f"{endpoint}:{custom_limit[1].value}"
                target_dict = self.endpoint_rate_limits[endpoint]
                
                if limit_key not in target_dict:
                    target_dict[limit_key] = RateLimit(
                        limit=custom_limit[0],
                        window=custom_limit[1],
                        remaining=custom_limit[0]
                    )
                    target_dict[limit_key]._calculate_next_reset()
                
                rate_limit = target_dict[limit_key]
                rate_limit.reset_if_needed()
                
                if rate_limit.is_exceeded():
                    return False, rate_limit
                
                rate_limit.remaining -= 1
                return True, rate_limit
            
            # Check user rate limits
            if user_id:
                allowed, rate_limit = self._check_user_limits(user_id)
                if not allowed:
                    return False, rate_limit
            
            # Check IP rate limits
            allowed, rate_limit = self._check_ip_limits(ip_address)
            if not allowed:
                return False, rate_limit
            
            # Check global endpoint limits
            allowed, rate_limit = self._check_endpoint_limits(endpoint)
            if not allowed:
                return False, rate_limit
            
            # Record successful request
            self._record_request(user_id, ip_address, endpoint)
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request on error
            return True, None
    
    def _check_user_limits(self, user_id: str) -> Tuple[bool, Optional[RateLimit]]:
        """Check rate limits for specific user"""
        user_limits = self.user_rate_limits[user_id]
        
        for limit_type, default_limit in self.default_limits.items():
            limit_key = limit_type.value
            
            if limit_key not in user_limits:
                user_limits[limit_key] = RateLimit(
                    limit=default_limit,
                    window=limit_type,
                    remaining=default_limit
                )
                user_limits[limit_key]._calculate_next_reset()
            
            rate_limit = user_limits[limit_key]
            rate_limit.reset_if_needed()
            
            if rate_limit.is_exceeded():
                return False, rate_limit
        
        # Decrement all rate limits
        for rate_limit in user_limits.values():
            rate_limit.remaining -= 1
        
        return True, None
    
    def _check_ip_limits(self, ip_address: str) -> Tuple[bool, Optional[RateLimit]]:
        """Check rate limits for IP address"""
        # More restrictive limits for IP-based limiting
        ip_limits = {
            RateLimitType.PER_SECOND: 5,
            RateLimitType.PER_MINUTE: 50,
            RateLimitType.PER_HOUR: 500,
            RateLimitType.PER_DAY: 5000
        }
        
        ip_rate_limits = self.ip_rate_limits[ip_address]
        
        for limit_type, limit_value in ip_limits.items():
            limit_key = limit_type.value
            
            if limit_key not in ip_rate_limits:
                ip_rate_limits[limit_key] = RateLimit(
                    limit=limit_value,
                    window=limit_type,
                    remaining=limit_value
                )
                ip_rate_limits[limit_key]._calculate_next_reset()
            
            rate_limit = ip_rate_limits[limit_key]
            rate_limit.reset_if_needed()
            
            if rate_limit.is_exceeded():
                return False, rate_limit
        
        # Decrement all rate limits
        for rate_limit in ip_rate_limits.values():
            rate_limit.remaining -= 1
        
        return True, None
    
    def _check_endpoint_limits(self, endpoint: str) -> Tuple[bool, Optional[RateLimit]]:
        """Check global rate limits for endpoint"""
        # Global endpoint limits (all users combined)
        endpoint_limits = {
            RateLimitType.PER_SECOND: 100,
            RateLimitType.PER_MINUTE: 1000,
            RateLimitType.PER_HOUR: 10000,
            RateLimitType.PER_DAY: 100000
        }
        
        global_limits = self.endpoint_rate_limits[endpoint]
        
        for limit_type, limit_value in endpoint_limits.items():
            limit_key = f"global:{limit_type.value}"
            
            if limit_key not in global_limits:
                global_limits[limit_key] = RateLimit(
                    limit=limit_value,
                    window=limit_type,
                    remaining=limit_value
                )
                global_limits[limit_key]._calculate_next_reset()
            
            rate_limit = global_limits[limit_key]
            rate_limit.reset_if_needed()
            
            if rate_limit.is_exceeded():
                return False, rate_limit
        
        # Decrement all rate limits
        for limit_key, rate_limit in global_limits.items():
            if limit_key.startswith("global:"):
                rate_limit.remaining -= 1
        
        return True, None
    
    def _record_request(self, user_id: Optional[str], ip_address: str, endpoint: str):
        """Record request for analytics"""
        timestamp = datetime.now()
        
        # Record for user
        if user_id:
            self.request_history[f"user:{user_id}"].append(timestamp)
        
        # Record for IP
        self.request_history[f"ip:{ip_address}"].append(timestamp)
        
        # Record for endpoint
        self.request_history[f"endpoint:{endpoint}"].append(timestamp)
    
    def set_user_rate_limit(self, user_id: str, limit_type: RateLimitType, limit: int):
        """Set custom rate limit for user"""
        user_limits = self.user_rate_limits[user_id]
        limit_key = limit_type.value
        
        user_limits[limit_key] = RateLimit(
            limit=limit,
            window=limit_type,
            remaining=limit
        )
        user_limits[limit_key]._calculate_next_reset()
        
        self.logger.info(f"Set rate limit for user {user_id}: {limit} {limit_type.value}")
    
    def get_user_rate_limit_status(self, user_id: str) -> Dict[str, Dict[str, any]]:
        """Get current rate limit status for user"""
        user_limits = self.user_rate_limits[user_id]
        status = {}
        
        for limit_key, rate_limit in user_limits.items():
            rate_limit.reset_if_needed()
            status[limit_key] = {
                "limit": rate_limit.limit,
                "remaining": rate_limit.remaining,
                "reset_at": rate_limit.reset_at.isoformat() if rate_limit.reset_at else None,
                "exceeded": rate_limit.is_exceeded()
            }
        
        return status
    
    def reset_user_rate_limits(self, user_id: str):
        """Reset all rate limits for user"""
        if user_id in self.user_rate_limits:
            user_limits = self.user_rate_limits[user_id]
            for rate_limit in user_limits.values():
                rate_limit.remaining = rate_limit.limit
                rate_limit._calculate_next_reset()
            
            self.logger.info(f"Reset rate limits for user {user_id}")
    
    def get_rate_limit_analytics(self) -> Dict[str, any]:
        """Get rate limiting analytics"""
        total_users_limited = len([
            user_id for user_id, limits in self.user_rate_limits.items()
            if any(limit.is_exceeded() for limit in limits.values())
        ])
        
        total_ips_limited = len([
            ip for ip, limits in self.ip_rate_limits.items()
            if any(limit.is_exceeded() for limit in limits.values())
        ])
        
        return {
            "total_users_with_limits": len(self.user_rate_limits),
            "users_currently_limited": total_users_limited,
            "total_ips_with_limits": len(self.ip_rate_limits),
            "ips_currently_limited": total_ips_limited,
            "total_endpoints_monitored": len(self.endpoint_rate_limits),
            "request_history_size": sum(len(history) for history in self.request_history.values())
        }
    
    def cleanup_expired_limits(self):
        """Clean up expired rate limit data"""
        now = datetime.now()
        cleanup_threshold = now - timedelta(hours=24)
        
        # Clean up old request history
        for key, history in self.request_history.items():
            while history and history[0] < cleanup_threshold:
                history.popleft()
        
        # Remove empty histories
        empty_keys = [key for key, history in self.request_history.items() if not history]
        for key in empty_keys:
            del self.request_history[key]
        
        self.logger.debug(f"Cleaned up rate limit data, removed {len(empty_keys)} empty histories")