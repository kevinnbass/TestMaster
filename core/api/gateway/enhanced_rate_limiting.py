"""
Enhanced Rate Limiting - Advanced algorithms extracted from unified gateway

This module provides advanced rate limiting algorithms that were extracted
from the unified gateway implementation during IRONCLAD consolidation.

Author: Agent E - Consolidated from unified_api_gateway.py
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Advanced rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"  
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitPolicy:
    """Rate limit policy configuration"""
    name: str
    requests: int
    window_seconds: int
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_limit: Optional[int] = None


class EnhancedRateLimitingEngine:
    """
    Enhanced rate limiting engine with multiple algorithms
    
    Extracted from unified gateway and enhanced for integration
    with the core TestMaster API Gateway.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("enhanced_rate_limiting")
        
        # Rate limit buckets
        self.rate_buckets: Dict[str, Dict[str, Any]] = {}
        self.bucket_lock = threading.Lock()
        
        # Predefined policies
        self.policies: Dict[str, RateLimitPolicy] = {
            "default": RateLimitPolicy("default", 1000, 3600),
            "premium": RateLimitPolicy("premium", 5000, 3600),
            "enterprise": RateLimitPolicy("enterprise", 20000, 3600, burst_limit=30000),
            "internal": RateLimitPolicy("internal", 100000, 3600),
            "development": RateLimitPolicy("development", 10000, 3600, burst_limit=50000)
        }
        
        # Algorithm implementations
        self.algorithms = {
            RateLimitAlgorithm.TOKEN_BUCKET: self._token_bucket_check,
            RateLimitAlgorithm.SLIDING_WINDOW: self._sliding_window_check,
            RateLimitAlgorithm.FIXED_WINDOW: self._fixed_window_check,
            RateLimitAlgorithm.LEAKY_BUCKET: self._leaky_bucket_check
        }
        
        self.logger.info("Enhanced rate limiting engine initialized")
    
    def add_policy(self, policy: RateLimitPolicy):
        """Add custom rate limiting policy"""
        self.policies[policy.name] = policy
        self.logger.info(f"Added rate limit policy: {policy.name}")
    
    def check_rate_limit(self, scope: str, identifier: str, 
                        policy_name: str = "default", 
                        algorithm: RateLimitAlgorithm = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request passes rate limit
        
        Args:
            scope: Rate limiting scope (user, ip, endpoint, etc.)
            identifier: Unique identifier for the scope
            policy_name: Name of rate limit policy to apply
            algorithm: Rate limiting algorithm to use (optional)
            
        Returns:
            Tuple of (allowed: bool, rate_info: dict)
        """
        try:
            policy = self.policies.get(policy_name, self.policies["default"])
            algorithm = algorithm or policy.algorithm
            bucket_key = f"{scope}:{identifier}:{policy_name}"
            
            with self.bucket_lock:
                # Initialize bucket if needed
                if bucket_key not in self.rate_buckets:
                    self.rate_buckets[bucket_key] = self._initialize_bucket(policy)
                
                bucket = self.rate_buckets[bucket_key]
                
                # Apply rate limiting algorithm
                if algorithm in self.algorithms:
                    allowed, bucket_info = self.algorithms[algorithm](bucket, policy)
                    
                    # Update bucket state
                    bucket.update(bucket_info)
                    self.rate_buckets[bucket_key] = bucket
                    
                    return allowed, {
                        "bucket_key": bucket_key,
                        "algorithm": algorithm.value,
                        "policy": policy_name,
                        "current_usage": bucket_info.get("current_usage", 0),
                        "limit": policy.requests,
                        "reset_time": bucket_info.get("reset_time"),
                        "retry_after": bucket_info.get("retry_after", 0)
                    }
                else:
                    self.logger.warning(f"Unknown rate limiting algorithm: {algorithm}")
                    return True, {"error": "unknown_algorithm"}
                    
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True, {"error": str(e)}  # Fail open for availability
    
    def _initialize_bucket(self, policy: RateLimitPolicy) -> Dict[str, Any]:
        """Initialize rate limit bucket"""
        return {
            "policy": policy.name,
            "algorithm": policy.algorithm.value,
            "requests": policy.requests,
            "window_seconds": policy.window_seconds,
            "burst_limit": policy.burst_limit or policy.requests,
            "current_usage": 0,
            "last_refill": time.time(),
            "window_start": time.time(),
            "request_times": []
        }
    
    def _token_bucket_check(self, bucket: Dict[str, Any], policy: RateLimitPolicy) -> Tuple[bool, Dict[str, Any]]:
        """
        Token bucket algorithm - allows bursts up to bucket capacity
        
        Extracted from unified gateway implementation
        """
        now = time.time()
        
        # Calculate tokens to add based on time passed
        time_passed = now - bucket["last_refill"]
        refill_rate = policy.requests / policy.window_seconds
        tokens_to_add = time_passed * refill_rate
        
        # Update bucket (don't exceed capacity)
        max_tokens = policy.burst_limit or policy.requests
        bucket["current_usage"] = max(0, min(max_tokens, bucket["current_usage"] - tokens_to_add))
        bucket["last_refill"] = now
        
        # Check if request allowed (need at least 1 token)
        if bucket["current_usage"] < max_tokens:
            bucket["current_usage"] += 1
            return True, bucket
        else:
            # Calculate retry after time
            retry_after = 1.0 / refill_rate
            bucket["retry_after"] = retry_after
            return False, bucket
    
    def _sliding_window_check(self, bucket: Dict[str, Any], policy: RateLimitPolicy) -> Tuple[bool, Dict[str, Any]]:
        """
        Sliding window algorithm - precise rate limiting
        
        Extracted from unified gateway implementation
        """
        now = time.time()
        window_start = now - policy.window_seconds
        
        # Remove requests outside the window
        bucket["request_times"] = [t for t in bucket["request_times"] if t > window_start]
        
        # Check if under limit
        if len(bucket["request_times"]) < policy.requests:
            bucket["request_times"].append(now)
            bucket["current_usage"] = len(bucket["request_times"])
            return True, bucket
        else:
            # Calculate retry after based on oldest request
            oldest_request = min(bucket["request_times"])
            retry_after = oldest_request + policy.window_seconds - now
            bucket["retry_after"] = max(0, retry_after)
            return False, bucket
    
    def _fixed_window_check(self, bucket: Dict[str, Any], policy: RateLimitPolicy) -> Tuple[bool, Dict[str, Any]]:
        """
        Fixed window algorithm - simple and efficient
        
        Extracted from unified gateway implementation
        """
        now = time.time()
        
        # Check if window reset is needed
        if now - bucket["window_start"] >= policy.window_seconds:
            bucket["current_usage"] = 0
            bucket["window_start"] = now
        
        # Check if under limit
        if bucket["current_usage"] < policy.requests:
            bucket["current_usage"] += 1
            return True, bucket
        else:
            # Calculate retry after
            retry_after = bucket["window_start"] + policy.window_seconds - now
            bucket["retry_after"] = max(0, retry_after)
            return False, bucket
    
    def _leaky_bucket_check(self, bucket: Dict[str, Any], policy: RateLimitPolicy) -> Tuple[bool, Dict[str, Any]]:
        """
        Leaky bucket algorithm - smooth rate limiting
        
        Extracted from unified gateway implementation
        """
        now = time.time()
        
        # Calculate leakage
        time_passed = now - bucket["last_refill"]
        leak_rate = policy.requests / policy.window_seconds
        leaked = time_passed * leak_rate
        
        # Update bucket
        bucket["current_usage"] = max(0, bucket["current_usage"] - leaked)
        bucket["last_refill"] = now
        
        # Check if under capacity
        if bucket["current_usage"] < policy.requests:
            bucket["current_usage"] += 1
            return True, bucket
        else:
            # Time for one token to leak
            retry_after = 1.0 / leak_rate
            bucket["retry_after"] = retry_after
            return False, bucket
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced rate limiting statistics"""
        with self.bucket_lock:
            return {
                "active_buckets": len(self.rate_buckets),
                "policies_configured": len(self.policies),
                "algorithms_available": len(self.algorithms),
                "bucket_statistics": self._calculate_bucket_stats()
            }
    
    def _calculate_bucket_stats(self) -> Dict[str, Any]:
        """Calculate detailed bucket statistics"""
        stats = {
            "by_algorithm": {},
            "by_policy": {},
            "total_current_usage": 0
        }
        
        for bucket_key, bucket in self.rate_buckets.items():
            algorithm = bucket.get("algorithm", "unknown")
            policy = bucket.get("policy", "unknown")
            usage = bucket.get("current_usage", 0)
            
            # Algorithm stats
            if algorithm not in stats["by_algorithm"]:
                stats["by_algorithm"][algorithm] = {"count": 0, "total_usage": 0}
            stats["by_algorithm"][algorithm]["count"] += 1
            stats["by_algorithm"][algorithm]["total_usage"] += usage
            
            # Policy stats
            if policy not in stats["by_policy"]:
                stats["by_policy"][policy] = {"count": 0, "total_usage": 0}
            stats["by_policy"][policy]["count"] += 1
            stats["by_policy"][policy]["total_usage"] += usage
            
            stats["total_current_usage"] += usage
        
        return stats
    
    def cleanup_expired_buckets(self, max_age_seconds: int = 86400):
        """Clean up old unused buckets"""
        now = time.time()
        expired_keys = []
        
        with self.bucket_lock:
            for bucket_key, bucket in self.rate_buckets.items():
                last_activity = max(
                    bucket.get("last_refill", 0),
                    bucket.get("window_start", 0)
                )
                if now - last_activity > max_age_seconds:
                    expired_keys.append(bucket_key)
            
            for key in expired_keys:
                del self.rate_buckets[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired rate limit buckets")


# Factory function for easy integration
def create_enhanced_rate_limiter() -> EnhancedRateLimitingEngine:
    """
    Create enhanced rate limiting engine
    
    Returns:
        Configured enhanced rate limiting engine
    """
    return EnhancedRateLimitingEngine()