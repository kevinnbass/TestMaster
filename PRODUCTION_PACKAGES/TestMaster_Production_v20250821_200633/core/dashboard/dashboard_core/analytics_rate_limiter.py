"""
Analytics Adaptive Rate Limiting and Backpressure Management
===========================================================

Advanced rate limiting system with adaptive throttling, backpressure
handling, and intelligent traffic shaping for analytics data flow.

Author: TestMaster Team
"""

import logging
import time
import threading
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import queue
import psutil

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE_AIMD = "adaptive_aimd"  # Additive Increase Multiplicative Decrease
    EXPONENTIAL_BACKOFF = "exponential_backoff"

class BackpressureAction(Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    COMPRESS_DATA = "compress_data"
    SAMPLE_DATA = "sample_data"
    DELAY_PROCESSING = "delay_processing"
    REDIRECT_TRAFFIC = "redirect_traffic"

class TrafficPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    rule_id: str
    strategy: RateLimitStrategy
    max_requests_per_second: float
    max_burst: int
    window_size_seconds: int = 60
    priority: TrafficPriority = TrafficPriority.NORMAL
    adaptive: bool = True
    backpressure_action: BackpressureAction = BackpressureAction.DELAY_PROCESSING
    source_pattern: str = "*"
    data_type_pattern: str = "*"
    description: str = ""

@dataclass
class TrafficMetrics:
    """Traffic measurement metrics."""
    requests_per_second: float
    bytes_per_second: float
    queue_depth: int
    processing_latency_ms: float
    error_rate: float
    timestamp: datetime

@dataclass
class BackpressureEvent:
    """Backpressure event information."""
    event_id: str
    timestamp: datetime
    trigger_metric: str
    threshold_value: float
    current_value: float
    action_taken: BackpressureAction
    source: str
    duration_seconds: float = 0
    recovery_time: Optional[datetime] = None

class AnalyticsRateLimiter:
    """
    Advanced adaptive rate limiting and backpressure management system.
    """
    
    def __init__(self, max_queue_size: int = 10000,
                 monitoring_interval: float = 1.0,
                 adaptive_adjustment_factor: float = 0.1):
        """
        Initialize analytics rate limiter.
        
        Args:
            max_queue_size: Maximum queue size before backpressure kicks in
            monitoring_interval: Interval for monitoring and adjustments
            adaptive_adjustment_factor: Factor for adaptive rate adjustments
        """
        self.max_queue_size = max_queue_size
        self.monitoring_interval = monitoring_interval
        self.adaptive_adjustment_factor = adaptive_adjustment_factor
        
        # Rate limiting rules
        self.rate_limit_rules = {}
        self.active_limiters = {}
        
        # Traffic tracking
        self.traffic_queues = defaultdict(lambda: deque(maxlen=max_queue_size))
        self.traffic_metrics = defaultdict(deque)
        self.request_timestamps = defaultdict(deque)
        
        # Backpressure management
        self.backpressure_events = deque(maxlen=1000)
        self.backpressure_active = defaultdict(bool)
        self.backpressure_thresholds = {
            'queue_depth': 0.8,  # 80% of max queue size
            'memory_usage': 0.85,  # 85% memory usage
            'cpu_usage': 0.80,  # 80% CPU usage
            'error_rate': 0.1,  # 10% error rate
            'latency_ms': 5000  # 5 second latency
        }
        
        # Adaptive control
        self.adaptive_rates = {}
        self.congestion_control = {}
        
        # Performance statistics
        self.limiter_stats = {
            'requests_processed': 0,
            'requests_throttled': 0,
            'requests_dropped': 0,
            'backpressure_events': 0,
            'adaptive_adjustments': 0,
            'total_delay_seconds': 0,
            'start_time': datetime.now()
        }
        
        # Threading
        self.limiter_active = False
        self.monitor_thread = None
        self.processor_thread = None
        
        # Processing queues by priority
        self.priority_queues = {
            priority: queue.PriorityQueue(maxsize=max_queue_size//5)
            for priority in TrafficPriority
        }
        
        # Setup default rules
        self._setup_default_rules()
        
        logger.info("Analytics Rate Limiter initialized")
    
    def start_rate_limiting(self):
        """Start rate limiting and monitoring."""
        if self.limiter_active:
            return
        
        self.limiter_active = True
        
        # Start monitoring and processing threads
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        
        self.monitor_thread.start()
        self.processor_thread.start()
        
        logger.info("Analytics rate limiting started")
    
    def stop_rate_limiting(self):
        """Stop rate limiting."""
        self.limiter_active = False
        
        # Wait for threads to finish
        for thread in [self.monitor_thread, self.processor_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("Analytics rate limiting stopped")
    
    def add_rate_limit_rule(self, rule: RateLimitRule):
        """Add a rate limiting rule."""
        self.rate_limit_rules[rule.rule_id] = rule
        self._initialize_limiter(rule)
        logger.info(f"Added rate limiting rule: {rule.rule_id}")
    
    def process_request(self, data: Dict[str, Any], source: str = "unknown",
                       data_type: str = "general", priority: TrafficPriority = TrafficPriority.NORMAL) -> Tuple[bool, Optional[float], str]:
        """
        Process a request through rate limiting.
        
        Args:
            data: Request data
            source: Data source identifier
            data_type: Type of data being processed
            priority: Request priority
        
        Returns:
            Tuple of (allowed, delay_seconds, reason)
        """
        try:
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(source, data_type)
            
            # Check rate limits
            for rule in applicable_rules:
                allowed, delay, reason = self._check_rate_limit(rule, source, data, priority)
                if not allowed:
                    self.limiter_stats['requests_throttled'] += 1
                    return False, delay, f"Rate limited by rule {rule.rule_id}: {reason}"
            
            # Check for backpressure
            backpressure_action = self._check_backpressure(source, data, priority)
            if backpressure_action:
                return self._handle_backpressure(backpressure_action, data, source, priority)
            
            # Queue request for processing
            success = self._queue_request(data, source, priority)
            
            if success:
                self.limiter_stats['requests_processed'] += 1
                return True, 0.0, "accepted"
            else:
                self.limiter_stats['requests_dropped'] += 1
                return False, 0.0, "queue_full"
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return False, 0.0, f"processing_error: {str(e)}"
    
    def get_rate_limiter_summary(self) -> Dict[str, Any]:
        """Get rate limiter system summary."""
        uptime = (datetime.now() - self.limiter_stats['start_time']).total_seconds()
        
        # Calculate current rates
        current_rates = {}
        for rule_id, rule in self.rate_limit_rules.items():
            if rule_id in self.active_limiters:
                limiter = self.active_limiters[rule_id]
                current_rates[rule_id] = self._calculate_current_rate(limiter, rule)
        
        # Recent backpressure events
        recent_events = [event for event in self.backpressure_events
                        if (datetime.now() - event.timestamp).total_seconds() < 3600]
        
        # Queue depths
        queue_depths = {priority.value: q.qsize() for priority, q in self.priority_queues.items()}
        
        return {
            'limiter_status': {
                'active': self.limiter_active,
                'uptime_seconds': uptime,
                'max_queue_size': self.max_queue_size
            },
            'statistics': self.limiter_stats.copy(),
            'rate_limiting_rules': {
                'total_rules': len(self.rate_limit_rules),
                'active_limiters': len(self.active_limiters),
                'current_rates': current_rates
            },
            'backpressure_management': {
                'active_backpressure': dict(self.backpressure_active),
                'recent_events_count': len(recent_events),
                'thresholds': self.backpressure_thresholds.copy()
            },
            'traffic_queues': {
                'queue_depths': queue_depths,
                'total_queued': sum(queue_depths.values()),
                'queue_utilization': sum(queue_depths.values()) / (self.max_queue_size * len(self.priority_queues))
            },
            'adaptive_control': {
                'adaptive_rates': dict(self.adaptive_rates),
                'congestion_control_active': len(self.congestion_control),
                'adjustment_factor': self.adaptive_adjustment_factor
            }
        }
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules."""
        
        # High-frequency analytics data
        high_freq_rule = RateLimitRule(
            rule_id='high_frequency_analytics',
            strategy=RateLimitStrategy.ADAPTIVE_AIMD,
            max_requests_per_second=100.0,
            max_burst=200,
            window_size_seconds=60,
            priority=TrafficPriority.HIGH,
            adaptive=True,
            backpressure_action=BackpressureAction.SAMPLE_DATA,
            data_type_pattern='*analytics*',
            description='High frequency analytics data limiting'
        )
        
        # System metrics
        system_metrics_rule = RateLimitRule(
            rule_id='system_metrics',
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            max_requests_per_second=50.0,
            max_burst=100,
            window_size_seconds=30,
            priority=TrafficPriority.NORMAL,
            adaptive=True,
            backpressure_action=BackpressureAction.DROP_OLDEST,
            data_type_pattern='*metrics*',
            description='System metrics rate limiting'
        )
        
        # Error reports (critical)
        error_rule = RateLimitRule(
            rule_id='error_reports',
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            max_requests_per_second=20.0,
            max_burst=50,
            window_size_seconds=300,  # 5 minutes
            priority=TrafficPriority.CRITICAL,
            adaptive=False,
            backpressure_action=BackpressureAction.DELAY_PROCESSING,
            data_type_pattern='*error*',
            description='Error report rate limiting'
        )
        
        # Background telemetry
        background_rule = RateLimitRule(
            rule_id='background_telemetry',
            strategy=RateLimitStrategy.EXPONENTIAL_BACKOFF,
            max_requests_per_second=10.0,
            max_burst=20,
            window_size_seconds=120,
            priority=TrafficPriority.BACKGROUND,
            adaptive=True,
            backpressure_action=BackpressureAction.COMPRESS_DATA,
            data_type_pattern='*telemetry*',
            description='Background telemetry rate limiting'
        )
        
        # Add rules
        for rule in [high_freq_rule, system_metrics_rule, error_rule, background_rule]:
            self.rate_limit_rules[rule.rule_id] = rule
            self._initialize_limiter(rule)
    
    def _initialize_limiter(self, rule: RateLimitRule):
        """Initialize rate limiter for a rule."""
        if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            limiter = {
                'tokens': rule.max_burst,
                'last_refill': time.time(),
                'max_tokens': rule.max_burst,
                'refill_rate': rule.max_requests_per_second
            }
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            limiter = {
                'requests': deque(),
                'window_size': rule.window_size_seconds
            }
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            limiter = {
                'window_start': time.time(),
                'window_requests': 0,
                'window_size': rule.window_size_seconds
            }
        elif rule.strategy == RateLimitStrategy.ADAPTIVE_AIMD:
            limiter = {
                'current_rate': rule.max_requests_per_second,
                'max_rate': rule.max_requests_per_second,
                'last_success': time.time(),
                'congestion_detected': False,
                'requests': deque()
            }
        else:  # EXPONENTIAL_BACKOFF
            limiter = {
                'current_delay': 0,
                'base_delay': 1.0 / rule.max_requests_per_second,
                'max_delay': 60.0,
                'consecutive_failures': 0
            }
        
        self.active_limiters[rule.rule_id] = limiter
    
    def _find_applicable_rules(self, source: str, data_type: str) -> List[RateLimitRule]:
        """Find rate limiting rules applicable to the request."""
        applicable_rules = []
        
        for rule in self.rate_limit_rules.values():
            # Check source pattern
            if rule.source_pattern != "*" and not self._matches_pattern(source, rule.source_pattern):
                continue
            
            # Check data type pattern
            if rule.data_type_pattern != "*" and not self._matches_pattern(data_type, rule.data_type_pattern):
                continue
            
            applicable_rules.append(rule)
        
        # Sort by priority (critical first)
        priority_order = {
            TrafficPriority.CRITICAL: 0,
            TrafficPriority.HIGH: 1,
            TrafficPriority.NORMAL: 2,
            TrafficPriority.LOW: 3,
            TrafficPriority.BACKGROUND: 4
        }
        
        applicable_rules.sort(key=lambda r: priority_order.get(r.priority, 5))
        return applicable_rules
    
    def _check_rate_limit(self, rule: RateLimitRule, source: str, data: Dict[str, Any],
                         priority: TrafficPriority) -> Tuple[bool, Optional[float], str]:
        """Check if request passes rate limit for a specific rule."""
        limiter = self.active_limiters.get(rule.rule_id)
        if not limiter:
            return True, 0.0, "no_limiter"
        
        current_time = time.time()
        
        if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(limiter, rule, current_time)
        
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(limiter, rule, current_time)
        
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(limiter, rule, current_time)
        
        elif rule.strategy == RateLimitStrategy.ADAPTIVE_AIMD:
            return self._check_adaptive_aimd(limiter, rule, current_time, source)
        
        elif rule.strategy == RateLimitStrategy.EXPONENTIAL_BACKOFF:
            return self._check_exponential_backoff(limiter, rule, current_time)
        
        return True, 0.0, "unknown_strategy"
    
    def _check_token_bucket(self, limiter: Dict[str, Any], rule: RateLimitRule,
                          current_time: float) -> Tuple[bool, Optional[float], str]:
        """Check token bucket rate limit."""
        # Refill tokens
        time_passed = current_time - limiter['last_refill']
        tokens_to_add = time_passed * limiter['refill_rate']
        limiter['tokens'] = min(limiter['max_tokens'], limiter['tokens'] + tokens_to_add)
        limiter['last_refill'] = current_time
        
        # Check if we have tokens
        if limiter['tokens'] >= 1:
            limiter['tokens'] -= 1
            return True, 0.0, "allowed"
        else:
            # Calculate delay until next token
            delay = (1 - limiter['tokens']) / limiter['refill_rate']
            return False, delay, "token_bucket_empty"
    
    def _check_sliding_window(self, limiter: Dict[str, Any], rule: RateLimitRule,
                            current_time: float) -> Tuple[bool, Optional[float], str]:
        """Check sliding window rate limit."""
        window_start = current_time - limiter['window_size']
        
        # Remove old requests
        while limiter['requests'] and limiter['requests'][0] < window_start:
            limiter['requests'].popleft()
        
        # Check if we're under the limit
        if len(limiter['requests']) < rule.max_requests_per_second * limiter['window_size']:
            limiter['requests'].append(current_time)
            return True, 0.0, "allowed"
        else:
            # Calculate delay until oldest request expires
            delay = limiter['requests'][0] + limiter['window_size'] - current_time
            return False, delay, "sliding_window_full"
    
    def _check_fixed_window(self, limiter: Dict[str, Any], rule: RateLimitRule,
                          current_time: float) -> Tuple[bool, Optional[float], str]:
        """Check fixed window rate limit."""
        # Check if we need a new window
        if current_time - limiter['window_start'] >= limiter['window_size']:
            limiter['window_start'] = current_time
            limiter['window_requests'] = 0
        
        # Check if we're under the limit
        max_requests = rule.max_requests_per_second * limiter['window_size']
        if limiter['window_requests'] < max_requests:
            limiter['window_requests'] += 1
            return True, 0.0, "allowed"
        else:
            # Calculate delay until next window
            delay = limiter['window_start'] + limiter['window_size'] - current_time
            return False, delay, "fixed_window_full"
    
    def _check_adaptive_aimd(self, limiter: Dict[str, Any], rule: RateLimitRule,
                           current_time: float, source: str) -> Tuple[bool, Optional[float], str]:
        """Check adaptive AIMD rate limit."""
        # Clean old requests
        window_start = current_time - 60  # 1 minute window
        while limiter['requests'] and limiter['requests'][0] < window_start:
            limiter['requests'].popleft()
        
        # Calculate current rate
        current_rate = len(limiter['requests']) / 60.0
        
        # Adaptive rate adjustment
        if rule.adaptive:
            if current_rate < limiter['current_rate'] * 0.8:  # Underutilized
                # Additive increase
                limiter['current_rate'] = min(limiter['max_rate'],
                                            limiter['current_rate'] + self.adaptive_adjustment_factor)
                self.limiter_stats['adaptive_adjustments'] += 1
            elif self._detect_congestion(source):  # Congestion detected
                # Multiplicative decrease
                limiter['current_rate'] = max(1.0, limiter['current_rate'] * 0.5)
                limiter['congestion_detected'] = True
                self.limiter_stats['adaptive_adjustments'] += 1
        
        # Check rate limit
        if current_rate < limiter['current_rate']:
            limiter['requests'].append(current_time)
            limiter['last_success'] = current_time
            return True, 0.0, "adaptive_allowed"
        else:
            delay = 1.0 / limiter['current_rate']
            return False, delay, "adaptive_rate_limited"
    
    def _check_exponential_backoff(self, limiter: Dict[str, Any], rule: RateLimitRule,
                                 current_time: float) -> Tuple[bool, Optional[float], str]:
        """Check exponential backoff rate limit."""
        if limiter['current_delay'] == 0:
            limiter['current_delay'] = limiter['base_delay']
            return True, 0.0, "allowed"
        else:
            # Double the delay (exponential backoff)
            limiter['current_delay'] = min(limiter['max_delay'], limiter['current_delay'] * 2)
            return False, limiter['current_delay'], "exponential_backoff"
    
    def _check_backpressure(self, source: str, data: Dict[str, Any],
                          priority: TrafficPriority) -> Optional[BackpressureAction]:
        """Check if backpressure should be applied."""
        # Check system resources
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # Check queue depth
            total_queued = sum(q.qsize() for q in self.priority_queues.values())
            queue_utilization = total_queued / self.max_queue_size
            
            # Determine if backpressure is needed
            if queue_utilization > self.backpressure_thresholds['queue_depth']:
                return BackpressureAction.DROP_OLDEST
            
            if memory_usage > self.backpressure_thresholds['memory_usage']:
                return BackpressureAction.COMPRESS_DATA
            
            if cpu_usage > self.backpressure_thresholds['cpu_usage']:
                return BackpressureAction.SAMPLE_DATA
            
        except Exception as e:
            logger.warning(f"Error checking backpressure: {e}")
        
        return None
    
    def _handle_backpressure(self, action: BackpressureAction, data: Dict[str, Any],
                           source: str, priority: TrafficPriority) -> Tuple[bool, Optional[float], str]:
        """Handle backpressure action."""
        self.limiter_stats['backpressure_events'] += 1
        
        event = BackpressureEvent(
            event_id=f"bp_{int(time.time())}_{hash(source)}",
            timestamp=datetime.now(),
            trigger_metric="system_load",
            threshold_value=0.8,
            current_value=0.9,  # Placeholder
            action_taken=action,
            source=source
        )
        self.backpressure_events.append(event)
        
        if action == BackpressureAction.DROP_OLDEST:
            # Drop oldest items from queue
            for q in self.priority_queues.values():
                if not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
            return True, 0.0, "backpressure_drop_oldest"
        
        elif action == BackpressureAction.DROP_NEWEST:
            return False, 0.0, "backpressure_drop_newest"
        
        elif action == BackpressureAction.COMPRESS_DATA:
            # Compress data before queueing
            compressed_data = self._compress_data(data)
            return self._queue_request(compressed_data, source, priority), 0.0, "backpressure_compressed"
        
        elif action == BackpressureAction.SAMPLE_DATA:
            # Sample data (keep every Nth request)
            if hash(source) % 2 == 0:  # Simple sampling
                return self._queue_request(data, source, priority), 0.0, "backpressure_sampled"
            else:
                return False, 0.0, "backpressure_sample_dropped"
        
        elif action == BackpressureAction.DELAY_PROCESSING:
            delay = 1.0  # 1 second delay
            self.limiter_stats['total_delay_seconds'] += delay
            return True, delay, "backpressure_delayed"
        
        return False, 0.0, "backpressure_unknown"
    
    def _queue_request(self, data: Dict[str, Any], source: str,
                      priority: TrafficPriority) -> bool:
        """Queue request for processing."""
        try:
            priority_queue = self.priority_queues[priority]
            
            # Create request item
            request_item = {
                'data': data,
                'source': source,
                'timestamp': time.time(),
                'priority': priority.value
            }
            
            # Queue with priority (lower number = higher priority)
            priority_value = list(TrafficPriority).index(priority)
            priority_queue.put_nowait((priority_value, request_item))
            
            return True
        
        except queue.Full:
            return False
        except Exception as e:
            logger.error(f"Error queueing request: {e}")
            return False
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.limiter_active:
            try:
                time.sleep(self.monitoring_interval)
                
                # Update traffic metrics
                self._update_traffic_metrics()
                
                # Adjust adaptive rates
                self._adjust_adaptive_rates()
                
                # Check for congestion recovery
                self._check_congestion_recovery()
                
                # Clean up old data
                self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _processing_loop(self):
        """Background processing loop for queued requests."""
        while self.limiter_active:
            try:
                # Process requests by priority
                for priority in TrafficPriority:
                    priority_queue = self.priority_queues[priority]
                    
                    if not priority_queue.empty():
                        try:
                            priority_value, request_item = priority_queue.get_nowait()
                            
                            # Process the request
                            self._process_queued_request(request_item)
                            
                        except queue.Empty:
                            continue
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
    
    def _process_queued_request(self, request_item: Dict[str, Any]):
        """Process a queued request."""
        # This is where the actual request processing would happen
        # For now, we just log that it was processed
        source = request_item.get('source', 'unknown')
        priority = request_item.get('priority', 'normal')
        
        logger.debug(f"Processed queued request from {source} with priority {priority}")
    
    def _update_traffic_metrics(self):
        """Update traffic metrics for monitoring."""
        current_time = datetime.now()
        
        for rule_id, limiter in self.active_limiters.items():
            rule = self.rate_limit_rules[rule_id]
            
            # Calculate current rate
            current_rate = self._calculate_current_rate(limiter, rule)
            
            # Calculate queue depth
            total_queued = sum(q.qsize() for q in self.priority_queues.values())
            
            # Create metrics
            metrics = TrafficMetrics(
                requests_per_second=current_rate,
                bytes_per_second=current_rate * 1024,  # Estimate
                queue_depth=total_queued,
                processing_latency_ms=10.0,  # Placeholder
                error_rate=0.01,  # Placeholder
                timestamp=current_time
            )
            
            self.traffic_metrics[rule_id].append(metrics)
            
            # Keep only recent metrics
            if len(self.traffic_metrics[rule_id]) > 100:
                self.traffic_metrics[rule_id].popleft()
    
    def _adjust_adaptive_rates(self):
        """Adjust adaptive rate limits based on current performance."""
        for rule_id, rule in self.rate_limit_rules.items():
            if not rule.adaptive:
                continue
            
            limiter = self.active_limiters.get(rule_id)
            if not limiter or rule.strategy != RateLimitStrategy.ADAPTIVE_AIMD:
                continue
            
            # Get recent metrics
            recent_metrics = list(self.traffic_metrics[rule_id])[-10:]  # Last 10 measurements
            
            if len(recent_metrics) < 5:
                continue
            
            # Calculate average error rate and latency
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            avg_latency = sum(m.processing_latency_ms for m in recent_metrics) / len(recent_metrics)
            
            # Adjust rate based on performance
            if avg_error_rate < 0.01 and avg_latency < 1000:  # Good performance
                # Increase rate (additive increase)
                limiter['current_rate'] = min(limiter['max_rate'],
                                            limiter['current_rate'] + self.adaptive_adjustment_factor)
            elif avg_error_rate > 0.05 or avg_latency > 5000:  # Poor performance
                # Decrease rate (multiplicative decrease)
                limiter['current_rate'] = max(1.0, limiter['current_rate'] * 0.8)
    
    def _check_congestion_recovery(self):
        """Check if congestion has been resolved."""
        for source, is_congested in list(self.congestion_control.items()):
            if is_congested:
                # Check if congestion indicators have improved
                if self._congestion_resolved(source):
                    del self.congestion_control[source]
                    logger.info(f"Congestion resolved for source: {source}")
    
    def _cleanup_old_data(self):
        """Clean up old data structures."""
        cutoff_time = time.time() - 3600  # 1 hour ago
        
        # Clean up request timestamps
        for source, timestamps in self.request_timestamps.items():
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()
        
        # Clean up backpressure events
        cutoff_datetime = datetime.now() - timedelta(hours=24)
        while (self.backpressure_events and 
               self.backpressure_events[0].timestamp < cutoff_datetime):
            self.backpressure_events.popleft()
    
    def _calculate_current_rate(self, limiter: Dict[str, Any], rule: RateLimitRule) -> float:
        """Calculate current request rate for a limiter."""
        if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return len(limiter.get('requests', [])) / rule.window_size_seconds
        elif rule.strategy == RateLimitStrategy.ADAPTIVE_AIMD:
            return limiter.get('current_rate', 0.0)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return rule.max_requests_per_second - (limiter.get('tokens', 0) / rule.max_burst)
        else:
            return 0.0
    
    def _detect_congestion(self, source: str) -> bool:
        """Detect if congestion is occurring for a source."""
        # Simple congestion detection based on system resources
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            if cpu_usage > 80 or memory_usage > 85:
                self.congestion_control[source] = True
                return True
        except:
            pass
        
        return False
    
    def _congestion_resolved(self, source: str) -> bool:
        """Check if congestion has been resolved for a source."""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            return cpu_usage < 60 and memory_usage < 70
        except:
            return True
    
    def _compress_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress data to reduce size (simplified implementation)."""
        # Simple compression: remove non-essential fields
        essential_fields = ['timestamp', 'component', 'metrics', 'cpu_usage_percent', 'memory_usage_percent']
        
        compressed = {}
        for field in essential_fields:
            if field in data:
                compressed[field] = data[field]
        
        compressed['_compressed'] = True
        return compressed
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        
        # Simple wildcard matching
        import fnmatch
        return fnmatch.fnmatch(text.lower(), pattern.lower())
    
    def get_backpressure_events(self, hours: int = 24) -> List[BackpressureEvent]:
        """Get recent backpressure events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [event for event in self.backpressure_events if event.timestamp >= cutoff_time]
    
    def shutdown(self):
        """Shutdown rate limiter."""
        self.stop_rate_limiting()
        logger.info("Analytics Rate Limiter shutdown")