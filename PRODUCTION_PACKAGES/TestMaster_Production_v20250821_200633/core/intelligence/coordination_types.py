"""
Coordination Types - Cross-System Coordination Type Definitions
===============================================================

Comprehensive type definitions and data structures for advanced cross-system
coordination protocols and communication systems with enterprise-grade
message handling, event subscription, and conflict resolution capabilities.

This module provides all type definitions, enums, and dataclasses required for
sophisticated coordination protocol management and cross-framework communication.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: coordination_types.py (300 lines)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid


class MessageType(Enum):
    """Types of coordination messages with comprehensive coverage"""
    COMMAND = "command"                      # Executable commands
    QUERY = "query"                         # Information requests
    EVENT = "event"                         # System events and notifications
    RESPONSE = "response"                   # Responses to commands/queries
    HEARTBEAT = "heartbeat"                 # System health indicators
    STATUS_UPDATE = "status_update"        # Status change notifications
    RESOURCE_REQUEST = "resource_request"   # Resource allocation requests
    COORDINATION_REQUEST = "coordination_request"  # Cross-system coordination requests
    EMERGENCY_ALERT = "emergency_alert"     # Critical system alerts
    SYSTEM_NOTIFICATION = "system_notification"  # General system notifications
    DATA_SYNC = "data_sync"                # Data synchronization messages
    CONFIGURATION_UPDATE = "configuration_update"  # Configuration changes
    MONITORING_DATA = "monitoring_data"     # Performance and monitoring data


class MessagePriority(Enum):
    """Message priority levels with numeric ordering"""
    LOW = 1         # Non-critical, background processing
    MEDIUM = 2      # Normal operational messages
    HIGH = 3        # Important system operations
    CRITICAL = 4    # Critical system functions
    EMERGENCY = 5   # Emergency situations requiring immediate attention


class CoordinationPattern(Enum):
    """Coordination communication patterns for different interaction types"""
    REQUEST_RESPONSE = "request_response"   # Synchronous request-response pattern
    PUBLISH_SUBSCRIBE = "publish_subscribe" # Asynchronous pub-sub pattern
    COMMAND_CONTROL = "command_control"     # Command and control pattern
    EVENT_DRIVEN = "event_driven"          # Event-driven architecture pattern
    PIPELINE = "pipeline"                   # Data pipeline processing pattern
    BROADCAST = "broadcast"                 # One-to-many broadcast pattern
    MULTICAST = "multicast"                # Selective multi-recipient pattern
    WORKFLOW = "workflow"                   # Workflow orchestration pattern


class ProtocolType(Enum):
    """Types of coordination protocols for different communication needs"""
    SYNCHRONOUS = "synchronous"             # Blocking, immediate response
    ASYNCHRONOUS = "asynchronous"          # Non-blocking, eventual response
    EVENT_DRIVEN = "event_driven"         # Event-based communication
    STREAM_BASED = "stream_based"          # Continuous data streaming


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving coordination conflicts"""
    PRIORITY_BASED = "priority_based"       # Resolve by message priority
    TIMESTAMP_BASED = "timestamp_based"     # Resolve by timestamp (first wins)
    CONSENSUS_BASED = "consensus_based"     # Resolve by consensus voting
    RESOURCE_BASED = "resource_based"       # Resolve by resource availability
    CUSTOM_RULE = "custom_rule"            # Apply custom resolution logic


class MessageStatus(Enum):
    """Status of messages in the coordination system"""
    PENDING = "pending"                     # Waiting to be processed
    PROCESSING = "processing"               # Currently being processed
    COMPLETED = "completed"                 # Successfully processed
    FAILED = "failed"                      # Processing failed
    RETRYING = "retrying"                  # Being retried after failure
    EXPIRED = "expired"                    # Message has expired
    CANCELLED = "cancelled"                # Message was cancelled


class FrameworkStatus(Enum):
    """Status of registered frameworks in coordination system"""
    ACTIVE = "active"                       # Framework is active and responsive
    INACTIVE = "inactive"                   # Framework is registered but not responding
    DISCONNECTED = "disconnected"          # Framework has disconnected
    MAINTENANCE = "maintenance"             # Framework is in maintenance mode
    ERROR = "error"                        # Framework is experiencing errors


@dataclass
class CoordinationMessage:
    """Comprehensive coordination message structure with enterprise features"""
    message_id: str
    sender_id: str
    recipient_id: str  # Can be specific framework or "broadcast"/"multicast"
    message_type: MessageType
    priority: MessagePriority
    pattern: CoordinationPattern
    payload: Dict[str, Any]
    created_at: datetime
    
    # Optional message attributes
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None  # For request-response correlation
    conversation_id: Optional[str] = None  # For multi-message conversations
    
    # Retry and reliability settings
    retry_count: int = 0
    max_retries: int = 3
    acknowledgment_required: bool = False
    delivery_confirmation: bool = False
    
    # Security and encryption
    encrypted: bool = False
    signature: Optional[str] = None
    
    # Message metadata and tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: MessageStatus = MessageStatus.PENDING
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    response_timeout: Optional[float] = None  # seconds
    
    # Performance tracking
    routing_path: List[str] = field(default_factory=list)
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation with proper serialization"""
        result = asdict(self)
        
        # Convert enums to their values
        result['message_type'] = self.message_type.value
        result['priority'] = self.priority.value
        result['pattern'] = self.pattern.value
        result['status'] = self.status.value
        
        # Convert datetime objects to ISO format
        result['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            result['expires_at'] = self.expires_at.isoformat()
        if self.processing_started_at:
            result['processing_started_at'] = self.processing_started_at.isoformat()
        if self.processing_completed_at:
            result['processing_completed_at'] = self.processing_completed_at.isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinationMessage':
        """Create from dictionary representation with proper deserialization"""
        # Convert enum values back to enums
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        data['pattern'] = CoordinationPattern(data['pattern'])
        data['status'] = MessageStatus(data.get('status', MessageStatus.PENDING.value))
        
        # Convert ISO format back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        if data.get('processing_started_at'):
            data['processing_started_at'] = datetime.fromisoformat(data['processing_started_at'])
        if data.get('processing_completed_at'):
            data['processing_completed_at'] = datetime.fromisoformat(data['processing_completed_at'])
        
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if the message has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def calculate_age(self) -> float:
        """Calculate message age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def calculate_processing_time(self) -> Optional[float]:
        """Calculate total processing time if available"""
        if self.processing_started_at and self.processing_completed_at:
            return (self.processing_completed_at - self.processing_started_at).total_seconds()
        return None
    
    def should_retry(self) -> bool:
        """Determine if message should be retried"""
        return (self.retry_count < self.max_retries and 
                self.status == MessageStatus.FAILED and
                not self.is_expired())
    
    def mark_processing_started(self):
        """Mark message as started processing"""
        self.status = MessageStatus.PROCESSING
        self.processing_started_at = datetime.now()
    
    def mark_processing_completed(self, success: bool = True):
        """Mark message processing as completed"""
        self.status = MessageStatus.COMPLETED if success else MessageStatus.FAILED
        self.processing_completed_at = datetime.now()
        if self.processing_started_at:
            self.processing_time = self.calculate_processing_time()


@dataclass
class ProtocolConfiguration:
    """Configuration for coordination protocols"""
    protocol_id: str
    protocol_type: ProtocolType
    enabled: bool = True
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    max_concurrent_messages: int = 100
    encryption_enabled: bool = False
    compression_enabled: bool = False
    
    # Protocol-specific settings
    heartbeat_interval: float = 30.0  # seconds
    cleanup_interval: float = 300.0   # seconds
    max_message_size: int = 1024 * 1024  # 1MB
    
    # Advanced features
    load_balancing_enabled: bool = False
    circuit_breaker_enabled: bool = False
    rate_limiting_enabled: bool = False
    
    # Performance tuning
    batch_processing_enabled: bool = False
    batch_size: int = 10
    batch_timeout: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate configuration and return any validation errors"""
        errors = []
        
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if self.retry_attempts < 0:
            errors.append("retry_attempts must be non-negative")
        
        if self.retry_delay_seconds < 0:
            errors.append("retry_delay_seconds must be non-negative")
        
        if self.max_concurrent_messages <= 0:
            errors.append("max_concurrent_messages must be positive")
        
        if self.heartbeat_interval <= 0:
            errors.append("heartbeat_interval must be positive")
        
        if self.max_message_size <= 0:
            errors.append("max_message_size must be positive")
        
        return errors


@dataclass
class EventSubscription:
    """Event subscription configuration with advanced filtering"""
    subscription_id: str
    subscriber_id: str
    event_pattern: str  # Can include wildcards like "system.*" or "error.critical.*"
    callback_url: Optional[str] = None
    callback_function: Optional[Callable] = None
    filter_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Subscription management
    active: bool = True
    max_events_per_second: Optional[int] = None
    batch_events: bool = False
    batch_size: int = 1
    batch_timeout: float = 1.0
    
    # Reliability settings
    delivery_guarantee: str = "at_least_once"  # "at_most_once", "at_least_once", "exactly_once"
    max_delivery_attempts: int = 3
    dead_letter_queue_enabled: bool = False
    
    def matches_event_type(self, event_type: str) -> bool:
        """Check if event type matches the subscription pattern"""
        # Simple wildcard matching - in production would use more sophisticated matching
        pattern = self.event_pattern.replace("*", ".*")
        import re
        return bool(re.match(f"^{pattern}$", event_type))
    
    def should_filter_event(self, event_data: Dict[str, Any]) -> bool:
        """Check if event should be filtered out based on criteria"""
        if not self.filter_criteria:
            return False
        
        for key, expected_value in self.filter_criteria.items():
            event_value = event_data.get(key)
            
            if isinstance(expected_value, dict) and "$gt" in expected_value:
                if event_value is None or event_value <= expected_value["$gt"]:
                    return True
            elif isinstance(expected_value, dict) and "$lt" in expected_value:
                if event_value is None or event_value >= expected_value["$lt"]:
                    return True
            elif event_value != expected_value:
                return True
        
        return False
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()


@dataclass
class ConflictResolutionRule:
    """Rule for resolving coordination conflicts"""
    rule_id: str
    rule_name: str
    conflict_pattern: str  # Pattern to match conflicts
    resolution_strategy: ConflictResolutionStrategy
    priority: int = 0  # Higher number = higher priority rule
    enabled: bool = True
    
    # Rule-specific parameters
    custom_logic: Optional[Callable] = None
    timeout_seconds: float = 10.0
    require_consensus_percentage: float = 0.75  # For consensus-based resolution
    
    # Rule metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    application_count: int = 0
    success_rate: float = 0.0
    
    def matches_conflict(self, conflict_info: Dict[str, Any]) -> bool:
        """Check if this rule applies to the given conflict"""
        # Simple pattern matching - in production would be more sophisticated
        conflict_type = conflict_info.get("conflict_type", "")
        return self.conflict_pattern in conflict_type
    
    def record_application(self, success: bool):
        """Record that this rule was applied"""
        self.last_applied = datetime.now()
        self.application_count += 1
        
        # Update success rate with exponential smoothing
        if self.application_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            alpha = 0.1  # Smoothing factor
            new_value = 1.0 if success else 0.0
            self.success_rate = alpha * new_value + (1 - alpha) * self.success_rate


@dataclass
class FrameworkRegistration:
    """Registration information for coordinated frameworks"""
    framework_id: str
    framework_name: str
    framework_type: str  # "analytics", "ml", "api", etc.
    status: FrameworkStatus
    connection_info: Dict[str, Any]
    capabilities: List[str] = field(default_factory=list)
    
    # Registration metadata
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Health and performance metrics
    health_score: float = 1.0  # 0.0 = unhealthy, 1.0 = fully healthy
    response_time_avg: float = 0.0  # Average response time in seconds
    error_rate: float = 0.0  # Error rate as percentage
    message_count: int = 0  # Total messages processed
    
    # Configuration
    max_concurrent_messages: int = 10
    timeout_seconds: float = 30.0
    priority_weight: float = 1.0  # For load balancing
    
    def is_healthy(self) -> bool:
        """Check if framework is considered healthy"""
        return (self.status == FrameworkStatus.ACTIVE and 
                self.health_score > 0.5 and
                (datetime.now() - self.last_heartbeat).total_seconds() < 120)
    
    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = datetime.now()
        if self.status == FrameworkStatus.INACTIVE:
            self.status = FrameworkStatus.ACTIVE
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        self.message_count += 1
    
    def calculate_load_score(self) -> float:
        """Calculate current load score for load balancing"""
        # Higher score means lower load (better for selection)
        base_score = self.priority_weight * self.health_score
        
        # Adjust for response time (faster = better)
        if self.response_time_avg > 0:
            time_factor = 1.0 / (1.0 + self.response_time_avg)
        else:
            time_factor = 1.0
        
        # Adjust for error rate (lower errors = better)
        error_factor = 1.0 - min(self.error_rate, 0.9)
        
        return base_score * time_factor * error_factor


@dataclass
class CoordinationMetrics:
    """Metrics for coordination system performance"""
    total_messages: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    retried_messages: int = 0
    expired_messages: int = 0
    
    # Performance metrics
    average_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    
    # Throughput metrics
    messages_per_second: float = 0.0
    peak_messages_per_second: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Resource utilization
    active_connections: int = 0
    queue_size: int = 0
    memory_usage_bytes: int = 0
    
    # Timing
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_message_metrics(self, success: bool, processing_time: float):
        """Update metrics with new message processing data"""
        self.total_messages += 1
        if success:
            self.successful_messages += 1
        else:
            self.failed_messages += 1
        
        # Update processing time statistics
        if self.min_processing_time == float('inf'):
            self.min_processing_time = processing_time
        else:
            self.min_processing_time = min(self.min_processing_time, processing_time)
        
        self.max_processing_time = max(self.max_processing_time, processing_time)
        
        # Update average with exponential smoothing
        alpha = 0.1
        self.average_processing_time = (alpha * processing_time + 
                                      (1 - alpha) * self.average_processing_time)
        
        # Update error rate
        self.error_rate = (self.failed_messages / self.total_messages) * 100.0
        
        self.last_updated = datetime.now()
    
    def calculate_success_rate(self) -> float:
        """Calculate overall success rate as percentage"""
        if self.total_messages == 0:
            return 100.0
        return (self.successful_messages / self.total_messages) * 100.0


# Export all coordination types and enums
__all__ = [
    'MessageType', 'MessagePriority', 'CoordinationPattern', 'ProtocolType',
    'ConflictResolutionStrategy', 'MessageStatus', 'FrameworkStatus',
    'CoordinationMessage', 'ProtocolConfiguration', 'EventSubscription',
    'ConflictResolutionRule', 'FrameworkRegistration', 'CoordinationMetrics'
]