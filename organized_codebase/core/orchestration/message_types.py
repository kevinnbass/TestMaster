"""
Message Types and Data Structures for Coordination

This module defines the core message types, enumerations, and data structures
used throughout the coordination protocol system.

Author: Agent B - Orchestration & Workflow Specialist  
Created: 2025-01-22
"""

import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable


class MessageType(Enum):
    """Types of coordination messages"""
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    COORDINATION_REQUEST = "coordination_request"
    EMERGENCY_ALERT = "emergency_alert"
    SYSTEM_NOTIFICATION = "system_notification"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class CoordinationPattern(Enum):
    """Coordination communication patterns"""
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    COMMAND_CONTROL = "command_control"
    EVENT_DRIVEN = "event_driven"
    PIPELINE = "pipeline"
    BROADCAST = "broadcast"


class ProtocolType(Enum):
    """Types of coordination protocols"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    EVENT_DRIVEN = "event_driven"
    STREAM_BASED = "stream_based"


@dataclass
class CoordinationMessage:
    """Coordination message structure"""
    message_id: str
    sender_id: str
    recipient_id: str  # Can be specific framework or "broadcast"
    message_type: MessageType
    priority: MessagePriority
    pattern: CoordinationPattern
    payload: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None  # For request-response correlation
    retry_count: int = 0
    max_retries: int = 3
    acknowledgment_required: bool = False
    encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            **asdict(self),
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'pattern': self.pattern.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class ProtocolConfiguration:
    """Configuration for coordination protocols"""
    protocol_type: ProtocolType
    pattern: CoordinationPattern
    timeout: timedelta
    retry_strategy: str  # 'exponential', 'linear', 'fixed'
    max_retries: int
    acknowledgment_required: bool
    encryption_enabled: bool
    compression_enabled: bool
    batch_size: int = 1
    buffer_size: int = 1000
    priority_queuing: bool = True
    dead_letter_queue: bool = True
    metrics_enabled: bool = True


@dataclass
class EventSubscription:
    """Event subscription information"""
    subscription_id: str
    subscriber_id: str
    event_pattern: str  # Pattern to match events (supports wildcards)
    callback: Optional[Callable[[CoordinationMessage], None]]
    filter_criteria: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConflictResolutionRule:
    """Rule for resolving coordination conflicts"""
    rule_id: str
    conflict_type: str
    resolution_strategy: str  # 'priority', 'timestamp', 'consensus', 'custom'
    priority_weights: Dict[str, float] = field(default_factory=dict)
    custom_resolver: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)