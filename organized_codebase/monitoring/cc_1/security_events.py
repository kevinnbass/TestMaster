#!/usr/bin/env python3
"""
Security Events Module
Agent D Hour 5 - Modularized Security Event Management

Handles security event data structures, threat levels, and response actions
following STEELCLAD Anti-Regression Modularization Protocol.
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import datetime

class ThreatLevel(Enum):
    """Threat severity levels for security classification"""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

    @property
    def severity_score(self) -> int:
        """Numerical severity score for comparison"""
        scores = {
            ThreatLevel.INFO: 1,
            ThreatLevel.LOW: 2,
            ThreatLevel.MEDIUM: 3,
            ThreatLevel.HIGH: 4,
            ThreatLevel.CRITICAL: 5,
            ThreatLevel.EMERGENCY: 6
        }
        return scores[self]

class ResponseAction(Enum):
    """Automated response actions for security events"""
    LOG_ONLY = "LOG_ONLY"
    ALERT = "ALERT"
    QUARANTINE = "QUARANTINE"
    BLOCK = "BLOCK"
    RESTART_SERVICE = "RESTART_SERVICE"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"
    
    @property
    def escalation_level(self) -> int:
        """Escalation level for response prioritization"""
        levels = {
            ResponseAction.LOG_ONLY: 1,
            ResponseAction.ALERT: 2,
            ResponseAction.QUARANTINE: 3,
            ResponseAction.BLOCK: 4,
            ResponseAction.RESTART_SERVICE: 5,
            ResponseAction.EMERGENCY_SHUTDOWN: 6
        }
        return levels[self]

@dataclass
class SecurityEvent:
    """Security event data structure with enhanced metadata"""
    timestamp: str
    event_type: str
    threat_level: ThreatLevel
    source_file: str
    description: str
    evidence: Dict[str, Any]
    response_action: ResponseAction
    resolved: bool = False
    resolution_time: Optional[str] = None
    event_id: Optional[str] = None
    correlation_id: Optional[str] = None
    severity_score: Optional[int] = None
    
    def __post_init__(self):
        """Initialize computed fields after dataclass creation"""
        if self.severity_score is None:
            self.severity_score = self.threat_level.severity_score
        if self.event_id is None:
            # Generate unique event ID based on timestamp and source
            import hashlib
            hash_input = f"{self.timestamp}_{self.source_file}_{self.event_type}"
            self.event_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security event to dictionary for serialization"""
        data = asdict(self)
        data['threat_level'] = self.threat_level.value
        data['response_action'] = self.response_action.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityEvent':
        """Create SecurityEvent from dictionary"""
        data['threat_level'] = ThreatLevel(data['threat_level'])
        data['response_action'] = ResponseAction(data['response_action'])
        return cls(**data)
    
    def should_escalate(self, current_time: datetime.datetime, 
                       escalation_timeout: int = 300) -> bool:
        """Check if event should be escalated based on resolution time"""
        if self.resolved:
            return False
            
        event_time = datetime.datetime.fromisoformat(self.timestamp)
        time_elapsed = (current_time - event_time).total_seconds()
        
        # Escalate critical events after 5 minutes without resolution
        if self.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
            return time_elapsed > escalation_timeout
        
        # Escalate high events after 10 minutes
        if self.threat_level == ThreatLevel.HIGH:
            return time_elapsed > (escalation_timeout * 2)
            
        return False

@dataclass
class SecurityMetrics:
    """Security metrics aggregation for dashboard reporting"""
    total_events: int = 0
    critical_events: int = 0
    high_events: int = 0
    medium_events: int = 0
    low_events: int = 0
    resolved_events: int = 0
    unresolved_events: int = 0
    average_resolution_time: float = 0.0
    threat_types: Dict[str, int] = None
    response_effectiveness: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default dictionaries"""
        if self.threat_types is None:
            self.threat_types = {}
        if self.response_effectiveness is None:
            self.response_effectiveness = {}
    
    def update_from_event(self, event: SecurityEvent):
        """Update metrics based on security event"""
        self.total_events += 1
        
        # Count by threat level
        if event.threat_level == ThreatLevel.CRITICAL:
            self.critical_events += 1
        elif event.threat_level == ThreatLevel.HIGH:
            self.high_events += 1
        elif event.threat_level == ThreatLevel.MEDIUM:
            self.medium_events += 1
        elif event.threat_level == ThreatLevel.LOW:
            self.low_events += 1
            
        # Count resolved/unresolved
        if event.resolved:
            self.resolved_events += 1
        else:
            self.unresolved_events += 1
            
        # Count threat types
        if event.event_type in self.threat_types:
            self.threat_types[event.event_type] += 1
        else:
            self.threat_types[event.event_type] = 1
    
    def calculate_resolution_rate(self) -> float:
        """Calculate percentage of resolved events"""
        if self.total_events == 0:
            return 0.0
        return (self.resolved_events / self.total_events) * 100

class SecurityEventProcessor:
    """Security event processing and validation utilities"""
    
    @staticmethod
    def validate_event(event: SecurityEvent) -> tuple[bool, str]:
        """Validate security event data integrity"""
        if not event.timestamp:
            return False, "Missing timestamp"
        if not event.event_type:
            return False, "Missing event type"
        if not event.source_file:
            return False, "Missing source file"
        if not event.description:
            return False, "Missing description"
        if not isinstance(event.evidence, dict):
            return False, "Evidence must be dictionary"
            
        # Validate timestamp format
        try:
            datetime.datetime.fromisoformat(event.timestamp)
        except ValueError:
            return False, "Invalid timestamp format"
            
        return True, "Valid event"
    
    @staticmethod
    def correlate_events(events: List[SecurityEvent], 
                        time_window: int = 300) -> Dict[str, List[SecurityEvent]]:
        """Correlate related security events within time window"""
        correlations = {}
        
        for event in events:
            correlation_key = f"{event.event_type}_{event.source_file}"
            if correlation_key not in correlations:
                correlations[correlation_key] = []
            correlations[correlation_key].append(event)
        
        # Filter correlations with multiple events
        return {k: v for k, v in correlations.items() if len(v) > 1}
    
    @staticmethod
    def prioritize_events(events: List[SecurityEvent]) -> List[SecurityEvent]:
        """Sort events by priority (threat level + escalation needs)"""
        def priority_score(event: SecurityEvent) -> int:
            score = event.threat_level.severity_score * 10
            
            # Add urgency bonus for unresolved critical events
            if not event.resolved and event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
                score += 50
                
            # Add bonus for events requiring escalated response
            if event.response_action.escalation_level >= 4:
                score += 20
                
            return score
        
        return sorted(events, key=priority_score, reverse=True)