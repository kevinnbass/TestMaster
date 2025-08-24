"""
Audit Event Data Structures

Defines the core data structures for audit logging.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime


class EventType(Enum):
    """Types of audit events."""
    ACCESS = "access"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    PERMISSION_CHANGE = "permission_change"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGE = "config_change"
    SECURITY_SCAN = "security_scan"
    VULNERABILITY_FOUND = "vulnerability_found"
    COMPLIANCE_CHECK = "compliance_check"
    FILE_CHANGE = "file_change"


class EventSeverity(Enum):
    """Severity levels for audit events."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    timestamp: str
    event_type: EventType
    severity: EventSeverity
    source: str
    actor: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    @classmethod
    def create(cls, 
               event_type: EventType,
               severity: EventSeverity,
               source: str,
               actor: str,
               resource: str,
               action: str,
               result: str,
               details: Optional[Dict[str, Any]] = None,
               session_id: Optional[str] = None,
               ip_address: Optional[str] = None) -> 'AuditEvent':
        """Create a new audit event with current timestamp."""
        try:
            return cls(
                timestamp=datetime.utcnow().isoformat() + "Z",
                event_type=event_type,
                severity=severity,
                source=source,
                actor=actor,
                resource=resource,
                action=action,
                result=result,
                details=details or {},
                session_id=session_id,
                ip_address=ip_address
            )
        except Exception as e:
            # Fallback to basic event if creation fails
            return cls(
                timestamp=datetime.utcnow().isoformat() + "Z",
                event_type=EventType.ACCESS,
                severity=EventSeverity.INFO,
                source="unknown",
                actor="system",
                resource="audit_system",
                action="event_creation_error",
                result="error",
                details={"error": str(e), "original_details": details or {}}
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        try:
            event_dict = asdict(self)
            event_dict['event_type'] = self.event_type.value
            event_dict['severity'] = self.severity.value
            return event_dict
        except Exception as e:
            # Return minimal dict if conversion fails
            return {
                'timestamp': self.timestamp,
                'event_type': self.event_type.value if hasattr(self.event_type, 'value') else str(self.event_type),
                'severity': self.severity.value if hasattr(self.severity, 'value') else str(self.severity),
                'source': self.source,
                'actor': self.actor,
                'resource': self.resource,
                'action': self.action,
                'result': self.result,
                'details': {'serialization_error': str(e)},
                'session_id': self.session_id,
                'ip_address': self.ip_address
            }


class EventFactory:
    """Factory for creating common audit events."""
    
    @staticmethod
    def authentication_event(user: str, 
                           success: bool, 
                           ip_address: str, 
                           session_id: str,
                           details: Optional[Dict] = None) -> AuditEvent:
        """Create authentication audit event."""
        try:
            return AuditEvent.create(
                event_type=EventType.AUTH_SUCCESS if success else EventType.AUTH_FAILURE,
                severity=EventSeverity.INFO if success else EventSeverity.HIGH,
                source="authentication_system",
                actor=user,
                resource="auth_endpoint",
                action="login",
                result="success" if success else "failure",
                details=details,
                session_id=session_id,
                ip_address=ip_address
            )
        except Exception as e:
            return AuditEvent.create(
                event_type=EventType.AUTH_FAILURE,
                severity=EventSeverity.CRITICAL,
                source="authentication_system",
                actor=user or "unknown",
                resource="auth_endpoint",
                action="login",
                result="error",
                details={"creation_error": str(e), "original_details": details}
            )
    
    @staticmethod
    def data_access_event(user: str,
                         resource: str,
                         action: str,
                         result: str,
                         sensitive: bool = False,
                         session_id: Optional[str] = None) -> AuditEvent:
        """Create data access audit event."""
        try:
            severity = EventSeverity.HIGH if sensitive else EventSeverity.MEDIUM
            
            return AuditEvent.create(
                event_type=EventType.DATA_ACCESS,
                severity=severity,
                source="data_access_layer",
                actor=user,
                resource=resource,
                action=action,
                result=result,
                details={"sensitive": sensitive},
                session_id=session_id
            )
        except Exception as e:
            return AuditEvent.create(
                event_type=EventType.DATA_ACCESS,
                severity=EventSeverity.CRITICAL,
                source="data_access_layer",
                actor=user or "unknown",
                resource=resource or "unknown",
                action=action or "unknown",
                result="error",
                details={"creation_error": str(e), "sensitive": sensitive}
            )
    
    @staticmethod
    def security_scan_event(scanner: str,
                           target: str,
                           vulnerabilities_found: int,
                           scan_type: str,
                           details: Optional[Dict] = None) -> AuditEvent:
        """Create security scan audit event."""
        try:
            severity = EventSeverity.HIGH if vulnerabilities_found > 0 else EventSeverity.INFO
            
            return AuditEvent.create(
                event_type=EventType.SECURITY_SCAN,
                severity=severity,
                source="security_scanner",
                actor=scanner,
                resource=target,
                action=f"{scan_type}_scan",
                result="completed",
                details={
                    "vulnerabilities_found": vulnerabilities_found,
                    "scan_type": scan_type,
                    **(details or {})
                }
            )
        except Exception as e:
            return AuditEvent.create(
                event_type=EventType.SECURITY_SCAN,
                severity=EventSeverity.CRITICAL,
                source="security_scanner",
                actor=scanner or "unknown",
                resource=target or "unknown",
                action="scan_error",
                result="error",
                details={"creation_error": str(e), "original_details": details}
            )
    
    @staticmethod
    def compliance_check_event(checker: str,
                              standard: str,
                              passed: bool,
                              issues_found: int,
                              details: Optional[Dict] = None) -> AuditEvent:
        """Create compliance check audit event."""
        try:
            severity = EventSeverity.MEDIUM if passed else EventSeverity.HIGH
            
            return AuditEvent.create(
                event_type=EventType.COMPLIANCE_CHECK,
                severity=severity,
                source="compliance_checker",
                actor=checker,
                resource=standard,
                action="compliance_check",
                result="passed" if passed else "failed",
                details={
                    "issues_found": issues_found,
                    "standard": standard,
                    **(details or {})
                }
            )
        except Exception as e:
            return AuditEvent.create(
                event_type=EventType.COMPLIANCE_CHECK,
                severity=EventSeverity.CRITICAL,
                source="compliance_checker",
                actor=checker or "unknown",
                resource=standard or "unknown",
                action="compliance_check_error",
                result="error",
                details={"creation_error": str(e), "original_details": details}
            )