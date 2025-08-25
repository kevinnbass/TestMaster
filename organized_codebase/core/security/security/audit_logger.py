"""
Streamlined Audit Logger Orchestrator

Main interface for the audit logging system - now modularized for simplicity.
"""

from typing import Dict, List, Any, Optional
import logging
from .audit import (
    AuditEvent, EventType, EventSeverity,
    AuditStorage, AuditReporter
)
from .audit.audit_interface import AuditInterface

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Streamlined audit logging system orchestrator.
    Provides enterprise-grade audit capabilities through modular components.
    """
    
    def __init__(self, log_file: str = "audit.log"):
        """
        Initialize the audit logger with modular components.
        
        Args:
            log_file: Path to audit log file
        """
        try:
            # Initialize core components
            self.storage = AuditStorage(log_file)
            self.reporter = AuditReporter(self.storage)
            self.interface = AuditInterface(self.storage, self.reporter)
            
            logger.info(f"Audit Logger orchestrator initialized: {log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize audit logger: {e}")
            # Initialize with fallback configuration
            try:
                self.storage = AuditStorage("fallback_audit.log")
                self.reporter = AuditReporter(self.storage)
                self.interface = AuditInterface(self.storage, self.reporter)
                logger.warning("Using fallback audit configuration")
            except Exception as critical_error:
                logger.critical(f"Critical audit logger initialization failure: {critical_error}")
                raise
    
    # High-level interface methods
    def log_user_action(self, user: str, action: str, resource: str, success: bool, **kwargs) -> str:
        """Log user action via interface."""
        try:
            return self.interface.log_user_action(user, action, resource, success, kwargs)
        except Exception as e:
            logger.error(f"Failed to log user action: {e}")
            return "error"
    
    def log_system_event(self, component: str, description: str, severity: EventSeverity = EventSeverity.INFO, **kwargs) -> str:
        """Log system event via interface."""
        try:
            return self.interface.log_system_event(component, description, severity, kwargs)
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return "error"
    
    def log_security_incident(self, incident_type: str, severity: EventSeverity, description: str, affected_resources: List[str], **kwargs) -> str:
        """Log security incident via interface."""
        try:
            return self.interface.log_security_incident(incident_type, severity, description, affected_resources, kwargs)
        except Exception as e:
            logger.error(f"Failed to log security incident: {e}")
            return "error"
    
    # Legacy compatibility methods (delegate to storage/reporter)
    def log_event(self, event_type: EventType, severity: EventSeverity, source: str, actor: str, 
                  resource: str, action: str, result: str, details: Optional[Dict[str, Any]] = None,
                  session_id: Optional[str] = None, ip_address: Optional[str] = None) -> str:
        """Legacy log_event method - delegates to storage."""
        try:
            event = AuditEvent.create(
                event_type=event_type, severity=severity, source=source, actor=actor,
                resource=resource, action=action, result=result, details=details,
                session_id=session_id, ip_address=ip_address
            )
            return self.storage.store_event(event)
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            return "error"
    
    def log_authentication(self, user: str, success: bool, ip_address: str, session_id: str, details: Optional[Dict] = None) -> str:
        """Legacy authentication logging."""
        try:
            return self.interface.log_user_action(
                user=user, 
                action="authentication", 
                resource="auth_endpoint", 
                success=success,
                ip_address=ip_address,
                session_id=session_id,
                **(details or {})
            )
        except Exception as e:
            logger.error(f"Failed to log authentication: {e}")
            return "error"
    
    def log_data_access(self, user: str, resource: str, action: str, result: str, 
                       sensitive: bool = False, session_id: Optional[str] = None) -> str:
        """Legacy data access logging."""
        try:
            return self.interface.log_user_action(
                user=user,
                action=action,
                resource=resource,
                success=(result == "success"),
                sensitive=sensitive,
                session_id=session_id,
                result=result
            )
        except Exception as e:
            logger.error(f"Failed to log data access: {e}")
            return "error"
    
    def log_security_scan(self, scanner: str, target: str, vulnerabilities_found: int, 
                         scan_type: str, details: Optional[Dict] = None) -> str:
        """Legacy security scan logging."""
        try:
            return self.interface.log_system_event(
                component=scanner,
                description=f"Security scan completed: {scan_type}",
                severity=EventSeverity.HIGH if vulnerabilities_found > 0 else EventSeverity.INFO,
                target=target,
                vulnerabilities_found=vulnerabilities_found,
                scan_type=scan_type,
                **(details or {})
            )
        except Exception as e:
            logger.error(f"Failed to log security scan: {e}")
            return "error"
    
    def log_compliance_check(self, checker: str, standard: str, passed: bool, 
                           issues_found: int, details: Optional[Dict] = None) -> str:
        """Legacy compliance check logging."""
        try:
            return self.interface.log_system_event(
                component=checker,
                description=f"Compliance check: {standard}",
                severity=EventSeverity.MEDIUM if passed else EventSeverity.HIGH,
                standard=standard,
                passed=passed,
                issues_found=issues_found,
                **(details or {})
            )
        except Exception as e:
            logger.error(f"Failed to log compliance check: {e}")
            return "error"
    
    # Query and reporting methods (delegate to components)
    def query_events(self, start_time: Optional[str] = None, end_time: Optional[str] = None,
                    event_type: Optional[EventType] = None, actor: Optional[str] = None,
                    severity: Optional[EventSeverity] = None) -> List[AuditEvent]:
        """Query events - delegates to storage."""
        try:
            return self.storage.query_events(start_time, end_time, event_type, actor, severity)
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            return []
    
    def generate_audit_report(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Generate audit report - delegates to reporter."""
        try:
            return self.reporter.generate_summary_report(start_time, end_time)
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
            return {'error': str(e)}
    
    def generate_compliance_report(self, standard: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Generate compliance report - delegates to reporter."""
        try:
            return self.reporter.generate_compliance_report(standard, start_time, end_time)
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {'error': str(e)}
    
    def generate_security_report(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Generate security report - delegates to reporter."""
        try:
            return self.reporter.generate_security_report(start_time, end_time)
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
            return {'error': str(e)}
    
    # Utility methods (delegate to appropriate components)
    def verify_integrity(self) -> bool:
        """Verify audit log integrity - delegates to storage."""
        try:
            return self.storage.verify_integrity()
        except Exception as e:
            logger.error(f"Failed to verify integrity: {e}")
            return False
    
    def export_audit_log(self, output_path: str, format: str = "json") -> bool:
        """Export audit log - delegates to interface for enhanced functionality."""
        try:
            return self.interface.export_audit_data(output_path, format)
        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics."""
        try:
            total_events = self.storage.get_event_count()
            integrity_status = self.storage.verify_integrity()
            
            return {
                'total_events': total_events,
                'integrity_verified': integrity_status,
                'system_health': 'healthy' if integrity_status else 'compromised',
                'components_active': {
                    'storage': True,
                    'reporter': True,
                    'interface': True
                }
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'total_events': 0,
                'integrity_verified': False,
                'system_health': 'error',
                'error': str(e)
            }
    
    # New enhanced methods via interface
    def get_recent_events(self, count: int = 50, severity_filter: Optional[EventSeverity] = None) -> List[Dict[str, Any]]:
        """Get recent events - uses interface."""
        try:
            return self.interface.get_recent_events(count, severity_filter)
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary - uses interface."""
        try:
            return self.interface.get_security_summary(hours)
        except Exception as e:
            logger.error(f"Failed to get security summary: {e}")
            return {'error': str(e)}
    
    def get_compliance_status(self, standard: str) -> Dict[str, Any]:
        """Get compliance status - uses interface."""
        try:
            return self.interface.get_compliance_status(standard)
        except Exception as e:
            logger.error(f"Failed to get compliance status: {e}")
            return {'error': str(e)}
    
    def search_events(self, query_params: Dict[str, Any], max_results: int = 100) -> List[Dict[str, Any]]:
        """Search events - uses interface."""
        try:
            return self.interface.search_events(query_params, max_results)
        except Exception as e:
            logger.error(f"Failed to search events: {e}")
            return []