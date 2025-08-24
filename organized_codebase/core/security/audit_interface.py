"""
Audit System Interface

High-level interface for audit operations with simplified API.
"""

from typing import Dict, List, Any, Optional
import logging
from .audit_events import EventType, EventSeverity, EventFactory
from .audit_storage import AuditStorage
from .audit_reporter import AuditReporter

logger = logging.getLogger(__name__)


class AuditInterface:
    """
    Simplified interface for common audit operations.
    Provides high-level methods for typical audit use cases.
    """
    
    def __init__(self, storage: AuditStorage, reporter: AuditReporter):
        """
        Initialize audit interface.
        
        Args:
            storage: Audit storage instance
            reporter: Audit reporter instance
        """
        try:
            self.storage = storage
            self.reporter = reporter
            self.event_factory = EventFactory()
            logger.info("Audit Interface initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audit interface: {e}")
            raise
    
    def log_user_action(self, 
                       user: str, 
                       action: str, 
                       resource: str, 
                       success: bool,
                       details: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a user action with automatic severity determination.
        
        Args:
            user: Username
            action: Action performed
            resource: Resource accessed
            success: Whether action succeeded
            details: Additional details
            
        Returns:
            Event ID
        """
        try:
            severity = EventSeverity.INFO if success else EventSeverity.HIGH
            event_type = EventType.ACCESS if success else EventType.AUTH_FAILURE
            
            event = self.event_factory.authentication_event(
                user=user,
                success=success,
                ip_address=details.get('ip_address', 'unknown') if details else 'unknown',
                session_id=details.get('session_id', 'unknown') if details else 'unknown',
                details={
                    'action': action,
                    'resource': resource,
                    **(details or {})
                }
            )
            
            return self.storage.store_event(event)
            
        except Exception as e:
            logger.error(f"Failed to log user action: {e}")
            return self._log_error_event("log_user_action", str(e))
    
    def log_system_event(self,
                        component: str,
                        event_description: str,
                        severity: EventSeverity = EventSeverity.INFO,
                        details: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a system event.
        
        Args:
            component: System component
            event_description: Description of event
            severity: Event severity
            details: Additional details
            
        Returns:
            Event ID
        """
        try:
            event = self.event_factory.data_access_event(
                user="system",
                resource=component,
                action="system_event",
                result="logged",
                sensitive=severity in [EventSeverity.CRITICAL, EventSeverity.HIGH],
                session_id=None
            )
            
            # Override with custom details
            event.details.update({
                'description': event_description,
                'component': component,
                **(details or {})
            })
            event.severity = severity
            
            return self.storage.store_event(event)
            
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return self._log_error_event("log_system_event", str(e))
    
    def log_security_incident(self,
                             incident_type: str,
                             severity: EventSeverity,
                             description: str,
                             affected_resources: List[str],
                             details: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a security incident.
        
        Args:
            incident_type: Type of security incident
            severity: Incident severity
            description: Incident description
            affected_resources: List of affected resources
            details: Additional incident details
            
        Returns:
            Event ID
        """
        try:
            event = self.event_factory.security_scan_event(
                scanner="security_monitor",
                target=",".join(affected_resources),
                vulnerabilities_found=1 if severity in [EventSeverity.CRITICAL, EventSeverity.HIGH] else 0,
                scan_type="incident_detection",
                details={
                    'incident_type': incident_type,
                    'description': description,
                    'affected_resources': affected_resources,
                    **(details or {})
                }
            )
            
            event.severity = severity
            event.event_type = EventType.VULNERABILITY_FOUND
            
            return self.storage.store_event(event)
            
        except Exception as e:
            logger.error(f"Failed to log security incident: {e}")
            return self._log_error_event("log_security_incident", str(e))
    
    def get_recent_events(self, 
                         count: int = 50,
                         severity_filter: Optional[EventSeverity] = None) -> List[Dict[str, Any]]:
        """
        Get recent audit events.
        
        Args:
            count: Number of events to retrieve
            severity_filter: Optional severity filter
            
        Returns:
            List of recent events
        """
        try:
            events = self.storage.query_events(severity=severity_filter)
            recent_events = events[-count:] if events else []
            
            return [event.to_dict() for event in recent_events]
            
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get security summary for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Security summary
        """
        try:
            from datetime import datetime, timedelta
            
            start_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            
            security_report = self.reporter.generate_security_report(start_time=start_time)
            
            return {
                'time_period_hours': hours,
                'security_overview': security_report.get('security_overview', {}),
                'threat_indicators': security_report.get('threat_indicators', []),
                'security_score': security_report.get('security_score', 0),
                'recommendations': security_report.get('recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get security summary: {e}")
            return {
                'error': str(e),
                'time_period_hours': hours,
                'security_overview': {},
                'threat_indicators': [],
                'security_score': 0,
                'recommendations': ['Security summary generation failed']
            }
    
    def get_compliance_status(self, standard: str) -> Dict[str, Any]:
        """
        Get compliance status for a specific standard.
        
        Args:
            standard: Compliance standard (e.g., 'GDPR', 'PCI-DSS')
            
        Returns:
            Compliance status report
        """
        try:
            compliance_report = self.reporter.generate_compliance_report(standard=standard)
            
            return {
                'standard': standard,
                'compliance_status': compliance_report.get('compliance_status', {}),
                'violations': compliance_report.get('violations', []),
                'remediation_needed': compliance_report.get('remediation_needed', []),
                'risk_assessment': compliance_report.get('risk_assessment', 'UNKNOWN')
            }
            
        except Exception as e:
            logger.error(f"Failed to get compliance status: {e}")
            return {
                'standard': standard,
                'error': str(e),
                'compliance_status': {},
                'violations': [],
                'remediation_needed': [f'Compliance check failed: {str(e)}'],
                'risk_assessment': 'ERROR'
            }
    
    def search_events(self,
                     query_params: Dict[str, Any],
                     max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search audit events with flexible parameters.
        
        Args:
            query_params: Search parameters (actor, event_type, severity, etc.)
            max_results: Maximum number of results
            
        Returns:
            List of matching events
        """
        try:
            # Extract search parameters
            start_time = query_params.get('start_time')
            end_time = query_params.get('end_time')
            event_type = query_params.get('event_type')
            actor = query_params.get('actor')
            severity = query_params.get('severity')
            
            # Convert string enums if needed
            if event_type and isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except ValueError:
                    event_type = None
                    
            if severity and isinstance(severity, str):
                try:
                    severity = EventSeverity(severity)
                except ValueError:
                    severity = None
            
            events = self.storage.query_events(
                start_time=start_time,
                end_time=end_time,
                event_type=event_type,
                actor=actor,
                severity=severity
            )
            
            # Limit results
            limited_events = events[-max_results:] if events else []
            
            return [event.to_dict() for event in limited_events]
            
        except Exception as e:
            logger.error(f"Failed to search events: {e}")
            return []
    
    def export_audit_data(self, 
                         output_path: str, 
                         format: str = "json",
                         filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export audit data with optional filtering.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv)
            filters: Optional filters to apply
            
        Returns:
            True if export successful
        """
        try:
            if filters:
                # If filters provided, export filtered data
                events = self.search_events(filters, max_results=10000)
                
                # Create temporary filtered export
                if format == "json":
                    import json
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'events': events,
                            'export_info': {
                                'filters_applied': filters,
                                'export_format': format,
                                'total_events': len(events)
                            }
                        }, f, indent=2, ensure_ascii=False)
                    return True
                else:
                    # CSV export for filtered data
                    import csv
                    if events:
                        with open(output_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=events[0].keys())
                            writer.writeheader()
                            writer.writerows(events)
                        return True
                    
            else:
                # Export all data using storage layer
                return self.storage.export_events(output_path, format)
                
        except Exception as e:
            logger.error(f"Failed to export audit data: {e}")
            return False
    
    def _log_error_event(self, operation: str, error: str) -> str:
        """Log an error event when other operations fail."""
        try:
            # Simple error event without recursion risk
            from .audit_events import AuditEvent
            error_event = AuditEvent.create(
                event_type=EventType.FILE_CHANGE,
                severity=EventSeverity.CRITICAL,
                source="audit_interface",
                actor="system",
                resource="audit_system",
                action=operation,
                result="error",
                details={"error": error}
            )
            return self.storage.store_event(error_event)
        except Exception:
            logger.critical(f"Critical audit system failure in {operation}: {error}")
            return "critical_error"