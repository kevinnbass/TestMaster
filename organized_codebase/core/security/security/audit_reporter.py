"""
Audit Report Generator

Generates comprehensive audit reports and analytics.
"""

from typing import Dict, List, Any, Optional
import logging
from .audit_events import AuditEvent, EventType, EventSeverity
from .audit_storage import AuditStorage

logger = logging.getLogger(__name__)


class AuditReporter:
    """
    Generates audit reports and analytics from stored audit events.
    Provides insights for compliance and security monitoring.
    """
    
    def __init__(self, storage: AuditStorage):
        """
        Initialize audit reporter.
        
        Args:
            storage: Audit storage instance
        """
        try:
            self.storage = storage
            logger.info("Audit Reporter initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audit reporter: {e}")
            raise
    
    def generate_summary_report(self,
                               start_time: Optional[str] = None,
                               end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive audit summary report.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Summary report dictionary
        """
        try:
            events = self.storage.query_events(start_time, end_time)
            
            if not events:
                return self._empty_report(start_time, end_time)
            
            # Calculate event statistics
            type_counts = self._count_by_type(events)
            severity_counts = self._count_by_severity(events)
            actor_stats = self._analyze_actors(events)
            
            # Generate security insights
            security_insights = self._generate_security_insights(events)
            
            report = {
                'report_period': {
                    'start': start_time or 'beginning',
                    'end': end_time or 'now'
                },
                'summary': {
                    'total_events': len(events),
                    'unique_actors': len(actor_stats),
                    'integrity_verified': self.storage.verify_integrity()
                },
                'event_breakdown': {
                    'by_type': type_counts,
                    'by_severity': severity_counts
                },
                'security_metrics': security_insights,
                'top_actors': sorted(actor_stats.items(), 
                                   key=lambda x: x[1]['event_count'], 
                                   reverse=True)[:10],
                'critical_events': self._get_critical_events(events),
                'recommendations': self._generate_recommendations(events)
            }
            
            logger.info(f"Generated summary report for {len(events)} events")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return self._error_report(str(e), start_time, end_time)
    
    def generate_compliance_report(self,
                                  standard: str,
                                  start_time: Optional[str] = None,
                                  end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate compliance-focused audit report.
        
        Args:
            standard: Compliance standard (e.g., 'GDPR', 'PCI-DSS')
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Compliance report dictionary
        """
        try:
            events = self.storage.query_events(start_time, end_time)
            
            # Filter for compliance-relevant events
            compliance_events = [
                e for e in events 
                if e.event_type in [EventType.COMPLIANCE_CHECK, EventType.DATA_ACCESS, 
                                  EventType.PERMISSION_CHANGE, EventType.AUTH_FAILURE]
            ]
            
            # Analyze compliance posture
            compliance_checks = [e for e in events if e.event_type == EventType.COMPLIANCE_CHECK]
            data_access_events = [e for e in events if e.event_type == EventType.DATA_ACCESS]
            auth_failures = [e for e in events if e.event_type == EventType.AUTH_FAILURE]
            
            report = {
                'standard': standard,
                'report_period': {
                    'start': start_time or 'beginning',
                    'end': end_time or 'now'
                },
                'compliance_status': {
                    'total_checks': len(compliance_checks),
                    'passed_checks': len([e for e in compliance_checks if e.result == 'passed']),
                    'failed_checks': len([e for e in compliance_checks if e.result == 'failed']),
                    'compliance_rate': self._calculate_compliance_rate(compliance_checks)
                },
                'data_protection': {
                    'total_data_access': len(data_access_events),
                    'sensitive_data_access': len([e for e in data_access_events 
                                                if e.details.get('sensitive', False)]),
                    'unauthorized_attempts': len(auth_failures)
                },
                'violations': self._identify_violations(compliance_events, standard),
                'remediation_needed': self._identify_remediation_needs(compliance_events),
                'risk_assessment': self._assess_compliance_risk(compliance_events)
            }
            
            logger.info(f"Generated compliance report for {standard}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {'error': str(e), 'standard': standard}
    
    def generate_security_report(self,
                               start_time: Optional[str] = None,
                               end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate security-focused audit report.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Security report dictionary
        """
        try:
            events = self.storage.query_events(start_time, end_time)
            
            # Filter for security events
            security_events = [
                e for e in events 
                if e.event_type in [EventType.SECURITY_SCAN, EventType.VULNERABILITY_FOUND,
                                  EventType.AUTH_FAILURE, EventType.PERMISSION_CHANGE]
            ]
            
            # Analyze security posture
            scan_results = self._analyze_security_scans(events)
            threat_indicators = self._identify_threat_indicators(events)
            access_patterns = self._analyze_access_patterns(events)
            
            report = {
                'report_period': {
                    'start': start_time or 'beginning',
                    'end': end_time or 'now'
                },
                'security_overview': {
                    'total_security_events': len(security_events),
                    'scans_performed': len([e for e in events if e.event_type == EventType.SECURITY_SCAN]),
                    'vulnerabilities_found': sum(e.details.get('vulnerabilities_found', 0) 
                                               for e in events if e.event_type == EventType.SECURITY_SCAN),
                    'failed_authentications': len([e for e in events if e.event_type == EventType.AUTH_FAILURE])
                },
                'scan_results': scan_results,
                'threat_indicators': threat_indicators,
                'access_analysis': access_patterns,
                'security_score': self._calculate_security_score(security_events),
                'recommendations': self._generate_security_recommendations(security_events)
            }
            
            logger.info(f"Generated security report for {len(security_events)} security events")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
            return {'error': str(e)}
    
    # Private helper methods
    def _count_by_type(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by type."""
        try:
            counts = {}
            for event in events:
                event_type = event.event_type.value
                counts[event_type] = counts.get(event_type, 0) + 1
            return counts
        except Exception as e:
            logger.error(f"Error counting by type: {e}")
            return {}
    
    def _count_by_severity(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by severity."""
        try:
            counts = {}
            for event in events:
                severity = event.severity.value
                counts[severity] = counts.get(severity, 0) + 1
            return counts
        except Exception as e:
            logger.error(f"Error counting by severity: {e}")
            return {}
    
    def _analyze_actors(self, events: List[AuditEvent]) -> Dict[str, Dict[str, Any]]:
        """Analyze actor activity patterns."""
        try:
            actors = {}
            for event in events:
                actor = event.actor
                if actor not in actors:
                    actors[actor] = {
                        'event_count': 0,
                        'last_activity': event.timestamp,
                        'event_types': set(),
                        'severity_counts': {}
                    }
                
                actors[actor]['event_count'] += 1
                actors[actor]['event_types'].add(event.event_type.value)
                
                severity = event.severity.value
                actors[actor]['severity_counts'][severity] = \
                    actors[actor]['severity_counts'].get(severity, 0) + 1
                
                # Update last activity if this event is more recent
                if event.timestamp > actors[actor]['last_activity']:
                    actors[actor]['last_activity'] = event.timestamp
            
            # Convert sets to lists for JSON serialization
            for actor_data in actors.values():
                actor_data['event_types'] = list(actor_data['event_types'])
            
            return actors
        except Exception as e:
            logger.error(f"Error analyzing actors: {e}")
            return {}
    
    def _generate_security_insights(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate security insights from events."""
        try:
            failed_auths = [e for e in events if e.event_type == EventType.AUTH_FAILURE]
            data_access = [e for e in events if e.event_type == EventType.DATA_ACCESS]
            security_scans = [e for e in events if e.event_type == EventType.SECURITY_SCAN]
            
            return {
                'authentication_health': {
                    'failed_attempts': len(failed_auths),
                    'unique_failed_users': len(set(e.actor for e in failed_auths)),
                    'brute_force_indicators': len(failed_auths) > 10  # Simple heuristic
                },
                'data_access_patterns': {
                    'total_access_events': len(data_access),
                    'sensitive_access': len([e for e in data_access 
                                           if e.details.get('sensitive', False)]),
                    'unique_data_actors': len(set(e.actor for e in data_access))
                },
                'vulnerability_status': {
                    'scans_performed': len(security_scans),
                    'total_vulnerabilities': sum(e.details.get('vulnerabilities_found', 0) 
                                               for e in security_scans)
                }
            }
        except Exception as e:
            logger.error(f"Error generating security insights: {e}")
            return {}
    
    def _get_critical_events(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Get critical severity events."""
        try:
            critical_events = [
                e.to_dict() for e in events 
                if e.severity == EventSeverity.CRITICAL
            ]
            return critical_events[:20]  # Limit to most recent 20
        except Exception as e:
            logger.error(f"Error getting critical events: {e}")
            return []
    
    def _generate_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """Generate recommendations based on event analysis."""
        try:
            recommendations = []
            
            # Check for common issues
            failed_auths = len([e for e in events if e.event_type == EventType.AUTH_FAILURE])
            if failed_auths > 10:
                recommendations.append("High number of authentication failures detected - review security policies")
            
            critical_events = len([e for e in events if e.severity == EventSeverity.CRITICAL])
            if critical_events > 0:
                recommendations.append(f"{critical_events} critical events require immediate attention")
            
            data_access = [e for e in events if e.event_type == EventType.DATA_ACCESS]
            sensitive_access = len([e for e in data_access if e.details.get('sensitive', False)])
            if sensitive_access > len(data_access) * 0.5:
                recommendations.append("High proportion of sensitive data access - review access controls")
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def _empty_report(self, start_time: Optional[str], end_time: Optional[str]) -> Dict[str, Any]:
        """Generate empty report structure."""
        return {
            'report_period': {'start': start_time or 'beginning', 'end': end_time or 'now'},
            'summary': {'total_events': 0, 'unique_actors': 0, 'integrity_verified': True},
            'event_breakdown': {'by_type': {}, 'by_severity': {}},
            'security_metrics': {},
            'top_actors': [],
            'critical_events': [],
            'recommendations': ['No events found in specified time period']
        }
    
    def _error_report(self, error: str, start_time: Optional[str], end_time: Optional[str]) -> Dict[str, Any]:
        """Generate error report structure."""
        return {
            'error': error,
            'report_period': {'start': start_time or 'beginning', 'end': end_time or 'now'},
            'summary': {'total_events': 0, 'error': True},
            'recommendations': ['Report generation failed - check audit system health']
        }
    
    def _calculate_compliance_rate(self, compliance_events: List[AuditEvent]) -> float:
        """Calculate compliance rate from events."""
        try:
            if not compliance_events:
                return 100.0
            
            passed = len([e for e in compliance_events if e.result == 'passed'])
            return (passed / len(compliance_events)) * 100.0
        except Exception as e:
            logger.error(f"Error calculating compliance rate: {e}")
            return 0.0
    
    def _identify_violations(self, events: List[AuditEvent], standard: str) -> List[Dict[str, Any]]:
        """Identify compliance violations."""
        try:
            violations = []
            failed_checks = [e for e in events 
                           if e.event_type == EventType.COMPLIANCE_CHECK and e.result == 'failed']
            
            for event in failed_checks:
                violations.append({
                    'timestamp': event.timestamp,
                    'standard': standard,
                    'violation_type': event.details.get('violation_type', 'unknown'),
                    'severity': event.severity.value,
                    'description': event.details.get('description', 'Compliance check failed')
                })
            
            return violations
        except Exception as e:
            logger.error(f"Error identifying violations: {e}")
            return []
    
    def _identify_remediation_needs(self, events: List[AuditEvent]) -> List[str]:
        """Identify remediation needs."""
        try:
            needs = []
            failed_checks = [e for e in events 
                           if e.event_type == EventType.COMPLIANCE_CHECK and e.result == 'failed']
            
            if failed_checks:
                needs.append(f"{len(failed_checks)} failed compliance checks require remediation")
            
            return needs
        except Exception as e:
            logger.error(f"Error identifying remediation needs: {e}")
            return []
    
    def _assess_compliance_risk(self, events: List[AuditEvent]) -> str:
        """Assess overall compliance risk level."""
        try:
            critical_events = len([e for e in events if e.severity == EventSeverity.CRITICAL])
            high_events = len([e for e in events if e.severity == EventSeverity.HIGH])
            
            if critical_events > 0:
                return "HIGH"
            elif high_events > 5:
                return "MEDIUM"
            else:
                return "LOW"
        except Exception as e:
            logger.error(f"Error assessing compliance risk: {e}")
            return "UNKNOWN"
    
    def _analyze_security_scans(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze security scan results."""
        try:
            scan_events = [e for e in events if e.event_type == EventType.SECURITY_SCAN]
            
            if not scan_events:
                return {'scans_performed': 0, 'vulnerabilities_found': 0}
            
            total_vulns = sum(e.details.get('vulnerabilities_found', 0) for e in scan_events)
            scan_types = list(set(e.details.get('scan_type', 'unknown') for e in scan_events))
            
            return {
                'scans_performed': len(scan_events),
                'vulnerabilities_found': total_vulns,
                'scan_types': scan_types,
                'avg_vulnerabilities_per_scan': total_vulns / len(scan_events) if scan_events else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing security scans: {e}")
            return {}
    
    def _identify_threat_indicators(self, events: List[AuditEvent]) -> List[str]:
        """Identify potential threat indicators."""
        try:
            indicators = []
            
            # Multiple failed authentications
            failed_auths = [e for e in events if e.event_type == EventType.AUTH_FAILURE]
            if len(failed_auths) > 10:
                indicators.append("Multiple authentication failures detected")
            
            # High number of vulnerabilities
            total_vulns = sum(e.details.get('vulnerabilities_found', 0) 
                            for e in events if e.event_type == EventType.SECURITY_SCAN)
            if total_vulns > 20:
                indicators.append("High number of vulnerabilities discovered")
            
            return indicators
        except Exception as e:
            logger.error(f"Error identifying threat indicators: {e}")
            return []
    
    def _analyze_access_patterns(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze access patterns."""
        try:
            access_events = [e for e in events if e.event_type == EventType.DATA_ACCESS]
            
            if not access_events:
                return {'total_access': 0}
            
            unique_actors = len(set(e.actor for e in access_events))
            unique_resources = len(set(e.resource for e in access_events))
            
            return {
                'total_access': len(access_events),
                'unique_actors': unique_actors,
                'unique_resources': unique_resources,
                'avg_access_per_actor': len(access_events) / unique_actors if unique_actors else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing access patterns: {e}")
            return {}
    
    def _calculate_security_score(self, events: List[AuditEvent]) -> int:
        """Calculate overall security score (0-100)."""
        try:
            base_score = 100
            
            # Deduct points for security issues
            critical_events = len([e for e in events if e.severity == EventSeverity.CRITICAL])
            high_events = len([e for e in events if e.severity == EventSeverity.HIGH])
            
            base_score -= critical_events * 20  # -20 points per critical event
            base_score -= high_events * 10      # -10 points per high event
            
            return max(0, min(100, base_score))
        except Exception as e:
            logger.error(f"Error calculating security score: {e}")
            return 0
    
    def _generate_security_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """Generate security-specific recommendations."""
        try:
            recommendations = []
            
            vulns = sum(e.details.get('vulnerabilities_found', 0) 
                       for e in events if e.event_type == EventType.SECURITY_SCAN)
            if vulns > 0:
                recommendations.append(f"Address {vulns} identified vulnerabilities")
            
            failed_auths = len([e for e in events if e.event_type == EventType.AUTH_FAILURE])
            if failed_auths > 5:
                recommendations.append("Implement stronger authentication controls")
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating security recommendations: {e}")
            return []