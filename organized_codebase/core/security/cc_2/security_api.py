
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
Security REST API

REST endpoints for all security scanning and monitoring functionality.
"""

from flask import Flask, request, jsonify, Response
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import logging

from .vulnerability_scanner import VulnerabilityScanner
from .compliance_checker import ComplianceChecker
from .threat_modeler import ThreatModeler
from .dependency_scanner import DependencyScanner
from .crypto_analyzer import CryptoAnalyzer
from .audit_logger import AuditLogger, EventType, EventSeverity
from .security_dashboard import SecurityDashboard
from .security_analytics import SecurityAnalytics

logger = logging.getLogger(__name__)


class SecurityAPI:
    """
    REST API for security services.
    Provides endpoints for all security scanning and monitoring functionality.
    """
    
    def __init__(self, app: Optional[Flask] = None):
        """
        Initialize security API.
        
        Args:
            app: Flask application instance
        """
        self.app = app or Flask(__name__)
        self.vuln_scanner = VulnerabilityScanner()
        self.compliance_checker = ComplianceChecker()
        self.threat_modeler = ThreatModeler()
        self.dep_scanner = DependencyScanner()
        self.crypto_analyzer = CryptoAnalyzer()
        self.audit_logger = AuditLogger()
        self.dashboard = SecurityDashboard()
        self.analytics = SecurityAnalytics()
        
        self._register_routes()
        logger.info("Security API initialized")
        
    def _register_routes(self) -> None:
        """Register API routes."""
        
        @self.app.route('/api/security/scan/vulnerabilities', methods=['POST'])
        def scan_vulnerabilities():
            """Scan for vulnerabilities."""
            try:
                data = request.get_json()
                directory = data.get('directory')
                recursive = data.get('recursive', True)
                
                if not directory:
                    return jsonify({'error': 'directory is required'}), 400
                    
                if not os.path.exists(directory):
                    return jsonify({'error': 'Directory does not exist'}), 404
                    
                results = self.vuln_scanner.scan_directory(directory, recursive)
                report = self.vuln_scanner.generate_report()
                
                # Log scan event
                self.audit_logger.log_security_scan(
                    scanner="vulnerability_scanner",
                    target=directory,
                    vulnerabilities_found=report['total_vulnerabilities'],
                    scan_type="vulnerability"
                )
                
                return jsonify({
                    'scan_results': {
                        file_path: [
                            {
                                'type': v.type,
                                'severity': v.severity,
                                'description': v.description,
                                'line': v.line_number
                            }
                            for v in vulns
                        ]
                        for file_path, vulns in results.items()
                    },
                    'summary': report
                })
                
            except Exception as e:
                logger.error(f"Error scanning vulnerabilities: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/scan/compliance', methods=['POST'])
        def check_compliance():
            """Check compliance against standards."""
            try:
                data = request.get_json()
                directory = data.get('directory')
                standards = data.get('standards', ['OWASP'])
                
                if not directory:
                    return jsonify({'error': 'directory is required'}), 400
                    
                results = {}
                
                for standard in standards:
                    if standard.upper() == 'OWASP':
                        issues = self.compliance_checker.check_owasp_compliance(directory)
                        results[standard] = len(issues)
                    elif standard.upper() == 'PCI-DSS':
                        issues = self.compliance_checker.check_pci_dss_compliance(directory)
                        results[standard] = len(issues)
                    elif standard.upper() == 'GDPR':
                        issues = self.compliance_checker.check_gdpr_compliance(directory)
                        results[standard] = len(issues)
                        
                report = self.compliance_checker.generate_compliance_report(standards)
                
                # Log compliance check
                self.audit_logger.log_compliance_check(
                    checker="compliance_checker",
                    standard=", ".join(standards),
                    passed=report.compliance_score > 70,
                    issues_found=report.total_issues
                )
                
                return jsonify({
                    'compliance_score': report.compliance_score,
                    'standards_checked': report.standards_checked,
                    'issues_by_standard': results,
                    'total_issues': report.total_issues
                })
                
            except Exception as e:
                logger.error(f"Error checking compliance: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/scan/dependencies', methods=['POST'])
        def scan_dependencies():
            """Scan dependencies for vulnerabilities."""
            try:
                data = request.get_json()
                project_path = data.get('project_path')
                language = data.get('language', 'python')
                
                if not project_path:
                    return jsonify({'error': 'project_path is required'}), 400
                    
                if language.lower() == 'python':
                    dependencies = self.dep_scanner.scan_python_dependencies(project_path)
                elif language.lower() == 'javascript':
                    dependencies = self.dep_scanner.scan_javascript_dependencies(project_path)
                else:
                    return jsonify({'error': 'Unsupported language'}), 400
                    
                report = self.dep_scanner.generate_report()
                suggestions = self.dep_scanner.suggest_updates()
                
                return jsonify({
                    'dependencies_scanned': len(dependencies),
                    'vulnerable_dependencies': len([d for d in dependencies if d.vulnerabilities]),
                    'report': report,
                    'update_suggestions': suggestions
                })
                
            except Exception as e:
                logger.error(f"Error scanning dependencies: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/scan/crypto', methods=['POST'])
        def analyze_crypto():
            """Analyze cryptographic implementations."""
            try:
                data = request.get_json()
                directory = data.get('directory')
                
                if not directory:
                    return jsonify({'error': 'directory is required'}), 400
                    
                from pathlib import Path
                crypto_issues = []
                
                for py_file in Path(directory).rglob("*.py"):
                    issues = self.crypto_analyzer.analyze_file(str(py_file))
                    crypto_issues.extend(issues)
                    
                report = self.crypto_analyzer.generate_report()
                recommendations = self.crypto_analyzer.get_recommendations()
                
                return jsonify({
                    'crypto_issues': [
                        {
                            'type': issue.type.value,
                            'severity': issue.severity,
                            'file': issue.file_path,
                            'line': issue.line_number,
                            'description': issue.description,
                            'recommendation': issue.recommendation
                        }
                        for issue in crypto_issues
                    ],
                    'summary': report,
                    'recommendations': recommendations
                })
                
            except Exception as e:
                logger.error(f"Error analyzing crypto: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/threat-model', methods=['POST'])
        def create_threat_model():
            """Create threat model for application."""
            try:
                data = request.get_json()
                directory = data.get('directory')
                
                if not directory:
                    return jsonify({'error': 'directory is required'}), 400
                    
                threat_model = self.threat_modeler.analyze_application(directory)
                report = self.threat_modeler.generate_threat_report(threat_model)
                prioritized_threats = self.threat_modeler.prioritize_threats(threat_model.threats)
                
                return jsonify({
                    'threat_model': {
                        'application': threat_model.application,
                        'total_threats': len(threat_model.threats),
                        'total_assets': len(threat_model.assets),
                        'risk_matrix': threat_model.risk_matrix
                    },
                    'prioritized_threats': [
                        {
                            'id': threat.id,
                            'name': threat.name,
                            'category': threat.category.value,
                            'risk_score': threat.risk_score,
                            'mitigations': threat.mitigations
                        }
                        for threat in prioritized_threats[:10]  # Top 10
                    ],
                    'report': report
                })
                
            except Exception as e:
                logger.error(f"Error creating threat model: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/dashboard/status', methods=['GET'])
        def get_dashboard_status():
            """Get security dashboard status."""
            try:
                status = self.dashboard.get_current_status()
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Error getting dashboard status: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/dashboard/start', methods=['POST'])
        def start_monitoring():
            """Start security monitoring."""
            try:
                data = request.get_json()
                project_path = data.get('project_path')
                
                if not project_path:
                    return jsonify({'error': 'project_path is required'}), 400
                    
                self.dashboard.start_monitoring(project_path)
                
                return jsonify({
                    'status': 'monitoring_started',
                    'project_path': project_path,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error starting monitoring: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/dashboard/alerts', methods=['GET'])
        def get_alerts():
            """Get active security alerts."""
            try:
                alerts = self.dashboard.get_active_alerts()
                
                return jsonify({
                    'active_alerts': [
                        {
                            'id': alert.id,
                            'severity': alert.severity,
                            'type': alert.type,
                            'title': alert.title,
                            'description': alert.description,
                            'timestamp': alert.timestamp
                        }
                        for alert in alerts
                    ],
                    'total_alerts': len(alerts)
                })
                
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/dashboard/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id: str):
            """Acknowledge a security alert."""
            try:
                success = self.dashboard.acknowledge_alert(alert_id)
                
                if success:
                    return jsonify({
                        'status': 'acknowledged',
                        'alert_id': alert_id,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Alert not found'}), 404
                    
            except Exception as e:
                logger.error(f"Error acknowledging alert: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/analytics/trends', methods=['GET'])
        def get_security_trends():
            """Get security trends analysis."""
            try:
                days = int(request.args.get('days', 7))
                trends = self.dashboard.get_security_trends(days)
                
                return jsonify({
                    'trends': trends,
                    'analysis_period_days': days,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting trends: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/analytics/risk-assessment', methods=['POST'])
        def assess_risk():
            """Perform risk assessment."""
            try:
                data = request.get_json()
                vuln_count = data.get('vulnerability_count', 0)
                compliance_score = data.get('compliance_score', 100)
                threat_indicators = data.get('threat_indicators', [])
                dependency_risks = data.get('dependency_risks', 0)
                
                risk_assessment = self.analytics.assess_overall_risk(
                    vuln_count, compliance_score, threat_indicators, dependency_risks
                )
                
                return jsonify({
                    'risk_score': risk_assessment.risk_score,
                    'risk_level': risk_assessment.risk_level,
                    'contributing_factors': risk_assessment.contributing_factors,
                    'recommendations': risk_assessment.recommendations,
                    'confidence': risk_assessment.confidence
                })
                
            except Exception as e:
                logger.error(f"Error assessing risk: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/audit/events', methods=['GET'])
        def get_audit_events():
            """Get audit events."""
            try:
                start_time = request.args.get('start_time')
                end_time = request.args.get('end_time')
                event_type = request.args.get('event_type')
                
                events = self.audit_logger.query_events(
                    start_time=start_time,
                    end_time=end_time,
                    event_type=EventType(event_type) if event_type else None
                )
                
                return jsonify({
                    'events': [
                        {
                            'timestamp': event.timestamp,
                            'type': event.event_type.value,
                            'severity': event.severity.value,
                            'actor': event.actor,
                            'resource': event.resource,
                            'action': event.action,
                            'result': event.result
                        }
                        for event in events
                    ],
                    'total_events': len(events)
                })
                
            except Exception as e:
                logger.error(f"Error getting audit events: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/security/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            try:
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'services': {
                        'vulnerability_scanner': 'operational',
                        'compliance_checker': 'operational',
                        'threat_modeler': 'operational',
                        'dependency_scanner': 'operational',
                        'crypto_analyzer': 'operational',
                        'audit_logger': 'operational',
                        'security_dashboard': 'operational',
                        'security_analytics': 'operational'
                    }
                })
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
                
    def run(self, host: str = '0.0.0.0', port: int = 5002, debug: bool = False) -> None:
        """
        Run the security API server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Debug mode
        """
        self.app.run(host=host, port=port, debug=debug)
        
    def get_app(self) -> Flask:
        """Get Flask application instance."""
        return self.app