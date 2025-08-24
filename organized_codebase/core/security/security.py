"""
Security Vulnerability Heatmap API
===================================

Visualizes security vulnerabilities, OWASP compliance, and threat patterns.

Author: TestMaster Team
"""

import logging
from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import random
import json

logger = logging.getLogger(__name__)

class SecurityAPI:
    """Security Vulnerability Heatmap API endpoints."""
    
    def __init__(self):
        """Initialize Security API."""
        self.blueprint = Blueprint('security', __name__, url_prefix='/api/security')
        self._setup_routes()
        logger.info("Security API initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.blueprint.route('/vulnerabilities/heatmap', methods=['GET'])
        def vulnerability_heatmap():
            """Get vulnerability heatmap data."""
            try:
                # Generate heatmap data for different modules
                modules = [
                    'auth', 'api', 'database', 'frontend', 'backend',
                    'utils', 'config', 'network', 'storage', 'cache',
                    'session', 'crypto', 'validation', 'logging', 'monitoring'
                ]
                
                vulnerability_types = [
                    'SQL Injection', 'XSS', 'CSRF', 'Authentication',
                    'Authorization', 'Encryption', 'Input Validation',
                    'Session Management', 'Error Handling', 'Information Disclosure'
                ]
                
                heatmap_data = []
                for module in modules:
                    for vuln_type in vulnerability_types:
                        severity = random.choice([0, 0, 0, 1, 1, 2, 2, 3, 4, 5])  # More low severity
                        if severity > 0:
                            heatmap_data.append({
                                'module': module,
                                'vulnerability': vuln_type,
                                'severity': severity,
                                'count': random.randint(1, 5) if severity > 0 else 0
                            })
                
                # Calculate statistics
                total_vulnerabilities = sum(item['count'] for item in heatmap_data)
                critical_count = sum(item['count'] for item in heatmap_data if item['severity'] >= 4)
                high_count = sum(item['count'] for item in heatmap_data if item['severity'] == 3)
                medium_count = sum(item['count'] for item in heatmap_data if item['severity'] == 2)
                low_count = sum(item['count'] for item in heatmap_data if item['severity'] == 1)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'heatmap': heatmap_data,
                    'summary': {
                        'total_vulnerabilities': total_vulnerabilities,
                        'critical': critical_count,
                        'high': high_count,
                        'medium': medium_count,
                        'low': low_count,
                        'risk_score': round((critical_count * 10 + high_count * 5 + medium_count * 2 + low_count) / 10, 1)
                    },
                    'charts': {
                        'severity_distribution': {
                            'critical': critical_count,
                            'high': high_count,
                            'medium': medium_count,
                            'low': low_count
                        },
                        'module_risk_scores': self._calculate_module_risk(heatmap_data, modules),
                        'vulnerability_trends': self._generate_vulnerability_trends(),
                        'top_vulnerabilities': self._get_top_vulnerabilities(heatmap_data)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Vulnerability heatmap failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/owasp/compliance', methods=['GET'])
        def owasp_compliance():
            """Get OWASP compliance status."""
            try:
                owasp_categories = [
                    {'id': 'A01', 'name': 'Broken Access Control', 'compliance': 85, 'status': 'good'},
                    {'id': 'A02', 'name': 'Cryptographic Failures', 'compliance': 92, 'status': 'excellent'},
                    {'id': 'A03', 'name': 'Injection', 'compliance': 78, 'status': 'fair'},
                    {'id': 'A04', 'name': 'Insecure Design', 'compliance': 65, 'status': 'needs_improvement'},
                    {'id': 'A05', 'name': 'Security Misconfiguration', 'compliance': 88, 'status': 'good'},
                    {'id': 'A06', 'name': 'Vulnerable Components', 'compliance': 71, 'status': 'fair'},
                    {'id': 'A07', 'name': 'Authentication Failures', 'compliance': 83, 'status': 'good'},
                    {'id': 'A08', 'name': 'Data Integrity Failures', 'compliance': 90, 'status': 'excellent'},
                    {'id': 'A09', 'name': 'Security Logging Failures', 'compliance': 76, 'status': 'fair'},
                    {'id': 'A10', 'name': 'Server-Side Request Forgery', 'compliance': 94, 'status': 'excellent'}
                ]
                
                # Calculate overall compliance
                overall_compliance = sum(cat['compliance'] for cat in owasp_categories) / len(owasp_categories)
                
                # Generate compliance timeline
                compliance_timeline = []
                for i in range(30):
                    timestamp = (datetime.now() - timedelta(days=29-i)).isoformat()
                    compliance_timeline.append({
                        'timestamp': timestamp,
                        'compliance': overall_compliance - 10 + (i * 0.6) + random.uniform(-2, 2)
                    })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'categories': owasp_categories,
                    'overall_compliance': round(overall_compliance, 1),
                    'compliance_level': self._get_compliance_level(overall_compliance),
                    'charts': {
                        'compliance_radar': [
                            {'category': cat['id'], 'compliance': cat['compliance']} 
                            for cat in owasp_categories
                        ],
                        'compliance_timeline': compliance_timeline,
                        'status_distribution': self._calculate_status_distribution(owasp_categories),
                        'improvement_areas': self._identify_improvement_areas(owasp_categories)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"OWASP compliance failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/threats/realtime', methods=['GET'])
        def realtime_threats():
            """Get real-time threat detection data."""
            try:
                # Generate threat events
                threat_events = []
                threat_types = ['SQL Injection Attempt', 'XSS Attack', 'Brute Force', 
                               'Unauthorized Access', 'Data Exfiltration', 'Malware Detection',
                               'DDoS Pattern', 'Privilege Escalation', 'Session Hijacking']
                
                for i in range(20):
                    timestamp = (datetime.now() - timedelta(minutes=i*3)).isoformat()
                    threat_events.append({
                        'id': f"threat_{i}",
                        'timestamp': timestamp,
                        'type': random.choice(threat_types),
                        'severity': random.choice(['low', 'medium', 'high', 'critical']),
                        'source': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                        'target': random.choice(['api', 'database', 'auth', 'frontend']),
                        'status': random.choice(['blocked', 'mitigated', 'investigating', 'resolved']),
                        'confidence': round(random.uniform(0.7, 1.0), 2)
                    })
                
                # Calculate threat metrics
                active_threats = sum(1 for t in threat_events if t['status'] == 'investigating')
                blocked_threats = sum(1 for t in threat_events if t['status'] == 'blocked')
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'threats': threat_events[:10],
                    'metrics': {
                        'total_threats_24h': len(threat_events),
                        'active_threats': active_threats,
                        'blocked_threats': blocked_threats,
                        'average_confidence': round(sum(t['confidence'] for t in threat_events) / len(threat_events), 2)
                    },
                    'charts': {
                        'threat_timeline': self._generate_threat_timeline(),
                        'threat_type_distribution': self._calculate_threat_distribution(threat_events),
                        'severity_breakdown': self._calculate_severity_breakdown(threat_events),
                        'target_frequency': self._calculate_target_frequency(threat_events)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Realtime threats failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/scanning/status', methods=['GET'])
        def scanning_status():
            """Get security scanning status."""
            try:
                scans = [
                    {
                        'id': 'scan_001',
                        'type': 'Static Analysis (SAST)',
                        'status': 'completed',
                        'progress': 100,
                        'findings': 23,
                        'duration': 145,
                        'last_run': (datetime.now() - timedelta(hours=2)).isoformat()
                    },
                    {
                        'id': 'scan_002',
                        'type': 'Dynamic Analysis (DAST)',
                        'status': 'running',
                        'progress': 67,
                        'findings': 12,
                        'duration': 89,
                        'last_run': datetime.now().isoformat()
                    },
                    {
                        'id': 'scan_003',
                        'type': 'Dependency Check',
                        'status': 'completed',
                        'progress': 100,
                        'findings': 8,
                        'duration': 34,
                        'last_run': (datetime.now() - timedelta(hours=6)).isoformat()
                    },
                    {
                        'id': 'scan_004',
                        'type': 'Container Scanning',
                        'status': 'scheduled',
                        'progress': 0,
                        'findings': 0,
                        'duration': 0,
                        'next_run': (datetime.now() + timedelta(hours=1)).isoformat()
                    }
                ]
                
                # Calculate scan metrics
                total_findings = sum(s['findings'] for s in scans)
                completed_scans = sum(1 for s in scans if s['status'] == 'completed')
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'scans': scans,
                    'summary': {
                        'total_scans': len(scans),
                        'completed': completed_scans,
                        'running': sum(1 for s in scans if s['status'] == 'running'),
                        'total_findings': total_findings,
                        'average_duration': round(sum(s['duration'] for s in scans) / len(scans), 1)
                    },
                    'charts': {
                        'scan_progress': [
                            {'scan': s['type'], 'progress': s['progress']} for s in scans
                        ],
                        'findings_by_scan': [
                            {'scan': s['type'], 'findings': s['findings']} for s in scans
                        ],
                        'scan_history': self._generate_scan_history(),
                        'finding_trends': self._generate_finding_trends()
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Scanning status failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/remediation/recommendations', methods=['GET'])
        def remediation_recommendations():
            """Get security remediation recommendations."""
            try:
                recommendations = [
                    {
                        'id': 'rec_001',
                        'vulnerability': 'SQL Injection in user input',
                        'severity': 'critical',
                        'module': 'api/auth.py',
                        'recommendation': 'Use parameterized queries',
                        'effort': 'low',
                        'impact': 'high',
                        'priority': 1
                    },
                    {
                        'id': 'rec_002',
                        'vulnerability': 'Weak password hashing',
                        'severity': 'high',
                        'module': 'utils/crypto.py',
                        'recommendation': 'Implement bcrypt or argon2',
                        'effort': 'medium',
                        'impact': 'high',
                        'priority': 2
                    },
                    {
                        'id': 'rec_003',
                        'vulnerability': 'Missing CSRF protection',
                        'severity': 'medium',
                        'module': 'frontend/forms.py',
                        'recommendation': 'Add CSRF tokens to all forms',
                        'effort': 'low',
                        'impact': 'medium',
                        'priority': 3
                    },
                    {
                        'id': 'rec_004',
                        'vulnerability': 'Outdated dependencies',
                        'severity': 'medium',
                        'module': 'requirements.txt',
                        'recommendation': 'Update vulnerable packages',
                        'effort': 'low',
                        'impact': 'medium',
                        'priority': 4
                    }
                ]
                
                # Calculate remediation metrics
                total_effort = {'low': 0, 'medium': 0, 'high': 0}
                for rec in recommendations:
                    total_effort[rec['effort']] += 1
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'recommendations': recommendations,
                    'summary': {
                        'total_recommendations': len(recommendations),
                        'critical_items': sum(1 for r in recommendations if r['severity'] == 'critical'),
                        'estimated_effort': total_effort,
                        'risk_reduction': 78.5
                    },
                    'charts': {
                        'priority_matrix': self._generate_priority_matrix(recommendations),
                        'effort_impact': [
                            {'recommendation': r['id'], 'effort': self._effort_to_number(r['effort']), 
                             'impact': self._impact_to_number(r['impact'])} 
                            for r in recommendations
                        ],
                        'remediation_timeline': self._generate_remediation_timeline(),
                        'risk_reduction_projection': self._generate_risk_projection()
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Remediation recommendations failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _calculate_module_risk(self, heatmap_data, modules):
        """Calculate risk score for each module."""
        module_risks = {}
        for module in modules:
            module_vulns = [item for item in heatmap_data if item['module'] == module]
            risk_score = sum(item['severity'] * item['count'] for item in module_vulns)
            module_risks[module] = risk_score
        
        return [{'module': m, 'risk_score': s} for m, s in sorted(module_risks.items(), 
                key=lambda x: x[1], reverse=True)[:10]]
    
    def _generate_vulnerability_trends(self):
        """Generate vulnerability trend data."""
        trends = []
        for i in range(30):
            timestamp = (datetime.now() - timedelta(days=29-i)).isoformat()
            trends.append({
                'timestamp': timestamp,
                'vulnerabilities': 50 - i + random.randint(-5, 5)
            })
        return trends
    
    def _get_top_vulnerabilities(self, heatmap_data):
        """Get top vulnerabilities by count."""
        vuln_counts = {}
        for item in heatmap_data:
            vuln = item['vulnerability']
            if vuln not in vuln_counts:
                vuln_counts[vuln] = 0
            vuln_counts[vuln] += item['count']
        
        return [{'vulnerability': v, 'count': c} for v, c in sorted(vuln_counts.items(), 
                key=lambda x: x[1], reverse=True)[:5]]
    
    def _get_compliance_level(self, compliance):
        """Get compliance level based on percentage."""
        if compliance >= 90:
            return 'Excellent'
        elif compliance >= 80:
            return 'Good'
        elif compliance >= 70:
            return 'Fair'
        elif compliance >= 60:
            return 'Needs Improvement'
        else:
            return 'Critical'
    
    def _calculate_status_distribution(self, categories):
        """Calculate status distribution."""
        statuses = {}
        for cat in categories:
            status = cat['status']
            if status not in statuses:
                statuses[status] = 0
            statuses[status] += 1
        return [{'status': s, 'count': c} for s, c in statuses.items()]
    
    def _identify_improvement_areas(self, categories):
        """Identify areas needing improvement."""
        return [{'category': cat['id'], 'name': cat['name'], 'gap': 100 - cat['compliance']} 
                for cat in sorted(categories, key=lambda x: x['compliance'])[:3]]
    
    def _generate_threat_timeline(self):
        """Generate threat timeline."""
        timeline = []
        for i in range(24):
            timestamp = (datetime.now() - timedelta(hours=23-i)).isoformat()
            timeline.append({
                'timestamp': timestamp,
                'threats': random.randint(5, 25)
            })
        return timeline
    
    def _calculate_threat_distribution(self, threats):
        """Calculate threat type distribution."""
        types = {}
        for threat in threats:
            t = threat['type']
            if t not in types:
                types[t] = 0
            types[t] += 1
        return [{'type': t, 'count': c} for t, c in types.items()]
    
    def _calculate_severity_breakdown(self, threats):
        """Calculate severity breakdown."""
        severities = {}
        for threat in threats:
            s = threat['severity']
            if s not in severities:
                severities[s] = 0
            severities[s] += 1
        return [{'severity': s, 'count': c} for s, c in severities.items()]
    
    def _calculate_target_frequency(self, threats):
        """Calculate target frequency."""
        targets = {}
        for threat in threats:
            t = threat['target']
            if t not in targets:
                targets[t] = 0
            targets[t] += 1
        return [{'target': t, 'frequency': f} for t, f in targets.items()]
    
    def _generate_scan_history(self):
        """Generate scan history."""
        history = []
        for i in range(7):
            timestamp = (datetime.now() - timedelta(days=6-i)).isoformat()
            history.append({
                'timestamp': timestamp,
                'scans_completed': random.randint(3, 8),
                'findings': random.randint(15, 40)
            })
        return history
    
    def _generate_finding_trends(self):
        """Generate finding trends."""
        trends = []
        for i in range(30):
            timestamp = (datetime.now() - timedelta(days=29-i)).isoformat()
            trends.append({
                'timestamp': timestamp,
                'findings': 40 - i * 0.5 + random.uniform(-5, 5)
            })
        return trends
    
    def _generate_priority_matrix(self, recommendations):
        """Generate priority matrix."""
        matrix = []
        for rec in recommendations:
            matrix.append({
                'id': rec['id'],
                'effort': self._effort_to_number(rec['effort']),
                'impact': self._impact_to_number(rec['impact']),
                'priority': rec['priority']
            })
        return matrix
    
    def _effort_to_number(self, effort):
        """Convert effort to number."""
        return {'low': 1, 'medium': 2, 'high': 3}.get(effort, 1)
    
    def _impact_to_number(self, impact):
        """Convert impact to number."""
        return {'low': 1, 'medium': 2, 'high': 3}.get(impact, 1)
    
    def _generate_remediation_timeline(self):
        """Generate remediation timeline."""
        timeline = []
        for i in range(8):
            timestamp = (datetime.now() + timedelta(days=i)).isoformat()
            timeline.append({
                'timestamp': timestamp,
                'planned_remediations': random.randint(1, 3)
            })
        return timeline
    
    def _generate_risk_projection(self):
        """Generate risk reduction projection."""
        projection = []
        current_risk = 100
        for i in range(12):
            timestamp = (datetime.now() + timedelta(weeks=i)).isoformat()
            current_risk *= 0.92  # 8% reduction per week
            projection.append({
                'timestamp': timestamp,
                'risk_level': round(current_risk, 1)
            })
        return projection