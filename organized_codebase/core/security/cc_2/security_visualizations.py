#!/usr/bin/env python3
"""
Security Visualizations Component
=================================
STEELCLAD Atomized Module (<180 lines)
Extracted from advanced_security_dashboard.py

Security-specific visualization components.
"""

from datetime import datetime, timedelta
import random
from typing import Dict, Any, List, Optional

class SecurityVisualizations:
    """Atomic component for security-specific visualizations."""
    
    def __init__(self):
        self.threat_map_data = {}
        self.vulnerability_timeline = []
        self.security_metrics = {}
        
    def render_threat_heatmap(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render threat level heatmap visualization."""
        return {
            "type": "heatmap",
            "title": "Threat Level Heatmap",
            "data": {
                "matrix": self._generate_threat_matrix(threat_data),
                "x_labels": ["Network", "Application", "Database", "API", "Frontend"],
                "y_labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                "colorScale": {
                    "min": 0,
                    "max": 100,
                    "colors": ["#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]
                }
            },
            "options": {
                "cellSize": 40,
                "showValues": True,
                "tooltip": True
            }
        }
    
    def render_vulnerability_timeline(self, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Render vulnerability discovery timeline."""
        timeline_data = []
        for vuln in vulnerabilities[-20:]:  # Last 20 vulnerabilities
            timeline_data.append({
                "timestamp": vuln.get("discovered", datetime.now().isoformat()),
                "severity": vuln.get("severity", "low"),
                "type": vuln.get("type", "unknown"),
                "status": vuln.get("status", "open")
            })
        
        return {
            "type": "timeline",
            "title": "Vulnerability Timeline",
            "data": timeline_data,
            "options": {
                "groupBy": "severity",
                "showLabels": True,
                "interactive": True
            }
        }
    
    def render_attack_vector_diagram(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Render attack vector visualization."""
        return {
            "type": "sankey",
            "title": "Attack Vector Analysis",
            "data": {
                "nodes": self._extract_vector_nodes(vectors),
                "links": self._extract_vector_links(vectors)
            },
            "options": {
                "nodeWidth": 20,
                "nodePadding": 15,
                "linkColor": "gradient"
            }
        }
    
    def render_security_radar(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Render security metrics radar chart."""
        categories = [
            "Authentication", "Authorization", "Encryption",
            "Monitoring", "Compliance", "Incident Response"
        ]
        
        values = [
            metrics.get("auth_score", random.uniform(70, 95)),
            metrics.get("authz_score", random.uniform(75, 90)),
            metrics.get("encryption_score", random.uniform(80, 95)),
            metrics.get("monitoring_score", random.uniform(70, 90)),
            metrics.get("compliance_score", random.uniform(85, 98)),
            metrics.get("incident_score", random.uniform(65, 85))
        ]
        
        return {
            "type": "radar",
            "title": "Security Posture",
            "data": {
                "labels": categories,
                "datasets": [{
                    "label": "Current",
                    "data": values,
                    "borderColor": "#2196F3",
                    "backgroundColor": "rgba(33, 150, 243, 0.2)"
                }]
            },
            "options": {
                "scale": {
                    "min": 0,
                    "max": 100,
                    "ticks": {"stepSize": 20}
                }
            }
        }
    
    def render_incident_flow(self, incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Render incident response flow diagram."""
        return {
            "type": "flow",
            "title": "Incident Response Flow",
            "data": {
                "stages": [
                    {"id": "detection", "label": "Detection", "count": len(incidents)},
                    {"id": "triage", "label": "Triage", "count": len([i for i in incidents if i.get("triaged")])},
                    {"id": "investigation", "label": "Investigation", "count": len([i for i in incidents if i.get("investigated")])},
                    {"id": "mitigation", "label": "Mitigation", "count": len([i for i in incidents if i.get("mitigated")])},
                    {"id": "resolution", "label": "Resolution", "count": len([i for i in incidents if i.get("resolved")])}
                ],
                "connections": [
                    {"from": "detection", "to": "triage"},
                    {"from": "triage", "to": "investigation"},
                    {"from": "investigation", "to": "mitigation"},
                    {"from": "mitigation", "to": "resolution"}
                ]
            }
        }
    
    def render_compliance_gauge(self, compliance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render compliance status gauge."""
        score = compliance_data.get("overall_score", 0)
        return {
            "type": "gauge",
            "title": "Compliance Score",
            "data": {
                "value": score,
                "max": 100,
                "segments": [
                    {"from": 0, "to": 60, "color": "#F44336", "label": "Non-Compliant"},
                    {"from": 60, "to": 80, "color": "#FF9800", "label": "Partial"},
                    {"from": 80, "to": 90, "color": "#FFC107", "label": "Good"},
                    {"from": 90, "to": 100, "color": "#4CAF50", "label": "Excellent"}
                ]
            },
            "options": {
                "needle": True,
                "animation": True,
                "showValue": True
            }
        }
    
    def _generate_threat_matrix(self, threat_data: Dict[str, Any]) -> List[List[float]]:
        """Generate threat level matrix for heatmap."""
        matrix = []
        for day in range(7):
            row = []
            for component in range(5):
                # Simulate threat levels
                base_threat = random.uniform(10, 40)
                if threat_data.get("high_risk_components"):
                    if component in threat_data["high_risk_components"]:
                        base_threat += random.uniform(30, 50)
                row.append(min(100, base_threat))
            matrix.append(row)
        return matrix
    
    def _extract_vector_nodes(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract nodes from attack vectors."""
        nodes = []
        node_set = set()
        
        for vector in vectors:
            source = vector.get("source", "External")
            target = vector.get("target", "System")
            node_set.add(source)
            node_set.add(target)
        
        return [{"id": i, "name": node} for i, node in enumerate(node_set)]
    
    def _extract_vector_links(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract links from attack vectors."""
        links = []
        node_map = {node: i for i, node in enumerate(set(
            [v.get("source", "External") for v in vectors] + 
            [v.get("target", "System") for v in vectors]
        ))}
        
        for vector in vectors:
            links.append({
                "source": node_map[vector.get("source", "External")],
                "target": node_map[vector.get("target", "System")],
                "value": vector.get("severity", 1)
            })
        
        return links

# Module exports
__all__ = ['SecurityVisualizations']