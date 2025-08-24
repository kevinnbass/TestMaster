#!/usr/bin/env python3
"""
Security Dashboard UI Component
================================
STEELCLAD Atomized Module (<200 lines)
Extracted from advanced_security_dashboard.py

Security dashboard interface and visualization components.
"""

from datetime import datetime, timedelta
import random
from typing import Dict, Any, List, Optional

class SecurityDashboardUI:
    """Atomic component for security dashboard interface."""
    
    def __init__(self):
        self.alert_queue = []
        self.vulnerability_cache = {}
        self.threat_levels = ["low", "medium", "high", "critical"]
        
    def render_security_overview(self, security_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render comprehensive security overview panel.
        
        Args:
            security_data: Security metrics and status data
            
        Returns:
            Formatted security overview for dashboard
        """
        overview = {
            "timestamp": datetime.now().isoformat(),
            "panels": {
                "threat_level": self._render_threat_indicator(security_data),
                "vulnerability_summary": self._render_vulnerability_panel(security_data),
                "security_score": self._render_security_score(security_data),
                "active_monitoring": self._render_monitoring_status(security_data),
                "recent_alerts": self._render_alert_panel()
            },
            "quick_stats": self._generate_quick_stats(security_data)
        }
        return overview
    
    def render_vulnerability_scanner(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Render vulnerability scanner results."""
        return {
            "scan_id": scan_results.get("scan_id", ""),
            "timestamp": scan_results.get("timestamp", datetime.now().isoformat()),
            "vulnerabilities": {
                "critical": scan_results.get("critical", 0),
                "high": scan_results.get("high", 0),
                "medium": scan_results.get("medium", 0),
                "low": scan_results.get("low", 0)
            },
            "affected_components": scan_results.get("affected_components", []),
            "recommendations": self._generate_recommendations(scan_results)
        }
    
    def render_threat_detection(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render threat detection panel."""
        return {
            "active_threats": threat_data.get("active_threats", 0),
            "threat_level": threat_data.get("threat_level", "low"),
            "detection_confidence": threat_data.get("confidence", 0),
            "threat_patterns": threat_data.get("patterns", []),
            "mitigation_status": threat_data.get("mitigation", "active")
        }
    
    def render_security_alerts(self, max_alerts: int = 10) -> List[Dict[str, Any]]:
        """Render recent security alerts."""
        alerts = []
        for alert in self.alert_queue[-max_alerts:]:
            alerts.append({
                "id": alert.get("id"),
                "severity": alert.get("severity"),
                "message": alert.get("message"),
                "timestamp": alert.get("timestamp"),
                "status": alert.get("status", "active")
            })
        return alerts
    
    def add_security_alert(self, message: str, severity: str = "info") -> None:
        """Add a new security alert to the queue."""
        alert = {
            "id": f"alert_{len(self.alert_queue)}",
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "status": "active"
        }
        self.alert_queue.append(alert)
        if len(self.alert_queue) > 100:
            self.alert_queue.pop(0)
    
    def _render_threat_indicator(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render threat level indicator."""
        threat_level = data.get("threat_level", "low")
        return {
            "level": threat_level,
            "color": self._get_threat_color(threat_level),
            "score": data.get("threat_score", 0),
            "trend": data.get("threat_trend", "stable")
        }
    
    def _render_vulnerability_panel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render vulnerability summary panel."""
        return {
            "total_vulnerabilities": data.get("vulnerability_count", 0),
            "critical": data.get("critical_vulns", 0),
            "last_scan": data.get("last_scan", datetime.now().isoformat()),
            "scan_status": data.get("scan_status", "completed")
        }
    
    def _render_security_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render security score gauge."""
        score = data.get("security_score", 0)
        return {
            "score": score,
            "grade": self._calculate_grade(score),
            "color": self._get_score_color(score),
            "improvement": data.get("score_improvement", 0)
        }
    
    def _render_monitoring_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render security monitoring status."""
        return {
            "active_scans": data.get("active_scans", 0),
            "monitored_endpoints": data.get("monitored_endpoints", 0),
            "detection_rate": data.get("detection_rate", 0),
            "false_positive_rate": data.get("false_positive_rate", 0)
        }
    
    def _render_alert_panel(self) -> List[Dict[str, Any]]:
        """Render recent alerts panel."""
        return self.render_security_alerts(5)
    
    def _generate_quick_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quick statistics for dashboard."""
        return {
            "alerts_today": len([a for a in self.alert_queue 
                               if datetime.fromisoformat(a["timestamp"]).date() == datetime.now().date()]),
            "vulnerabilities_fixed": data.get("vulnerabilities_fixed", 0),
            "security_incidents": data.get("incidents", 0),
            "compliance_score": data.get("compliance_score", 0)
        }
    
    def _generate_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        if scan_results.get("critical", 0) > 0:
            recommendations.append("Address critical vulnerabilities immediately")
        if scan_results.get("high", 0) > 2:
            recommendations.append("Schedule patches for high-severity issues")
        return recommendations
    
    def _get_threat_color(self, level: str) -> str:
        """Get color code for threat level."""
        colors = {"low": "#4caf50", "medium": "#ff9800", "high": "#ff5722", "critical": "#d32f2f"}
        return colors.get(level, "#9e9e9e")
    
    def _get_score_color(self, score: float) -> str:
        """Get color for security score."""
        if score >= 90: return "#4caf50"
        elif score >= 75: return "#8bc34a"
        elif score >= 60: return "#ff9800"
        elif score >= 40: return "#ff5722"
        return "#d32f2f"
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 90: return "A"
        elif score >= 80: return "B"
        elif score >= 70: return "C"
        elif score >= 60: return "D"
        return "F"

# Module exports
__all__ = ['SecurityDashboardUI']