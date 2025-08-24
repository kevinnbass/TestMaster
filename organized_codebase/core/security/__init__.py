"""
Security Features Module Registry
Pluggable security modules extracted for Agent X's Epsilon base integration

All modules comply with STEELCLAD protocol (< 200 lines each)
Ready for integration into unified dashboard architecture
"""

from .websocket_security_stream import WebSocketSecurityStream, create_security_stream_plugin
from .ml_threat_correlation import MLThreatCorrelationEngine, create_threat_correlation_plugin  
from .predictive_security_analytics import PredictiveSecurityAnalytics, create_predictive_analytics_plugin
from .vulnerability_scanner import SecurityVulnerabilityScanner, create_vulnerability_scanner_plugin

# Security module registry for Agent X integration
SECURITY_MODULES = {
    'websocket_security': {
        'class': WebSocketSecurityStream,
        'factory': create_security_stream_plugin,
        'description': 'Real-time WebSocket security monitoring and event streaming',
        'features': ['client_management', 'security_broadcasting', 'health_monitoring']
    },
    'threat_correlation': {
        'class': MLThreatCorrelationEngine,
        'factory': create_threat_correlation_plugin,
        'description': 'ML-based threat correlation and behavioral analysis',
        'features': ['system_correlation', 'threat_assessment', 'pattern_detection']
    },
    'predictive_analytics': {
        'class': PredictiveSecurityAnalytics,
        'factory': create_predictive_analytics_plugin,
        'description': 'Forward-looking security threat prediction and forecasting',
        'features': ['threat_prediction', 'risk_scoring', 'mitigation_suggestions']
    },
    'vulnerability_scanner': {
        'class': SecurityVulnerabilityScanner,
        'factory': create_vulnerability_scanner_plugin,
        'description': 'Comprehensive security vulnerability assessment and scanning',
        'features': ['code_scanning', 'pattern_analysis', 'risk_assessment']
    }
}

def create_security_suite(config=None):
    """Create complete security feature suite for dashboard integration"""
    suite = {}
    for module_name, module_info in SECURITY_MODULES.items():
        suite[module_name] = module_info['factory'](config)
    return suite

def get_security_features():
    """Get list of all available security features"""
    features = []
    for module_name, module_info in SECURITY_MODULES.items():
        features.extend([f"{module_name}_{feature}" for feature in module_info['features']])
    return features

__all__ = [
    'WebSocketSecurityStream', 'MLThreatCorrelationEngine', 
    'PredictiveSecurityAnalytics', 'SecurityVulnerabilityScanner',
    'create_security_stream_plugin', 'create_threat_correlation_plugin',
    'create_predictive_analytics_plugin', 'create_vulnerability_scanner_plugin',
    'SECURITY_MODULES', 'create_security_suite', 'get_security_features'
]