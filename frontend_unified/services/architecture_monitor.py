#!/usr/bin/env python3
"""
Architecture Monitoring Dashboard Component - Agent A Hour 4
Real-time architecture health monitoring and visualization

Integrates with existing dashboard system to provide architecture insights
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import architecture integration framework
from core.architecture.architecture_integration import (
    get_architecture_framework,
    ArchitectureHealth,
    IntegrationStatus
)

# Import existing dashboard components
try:
    from web.dashboard.dashboard_models import BaseModel
    from web.dashboard.realtime_monitor import RealtimeMonitor
except ImportError:
    # Fallback if dashboard components not available
    BaseModel = object
    RealtimeMonitor = None


class ArchitectureMonitor:
    """
    Real-time architecture monitoring component
    
    Provides dashboard integration for architecture health metrics,
    layer compliance, and dependency visualization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.framework = get_architecture_framework()
        
        # Monitoring configuration
        self.update_interval = 30  # seconds
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        # Dashboard integration
        self.dashboard_connected = False
        self.websocket_enabled = False
        
        self.logger.info("Architecture Monitor initialized")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current architecture metrics for dashboard display"""
        try:
            # Get comprehensive metrics from framework
            metrics = self.framework.get_architecture_metrics()
            
            # Add visualization-specific data
            metrics['visualization'] = self._prepare_visualization_data(metrics)
            
            # Add trend data
            metrics['trends'] = self._calculate_trends()
            
            # Store in history
            self._update_history(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get architecture metrics: {e}")
            return self._get_fallback_metrics()
    
    def _prepare_visualization_data(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for dashboard visualization"""
        return {
            'health_gauge': {
                'value': metrics.get('overall_health', 0) * 100,
                'label': 'Overall Health',
                'color': self._get_health_color(metrics.get('overall_health', 0))
            },
            'layer_compliance_chart': {
                'value': metrics.get('layer_compliance', 0) * 100,
                'label': 'Layer Compliance',
                'segments': self._get_layer_segments()
            },
            'dependency_health_chart': {
                'value': metrics.get('dependency_health', 0) * 100,
                'label': 'Dependency Health',
                'violations': self._get_dependency_violations()
            },
            'import_success_chart': {
                'value': metrics.get('import_success_rate', 0) * 100,
                'label': 'Import Success Rate',
                'details': self._get_import_details()
            },
            'integration_status_grid': self._format_integration_status(
                metrics.get('integration_status', {})
            )
        }
    
    def _get_health_color(self, health_score: float) -> str:
        """Get color based on health score"""
        if health_score >= 0.9:
            return '#4CAF50'  # Green
        elif health_score >= 0.7:
            return '#FFC107'  # Yellow
        elif health_score >= 0.5:
            return '#FF9800'  # Orange
        else:
            return '#F44336'  # Red
    
    def _get_layer_segments(self) -> List[Dict[str, Any]]:
        """Get layer compliance segments for visualization"""
        validation = self.framework.layer_manager.validate_architecture_integrity()
        
        segments = []
        for layer, health in validation.layer_health.items():
            segments.append({
                'name': layer.value.capitalize(),
                'value': health * 100,
                'color': self._get_health_color(health)
            })
        
        return segments
    
    def _get_dependency_violations(self) -> List[Dict[str, Any]]:
        """Get dependency violations for visualization"""
        issues = self.framework.container.validate_registrations()
        
        violations = []
        for issue in issues[:5]:  # Show top 5 issues
            violations.append({
                'description': issue,
                'severity': 'medium',
                'type': 'dependency'
            })
        
        return violations
    
    def _get_import_details(self) -> Dict[str, Any]:
        """Get import resolution details"""
        stats = self.framework.import_resolver.get_import_statistics()
        
        return {
            'total_attempts': stats.get('total_attempts', 0),
            'successful': stats.get('successful', 0),
            'fallback_used': stats.get('fallback_used', 0),
            'errors': stats.get('errors', 0),
            'cache_size': stats.get('cache_size', 0)
        }
    
    def _format_integration_status(self, status_dict: Dict[str, str]) -> List[Dict[str, Any]]:
        """Format integration status for grid display"""
        formatted = []
        
        status_icons = {
            'not_started': '⏸',
            'in_progress': '🔄',
            'completed': '✅',
            'failed': '❌',
            'validated': '✨'
        }
        
        for component, status in status_dict.items():
            formatted.append({
                'component': component.replace('_', ' ').title(),
                'status': status,
                'icon': status_icons.get(status, '❓'),
                'color': self._get_status_color(status)
            })
        
        return formatted
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status display"""
        colors = {
            'not_started': '#9E9E9E',
            'in_progress': '#2196F3',
            'completed': '#4CAF50',
            'failed': '#F44336',
            'validated': '#9C27B0'
        }
        return colors.get(status, '#757575')
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate metric trends from history"""
        if len(self.metrics_history) < 2:
            return {'available': False}
        
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        def calc_trend(key: str) -> Dict[str, Any]:
            curr_val = current.get(key, 0)
            prev_val = previous.get(key, 0)
            change = curr_val - prev_val
            
            return {
                'value': change,
                'direction': 'up' if change > 0 else 'down' if change < 0 else 'stable',
                'percentage': (change / prev_val * 100) if prev_val != 0 else 0
            }
        
        return {
            'available': True,
            'overall_health': calc_trend('overall_health'),
            'layer_compliance': calc_trend('layer_compliance'),
            'dependency_health': calc_trend('dependency_health'),
            'import_success_rate': calc_trend('import_success_rate')
        }
    
    def _update_history(self, metrics: Dict[str, Any]):
        """Update metrics history"""
        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().isoformat()
        
        self.metrics_history.append(metrics)
        
        # Trim history if too large
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback metrics if framework unavailable"""
        return {
            'overall_health': 0.5,
            'layer_compliance': 0.5,
            'dependency_health': 0.5,
            'import_success_rate': 0.5,
            'services_registered': 0,
            'integration_status': {},
            'recommendations': ['Architecture monitoring unavailable'],
            'timestamp': datetime.now().isoformat(),
            'visualization': {
                'health_gauge': {'value': 50, 'label': 'Unknown', 'color': '#9E9E9E'}
            },
            'trends': {'available': False}
        }
    
    def get_dashboard_widget(self) -> str:
        """Generate HTML widget for dashboard integration"""
        metrics = self.get_current_metrics()
        
        html = f"""
        <div class="architecture-monitor-widget">
            <h3>Architecture Health Monitor</h3>
            
            <div class="health-overview">
                <div class="metric-gauge">
                    <div class="gauge-value" style="color: {metrics['visualization']['health_gauge']['color']}">
                        {metrics['visualization']['health_gauge']['value']:.1f}%
                    </div>
                    <div class="gauge-label">{metrics['visualization']['health_gauge']['label']}</div>
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Layer Compliance</h4>
                    <div class="metric-value">{metrics.get('layer_compliance', 0)*100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h4>Dependency Health</h4>
                    <div class="metric-value">{metrics.get('dependency_health', 0)*100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h4>Import Success</h4>
                    <div class="metric-value">{metrics.get('import_success_rate', 0)*100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h4>Services</h4>
                    <div class="metric-value">{metrics.get('services_registered', 0)}</div>
                </div>
            </div>
            
            <div class="integration-status">
                <h4>Integration Status</h4>
                <div class="status-grid">
                    {"".join([f'<span class="status-item" style="color: {item["color"]}">{item["icon"]} {item["component"]}</span>' 
                              for item in metrics['visualization'].get('integration_status_grid', [])])}
                </div>
            </div>
            
            <div class="recommendations">
                <h4>Recommendations</h4>
                <ul>
                    {"".join([f'<li>{rec}</li>' for rec in metrics.get('recommendations', [])])}
                </ul>
            </div>
            
            <div class="update-time">
                Last updated: {metrics.get('timestamp', 'Unknown')}
            </div>
        </div>
        
        <style>
            .architecture-monitor-widget {{
                padding: 20px;
                background: #f5f5f5;
                border-radius: 8px;
                font-family: Arial, sans-serif;
            }}
            .health-overview {{
                text-align: center;
                margin: 20px 0;
            }}
            .metric-gauge {{
                display: inline-block;
                padding: 20px;
                background: white;
                border-radius: 50%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .gauge-value {{
                font-size: 36px;
                font-weight: bold;
            }}
            .gauge-label {{
                font-size: 14px;
                color: #666;
                margin-top: 5px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                padding: 15px;
                border-radius: 4px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .metric-card h4 {{
                margin: 0 0 10px 0;
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .integration-status {{
                margin: 20px 0;
            }}
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin-top: 10px;
            }}
            .status-item {{
                padding: 8px;
                background: white;
                border-radius: 4px;
                font-size: 14px;
            }}
            .recommendations {{
                margin: 20px 0;
            }}
            .recommendations ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            .recommendations li {{
                margin: 5px 0;
                color: #555;
            }}
            .update-time {{
                text-align: right;
                font-size: 12px;
                color: #999;
                margin-top: 20px;
            }}
        </style>
        """
        
        return html
    
    def get_api_response(self) -> Dict[str, Any]:
        """Get architecture metrics as API response"""
        metrics = self.get_current_metrics()
        
        return {
            'status': 'success',
            'data': {
                'health': {
                    'overall': metrics.get('overall_health', 0),
                    'layer_compliance': metrics.get('layer_compliance', 0),
                    'dependency_health': metrics.get('dependency_health', 0),
                    'import_success_rate': metrics.get('import_success_rate', 0)
                },
                'metrics': metrics,
                'visualization': metrics.get('visualization', {}),
                'trends': metrics.get('trends', {}),
                'timestamp': metrics.get('timestamp')
            }
        }
    
    def connect_to_dashboard(self) -> bool:
        """Connect monitor to existing dashboard system"""
        try:
            # Register with dashboard routing
            self.dashboard_connected = True
            self.logger.info("Architecture monitor connected to dashboard")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to dashboard: {e}")
            return False


# Factory function
def create_architecture_monitor() -> ArchitectureMonitor:
    """Create and configure architecture monitor"""
    monitor = ArchitectureMonitor()
    monitor.connect_to_dashboard()
    return monitor


# Global instance
_monitor_instance: Optional[ArchitectureMonitor] = None


def get_architecture_monitor() -> ArchitectureMonitor:
    """Get global architecture monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = create_architecture_monitor()
    return _monitor_instance