#!/usr/bin/env python3
"""
📊 ATOM: Swarm Status Display Component
=======================================
Real-time swarm status visualization and monitoring.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

@dataclass
class SwarmMetrics:
    """Swarm performance and health metrics"""
    active_agents: int = 0
    total_agents: int = 0
    average_health_score: float = 0.0
    load_distribution_score: float = 0.0
    total_coordination_messages: int = 0
    successful_coordinations: int = 0
    failed_coordinations: int = 0
    average_response_time: float = 0.0
    network_latency: float = 0.0
    data_sync_status: str = "synchronized"
    last_update: datetime = field(default_factory=datetime.utcnow)

class SwarmStatusDisplay:
    """Swarm status visualization component"""
    
    def __init__(self):
        self.current_metrics = SwarmMetrics()
        self.metrics_history = []
        self.alert_thresholds = self._initialize_thresholds()
        self.display_config = self._initialize_display_config()
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds"""
        return {
            'health_score': {'critical': 0.3, 'warning': 0.6, 'good': 0.8},
            'response_time': {'critical': 1000, 'warning': 500, 'good': 100},
            'error_rate': {'critical': 0.1, 'warning': 0.05, 'good': 0.01},
            'load_distribution': {'critical': 0.3, 'warning': 0.5, 'good': 0.7}
        }
    
    def _initialize_display_config(self) -> Dict[str, Any]:
        """Initialize display configuration"""
        return {
            'update_interval': 5000,
            'history_points': 50,
            'enable_alerts': True,
            'chart_animations': True,
            'color_coding': True
        }
    
    def render_status_display(self, swarm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render swarm status display
        
        Args:
            swarm_data: Current swarm metrics and status
            
        Returns:
            Status display UI configuration
        """
        self._update_metrics(swarm_data)
        
        return {
            'display_type': 'swarm_status',
            'layout': self._get_status_layout(),
            'components': {
                'health_overview': self._render_health_overview(),
                'metrics_cards': self._render_metrics_cards(),
                'status_timeline': self._render_status_timeline(),
                'alerts_panel': self._render_alerts_panel()
            },
            'real_time': True,
            'refresh_rate': self.display_config['update_interval']
        }
    
    def _get_status_layout(self) -> Dict[str, Any]:
        """Get status display layout"""
        return {
            'type': 'dashboard',
            'sections': [
                {'id': 'overview', 'size': 'full', 'priority': 1},
                {'id': 'metrics', 'size': 'half', 'priority': 2},
                {'id': 'timeline', 'size': 'half', 'priority': 2},
                {'id': 'alerts', 'size': 'full', 'priority': 3}
            ]
        }
    
    def _render_health_overview(self) -> Dict[str, Any]:
        """Render health overview section"""
        health_status = self._calculate_health_status()
        
        return {
            'title': 'Swarm Health Overview',
            'status': health_status['level'],
            'score': f"{self.current_metrics.average_health_score * 100:.1f}%",
            'indicators': [
                {
                    'name': 'System Health',
                    'value': self.current_metrics.average_health_score,
                    'status': health_status['level'],
                    'trend': self._calculate_trend('health')
                },
                {
                    'name': 'Load Balance',
                    'value': self.current_metrics.load_distribution_score,
                    'status': self._get_load_status(),
                    'trend': self._calculate_trend('load')
                },
                {
                    'name': 'Coordination',
                    'value': self._calculate_coordination_rate(),
                    'status': self._get_coordination_status(),
                    'trend': self._calculate_trend('coordination')
                }
            ],
            'visualization': {
                'type': 'radial_gauge',
                'value': self.current_metrics.average_health_score * 100,
                'thresholds': self.alert_thresholds['health_score']
            }
        }
    
    def _render_metrics_cards(self) -> List[Dict[str, Any]]:
        """Render metric cards"""
        return [
            {
                'title': 'Active Agents',
                'value': self.current_metrics.active_agents,
                'total': self.current_metrics.total_agents,
                'icon': '🤖',
                'color': self._get_agent_status_color(),
                'sparkline': self._get_metric_sparkline('active_agents')
            },
            {
                'title': 'Response Time',
                'value': f"{self.current_metrics.average_response_time * 1000:.0f}ms",
                'icon': '⚡',
                'color': self._get_response_time_color(),
                'sparkline': self._get_metric_sparkline('response_time')
            },
            {
                'title': 'Coordination Success',
                'value': f"{self._calculate_success_rate():.1f}%",
                'icon': '✅',
                'color': self._get_success_rate_color(),
                'sparkline': self._get_metric_sparkline('success_rate')
            },
            {
                'title': 'Network Status',
                'value': self.current_metrics.data_sync_status.capitalize(),
                'icon': '🌐',
                'color': self._get_sync_status_color(),
                'details': f"Latency: {self.current_metrics.network_latency:.0f}ms"
            }
        ]
    
    def _render_status_timeline(self) -> Dict[str, Any]:
        """Render status timeline"""
        timeline_data = self._prepare_timeline_data()
        
        return {
            'title': 'Status Timeline',
            'type': 'time_series',
            'data': timeline_data,
            'options': {
                'show_grid': True,
                'show_legend': True,
                'interactive': True,
                'zoom': True
            }
        }
    
    def _render_alerts_panel(self) -> Dict[str, Any]:
        """Render alerts panel"""
        alerts = self._check_alerts()
        
        return {
            'title': 'System Alerts',
            'count': len(alerts),
            'alerts': alerts,
            'settings': {
                'auto_dismiss': False,
                'priority_sort': True,
                'show_timestamp': True
            }
        }
    
    def _update_metrics(self, swarm_data: Dict[str, Any]):
        """Update current metrics from swarm data"""
        metrics = swarm_data.get('metrics', {})
        
        self.current_metrics = SwarmMetrics(
            active_agents=metrics.get('active_agents', 0),
            total_agents=metrics.get('total_agents', 0),
            average_health_score=metrics.get('average_health_score', 0.0),
            load_distribution_score=metrics.get('load_distribution_score', 0.0),
            total_coordination_messages=metrics.get('total_coordination_messages', 0),
            successful_coordinations=metrics.get('successful_coordinations', 0),
            failed_coordinations=metrics.get('failed_coordinations', 0),
            average_response_time=metrics.get('average_response_time', 0.0),
            network_latency=metrics.get('network_latency', 0.0),
            data_sync_status=metrics.get('data_sync_status', 'unknown')
        )
        
        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': self.current_metrics.__dict__.copy()
        })
        
        # Keep only recent history
        max_history = self.display_config['history_points']
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]
    
    def _calculate_health_status(self) -> Dict[str, str]:
        """Calculate overall health status"""
        score = self.current_metrics.average_health_score
        thresholds = self.alert_thresholds['health_score']
        
        if score >= thresholds['good']:
            return {'level': 'healthy', 'color': 'success'}
        elif score >= thresholds['warning']:
            return {'level': 'warning', 'color': 'warning'}
        else:
            return {'level': 'critical', 'color': 'danger'}
    
    def _calculate_coordination_rate(self) -> float:
        """Calculate coordination success rate"""
        total = self.current_metrics.successful_coordinations + self.current_metrics.failed_coordinations
        if total == 0:
            return 1.0
        return self.current_metrics.successful_coordinations / total
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        return self._calculate_coordination_rate() * 100
    
    def _get_load_status(self) -> str:
        """Get load distribution status"""
        score = self.current_metrics.load_distribution_score
        thresholds = self.alert_thresholds['load_distribution']
        
        if score >= thresholds['good']:
            return 'balanced'
        elif score >= thresholds['warning']:
            return 'uneven'
        else:
            return 'imbalanced'
    
    def _get_coordination_status(self) -> str:
        """Get coordination status"""
        rate = self._calculate_coordination_rate()
        if rate >= 0.95:
            return 'excellent'
        elif rate >= 0.85:
            return 'good'
        elif rate >= 0.70:
            return 'fair'
        else:
            return 'poor'
    
    def _get_agent_status_color(self) -> str:
        """Get color for agent status"""
        ratio = self.current_metrics.active_agents / max(self.current_metrics.total_agents, 1)
        if ratio >= 0.9:
            return 'success'
        elif ratio >= 0.7:
            return 'warning'
        else:
            return 'danger'
    
    def _get_response_time_color(self) -> str:
        """Get color for response time"""
        time_ms = self.current_metrics.average_response_time * 1000
        thresholds = self.alert_thresholds['response_time']
        
        if time_ms <= thresholds['good']:
            return 'success'
        elif time_ms <= thresholds['warning']:
            return 'warning'
        else:
            return 'danger'
    
    def _get_success_rate_color(self) -> str:
        """Get color for success rate"""
        rate = self._calculate_success_rate()
        if rate >= 95:
            return 'success'
        elif rate >= 85:
            return 'warning'
        else:
            return 'danger'
    
    def _get_sync_status_color(self) -> str:
        """Get color for sync status"""
        status = self.current_metrics.data_sync_status.lower()
        if status == 'synchronized':
            return 'success'
        elif status == 'syncing':
            return 'warning'
        else:
            return 'danger'
    
    def _calculate_trend(self, metric: str) -> str:
        """Calculate trend for a metric"""
        if len(self.metrics_history) < 2:
            return 'stable'
        
        # Compare last two values
        # Placeholder implementation
        return 'stable'
    
    def _get_metric_sparkline(self, metric: str) -> List[float]:
        """Get sparkline data for a metric"""
        # Return last 10 values for sparkline
        # Placeholder implementation
        return [50 + (i * 5) for i in range(10)]
    
    def _prepare_timeline_data(self) -> Dict[str, Any]:
        """Prepare timeline chart data"""
        return {
            'labels': [h['timestamp'] for h in self.metrics_history[-20:]],
            'datasets': [
                {
                    'label': 'Health Score',
                    'data': [h['metrics']['average_health_score'] * 100 
                            for h in self.metrics_history[-20:]]
                },
                {
                    'label': 'Active Agents',
                    'data': [h['metrics']['active_agents'] 
                            for h in self.metrics_history[-20:]]
                }
            ]
        }
    
    def _check_alerts(self) -> List[Dict[str, Any]]:
        """Check for system alerts"""
        alerts = []
        
        # Check health score
        if self.current_metrics.average_health_score < self.alert_thresholds['health_score']['warning']:
            alerts.append({
                'type': 'warning' if self.current_metrics.average_health_score >= self.alert_thresholds['health_score']['critical'] else 'critical',
                'message': f"Low system health: {self.current_metrics.average_health_score * 100:.1f}%",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Check response time
        if self.current_metrics.average_response_time * 1000 > self.alert_thresholds['response_time']['warning']:
            alerts.append({
                'type': 'warning',
                'message': f"High response time: {self.current_metrics.average_response_time * 1000:.0f}ms",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Check coordination failures
        if self.current_metrics.failed_coordinations > 10:
            alerts.append({
                'type': 'warning',
                'message': f"Multiple coordination failures: {self.current_metrics.failed_coordinations}",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return alerts
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export current metrics for external use"""
        return {
            'current': self.current_metrics.__dict__,
            'history': self.metrics_history,
            'alerts': self._check_alerts()
        }