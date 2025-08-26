#!/usr/bin/env python3
"""
📈 ATOM: Agent Status Panels Component
======================================
Individual agent status display panels.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class PanelType(Enum):
    """Types of status panels"""
    COMPACT = "compact"
    DETAILED = "detailed"
    MINI = "mini"
    EXPANDED = "expanded"

@dataclass
class AgentStatusData:
    """Agent status data structure"""
    agent_id: str
    name: str
    type: str
    status: str
    health_score: float
    resource_usage: Dict[str, float]
    active_tasks: int
    completed_tasks: int
    error_count: int
    uptime: timedelta
    last_activity: datetime

class AgentStatusPanels:
    """Agent status display panels component"""
    
    def __init__(self):
        self.panel_configs = {}
        self.status_cache = {}
        self.display_preferences = self._initialize_preferences()
        self.alert_rules = self._initialize_alert_rules()
    
    def _initialize_preferences(self) -> Dict[str, Any]:
        """Initialize display preferences"""
        return {
            'default_panel_type': PanelType.DETAILED,
            'show_resource_usage': True,
            'show_task_counts': True,
            'show_health_indicators': True,
            'enable_sparklines': True,
            'update_animation': True,
            'color_coding': True
        }
    
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert rules"""
        return {
            'health_score': {'threshold': 0.7, 'type': 'below'},
            'error_count': {'threshold': 5, 'type': 'above'},
            'resource_usage': {'threshold': 0.9, 'type': 'above'},
            'inactive_time': {'threshold': 300, 'type': 'seconds'}
        }
    
    def render_status_panel(self, agent_id: str, status: Dict[str, Any], 
                           panel_type: PanelType = None) -> Dict[str, Any]:
        """
        Render individual agent status panel
        
        Args:
            agent_id: Agent identifier
            status: Agent status data
            panel_type: Type of panel to render
            
        Returns:
            Status panel configuration
        """
        panel_type = panel_type or self.display_preferences['default_panel_type']
        
        # Convert status dict to dataclass
        status_data = self._parse_status_data(agent_id, status)
        
        # Cache status for history tracking
        self.status_cache[agent_id] = status_data
        
        # Render based on panel type
        if panel_type == PanelType.COMPACT:
            return self._render_compact_panel(status_data)
        elif panel_type == PanelType.DETAILED:
            return self._render_detailed_panel(status_data)
        elif panel_type == PanelType.MINI:
            return self._render_mini_panel(status_data)
        else:
            return self._render_expanded_panel(status_data)
    
    def _render_compact_panel(self, status: AgentStatusData) -> Dict[str, Any]:
        """Render compact status panel"""
        return {
            'panel_type': 'compact',
            'agent_id': status.agent_id,
            'layout': 'horizontal',
            'content': {
                'header': {
                    'title': status.name,
                    'subtitle': status.type,
                    'status_badge': {
                        'text': status.status,
                        'color': self._get_status_color(status.status)
                    }
                },
                'metrics': [
                    {
                        'label': 'Health',
                        'value': f"{status.health_score * 100:.0f}%",
                        'indicator': self._get_health_indicator(status.health_score)
                    },
                    {
                        'label': 'Tasks',
                        'value': f"{status.active_tasks}/{status.completed_tasks}",
                        'tooltip': f"Active/Completed"
                    },
                    {
                        'label': 'Uptime',
                        'value': self._format_uptime(status.uptime)
                    }
                ],
                'quick_actions': ['restart', 'pause', 'details']
            },
            'alerts': self._check_alerts(status)
        }
    
    def _render_detailed_panel(self, status: AgentStatusData) -> Dict[str, Any]:
        """Render detailed status panel"""
        return {
            'panel_type': 'detailed',
            'agent_id': status.agent_id,
            'layout': 'card',
            'content': {
                'header': {
                    'title': status.name,
                    'subtitle': f"{status.type} Agent",
                    'icon': self._get_agent_icon(status.type),
                    'status': {
                        'text': status.status,
                        'color': self._get_status_color(status.status),
                        'pulse': status.status == 'active'
                    }
                },
                'health_section': {
                    'title': 'System Health',
                    'score': status.health_score,
                    'visual': 'radial_gauge',
                    'color': self._get_health_color(status.health_score),
                    'breakdown': self._get_health_breakdown(status)
                },
                'resource_section': {
                    'title': 'Resource Usage',
                    'metrics': [
                        {
                            'name': 'CPU',
                            'value': status.resource_usage.get('cpu', 0),
                            'max': 100,
                            'unit': '%',
                            'chart': self._get_resource_sparkline('cpu', status.agent_id)
                        },
                        {
                            'name': 'Memory',
                            'value': status.resource_usage.get('memory', 0),
                            'max': 100,
                            'unit': '%',
                            'chart': self._get_resource_sparkline('memory', status.agent_id)
                        },
                        {
                            'name': 'Network',
                            'value': status.resource_usage.get('network', 0),
                            'max': 1000,
                            'unit': 'Mbps',
                            'chart': self._get_resource_sparkline('network', status.agent_id)
                        }
                    ]
                },
                'activity_section': {
                    'title': 'Activity',
                    'stats': [
                        {'label': 'Active Tasks', 'value': status.active_tasks, 'color': 'primary'},
                        {'label': 'Completed', 'value': status.completed_tasks, 'color': 'success'},
                        {'label': 'Errors', 'value': status.error_count, 'color': 'danger'},
                        {'label': 'Last Activity', 'value': self._format_time_ago(status.last_activity)}
                    ]
                },
                'actions': {
                    'primary': ['configure', 'restart'],
                    'secondary': ['pause', 'resume', 'logs', 'metrics']
                }
            },
            'alerts': self._check_alerts(status),
            'refresh_interval': 5000
        }
    
    def _render_mini_panel(self, status: AgentStatusData) -> Dict[str, Any]:
        """Render mini status panel"""
        return {
            'panel_type': 'mini',
            'agent_id': status.agent_id,
            'layout': 'inline',
            'content': {
                'name': status.name[:10],  # Truncate for mini view
                'status_icon': self._get_status_icon(status.status),
                'health': f"{status.health_score * 100:.0f}%",
                'alert': len(self._check_alerts(status)) > 0
            },
            'tooltip': self._get_mini_tooltip(status)
        }
    
    def _render_expanded_panel(self, status: AgentStatusData) -> Dict[str, Any]:
        """Render expanded status panel with full details"""
        return {
            'panel_type': 'expanded',
            'agent_id': status.agent_id,
            'layout': 'full_width',
            'tabs': [
                {
                    'id': 'overview',
                    'label': 'Overview',
                    'content': self._render_detailed_panel(status)['content']
                },
                {
                    'id': 'performance',
                    'label': 'Performance',
                    'content': self._render_performance_tab(status)
                },
                {
                    'id': 'history',
                    'label': 'History',
                    'content': self._render_history_tab(status)
                },
                {
                    'id': 'configuration',
                    'label': 'Configuration',
                    'content': self._render_configuration_tab(status)
                }
            ],
            'alerts': self._check_alerts(status)
        }
    
    def _render_performance_tab(self, status: AgentStatusData) -> Dict[str, Any]:
        """Render performance tab content"""
        return {
            'charts': [
                {
                    'title': 'Task Throughput',
                    'type': 'line',
                    'data': self._get_throughput_data(status.agent_id)
                },
                {
                    'title': 'Response Times',
                    'type': 'histogram',
                    'data': self._get_response_time_data(status.agent_id)
                },
                {
                    'title': 'Error Rate',
                    'type': 'area',
                    'data': self._get_error_rate_data(status.agent_id)
                }
            ],
            'metrics_table': self._get_performance_metrics_table(status)
        }
    
    def _render_history_tab(self, status: AgentStatusData) -> Dict[str, Any]:
        """Render history tab content"""
        return {
            'timeline': self._get_agent_timeline(status.agent_id),
            'events': self._get_agent_events(status.agent_id),
            'statistics': self._get_historical_statistics(status.agent_id)
        }
    
    def _render_configuration_tab(self, status: AgentStatusData) -> Dict[str, Any]:
        """Render configuration tab content"""
        return {
            'settings': self._get_agent_settings(status.agent_id),
            'capabilities': self._get_agent_capabilities(status.agent_id),
            'connections': self._get_agent_connections(status.agent_id)
        }
    
    def _parse_status_data(self, agent_id: str, status: Dict[str, Any]) -> AgentStatusData:
        """Parse status dictionary into dataclass"""
        return AgentStatusData(
            agent_id=agent_id,
            name=status.get('name', agent_id),
            type=status.get('type', 'unknown'),
            status=status.get('status', 'offline'),
            health_score=status.get('health_score', 0.0),
            resource_usage=status.get('resource_usage', {}),
            active_tasks=status.get('active_tasks', 0),
            completed_tasks=status.get('completed_tasks', 0),
            error_count=status.get('error_count', 0),
            uptime=timedelta(seconds=status.get('uptime_seconds', 0)),
            last_activity=datetime.fromisoformat(status.get('last_activity', datetime.utcnow().isoformat()))
        )
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status"""
        colors = {
            'active': 'success',
            'idle': 'info',
            'processing': 'warning',
            'error': 'danger',
            'offline': 'secondary'
        }
        return colors.get(status.lower(), 'secondary')
    
    def _get_health_indicator(self, score: float) -> str:
        """Get health indicator symbol"""
        if score >= 0.9:
            return '🟢'
        elif score >= 0.7:
            return '🟡'
        else:
            return '🔴'
    
    def _get_health_color(self, score: float) -> str:
        """Get health color"""
        if score >= 0.9:
            return 'success'
        elif score >= 0.7:
            return 'warning'
        else:
            return 'danger'
    
    def _get_agent_icon(self, agent_type: str) -> str:
        """Get icon for agent type"""
        icons = {
            'intelligence': '🧠',
            'performance': '⚡',
            'visualization': '📊',
            'coordination': '🔄',
            'integration': '🔗'
        }
        return icons.get(agent_type.lower(), '🤖')
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for status"""
        icons = {
            'active': '✅',
            'idle': '💤',
            'processing': '⚙️',
            'error': '❌',
            'offline': '🔌'
        }
        return icons.get(status.lower(), '❓')
    
    def _format_uptime(self, uptime: timedelta) -> str:
        """Format uptime duration"""
        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format time ago"""
        delta = datetime.utcnow() - timestamp
        
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"
    
    def _check_alerts(self, status: AgentStatusData) -> List[Dict[str, Any]]:
        """Check for alerts based on status"""
        alerts = []
        
        # Health score alert
        if status.health_score < self.alert_rules['health_score']['threshold']:
            alerts.append({
                'type': 'health',
                'severity': 'warning',
                'message': f"Low health score: {status.health_score * 100:.0f}%"
            })
        
        # Error count alert
        if status.error_count > self.alert_rules['error_count']['threshold']:
            alerts.append({
                'type': 'errors',
                'severity': 'danger',
                'message': f"High error count: {status.error_count}"
            })
        
        # Resource usage alerts
        for resource, usage in status.resource_usage.items():
            if usage / 100 > self.alert_rules['resource_usage']['threshold']:
                alerts.append({
                    'type': 'resource',
                    'severity': 'warning',
                    'message': f"High {resource} usage: {usage:.0f}%"
                })
        
        return alerts
    
    # Placeholder methods for data retrieval
    def _get_health_breakdown(self, status: AgentStatusData) -> List[Dict[str, Any]]:
        return []
    
    def _get_resource_sparkline(self, resource: str, agent_id: str) -> List[float]:
        return [50 + (i * 5) for i in range(10)]
    
    def _get_mini_tooltip(self, status: AgentStatusData) -> str:
        return f"{status.name}: {status.status} | Health: {status.health_score * 100:.0f}%"
    
    def _get_throughput_data(self, agent_id: str) -> Dict[str, Any]:
        return {'labels': [], 'values': []}
    
    def _get_response_time_data(self, agent_id: str) -> Dict[str, Any]:
        return {'labels': [], 'values': []}
    
    def _get_error_rate_data(self, agent_id: str) -> Dict[str, Any]:
        return {'labels': [], 'values': []}
    
    def _get_performance_metrics_table(self, status: AgentStatusData) -> List[Dict[str, Any]]:
        return []
    
    def _get_agent_timeline(self, agent_id: str) -> List[Dict[str, Any]]:
        return []
    
    def _get_agent_events(self, agent_id: str) -> List[Dict[str, Any]]:
        return []
    
    def _get_historical_statistics(self, agent_id: str) -> Dict[str, Any]:
        return {}
    
    def _get_agent_settings(self, agent_id: str) -> Dict[str, Any]:
        return {}
    
    def _get_agent_capabilities(self, agent_id: str) -> List[str]:
        return []
    
    def _get_agent_connections(self, agent_id: str) -> List[Dict[str, Any]]:
        return []