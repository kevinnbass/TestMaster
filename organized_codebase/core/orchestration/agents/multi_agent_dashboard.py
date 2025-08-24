#!/usr/bin/env python3
"""
🤖 ATOM: Multi-Agent Dashboard Component
========================================
Provides multi-agent overview and monitoring interface.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    """Agent type enumeration"""
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    DELTA = "delta"
    EPSILON = "epsilon"

class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    response_time: float
    throughput: int
    error_rate: float
    uptime: float
    resource_usage: Dict[str, float]

class MultiAgentDashboard:
    """Multi-agent dashboard UI component"""
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.agent_connections: Dict[str, Set[str]] = {}
        self.dashboard_config = self._initialize_config()
    
    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize dashboard configuration"""
        return {
            'refresh_interval': 5000,
            'max_agents_display': 20,
            'enable_animations': True,
            'color_scheme': 'multi_agent',
            'layout_mode': 'grid'
        }
    
    def render_agent_overview(self, agents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render multi-agent overview dashboard
        
        Args:
            agents: Dictionary of agent data
            
        Returns:
            Agent overview UI configuration
        """
        self.agents = agents
        
        return {
            'view_type': 'multi_agent_overview',
            'layout': self._get_overview_layout(),
            'sections': {
                'summary': self._render_summary_section(),
                'agent_grid': self._render_agent_grid(),
                'performance': self._render_performance_section(),
                'connections': self._render_connections_view()
            },
            'interactions': self._get_interaction_config()
        }
    
    def _get_overview_layout(self) -> Dict[str, Any]:
        """Get overview layout configuration"""
        return {
            'type': 'responsive_grid',
            'columns': {
                'desktop': 4,
                'tablet': 2,
                'mobile': 1
            },
            'spacing': 'medium',
            'animation': 'fade-in'
        }
    
    def _render_summary_section(self) -> Dict[str, Any]:
        """Render summary statistics section"""
        total_agents = len(self.agents)
        active_agents = sum(1 for a in self.agents.values() 
                          if a.get('status') == AgentStatus.ACTIVE.value)
        
        return {
            'title': 'System Overview',
            'stats': [
                {
                    'label': 'Total Agents',
                    'value': total_agents,
                    'icon': '🤖',
                    'trend': self._calculate_trend('total')
                },
                {
                    'label': 'Active',
                    'value': active_agents,
                    'icon': '✅',
                    'color': 'success'
                },
                {
                    'label': 'Processing',
                    'value': sum(1 for a in self.agents.values() 
                               if a.get('status') == AgentStatus.PROCESSING.value),
                    'icon': '⚙️',
                    'color': 'warning'
                },
                {
                    'label': 'Errors',
                    'value': sum(1 for a in self.agents.values() 
                               if a.get('status') == AgentStatus.ERROR.value),
                    'icon': '❌',
                    'color': 'danger'
                }
            ]
        }
    
    def _render_agent_grid(self) -> List[Dict[str, Any]]:
        """Render agent grid cards"""
        agent_cards = []
        
        for agent_id, agent_data in self.agents.items():
            metrics = self.agent_metrics.get(agent_id, None)
            
            card = {
                'id': agent_id,
                'type': agent_data.get('agent_type', 'unknown'),
                'status': agent_data.get('status', AgentStatus.OFFLINE.value),
                'health': agent_data.get('health_score', 0),
                'details': {
                    'host': agent_data.get('host', 'N/A'),
                    'port': agent_data.get('port', 0),
                    'uptime': self._format_uptime(agent_data.get('last_heartbeat')),
                    'capabilities': agent_data.get('capabilities', [])
                }
            }
            
            if metrics:
                card['metrics'] = {
                    'response_time': f"{metrics.response_time:.2f}ms",
                    'throughput': f"{metrics.throughput}/s",
                    'error_rate': f"{metrics.error_rate:.1%}"
                }
            
            agent_cards.append(card)
        
        return sorted(agent_cards, key=lambda x: (x['status'] != 'active', x['id']))
    
    def _render_performance_section(self) -> Dict[str, Any]:
        """Render performance metrics section"""
        return {
            'title': 'Performance Metrics',
            'charts': {
                'response_times': self._get_response_time_chart(),
                'throughput': self._get_throughput_chart(),
                'resource_usage': self._get_resource_usage_chart()
            },
            'summary': self._get_performance_summary()
        }
    
    def _render_connections_view(self) -> Dict[str, Any]:
        """Render agent connections network view"""
        nodes = []
        edges = []
        
        for agent_id in self.agents:
            nodes.append({
                'id': agent_id,
                'label': self.agents[agent_id].get('agent_type', agent_id),
                'status': self.agents[agent_id].get('status', 'offline')
            })
        
        for source, targets in self.agent_connections.items():
            for target in targets:
                edges.append({
                    'source': source,
                    'target': target,
                    'type': 'coordination'
                })
        
        return {
            'type': 'network_graph',
            'nodes': nodes,
            'edges': edges,
            'layout': 'force-directed',
            'interactive': True
        }
    
    def _get_interaction_config(self) -> Dict[str, Any]:
        """Get interaction configuration"""
        return {
            'click_actions': {
                'agent_card': 'show_details',
                'connection': 'highlight_path',
                'metric': 'show_history'
            },
            'hover_effects': True,
            'tooltips': True,
            'context_menu': ['inspect', 'restart', 'configure']
        }
    
    def _calculate_trend(self, metric_type: str) -> str:
        """Calculate trend indicator"""
        # Placeholder for trend calculation
        return 'stable'
    
    def _format_uptime(self, last_heartbeat: Any) -> str:
        """Format uptime duration"""
        if not last_heartbeat:
            return 'Unknown'
        
        try:
            if isinstance(last_heartbeat, str):
                last_time = datetime.fromisoformat(last_heartbeat)
            else:
                last_time = last_heartbeat
            
            delta = datetime.utcnow() - last_time
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            
            return f"{hours}h {minutes}m"
        except:
            return 'Unknown'
    
    def _get_response_time_chart(self) -> Dict[str, Any]:
        """Get response time chart data"""
        return {
            'type': 'line',
            'data': {
                'labels': [f"T-{i}" for i in range(10, 0, -1)],
                'datasets': [
                    {
                        'label': agent_id,
                        'data': [50 + (i * 5) for i in range(10)]  # Placeholder
                    }
                    for agent_id in list(self.agents.keys())[:5]
                ]
            }
        }
    
    def _get_throughput_chart(self) -> Dict[str, Any]:
        """Get throughput chart data"""
        return {
            'type': 'bar',
            'data': {
                'labels': [a.get('agent_type', 'unknown') for a in self.agents.values()],
                'values': [100 + (i * 10) for i in range(len(self.agents))]  # Placeholder
            }
        }
    
    def _get_resource_usage_chart(self) -> Dict[str, Any]:
        """Get resource usage chart data"""
        return {
            'type': 'radar',
            'metrics': ['CPU', 'Memory', 'Network', 'Disk'],
            'agents': {
                agent_id: [65, 75, 70, 60]  # Placeholder values
                for agent_id in list(self.agents.keys())[:3]
            }
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        return {
            'avg_response_time': '45ms',
            'total_throughput': '1250/s',
            'system_efficiency': '87%',
            'resource_utilization': '62%'
        }
    
    def update_agent_status(self, agent_id: str, status: Dict[str, Any]):
        """Update individual agent status"""
        if agent_id not in self.agents:
            self.agents[agent_id] = {}
        self.agents[agent_id].update(status)
    
    def update_agent_metrics(self, agent_id: str, metrics: AgentMetrics):
        """Update agent performance metrics"""
        self.agent_metrics[agent_id] = metrics
    
    def add_agent_connection(self, source: str, target: str):
        """Add connection between agents"""
        if source not in self.agent_connections:
            self.agent_connections[source] = set()
        self.agent_connections[source].add(target)