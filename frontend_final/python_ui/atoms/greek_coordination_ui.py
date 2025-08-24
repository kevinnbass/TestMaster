#!/usr/bin/env python3
"""
🌐 ATOM: Greek Coordination UI Component
=========================================
Handles Greek swarm coordination interface rendering and updates.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class GreekSwarmStatus:
    """Status representation for Greek swarm"""
    total_agents: int
    active_agents: int
    health_score: float
    load_distribution: float
    coordination_messages: int
    last_update: datetime

class GreekCoordinationUI:
    """Greek swarm coordination user interface component"""
    
    def __init__(self):
        self.swarm_status = None
        self.agent_statuses = {}
        self.coordination_log = []
        self.ui_state = {
            'selected_agent': None,
            'view_mode': 'overview',
            'auto_refresh': True
        }
    
    def render_swarm_dashboard(self, swarm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render Greek swarm coordination dashboard
        
        Args:
            swarm_data: Current swarm status and metrics
            
        Returns:
            Dashboard UI configuration and data
        """
        return {
            'dashboard_type': 'greek_swarm',
            'layout': self._get_dashboard_layout(),
            'components': {
                'header': self._render_header(swarm_data),
                'status_panel': self._render_status_panel(swarm_data),
                'agent_grid': self._render_agent_grid(swarm_data.get('agents', {})),
                'coordination_view': self._render_coordination_view()
            },
            'update_frequency': 5000,  # 5 seconds
            'theme': 'greek_coordination'
        }
    
    def _get_dashboard_layout(self) -> Dict[str, Any]:
        """Get dashboard layout configuration"""
        return {
            'grid': {
                'columns': 3,
                'rows': 'auto',
                'gap': '20px'
            },
            'responsive': True,
            'breakpoints': {
                'mobile': 768,
                'tablet': 1024
            }
        }
    
    def _render_header(self, swarm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render dashboard header"""
        return {
            'title': '🤝 Greek Swarm Coordination',
            'subtitle': f"Active Agents: {swarm_data.get('active_agents', 0)}",
            'status_indicator': self._get_swarm_health_indicator(swarm_data),
            'last_update': datetime.utcnow().isoformat()
        }
    
    def _render_status_panel(self, swarm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render swarm status panel"""
        metrics = swarm_data.get('metrics', {})
        return {
            'metrics': [
                {'label': 'Total Agents', 'value': metrics.get('total_agents', 0)},
                {'label': 'Active Agents', 'value': metrics.get('active_agents', 0)},
                {'label': 'Health Score', 'value': f"{metrics.get('average_health_score', 0) * 100:.1f}%"},
                {'label': 'Load Distribution', 'value': f"{metrics.get('load_distribution_score', 0) * 100:.1f}%"},
                {'label': 'Coordination Messages', 'value': metrics.get('total_coordination_messages', 0)}
            ],
            'charts': {
                'health_trend': self._get_health_trend_data(),
                'load_distribution': self._get_load_distribution_data(swarm_data)
            }
        }
    
    def _render_agent_grid(self, agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Render agent grid view"""
        agent_cards = []
        for agent_id, agent_info in agents.items():
            agent_cards.append({
                'id': agent_id,
                'type': agent_info.get('agent_type', 'unknown'),
                'status': agent_info.get('status', 'inactive'),
                'health': agent_info.get('health_score', 0),
                'load': agent_info.get('load_factor', 0),
                'host': f"{agent_info.get('host', 'localhost')}:{agent_info.get('port', 0)}",
                'last_heartbeat': agent_info.get('last_heartbeat', ''),
                'capabilities': agent_info.get('capabilities', [])
            })
        return agent_cards
    
    def _render_coordination_view(self) -> Dict[str, Any]:
        """Render coordination message view"""
        return {
            'recent_messages': self.coordination_log[-20:],
            'message_stats': self._get_coordination_stats(),
            'active_coordinations': self._get_active_coordinations()
        }
    
    def _get_swarm_health_indicator(self, swarm_data: Dict[str, Any]) -> str:
        """Get swarm health status indicator"""
        health_score = swarm_data.get('metrics', {}).get('average_health_score', 0)
        if health_score > 0.8:
            return 'healthy'
        elif health_score > 0.5:
            return 'warning'
        else:
            return 'critical'
    
    def _get_health_trend_data(self) -> List[Dict[str, Any]]:
        """Get health trend chart data"""
        # Return last 20 data points for trend visualization
        return [
            {'time': i, 'value': 0.75 + (i * 0.01)}  # Placeholder data
            for i in range(20)
        ]
    
    def _get_load_distribution_data(self, swarm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get load distribution chart data"""
        agents = swarm_data.get('agents', {})
        return {
            'labels': [a.get('agent_type', 'unknown') for a in agents.values()],
            'values': [a.get('load_factor', 0) * 100 for a in agents.values()]
        }
    
    def _get_coordination_stats(self) -> Dict[str, int]:
        """Get coordination message statistics"""
        return {
            'total': len(self.coordination_log),
            'successful': sum(1 for m in self.coordination_log if m.get('status') == 'success'),
            'failed': sum(1 for m in self.coordination_log if m.get('status') == 'failed'),
            'pending': sum(1 for m in self.coordination_log if m.get('status') == 'pending')
        }
    
    def _get_active_coordinations(self) -> List[Dict[str, Any]]:
        """Get currently active coordinations"""
        return [
            msg for msg in self.coordination_log
            if msg.get('status') == 'active'
        ][-5:]  # Last 5 active coordinations
    
    def update_swarm_status(self, status: Dict[str, Any]):
        """Update swarm status data"""
        self.swarm_status = GreekSwarmStatus(
            total_agents=status.get('total_agents', 0),
            active_agents=status.get('active_agents', 0),
            health_score=status.get('health_score', 0),
            load_distribution=status.get('load_distribution', 0),
            coordination_messages=status.get('coordination_messages', 0),
            last_update=datetime.utcnow()
        )
    
    def add_coordination_message(self, message: Dict[str, Any]):
        """Add coordination message to log"""
        self.coordination_log.append({
            **message,
            'timestamp': datetime.utcnow().isoformat()
        })
        # Keep only last 100 messages
        if len(self.coordination_log) > 100:
            self.coordination_log = self.coordination_log[-100:]
    
    def get_ui_state(self) -> Dict[str, Any]:
        """Get current UI state"""
        return self.ui_state
    
    def update_ui_state(self, updates: Dict[str, Any]):
        """Update UI state"""
        self.ui_state.update(updates)