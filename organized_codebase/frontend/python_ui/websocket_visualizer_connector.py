#!/usr/bin/env python3
"""
WebSocket Visualizer Connector - Agent Z Integration
Connects WebSocket streams to dashboard visualizers
Agent Z - Service Infrastructure to Frontend Bridge
"""

import asyncio
from typing import Dict, Any, List, Callable
from datetime import datetime
from enum import Enum

# Import atomic components
from .websocket_frontend_stream import WebSocketFrontendStream, MessageType
from .realtime_dashboard_updates import RealtimeDashboardUpdates, UpdateType
from .dashboard_broadcast import DashboardBroadcast
from .dashboard_metrics_stream import DashboardMetricsStream, StreamType, MetricCategory
from .viz_engine import VizEngine
from .linkage_visualizer import LinkageVisualizer
from .performance_charts import PerformanceCharts
from .security_visualizations import SecurityVisualizations


class VisualizerType(Enum):
    """Types of visualizers"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    LINKAGE = "linkage"
    METRICS = "metrics"
    SWARM = "swarm"
    ANALYTICS = "analytics"


class WebSocketVisualizerConnector:
    """
    Connects WebSocket streams to visualization components
    Bridges Agent Z service infrastructure with frontend visualizers
    """
    
    def __init__(self):
        # WebSocket streaming components
        self.websocket_stream = WebSocketFrontendStream(port=8765)
        self.dashboard_updates = RealtimeDashboardUpdates()
        self.dashboard_broadcast = DashboardBroadcast()
        self.metrics_stream = DashboardMetricsStream()
        
        # Visualization components
        self.viz_engine = VizEngine()
        self.linkage_viz = LinkageVisualizer()
        self.performance_charts = PerformanceCharts()
        self.security_viz = SecurityVisualizations()
        
        # Connection mappings
        self.stream_to_viz_map: Dict[MessageType, List[VisualizerType]] = {
            MessageType.METRICS_UPDATE: [VisualizerType.PERFORMANCE, VisualizerType.METRICS],
            MessageType.PERFORMANCE_UPDATE: [VisualizerType.PERFORMANCE],
            MessageType.SYSTEM_ALERT: [VisualizerType.SECURITY],
            MessageType.ARCHITECTURE_HEALTH: [VisualizerType.LINKAGE],
            MessageType.SWARM_STATUS: [VisualizerType.SWARM],
            MessageType.SERVICE_STATUS: [VisualizerType.METRICS, VisualizerType.LINKAGE]
        }
        
        # Active connections
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.connector_metrics = {
            'streams_connected': 0,
            'visualizers_active': 0,
            'messages_routed': 0,
            'avg_routing_time_ms': 0.0
        }
    
    async def connect_stream_to_visualizer(self, stream_type: MessageType, 
                                          viz_type: VisualizerType) -> bool:
        """
        Connect WebSocket stream to visualizer
        Main interface for Agent Z integration
        """
        connection_id = f"{stream_type.value}:{viz_type.value}"
        
        if connection_id in self.active_connections:
            return True  # Already connected
        
        try:
            # Setup stream handler
            stream_handler = self._create_stream_handler(viz_type)
            
            # Subscribe to WebSocket updates
            await self.websocket_stream.add_update_callback(stream_handler)
            
            # Setup visualizer callback
            viz_callback = self._create_viz_callback(viz_type)
            self.dashboard_updates.subscribe_to_updates(connection_id, viz_callback)
            
            # Register connection
            self.active_connections[connection_id] = {
                'stream_type': stream_type,
                'viz_type': viz_type,
                'connected_at': datetime.now(),
                'handler': stream_handler,
                'callback': viz_callback
            }
            
            self.connector_metrics['streams_connected'] += 1
            self.connector_metrics['visualizers_active'] = len(
                set(conn['viz_type'] for conn in self.active_connections.values())
            )
            
            return True
            
        except Exception:
            return False
    
    def _create_stream_handler(self, viz_type: VisualizerType) -> Callable:
        """Create handler for WebSocket stream data"""
        async def handler(data: Dict[str, Any]):
            import time
            start_time = time.time()
            
            # Route to appropriate visualizer
            if viz_type == VisualizerType.PERFORMANCE:
                await self._handle_performance_data(data)
            elif viz_type == VisualizerType.SECURITY:
                await self._handle_security_data(data)
            elif viz_type == VisualizerType.LINKAGE:
                await self._handle_linkage_data(data)
            elif viz_type == VisualizerType.METRICS:
                await self._handle_metrics_data(data)
            
            # Update routing metrics
            routing_time = (time.time() - start_time) * 1000
            self._update_routing_metrics(routing_time)
            
        return handler
    
    def _create_viz_callback(self, viz_type: VisualizerType) -> Callable:
        """Create callback for visualizer updates"""
        def callback(update_data: Dict[str, Any]):
            # Process visualizer update
            if viz_type == VisualizerType.PERFORMANCE:
                self.performance_charts.update_chart(update_data)
            elif viz_type == VisualizerType.SECURITY:
                self.security_viz.update_visualization(update_data)
            elif viz_type == VisualizerType.LINKAGE:
                self.linkage_viz.update_linkage_display(update_data)
            
        return callback
    
    async def _handle_performance_data(self, data: Dict[str, Any]):
        """Handle performance data for visualization"""
        # Transform data for performance charts
        chart_data = {
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'metrics': data.get('data', {}),
            'type': 'performance'
        }
        
        # Update dashboard
        self.dashboard_updates.broadcast_dashboard_update(
            chart_data,
            UpdateType.PERFORMANCE,
            priority="normal"
        )
    
    async def _handle_security_data(self, data: Dict[str, Any]):
        """Handle security data for visualization"""
        # Transform for security visualizations
        security_data = {
            'alerts': data.get('alerts', []),
            'threat_level': data.get('threat_level', 'low'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Broadcast to security dashboard
        await self.dashboard_broadcast.broadcast_to_dashboard(
            'security',
            security_data,
            priority="high"
        )
    
    async def _handle_linkage_data(self, data: Dict[str, Any]):
        """Handle linkage data for visualization"""
        # Process for linkage visualizer
        linkage_data = {
            'connections': data.get('connections', []),
            'health': data.get('health', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update linkage display
        self.linkage_viz.process_linkage_data(linkage_data)
    
    async def _handle_metrics_data(self, data: Dict[str, Any]):
        """Handle metrics data for visualization"""
        # Stream metrics to dashboard
        await self.metrics_stream.stream_metrics_to_dashboard(
            data.get('metrics', {}),
            MetricCategory.SYSTEM
        )
    
    def _update_routing_metrics(self, routing_time_ms: float):
        """Update routing performance metrics"""
        self.connector_metrics['messages_routed'] += 1
        
        # Update average
        current_avg = self.connector_metrics['avg_routing_time_ms']
        total = self.connector_metrics['messages_routed']
        self.connector_metrics['avg_routing_time_ms'] = (
            (current_avg * (total - 1) + routing_time_ms) / total
        )
    
    async def setup_default_connections(self):
        """Setup default stream to visualizer connections"""
        # Connect performance streams
        await self.connect_stream_to_visualizer(
            MessageType.METRICS_UPDATE,
            VisualizerType.PERFORMANCE
        )
        
        # Connect security streams
        await self.connect_stream_to_visualizer(
            MessageType.SYSTEM_ALERT,
            VisualizerType.SECURITY
        )
        
        # Connect architecture streams
        await self.connect_stream_to_visualizer(
            MessageType.ARCHITECTURE_HEALTH,
            VisualizerType.LINKAGE
        )
        
        # Connect service status
        await self.connect_stream_to_visualizer(
            MessageType.SERVICE_STATUS,
            VisualizerType.METRICS
        )
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get status of all connections"""
        return {
            'active_connections': len(self.active_connections),
            'connections': [
                {
                    'id': conn_id,
                    'stream': conn['stream_type'].value,
                    'visualizer': conn['viz_type'].value,
                    'connected_at': conn['connected_at'].isoformat()
                }
                for conn_id, conn in self.active_connections.items()
            ],
            'metrics': self.connector_metrics,
            'latency_compliant': self.connector_metrics['avg_routing_time_ms'] < 50
        }
    
    def disconnect_stream(self, connection_id: str):
        """Disconnect a stream from visualizer"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            self.connector_metrics['streams_connected'] -= 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics"""
        return {
            **self.connector_metrics,
            'active_connections': len(self.active_connections),
            'latency_target_met': self.connector_metrics['avg_routing_time_ms'] < 50
        }


# Global connector instance
_websocket_viz_connector = None

def get_websocket_visualizer_connector() -> WebSocketVisualizerConnector:
    """Get global WebSocket visualizer connector instance"""
    global _websocket_viz_connector
    if _websocket_viz_connector is None:
        _websocket_viz_connector = WebSocketVisualizerConnector()
    return _websocket_viz_connector

async def initialize_websocket_visualizers():
    """Initialize WebSocket to visualizer connections"""
    connector = get_websocket_visualizer_connector()
    await connector.setup_default_connections()
    return connector