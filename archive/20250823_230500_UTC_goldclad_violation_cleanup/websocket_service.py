#!/usr/bin/env python3
"""
Unified WebSocket Service Module - Agent Z Phase 2
Real-time communication service layer for dashboard consolidation

Provides unified WebSocket communication with support for:
- Architecture health streaming
- Multi-agent coordination 
- API cost tracking
- Cross-agent synthesis
- Performance optimization with <50ms latency targets
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
from dataclasses import asdict

from ..services.websocket_architecture_stream import (
    ArchitectureWebSocketStream,
    MessageType,
    WebSocketMessage,
    get_websocket_stream
)

logger = logging.getLogger(__name__)


class UnifiedWebSocketService:
    """
    Unified WebSocket service providing all real-time communication
    capabilities for the consolidated dashboard system.
    """
    
    def __init__(self, port: int = 8765, update_interval: int = 2):
        self.port = port
        self.update_interval = update_interval
        self.stream = get_websocket_stream(port)
        self.running = False
        
        # Service configuration for <50ms latency
        self.performance_config = {
            'max_concurrent_connections': 100,
            'message_batch_size': 5,  # Smaller batches for speed
            'compression_threshold': 512,  # Lower threshold
            'heartbeat_interval': 10,  # More frequent heartbeats
            'connection_timeout': 30,  # Faster timeouts
            'enable_connection_pooling': True
        }
        
        # Update stream configuration for performance
        self.stream.stream_config.update({
            'enable_compression': True,
            'enable_batching': True,
            'heartbeat_interval': self.performance_config['heartbeat_interval'],
            'max_message_size': 4096,  # Smaller message size for speed
            'connection_timeout': self.performance_config['connection_timeout']
        })
        
        # Reduce batch settings for lower latency
        self.stream.max_batch_size = self.performance_config['message_batch_size']
        self.stream.batch_timeout = 0.5  # Faster batch processing
        
        logger.info(f"Unified WebSocket Service initialized on port {port} with <50ms latency optimization")
    
    async def start(self) -> None:
        """Start the unified WebSocket service"""
        self.running = True
        logger.info("Starting Unified WebSocket Service...")
        
        try:
            await self.stream.start_streaming()
        except Exception as e:
            logger.error(f"Failed to start WebSocket service: {e}")
            self.running = False
            raise
    
    def stop(self) -> None:
        """Stop the unified WebSocket service"""
        self.running = False
        self.stream.stop_streaming()
        logger.info("Unified WebSocket Service stopped")
    
    def broadcast_architecture_health(self, health_data: Dict[str, Any]) -> None:
        """Broadcast architecture health update to all clients"""
        self.stream.queue_alert("architecture_health", json.dumps(health_data), priority="high")
    
    def broadcast_agent_status(self, agent_id: str, status_data: Dict[str, Any]) -> None:
        """Broadcast agent status update for multi-agent coordination"""
        self.stream.agent_status_cache[agent_id] = status_data
        message_data = {"agent_id": agent_id, "status": status_data}
        self.stream.queue_alert("agents_update", json.dumps(message_data), priority="normal")
    
    def broadcast_cost_update(self, provider: str, model: str, cost: float, tokens: int) -> None:
        """Broadcast API cost update for cost tracking"""
        cost_data = {
            "provider": provider,
            "model": model, 
            "cost": cost,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat()
        }
        self.stream.queue_alert("cost_update", json.dumps(cost_data), priority="normal")
    
    def broadcast_synthesis_insight(self, synthesis_id: str, insight_data: Dict[str, Any]) -> None:
        """Broadcast cross-agent synthesis insight"""
        self.stream.synthesis_processes[synthesis_id] = insight_data
        message_data = {"synthesis_id": synthesis_id, "insight": insight_data}
        self.stream.queue_alert("agent_synthesis", json.dumps(message_data), priority="normal")
    
    def broadcast_pattern_insight(self, pattern_id: str, pattern_data: Dict[str, Any]) -> None:
        """Broadcast pattern detection insight"""
        self.stream.pattern_insights[pattern_id] = pattern_data
        message_data = {"pattern_id": pattern_id, "pattern": pattern_data}
        self.stream.queue_alert("pattern_insight", json.dumps(message_data), priority="normal")
    
    def broadcast_coordination_message(self, message_type: str, data: Dict[str, Any]) -> None:
        """Broadcast inter-agent coordination message"""
        coordination_data = {
            "message_type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.stream.swarm_coordination_data[message_type] = coordination_data
        self.stream.queue_alert("coordination_message", json.dumps(coordination_data), priority="high")
    
    def get_connection_count(self) -> int:
        """Get current number of connected clients"""
        return len(self.stream.clients)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics optimized for <50ms latency monitoring"""
        base_metrics = self.stream.get_stream_metrics()
        
        # Add latency-specific metrics
        latency_metrics = {
            'target_latency_ms': 50,
            'current_avg_latency_ms': base_metrics.get('avg_response_time_ms', 0),
            'latency_performance': 'GOOD' if base_metrics.get('avg_response_time_ms', 0) < 50 else 'NEEDS_OPTIMIZATION',
            'connection_pooling_enabled': self.performance_config['enable_connection_pooling'],
            'batch_processing_optimized': self.stream.max_batch_size <= 5,
            'compression_optimized': self.performance_config['compression_threshold'] <= 512
        }
        
        base_metrics['latency_optimization'] = latency_metrics
        return base_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Service health check for monitoring"""
        return {
            'service': 'UnifiedWebSocketService',
            'status': 'healthy' if self.running else 'stopped',
            'port': self.port,
            'clients_connected': self.get_connection_count(),
            'performance_optimized': self.get_performance_metrics()['latency_optimization']['latency_performance'] == 'GOOD',
            'timestamp': datetime.now().isoformat()
        }


# Global service instance for unified access
_websocket_service: Optional[UnifiedWebSocketService] = None


def get_websocket_service(port: int = 8765) -> UnifiedWebSocketService:
    """Get global unified WebSocket service instance"""
    global _websocket_service
    if _websocket_service is None:
        _websocket_service = UnifiedWebSocketService(port=port)
    return _websocket_service


async def start_websocket_service(port: int = 8765) -> None:
    """Start the unified WebSocket service"""
    service = get_websocket_service(port)
    await service.start()