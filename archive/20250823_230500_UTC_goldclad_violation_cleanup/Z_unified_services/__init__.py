#!/usr/bin/env python3
"""
Unified Services Package - Agent Z Phase 2
Service layer initialization and orchestration

Provides unified service layer architecture with:
- WebSocket communication service
- Multi-agent coordination service  
- Unified API endpoints service
- Performance monitoring service
- Service registry and orchestration
"""

import logging
from typing import Optional, Dict, Any
from flask import Flask

from .websocket_service import UnifiedWebSocketService, get_websocket_service
from .coordination_service import MultiAgentCoordinationService, get_coordination_service
from .api_service import UnifiedAPIService, get_api_service
from .monitoring_service import UnifiedMonitoringService, get_monitoring_service

logger = logging.getLogger(__name__)


class UnifiedServiceRegistry:
    """
    Service registry and orchestration for the unified service layer.
    Manages lifecycle of all services and provides unified access.
    """
    
    def __init__(self, websocket_port: int = 8765):
        self.websocket_port = websocket_port
        self.running = False
        
        # Service instances
        self.websocket_service: Optional[UnifiedWebSocketService] = None
        self.coordination_service: Optional[MultiAgentCoordinationService] = None
        self.api_service: Optional[UnifiedAPIService] = None
        self.monitoring_service: Optional[UnifiedMonitoringService] = None
        self.flask_app: Optional[Flask] = None
        
        logger.info("Unified Service Registry initialized")
    
    async def start_all_services(self, flask_app: Optional[Flask] = None) -> bool:
        """Start all unified services"""
        try:
            logger.info("Starting unified service layer...")
            
            # Initialize Flask app if provided
            if flask_app:
                self.flask_app = flask_app
            
            # Start coordination service
            self.coordination_service = get_coordination_service()
            logger.info("✅ Coordination service started")
            
            # Start monitoring service
            self.monitoring_service = get_monitoring_service()
            self.monitoring_service.start()
            logger.info("✅ Monitoring service started")
            
            # Start API service
            self.api_service = get_api_service(self.flask_app)
            logger.info("✅ API service started")
            
            # Start WebSocket service
            self.websocket_service = get_websocket_service(self.websocket_port)
            await self.websocket_service.start()
            logger.info(f"✅ WebSocket service started on port {self.websocket_port}")
            
            self.running = True
            logger.info("🚀 All unified services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start unified services: {e}")
            await self.stop_all_services()
            return False
    
    async def stop_all_services(self):
        """Stop all unified services"""
        logger.info("Stopping unified service layer...")
        
        try:
            # Stop WebSocket service
            if self.websocket_service:
                self.websocket_service.stop()
                logger.info("✅ WebSocket service stopped")
            
            # Stop monitoring service
            if self.monitoring_service:
                self.monitoring_service.stop()
                logger.info("✅ Monitoring service stopped")
            
            # API service stops with Flask app
            logger.info("✅ API service stopped")
            
            # Coordination service cleanup
            logger.info("✅ Coordination service stopped")
            
            self.running = False
            logger.info("🛑 All unified services stopped")
            
        except Exception as e:
            logger.error(f"Error stopping services: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {
            'registry_running': self.running,
            'websocket_service': None,
            'coordination_service': None,
            'api_service': None,
            'monitoring_service': None,
            'timestamp': None
        }
        
        try:
            from datetime import datetime
            status['timestamp'] = datetime.now().isoformat()
            
            # WebSocket service status
            if self.websocket_service:
                status['websocket_service'] = self.websocket_service.health_check()
            
            # Coordination service status
            if self.coordination_service:
                swarm_status = self.coordination_service.get_swarm_status()
                status['coordination_service'] = {
                    'active_agents': swarm_status.get('active_agents', 0),
                    'total_agents': swarm_status.get('total_agents', 0),
                    'message_queue_size': swarm_status.get('message_queue_size', 0),
                    'status': 'healthy'
                }
            
            # Monitoring service status
            if self.monitoring_service:
                status['monitoring_service'] = {
                    'running': self.monitoring_service.running,
                    'checks_performed': self.monitoring_service.monitoring_stats['checks_performed'],
                    'active_alerts': len(self.monitoring_service.active_alerts),
                    'status': 'healthy'
                }
            
            # API service status
            if self.api_service:
                api_stats = self.api_service._get_api_stats()
                status['api_service'] = {
                    'total_requests': api_stats['total_requests'],
                    'active_clients': api_stats['active_clients'],
                    'cache_size': api_stats['cache_size'],
                    'status': 'healthy'
                }
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            status['error'] = str(e)
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all services"""
        try:
            summary = {
                'websocket_performance': {},
                'coordination_performance': {},
                'monitoring_performance': {},
                'overall_health': 'unknown'
            }
            
            # WebSocket performance
            if self.websocket_service:
                ws_metrics = self.websocket_service.get_performance_metrics()
                summary['websocket_performance'] = {
                    'avg_response_time_ms': ws_metrics.get('avg_response_time_ms', 0),
                    'messages_per_second': ws_metrics.get('messages_per_second', 0),
                    'clients_connected': ws_metrics.get('clients_connected', 0),
                    'latency_target_met': ws_metrics.get('latency_optimization', {}).get('latency_performance') == 'GOOD'
                }
            
            # Coordination performance
            if self.coordination_service:
                coord_stats = self.coordination_service.coordination_stats
                summary['coordination_performance'] = {
                    'messages_processed': coord_stats['messages_processed'],
                    'handoffs_completed': coord_stats['handoffs_completed'],
                    'errors': coord_stats['errors']
                }
            
            # Monitoring performance
            if self.monitoring_service:
                monitoring_stats = self.monitoring_service.monitoring_stats
                summary['monitoring_performance'] = {
                    'checks_performed': monitoring_stats['checks_performed'],
                    'alerts_generated': monitoring_stats['alerts_generated'],
                    'active_alerts': len(self.monitoring_service.active_alerts)
                }
            
            # Determine overall health
            latency_good = summary['websocket_performance'].get('latency_target_met', False)
            low_errors = summary['coordination_performance'].get('errors', 0) < 5
            few_alerts = summary['monitoring_performance'].get('active_alerts', 0) < 3
            
            if latency_good and low_errors and few_alerts:
                summary['overall_health'] = 'healthy'
            elif latency_good or (low_errors and few_alerts):
                summary['overall_health'] = 'warning'
            else:
                summary['overall_health'] = 'critical'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e), 'overall_health': 'error'}


# Global registry instance
_service_registry: Optional[UnifiedServiceRegistry] = None


def get_service_registry(websocket_port: int = 8765) -> UnifiedServiceRegistry:
    """Get global service registry instance"""
    global _service_registry
    if _service_registry is None:
        _service_registry = UnifiedServiceRegistry(websocket_port)
    return _service_registry


async def start_unified_services(websocket_port: int = 8765, 
                                flask_app: Optional[Flask] = None) -> bool:
    """Start all unified services through registry"""
    registry = get_service_registry(websocket_port)
    return await registry.start_all_services(flask_app)


async def stop_unified_services():
    """Stop all unified services through registry"""
    if _service_registry:
        await _service_registry.stop_all_services()


# Export all service classes and functions
__all__ = [
    'UnifiedWebSocketService',
    'MultiAgentCoordinationService', 
    'UnifiedAPIService',
    'UnifiedMonitoringService',
    'UnifiedServiceRegistry',
    'get_websocket_service',
    'get_coordination_service',
    'get_api_service', 
    'get_monitoring_service',
    'get_service_registry',
    'start_unified_services',
    'stop_unified_services'
]