"""
System Integration Core - Modularized Main Class
Extracted from advanced_system_integration.py for modularization

Main orchestration class using modular components.
"""

import json
import asyncio
import time
import logging
import threading
import statistics
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from integration_models import (
    ServiceStatus, IntegrationType, ServiceHealth, 
    IntegrationEndpoint, SystemMetrics
)
from integration_database import IntegrationDatabase
from service_health_monitor import ServiceHealthMonitor

logger = logging.getLogger(__name__)


class AdvancedSystemIntegration:
    """
    Advanced System Integration Engine - Modularized Version
    
    Provides comprehensive integration capabilities using modular components:
    - Service discovery and health monitoring
    - Cross-system communication protocols
    - Integration validation and testing
    - Unified configuration management
    - Service orchestration and lifecycle management
    """
    
    def __init__(self, config_file: str = "integration_config.json"):
        self.config_file = config_file
        self.services: Dict[str, IntegrationEndpoint] = {}
        self.health_status: Dict[str, ServiceHealth] = {}
        self.integration_metrics = SystemMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, datetime.now())
        self.monitoring_active = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize modular components
        self.database = IntegrationDatabase()
        self.health_monitor = ServiceHealthMonitor(self.database)
        
        # Initialize system
        self._load_configuration()
        self._register_core_services()
        
        logger.info("Advanced System Integration Engine initialized successfully")
    
    def _load_configuration(self):
        """Load integration configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Configuration loaded from {self.config_file}")
            else:
                # Create default configuration
                config = self._create_default_configuration()
                self._save_configuration(config)
                logger.info("Default configuration created")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            config = self._create_default_configuration()
        
        return config
    
    def _create_default_configuration(self) -> Dict[str, Any]:
        """Create default integration configuration"""
        return {
            "monitoring": {
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "retry_count": 3,
                "health_check_enabled": True
            },
            "alerts": {
                "enabled": True,
                "email_notifications": False,
                "slack_notifications": False,
                "threshold_response_time_ms": 5000,
                "threshold_error_rate": 0.05
            },
            "integration": {
                "max_concurrent_checks": 10,
                "circuit_breaker_enabled": True,
                "rate_limiting_enabled": True,
                "retry_backoff_factor": 2.0
            }
        }
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _register_core_services(self):
        """Register core system services for integration"""
        core_services = [
            IntegrationEndpoint(
                name="ai_intelligence_engine",
                type=IntegrationType.API,
                url="http://localhost:8001",
                health_check_path="/api/health",
                required_dependencies=["database_monitor"]
            ),
            IntegrationEndpoint(
                name="enterprise_analytics_engine",
                type=IntegrationType.API,
                url="http://localhost:8002",
                health_check_path="/api/health",
                required_dependencies=["multi_database_integration"]
            ),
            IntegrationEndpoint(
                name="commercial_features_suite",
                type=IntegrationType.API,
                url="http://localhost:8003",
                health_check_path="/api/health",
                required_dependencies=[]
            ),
            IntegrationEndpoint(
                name="multi_database_integration",
                type=IntegrationType.DATABASE,
                url="http://localhost:8004",
                health_check_path="/api/health",
                required_dependencies=[]
            ),
            IntegrationEndpoint(
                name="automated_optimization_system",
                type=IntegrationType.API,
                url="http://localhost:8005",
                health_check_path="/api/health",
                required_dependencies=["multi_database_integration", "enterprise_analytics_engine"]
            ),
            IntegrationEndpoint(
                name="production_deployment_system",
                type=IntegrationType.API,
                url="http://localhost:8006",
                health_check_path="/api/health",
                required_dependencies=[]
            )
        ]
        
        for service in core_services:
            self.register_service(service)
        
        logger.info(f"Registered {len(core_services)} core services")
    
    def register_service(self, service: IntegrationEndpoint):
        """Register a service for integration monitoring"""
        try:
            self.services[service.name] = service
            self.database.register_service(service)
            logger.info(f"Service '{service.name}' registered successfully")
        except Exception as e:
            logger.error(f"Error registering service '{service.name}': {e}")
    
    def unregister_service(self, service_name: str):
        """Unregister a service from integration monitoring"""
        try:
            if service_name in self.services:
                del self.services[service_name]
                self.database.unregister_service(service_name)
                logger.info(f"Service '{service_name}' unregistered successfully")
        except Exception as e:
            logger.error(f"Error unregistering service '{service_name}': {e}")
    
    async def check_all_services_health(self) -> Dict[str, ServiceHealth]:
        """Check health of all registered services using health monitor"""
        health_results = await self.health_monitor.check_all_services_health(self.services)
        
        # Update local health status
        self.health_status.update(health_results)
        
        # Update system metrics
        self.integration_metrics = self.health_monitor.calculate_system_metrics(health_results)
        self.database.store_system_metrics(self.integration_metrics)
        
        return health_results
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Health monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Run health checks asynchronously
                asyncio.run(self.check_all_services_health())
                
                # Log current status
                logger.info(f"Health check completed - Score: {self.integration_metrics.integration_score:.1f}%, "
                          f"Healthy: {self.integration_metrics.healthy_services}/{self.integration_metrics.total_services}")
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "integration_metrics": asdict(self.integration_metrics),
            "service_health": {
                name: asdict(health) if health else None 
                for name, health in self.health_status.items()
            },
            "monitoring_active": self.monitoring_active,
            "registered_services": len(self.services),
            "system_health_grade": (
                "A" if self.integration_metrics.integration_score >= 95 else
                "B" if self.integration_metrics.integration_score >= 85 else
                "C" if self.integration_metrics.integration_score >= 75 else
                "D" if self.integration_metrics.integration_score >= 60 else
                "F"
            )
        }