"""
Service Health Monitoring
Extracted from advanced_system_integration.py for modularization

Handles health checking and monitoring of integration services.
"""

import asyncio
import time
import logging
import random
import statistics
from datetime import datetime
from typing import Dict

from integration_models import ServiceHealth, ServiceStatus, IntegrationEndpoint, SystemMetrics

logger = logging.getLogger(__name__)


class ServiceHealthMonitor:
    """Handles service health checking and monitoring"""
    
    def __init__(self, database_handler):
        self.database_handler = database_handler
    
    async def check_service_health(self, service: IntegrationEndpoint) -> ServiceHealth:
        """Check the health of a specific service"""
        start_time = time.time()
        
        try:
            # Simulate health check (in real implementation, this would be HTTP request)
            await asyncio.sleep(random.uniform(0.01, 0.1))  # Simulate network latency
            
            # Simulate service responses with realistic patterns
            success_rate = 0.95  # 95% success rate
            is_healthy = random.random() < success_rate
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if is_healthy:
                status = ServiceStatus.HEALTHY
                error_message = None
            else:
                status = ServiceStatus.DEGRADED if random.random() < 0.7 else ServiceStatus.UNHEALTHY
                error_message = "Service temporarily unavailable" if status == ServiceStatus.DEGRADED else "Service critical error"
            
            health = ServiceHealth(
                service_name=service.name,
                status=status,
                response_time_ms=response_time_ms,
                last_check=datetime.now(),
                error_message=error_message,
                uptime_percentage=95.5 if is_healthy else 85.2
            )
            
            # Store health check result
            self.database_handler.store_health_result(health)
            
            return health
            
        except Exception as e:
            error_health = ServiceHealth(
                service_name=service.name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(),
                error_message=str(e),
                uptime_percentage=0.0
            )
            
            self.database_handler.store_health_result(error_health)
            return error_health
    
    async def check_all_services_health(self, services: Dict[str, IntegrationEndpoint]) -> Dict[str, ServiceHealth]:
        """Check health of all registered services"""
        health_results = {}
        
        # Use async tasks for concurrent health checks
        tasks = []
        for service in services.values():
            task = asyncio.create_task(self.check_service_health(service))
            tasks.append((service.name, task))
        
        # Wait for all health checks to complete
        for service_name, task in tasks:
            try:
                health = await task
                health_results[service_name] = health
            except Exception as e:
                logger.error(f"Error checking health for service '{service_name}': {e}")
        
        return health_results
    
    def calculate_system_metrics(self, health_results: Dict[str, ServiceHealth]) -> SystemMetrics:
        """Calculate system-wide integration metrics"""
        if not health_results:
            return SystemMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, datetime.now())
        
        total_services = len(health_results)
        healthy_services = sum(1 for h in health_results.values() if h.status == ServiceStatus.HEALTHY)
        degraded_services = sum(1 for h in health_results.values() if h.status == ServiceStatus.DEGRADED)
        unhealthy_services = sum(1 for h in health_results.values() if h.status == ServiceStatus.UNHEALTHY)
        
        response_times = [h.response_time_ms for h in health_results.values()]
        average_response_time = statistics.mean(response_times) if response_times else 0.0
        
        uptime_percentages = [h.uptime_percentage for h in health_results.values()]
        overall_uptime = statistics.mean(uptime_percentages) if uptime_percentages else 0.0
        
        # Calculate integration score (0-100)
        health_score = (healthy_services / total_services) * 100 if total_services > 0 else 0
        performance_score = max(0, 100 - (average_response_time / 50))  # Penalize high response times
        integration_score = (health_score * 0.7) + (performance_score * 0.3)
        
        return SystemMetrics(
            total_services=total_services,
            healthy_services=healthy_services,
            degraded_services=degraded_services,
            unhealthy_services=unhealthy_services,
            average_response_time=average_response_time,
            uptime_percentage=overall_uptime,
            integration_score=integration_score,
            last_updated=datetime.now()
        )