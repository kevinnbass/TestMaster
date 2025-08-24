"""
Service Registry System
=======================

Centralized service discovery and registration system.
Provides service location, health tracking, and metadata management.

Author: TestMaster Team
"""

import json
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Health status of a service"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    CRITICAL = "critical"


@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    endpoint_id: str = field(default_factory=lambda: f"endpoint_{uuid.uuid4().hex[:12]}")
    protocol: str = "http"
    host: str = "localhost"
    port: int = 8080
    path: str = "/"
    weight: int = 100  # For weighted load balancing
    zone: str = "default"  # For zone-aware routing
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        """Get full endpoint URL"""
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"


@dataclass
class ServiceDescriptor:
    """Complete service description"""
    service_id: str = field(default_factory=lambda: f"service_{uuid.uuid4().hex[:12]}")
    name: str = ""
    service_type: str = ""
    version: str = "1.0.0"
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    health_status: ServiceHealth = ServiceHealth.UNKNOWN
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.health_status in [ServiceHealth.HEALTHY, ServiceHealth.DEGRADED]
    
    def is_stale(self, timeout_seconds: int = 60) -> bool:
        """Check if service heartbeat is stale"""
        if not self.last_heartbeat:
            return True
        return (datetime.now() - self.last_heartbeat).total_seconds() > timeout_seconds


class ServiceRegistry:
    """
    Centralized service registry with discovery capabilities.
    
    Features:
    - Service registration and deregistration
    - Health tracking and monitoring
    - Service discovery by type, capabilities, or tags
    - Load balancing support
    - Zone-aware routing
    - Service versioning
    - Dependency tracking
    """
    
    def __init__(self, heartbeat_timeout: int = 60, cleanup_interval: int = 120):
        self.registry_id = f"registry_{uuid.uuid4().hex[:12]}"
        self.services: Dict[str, ServiceDescriptor] = {}
        self.service_types: Dict[str, Set[str]] = defaultdict(set)
        self.service_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.service_tags: Dict[str, Set[str]] = defaultdict(set)
        self.service_zones: Dict[str, Set[str]] = defaultdict(set)
        self.heartbeat_timeout = heartbeat_timeout
        self.cleanup_interval = cleanup_interval
        self.started_at = datetime.now()
        self.statistics = {
            "registrations": 0,
            "deregistrations": 0,
            "discoveries": 0,
            "heartbeats": 0,
            "health_checks": 0
        }
        
        # Start cleanup task
        self.cleanup_task = None
    
    def register_service(self, descriptor: ServiceDescriptor) -> str:
        """Register a new service"""
        service_id = descriptor.service_id
        
        # Store service descriptor
        self.services[service_id] = descriptor
        
        # Update indexes
        self.service_types[descriptor.service_type].add(service_id)
        
        for capability in descriptor.capabilities:
            self.service_capabilities[capability].add(service_id)
        
        for tag in descriptor.tags:
            self.service_tags[tag].add(service_id)
        
        for endpoint in descriptor.endpoints:
            self.service_zones[endpoint.zone].add(service_id)
        
        self.statistics["registrations"] += 1
        
        logger.info(f"Registered service {descriptor.name} ({service_id}) of type {descriptor.service_type}")
        return service_id
    
    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service"""
        if service_id not in self.services:
            return False
        
        descriptor = self.services[service_id]
        
        # Remove from indexes
        self.service_types[descriptor.service_type].discard(service_id)
        
        for capability in descriptor.capabilities:
            self.service_capabilities[capability].discard(service_id)
        
        for tag in descriptor.tags:
            self.service_tags[tag].discard(service_id)
        
        for endpoint in descriptor.endpoints:
            self.service_zones[endpoint.zone].discard(service_id)
        
        # Remove service
        del self.services[service_id]
        
        self.statistics["deregistrations"] += 1
        
        logger.info(f"Deregistered service {descriptor.name} ({service_id})")
        return True
    
    def update_heartbeat(self, service_id: str) -> bool:
        """Update service heartbeat"""
        if service_id not in self.services:
            return False
        
        self.services[service_id].last_heartbeat = datetime.now()
        self.statistics["heartbeats"] += 1
        return True
    
    def update_health(self, service_id: str, health_status: ServiceHealth) -> bool:
        """Update service health status"""
        if service_id not in self.services:
            return False
        
        old_status = self.services[service_id].health_status
        self.services[service_id].health_status = health_status
        
        if old_status != health_status:
            logger.info(f"Service {service_id} health changed from {old_status.value} to {health_status.value}")
        
        self.statistics["health_checks"] += 1
        return True
    
    def discover_services(self, 
                         service_type: Optional[str] = None,
                         capabilities: Optional[List[str]] = None,
                         tags: Optional[Set[str]] = None,
                         zone: Optional[str] = None,
                         health_only: bool = True) -> List[ServiceDescriptor]:
        """Discover services matching criteria"""
        candidates = set(self.services.keys())
        
        # Filter by type
        if service_type:
            candidates &= self.service_types.get(service_type, set())
        
        # Filter by capabilities
        if capabilities:
            for capability in capabilities:
                candidates &= self.service_capabilities.get(capability, set())
        
        # Filter by tags
        if tags:
            for tag in tags:
                candidates &= self.service_tags.get(tag, set())
        
        # Filter by zone
        if zone:
            candidates &= self.service_zones.get(zone, set())
        
        # Get service descriptors
        services = [self.services[sid] for sid in candidates]
        
        # Filter by health if requested
        if health_only:
            services = [s for s in services if s.is_healthy() and not s.is_stale(self.heartbeat_timeout)]
        
        self.statistics["discoveries"] += 1
        
        return services
    
    def get_service(self, service_id: str) -> Optional[ServiceDescriptor]:
        """Get a specific service"""
        return self.services.get(service_id)
    
    def get_service_endpoint(self, 
                            service_type: str,
                            zone: Optional[str] = None,
                            load_balance: bool = True) -> Optional[ServiceEndpoint]:
        """Get a service endpoint with optional load balancing"""
        services = self.discover_services(service_type=service_type, zone=zone)
        
        if not services:
            return None
        
        # Collect all healthy endpoints
        endpoints = []
        for service in services:
            for endpoint in service.endpoints:
                if not zone or endpoint.zone == zone:
                    endpoints.append(endpoint)
        
        if not endpoints:
            return None
        
        if not load_balance:
            return endpoints[0]
        
        # Weighted random selection
        import random
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return endpoints[0]
        
        rand = random.uniform(0, total_weight)
        cumulative = 0
        
        for endpoint in endpoints:
            cumulative += endpoint.weight
            if rand <= cumulative:
                return endpoint
        
        return endpoints[-1]
    
    def get_service_dependencies(self, service_id: str, recursive: bool = True) -> Set[str]:
        """Get service dependencies"""
        if service_id not in self.services:
            return set()
        
        dependencies = set()
        to_process = [service_id]
        processed = set()
        
        while to_process:
            current = to_process.pop(0)
            if current in processed:
                continue
            
            processed.add(current)
            
            if current in self.services:
                service_deps = self.services[current].dependencies
                dependencies.update(service_deps)
                
                if recursive:
                    to_process.extend(service_deps)
        
        return dependencies
    
    def check_service_health(self, service_id: str) -> ServiceHealth:
        """Check and update service health"""
        if service_id not in self.services:
            return ServiceHealth.UNKNOWN
        
        service = self.services[service_id]
        
        # Check if heartbeat is stale
        if service.is_stale(self.heartbeat_timeout):
            service.health_status = ServiceHealth.UNHEALTHY
            return ServiceHealth.UNHEALTHY
        
        # Check dependencies
        dependencies = self.get_service_dependencies(service_id, recursive=False)
        unhealthy_deps = 0
        
        for dep_id in dependencies:
            if dep_id in self.services:
                dep = self.services[dep_id]
                if not dep.is_healthy():
                    unhealthy_deps += 1
        
        # Update health based on dependencies
        if unhealthy_deps > 0:
            if unhealthy_deps >= len(dependencies) / 2:
                service.health_status = ServiceHealth.CRITICAL
            else:
                service.health_status = ServiceHealth.DEGRADED
        else:
            # Service and dependencies are healthy
            service.health_status = ServiceHealth.HEALTHY
        
        return service.health_status
    
    async def cleanup_stale_services(self):
        """Remove stale services from registry"""
        stale_services = []
        
        for service_id, service in self.services.items():
            if service.is_stale(self.heartbeat_timeout * 3):  # 3x timeout for removal
                stale_services.append(service_id)
        
        for service_id in stale_services:
            self.deregister_service(service_id)
            logger.info(f"Cleaned up stale service {service_id}")
        
        return len(stale_services)
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                removed = await self.cleanup_stale_services()
                if removed > 0:
                    logger.info(f"Cleanup task removed {removed} stale services")
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status and statistics"""
        healthy_services = sum(1 for s in self.services.values() if s.is_healthy())
        unhealthy_services = sum(1 for s in self.services.values() if not s.is_healthy())
        stale_services = sum(1 for s in self.services.values() if s.is_stale(self.heartbeat_timeout))
        
        return {
            "registry_id": self.registry_id,
            "uptime_seconds": (datetime.now() - self.started_at).total_seconds(),
            "total_services": len(self.services),
            "healthy_services": healthy_services,
            "unhealthy_services": unhealthy_services,
            "stale_services": stale_services,
            "service_types": list(self.service_types.keys()),
            "zones": list(self.service_zones.keys()),
            "statistics": self.statistics,
            "heartbeat_timeout": self.heartbeat_timeout,
            "cleanup_interval": self.cleanup_interval
        }
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry state for backup/migration"""
        return {
            "registry_id": self.registry_id,
            "exported_at": datetime.now().isoformat(),
            "services": {
                sid: {
                    "name": s.name,
                    "type": s.service_type,
                    "version": s.version,
                    "health": s.health_status.value,
                    "endpoints": [{
                        "url": ep.url,
                        "zone": ep.zone,
                        "weight": ep.weight
                    } for ep in s.endpoints],
                    "capabilities": s.capabilities,
                    "dependencies": s.dependencies,
                    "tags": list(s.tags),
                    "metadata": s.metadata
                }
                for sid, s in self.services.items()
            },
            "statistics": self.statistics
        }
    
    def import_registry(self, data: Dict[str, Any]) -> bool:
        """Import registry state from export"""
        try:
            # Clear current registry
            self.services.clear()
            self.service_types.clear()
            self.service_capabilities.clear()
            self.service_tags.clear()
            self.service_zones.clear()
            
            # Import services
            for service_id, service_data in data["services"].items():
                endpoints = [
                    ServiceEndpoint(
                        protocol=ep["url"].split("://")[0],
                        host=ep["url"].split("://")[1].split(":")[0],
                        port=int(ep["url"].split(":")[-1].split("/")[0]),
                        zone=ep["zone"],
                        weight=ep["weight"]
                    )
                    for ep in service_data["endpoints"]
                ]
                
                descriptor = ServiceDescriptor(
                    service_id=service_id,
                    name=service_data["name"],
                    service_type=service_data["type"],
                    version=service_data["version"],
                    endpoints=endpoints,
                    health_status=ServiceHealth(service_data["health"]),
                    capabilities=service_data["capabilities"],
                    dependencies=service_data["dependencies"],
                    tags=set(service_data["tags"]),
                    metadata=service_data["metadata"]
                )
                
                self.register_service(descriptor)
            
            # Import statistics
            if "statistics" in data:
                self.statistics.update(data["statistics"])
            
            logger.info(f"Imported {len(data['services'])} services into registry")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
            return False