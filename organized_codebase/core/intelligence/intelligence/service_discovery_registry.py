"""
Service Discovery Registry
=========================

Enterprise service discovery system providing dynamic service registration,
health monitoring, load balancing, and automatic failover capabilities.

Features:
- Dynamic service registration and deregistration
- Health check monitoring with configurable intervals
- Service versioning and blue-green deployment support
- Load balancing with multiple algorithms
- Service metadata and tagging system
- Circuit breaker integration
- Service dependency mapping
- Automatic cleanup of stale services

Author: TestMaster Intelligence Team
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
import threading
from collections import defaultdict, deque
import hashlib
import weakref

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    IP_HASH = "ip_hash"

class ServiceType(Enum):
    """Service types in the system"""
    MICROSERVICE = "microservice"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    API_GATEWAY = "api_gateway"
    LOAD_BALANCER = "load_balancer"
    MONITORING = "monitoring"
    LOGGING = "logging"
    STORAGE = "storage"
    EXTERNAL_API = "external_api"

@dataclass
class HealthCheck:
    """Health check configuration"""
    endpoint: str = "/health"
    interval_seconds: int = 30
    timeout_seconds: int = 5
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    method: str = "GET"
    expected_status: int = 200
    expected_response: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

@dataclass
class ServiceMetadata:
    """Service metadata and configuration"""
    service_id: str
    service_name: str
    service_type: ServiceType
    version: str = "1.0.0"
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    environment: str = "production"
    region: str = "default"
    datacenter: str = "default"
    
    # Network configuration
    host: str = "localhost"
    port: int = 8080
    protocol: str = "http"
    base_path: str = "/"
    
    # Service configuration
    max_connections: int = 100
    weight: int = 100
    priority: int = 1
    
    # Health monitoring
    health_check: HealthCheck = field(default_factory=HealthCheck)
    
    # Lifecycle
    registered_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_full_url(self) -> str:
        """Get full service URL"""
        return f"{self.protocol}://{self.host}:{self.port}{self.base_path}"
    
    def get_health_check_url(self) -> str:
        """Get health check URL"""
        return f"{self.protocol}://{self.host}:{self.port}{self.health_check.endpoint}"

@dataclass
class ServiceInstance:
    """Runtime service instance with health tracking"""
    metadata: ServiceMetadata
    status: ServiceStatus = ServiceStatus.UNKNOWN
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_health_check: Optional[datetime] = None
    health_check_streak: int = 0  # Consecutive successful/failed checks
    response_time_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def get_success_rate(self) -> float:
        """Get request success rate"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100
    
    def get_average_response_time(self) -> float:
        """Get average response time"""
        if not self.response_time_history:
            return 0.0
        return sum(self.response_time_history) / len(self.response_time_history)
    
    def record_request(self, success: bool, response_time_ms: float):
        """Record request statistics"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.response_time_history.append(response_time_ms)

@dataclass
class ServiceDependency:
    """Service dependency definition"""
    dependent_service: str
    required_service: str
    dependency_type: str = "required"  # required, optional, preferred
    min_instances: int = 1
    max_latency_ms: float = 1000.0
    created_at: datetime = field(default_factory=datetime.now)

class LoadBalancer:
    """Service load balancer with multiple algorithms"""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN):
        self.algorithm = algorithm
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.consistent_hash_ring: Dict[str, List[Tuple[int, str]]] = {}
    
    def select_instance(self, service_name: str, instances: List[ServiceInstance],
                       client_key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select service instance using configured algorithm"""
        healthy_instances = [inst for inst in instances if inst.status == ServiceStatus.HEALTHY]
        
        if not healthy_instances:
            # Fallback to degraded instances if no healthy ones
            degraded_instances = [inst for inst in instances if inst.status == ServiceStatus.DEGRADED]
            if degraded_instances:
                healthy_instances = degraded_instances
            else:
                return None
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_select(service_name, healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(service_name, healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.RANDOM:
            return self._random_select(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            return self._consistent_hash_select(service_name, healthy_instances, client_key)
        elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
            return self._ip_hash_select(healthy_instances, client_key)
        else:
            return healthy_instances[0] if healthy_instances else None
    
    def _round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection"""
        counter = self.round_robin_counters[service_name]
        selected = instances[counter % len(instances)]
        self.round_robin_counters[service_name] = (counter + 1) % len(instances)
        return selected
    
    def _weighted_round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection"""
        total_weight = sum(inst.metadata.weight for inst in instances)
        if total_weight == 0:
            return self._round_robin_select(service_name, instances)
        
        counter = self.round_robin_counters[service_name]
        target_weight = (counter % total_weight) + 1
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.metadata.weight
            if current_weight >= target_weight:
                self.round_robin_counters[service_name] = counter + 1
                return instance
        
        return instances[0]  # Fallback
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection"""
        return min(instances, key=lambda inst: inst.current_connections)
    
    def _random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random selection"""
        import random
        return random.choice(instances)
    
    def _consistent_hash_select(self, service_name: str, instances: List[ServiceInstance],
                               client_key: Optional[str]) -> ServiceInstance:
        """Consistent hash selection"""
        if not client_key:
            return self._round_robin_select(service_name, instances)
        
        # Build hash ring if not exists or outdated
        instance_ids = [inst.metadata.service_id for inst in instances]
        ring_key = f"{service_name}:{':'.join(sorted(instance_ids))}"
        
        if ring_key not in self.consistent_hash_ring:
            self.consistent_hash_ring[ring_key] = self._build_hash_ring(instances)
        
        ring = self.consistent_hash_ring[ring_key]
        client_hash = hash(client_key) & 0x7FFFFFFF  # Ensure positive
        
        # Find first instance with hash >= client_hash
        for ring_hash, instance_id in ring:
            if ring_hash >= client_hash:
                return next(inst for inst in instances if inst.metadata.service_id == instance_id)
        
        # Wrap around to first instance
        return next(inst for inst in instances if inst.metadata.service_id == ring[0][1])
    
    def _build_hash_ring(self, instances: List[ServiceInstance], virtual_nodes: int = 150) -> List[Tuple[int, str]]:
        """Build consistent hash ring"""
        ring = []
        for instance in instances:
            for i in range(virtual_nodes):
                virtual_key = f"{instance.metadata.service_id}:{i}"
                ring_hash = hash(virtual_key) & 0x7FFFFFFF
                ring.append((ring_hash, instance.metadata.service_id))
        
        ring.sort()
        return ring
    
    def _ip_hash_select(self, instances: List[ServiceInstance], client_key: Optional[str]) -> ServiceInstance:
        """IP hash selection"""
        if not client_key:
            return self._round_robin_select("default", instances)
        
        client_hash = hash(client_key)
        return instances[client_hash % len(instances)]

class ServiceDiscoveryRegistry:
    """
    Enterprise service discovery registry providing dynamic service registration,
    health monitoring, and intelligent load balancing.
    """
    
    def __init__(self, health_check_interval: int = 30, cleanup_interval: int = 300):
        self.health_check_interval = health_check_interval
        self.cleanup_interval = cleanup_interval
        
        # Service storage
        self.services: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self.service_metadata: Dict[str, ServiceMetadata] = {}
        self.service_dependencies: Dict[str, List[ServiceDependency]] = defaultdict(list)
        
        # Load balancer
        self.load_balancer = LoadBalancer()
        
        # Health monitoring
        self.health_monitoring_active = False
        self.health_check_tasks: Set[asyncio.Task] = set()
        
        # Service watchers
        self.service_watchers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Registry statistics
        self.registry_stats = {
            'total_services': 0,
            'healthy_services': 0,
            'unhealthy_services': 0,
            'registrations': 0,
            'deregistrations': 0,
            'health_checks_performed': 0,
            'start_time': datetime.now()
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Service Discovery Registry initialized")
    
    def register_service(self, metadata: ServiceMetadata) -> bool:
        """Register a new service instance"""
        with self.lock:
            try:
                instance = ServiceInstance(metadata=metadata, status=ServiceStatus.STARTING)
                
                # Add to services
                self.services[metadata.service_name].append(instance)
                self.service_metadata[metadata.service_id] = metadata
                
                # Update statistics
                self.registry_stats['registrations'] += 1
                self.registry_stats['total_services'] = sum(len(instances) for instances in self.services.values())
                
                logger.info(f"Registered service: {metadata.service_name} ({metadata.service_id})")
                
                # Notify watchers
                self._notify_watchers(metadata.service_name, "registered", instance)
                
                # Start health checking if monitoring is active
                if self.health_monitoring_active:
                    self._start_health_monitoring_for_instance(instance)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to register service {metadata.service_id}: {e}")
                return False
    
    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance"""
        with self.lock:
            try:
                # Find and remove service instance
                for service_name, instances in self.services.items():
                    for i, instance in enumerate(instances):
                        if instance.metadata.service_id == service_id:
                            # Remove from list
                            removed_instance = instances.pop(i)
                            
                            # Clean up empty service lists
                            if not instances:
                                del self.services[service_name]
                            
                            # Remove metadata
                            if service_id in self.service_metadata:
                                del self.service_metadata[service_id]
                            
                            # Update statistics
                            self.registry_stats['deregistrations'] += 1
                            self.registry_stats['total_services'] = sum(len(instances) for instances in self.services.values())
                            
                            logger.info(f"Deregistered service: {service_name} ({service_id})")
                            
                            # Notify watchers
                            self._notify_watchers(service_name, "deregistered", removed_instance)
                            
                            return True
                
                logger.warning(f"Service not found for deregistration: {service_id}")
                return False
                
            except Exception as e:
                logger.error(f"Failed to deregister service {service_id}: {e}")
                return False
    
    def discover_services(self, service_name: str, tags: Optional[Set[str]] = None,
                         environment: Optional[str] = None) -> List[ServiceInstance]:
        """Discover services by name and optional filters"""
        with self.lock:
            instances = self.services.get(service_name, [])
            
            # Apply filters
            filtered_instances = []
            for instance in instances:
                # Filter by tags
                if tags and not tags.issubset(instance.metadata.tags):
                    continue
                
                # Filter by environment
                if environment and instance.metadata.environment != environment:
                    continue
                
                filtered_instances.append(instance)
            
            return filtered_instances
    
    def get_service_instance(self, service_name: str, client_key: Optional[str] = None,
                           tags: Optional[Set[str]] = None) -> Optional[ServiceInstance]:
        """Get a service instance using load balancing"""
        instances = self.discover_services(service_name, tags)
        return self.load_balancer.select_instance(service_name, instances, client_key)
    
    def update_service_status(self, service_id: str, status: ServiceStatus):
        """Update service status"""
        with self.lock:
            for instances in self.services.values():
                for instance in instances:
                    if instance.metadata.service_id == service_id:
                        old_status = instance.status
                        instance.status = status
                        
                        # Update statistics
                        if old_status != status:
                            self._update_health_stats()
                        
                        # Notify watchers
                        self._notify_watchers(instance.metadata.service_name, "status_changed", instance)
                        
                        logger.debug(f"Updated service {service_id} status: {old_status.value} -> {status.value}")
                        return
    
    def add_service_dependency(self, dependency: ServiceDependency):
        """Add service dependency"""
        with self.lock:
            self.service_dependencies[dependency.dependent_service].append(dependency)
            logger.info(f"Added dependency: {dependency.dependent_service} -> {dependency.required_service}")
    
    def get_service_dependencies(self, service_name: str) -> List[ServiceDependency]:
        """Get service dependencies"""
        return self.service_dependencies.get(service_name, [])
    
    def watch_service(self, service_name: str, callback: Callable):
        """Watch service changes"""
        self.service_watchers[service_name].append(callback)
        logger.info(f"Added watcher for service: {service_name}")
    
    def unwatch_service(self, service_name: str, callback: Callable):
        """Stop watching service changes"""
        if callback in self.service_watchers[service_name]:
            self.service_watchers[service_name].remove(callback)
            logger.info(f"Removed watcher for service: {service_name}")
    
    async def start_health_monitoring(self):
        """Start health monitoring for all services"""
        if self.health_monitoring_active:
            return
        
        logger.info("Starting service health monitoring")
        self.health_monitoring_active = True
        
        # Start health monitoring for existing services
        with self.lock:
            for instances in self.services.values():
                for instance in instances:
                    self._start_health_monitoring_for_instance(instance)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.health_check_tasks.add(cleanup_task)
        
        logger.info("Service health monitoring started")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        if not self.health_monitoring_active:
            return
        
        logger.info("Stopping service health monitoring")
        self.health_monitoring_active = False
        
        # Cancel all health check tasks
        for task in self.health_check_tasks:
            task.cancel()
        
        await asyncio.gather(*self.health_check_tasks, return_exceptions=True)
        self.health_check_tasks.clear()
        
        logger.info("Service health monitoring stopped")
    
    def _start_health_monitoring_for_instance(self, instance: ServiceInstance):
        """Start health monitoring for a specific instance"""
        if self.health_monitoring_active:
            task = asyncio.create_task(self._health_check_loop(instance))
            self.health_check_tasks.add(task)
    
    async def _health_check_loop(self, instance: ServiceInstance):
        """Health check loop for a service instance"""
        service_id = instance.metadata.service_id
        
        while self.health_monitoring_active:
            try:
                # Perform health check
                is_healthy = await self._perform_health_check(instance)
                
                # Update health check tracking
                instance.last_health_check = datetime.now()
                self.registry_stats['health_checks_performed'] += 1
                
                # Update status based on health check result
                if is_healthy:
                    instance.health_check_streak += 1
                    if (instance.status in [ServiceStatus.UNHEALTHY, ServiceStatus.STARTING] and
                        instance.health_check_streak >= instance.metadata.health_check.healthy_threshold):
                        self.update_service_status(service_id, ServiceStatus.HEALTHY)
                else:
                    instance.health_check_streak = max(0, instance.health_check_streak - 1)
                    if (instance.status == ServiceStatus.HEALTHY and
                        (instance.health_check_streak * -1) >= instance.metadata.health_check.unhealthy_threshold):
                        self.update_service_status(service_id, ServiceStatus.UNHEALTHY)
                
                # Wait for next check
                await asyncio.sleep(instance.metadata.health_check.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {service_id}: {e}")
                await asyncio.sleep(instance.metadata.health_check.interval_seconds)
    
    async def _perform_health_check(self, instance: ServiceInstance) -> bool:
        """Perform health check on service instance"""
        try:
            start_time = time.time()
            
            # Simulate HTTP health check (in real implementation, use aiohttp)
            await asyncio.sleep(0.01)  # Simulate network delay
            
            # Simulate success/failure based on service behavior
            import random
            is_healthy = random.random() > 0.05  # 95% healthy rate
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record response time
            instance.response_time_history.append(response_time_ms)
            
            return is_healthy
            
        except Exception as e:
            logger.warning(f"Health check failed for {instance.metadata.service_id}: {e}")
            return False
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.health_monitoring_active:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_services()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_stale_services(self):
        """Clean up stale service instances"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=10)
            stale_services = []
            
            for service_name, instances in self.services.items():
                for instance in instances:
                    if (instance.last_health_check and
                        instance.last_health_check < cutoff_time and
                        instance.status == ServiceStatus.UNHEALTHY):
                        stale_services.append((service_name, instance.metadata.service_id))
            
            # Remove stale services
            for service_name, service_id in stale_services:
                logger.info(f"Cleaning up stale service: {service_name} ({service_id})")
                self.deregister_service(service_id)
    
    def _notify_watchers(self, service_name: str, event_type: str, instance: ServiceInstance):
        """Notify service watchers of changes"""
        for callback in self.service_watchers.get(service_name, []):
            try:
                callback(event_type, instance)
            except Exception as e:
                logger.error(f"Service watcher callback error: {e}")
    
    def _update_health_stats(self):
        """Update health statistics"""
        healthy_count = 0
        unhealthy_count = 0
        
        for instances in self.services.values():
            for instance in instances:
                if instance.status == ServiceStatus.HEALTHY:
                    healthy_count += 1
                elif instance.status == ServiceStatus.UNHEALTHY:
                    unhealthy_count += 1
        
        self.registry_stats['healthy_services'] = healthy_count
        self.registry_stats['unhealthy_services'] = unhealthy_count
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status"""
        with self.lock:
            uptime = (datetime.now() - self.registry_stats['start_time']).total_seconds()
            
            # Service summary by status
            status_summary = defaultdict(int)
            service_summary = {}
            
            for service_name, instances in self.services.items():
                instance_statuses = defaultdict(int)
                for instance in instances:
                    status_summary[instance.status.value] += 1
                    instance_statuses[instance.status.value] += 1
                
                service_summary[service_name] = {
                    'total_instances': len(instances),
                    'status_breakdown': dict(instance_statuses),
                    'healthy_instances': instance_statuses['healthy'],
                    'load_balancing_algorithm': self.load_balancer.algorithm.value
                }
            
            return {
                'status': 'active' if self.health_monitoring_active else 'inactive',
                'uptime_seconds': uptime,
                'statistics': self.registry_stats.copy(),
                'service_summary': service_summary,
                'status_distribution': dict(status_summary),
                'total_services': len(self.services),
                'total_instances': sum(len(instances) for instances in self.services.values()),
                'dependencies': {
                    service: len(deps) for service, deps in self.service_dependencies.items()
                },
                'watchers': {
                    service: len(watchers) for service, watchers in self.service_watchers.items()
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_service_topology(self) -> Dict[str, Any]:
        """Get service topology and dependency graph"""
        with self.lock:
            topology = {
                'services': {},
                'dependencies': [],
                'clusters': {}
            }
            
            # Build service information
            for service_name, instances in self.services.items():
                service_info = {
                    'name': service_name,
                    'instances': []
                }
                
                for instance in instances:
                    instance_info = {
                        'id': instance.metadata.service_id,
                        'host': instance.metadata.host,
                        'port': instance.metadata.port,
                        'status': instance.status.value,
                        'version': instance.metadata.version,
                        'environment': instance.metadata.environment,
                        'tags': list(instance.metadata.tags),
                        'connections': instance.current_connections,
                        'success_rate': instance.get_success_rate(),
                        'avg_response_time': instance.get_average_response_time()
                    }
                    service_info['instances'].append(instance_info)
                
                topology['services'][service_name] = service_info
            
            # Build dependency information
            for service_name, dependencies in self.service_dependencies.items():
                for dep in dependencies:
                    topology['dependencies'].append({
                        'from': dep.dependent_service,
                        'to': dep.required_service,
                        'type': dep.dependency_type,
                        'min_instances': dep.min_instances,
                        'max_latency_ms': dep.max_latency_ms
                    })
            
            return topology
    
    def shutdown(self):
        """Shutdown service discovery registry"""
        if self.health_monitoring_active:
            # Create and run shutdown task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop_health_monitoring())
            loop.close()
        
        logger.info("Service Discovery Registry shutdown")

# Global service discovery registry instance
service_discovery_registry = ServiceDiscoveryRegistry()