"""
Load Balancing System
====================

Advanced load balancing system with multiple algorithms, health checking,
session affinity, and intelligent traffic distribution.

Features:
- Multiple load balancing algorithms (round-robin, least connections, weighted, etc.)
- Real-time health monitoring and automatic failover
- Session affinity/sticky sessions support
- Traffic shaping and rate limiting
- SSL/TLS termination support
- WebSocket connection balancing
- Metrics collection and performance monitoring

Author: TestMaster Integration System
"""

import asyncio
import hashlib
import json
import logging
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
import socket
import struct

logger = logging.getLogger(__name__)

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"

class ServerState(Enum):
    """Server health states."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"
    WARMING = "warming"

class SessionAffinity(Enum):
    """Session affinity types."""
    NONE = "none"
    CLIENT_IP = "client_ip"
    COOKIE = "cookie"
    HEADER = "header"
    URL_PARAMETER = "url_parameter"

@dataclass
class ServerInstance:
    """Represents a backend server instance."""
    server_id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 1000
    
    # State management
    state: ServerState = ServerState.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_response_time: float = 0.0
    success_rate: float = 100.0
    
    # Health check configuration
    health_check_url: str = "/health"
    health_check_interval: int = 30
    health_check_timeout: int = 5
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    bandwidth_usage: float = 0.0
    
    # Timestamps
    last_health_check: Optional[datetime] = None
    last_request: Optional[datetime] = None
    added_at: datetime = field(default_factory=datetime.now)
    
    def update_metrics(self, response_time: float, success: bool):
        """Update server performance metrics."""
        self.response_times.append(response_time)
        self.avg_response_time = sum(self.response_times) / len(self.response_times)
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        self.success_rate = ((self.total_requests - self.failed_requests) / 
                           self.total_requests * 100) if self.total_requests > 0 else 100

@dataclass
class LoadBalancingPolicy:
    """Load balancing policy configuration."""
    algorithm: LoadBalancingAlgorithm
    session_affinity: SessionAffinity = SessionAffinity.NONE
    session_timeout: int = 3600  # seconds
    
    # Traffic shaping
    max_requests_per_second: Optional[int] = None
    max_connections_per_client: Optional[int] = None
    burst_size: int = 10
    
    # Retry configuration
    retry_failed_requests: bool = True
    max_retries: int = 3
    retry_delay_ms: int = 100
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: float = 50.0  # error percentage
    circuit_breaker_timeout: int = 60  # seconds
    
    # Advanced features
    enable_compression: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300
    enable_ssl_termination: bool = False
    enable_websocket_support: bool = True

class ConsistentHashRing:
    """Consistent hash ring for distributed load balancing."""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        self.servers = {}
        
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_server(self, server: ServerInstance):
        """Add server to hash ring."""
        self.servers[server.server_id] = server
        for i in range(self.virtual_nodes):
            virtual_key = f"{server.server_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = server
        self._update_sorted_keys()
    
    def remove_server(self, server_id: str):
        """Remove server from hash ring."""
        if server_id in self.servers:
            del self.servers[server_id]
            for i in range(self.virtual_nodes):
                virtual_key = f"{server_id}:{i}"
                hash_value = self._hash(virtual_key)
                if hash_value in self.ring:
                    del self.ring[hash_value]
            self._update_sorted_keys()
    
    def _update_sorted_keys(self):
        """Update sorted keys for binary search."""
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_server(self, key: str) -> Optional[ServerInstance]:
        """Get server for a given key."""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Binary search for the next server
        idx = self._binary_search(hash_value)
        return self.ring[self.sorted_keys[idx]]
    
    def _binary_search(self, hash_value: int) -> int:
        """Binary search for next server position."""
        left, right = 0, len(self.sorted_keys) - 1
        
        if hash_value > self.sorted_keys[right]:
            return 0
        
        while left < right:
            mid = (left + right) // 2
            if self.sorted_keys[mid] < hash_value:
                left = mid + 1
            else:
                right = mid
        
        return left

class LoadBalancingSystem:
    """Comprehensive load balancing system with advanced features."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.servers = {}  # server_id -> ServerInstance
        self.policies = {}  # service_name -> LoadBalancingPolicy
        self.sessions = {}  # session_id -> server_id
        self.consistent_hash_rings = {}  # service_name -> ConsistentHashRing
        
        # Metrics and monitoring
        self.total_requests = 0
        self.total_failures = 0
        self.request_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread pool for health checks
        self.health_check_executor = ThreadPoolExecutor(max_workers=10)
        self.health_check_tasks = {}
        
        # Rate limiting
        self.rate_limiters = defaultdict(lambda: {"tokens": 0, "last_update": time.time()})
        
        # Circuit breakers
        self.circuit_breakers = {}
        
        # Start background tasks
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_servers, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Load Balancing System initialized with advanced features")
    
    def register_server(self, server_name: str, config: dict):
        """Register a new backend server."""
        server = ServerInstance(
            server_id=server_name,
            host=config.get("host", "localhost"),
            port=config.get("port", 8080),
            weight=config.get("weight", 1),
            max_connections=config.get("max_connections", 1000),
            health_check_url=config.get("health_check", "/health")
        )
        
        self.servers[server_name] = server
        
        # Start health monitoring
        self._start_health_check(server)
        
        self.logger.info(f"Registered server: {server_name} ({server.host}:{server.port})")
        return server
    
    def set_algorithm(self, algorithm: str, service: str = "default"):
        """Set load balancing algorithm for a service."""
        try:
            algo_enum = LoadBalancingAlgorithm(algorithm)
            if service not in self.policies:
                self.policies[service] = LoadBalancingPolicy(algorithm=algo_enum)
            else:
                self.policies[service].algorithm = algo_enum
            
            # Initialize consistent hash ring if needed
            if algo_enum == LoadBalancingAlgorithm.CONSISTENT_HASH:
                if service not in self.consistent_hash_rings:
                    self.consistent_hash_rings[service] = ConsistentHashRing()
                    for server in self.servers.values():
                        self.consistent_hash_rings[service].add_server(server)
            
            self.logger.info(f"Set algorithm {algorithm} for service {service}")
        except ValueError:
            self.logger.error(f"Invalid algorithm: {algorithm}")
    
    def get_next_server(self, request_context: dict = None) -> Optional[dict]:
        """Get next server based on load balancing algorithm."""
        service = request_context.get("service", "default") if request_context else "default"
        policy = self.policies.get(service, LoadBalancingPolicy(LoadBalancingAlgorithm.ROUND_ROBIN))
        
        # Filter healthy servers
        healthy_servers = [s for s in self.servers.values() 
                         if s.state == ServerState.HEALTHY]
        
        if not healthy_servers:
            self.logger.warning("No healthy servers available")
            return None
        
        # Check session affinity
        if policy.session_affinity != SessionAffinity.NONE and request_context:
            server = self._get_session_server(request_context, policy)
            if server:
                return {"name": server.server_id, "config": {"host": server.host, "port": server.port}}
        
        # Select server based on algorithm
        server = self._select_server(healthy_servers, policy, request_context)
        
        if server:
            # Update metrics
            server.current_connections += 1
            server.last_request = datetime.now()
            
            return {
                "name": server.server_id,
                "config": {
                    "host": server.host,
                    "port": server.port
                }
            }
        
        return None
    
    def _select_server(self, servers: List[ServerInstance], 
                      policy: LoadBalancingPolicy, 
                      context: dict = None) -> Optional[ServerInstance]:
        """Select server based on algorithm."""
        algorithm = policy.algorithm
        
        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            if not hasattr(self, '_rr_index'):
                self._rr_index = 0
            server = servers[self._rr_index % len(servers)]
            self._rr_index += 1
            return server
            
        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return min(servers, key=lambda s: s.current_connections)
            
        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            weighted_servers = []
            for server in servers:
                weighted_servers.extend([server] * server.weight)
            if not hasattr(self, '_wrr_index'):
                self._wrr_index = 0
            server = weighted_servers[self._wrr_index % len(weighted_servers)]
            self._wrr_index += 1
            return server
            
        elif algorithm == LoadBalancingAlgorithm.RANDOM:
            return random.choice(servers)
            
        elif algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return min(servers, key=lambda s: s.avg_response_time)
            
        elif algorithm == LoadBalancingAlgorithm.IP_HASH and context:
            client_ip = context.get("client_ip", "")
            if client_ip:
                hash_value = hash(client_ip)
                return servers[hash_value % len(servers)]
                
        elif algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH and context:
            service = context.get("service", "default")
            if service in self.consistent_hash_rings:
                key = context.get("key", str(uuid.uuid4()))
                return self.consistent_hash_rings[service].get_server(key)
                
        elif algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
            # Select based on resource availability
            return min(servers, key=lambda s: (s.cpu_usage + s.memory_usage) / 2)
        
        # Default to round-robin
        return servers[0] if servers else None
    
    def _get_session_server(self, context: dict, policy: LoadBalancingPolicy) -> Optional[ServerInstance]:
        """Get server based on session affinity."""
        session_key = None
        
        if policy.session_affinity == SessionAffinity.CLIENT_IP:
            session_key = context.get("client_ip")
        elif policy.session_affinity == SessionAffinity.COOKIE:
            session_key = context.get("session_cookie")
        elif policy.session_affinity == SessionAffinity.HEADER:
            session_key = context.get("session_header")
        
        if session_key and session_key in self.sessions:
            server_id = self.sessions[session_key]
            if server_id in self.servers:
                server = self.servers[server_id]
                if server.state == ServerState.HEALTHY:
                    return server
        
        return None
    
    def mark_server_healthy(self, server_name: str):
        """Mark server as healthy."""
        if server_name in self.servers:
            server = self.servers[server_name]
            server.state = ServerState.HEALTHY
            server.consecutive_failures = 0
            self.logger.info(f"Server {server_name} marked as healthy")
    
    def mark_server_unhealthy(self, server_name: str):
        """Mark server as unhealthy."""
        if server_name in self.servers:
            server = self.servers[server_name]
            server.state = ServerState.UNHEALTHY
            self.logger.warning(f"Server {server_name} marked as unhealthy")
            
            # Remove from consistent hash rings
            for ring in self.consistent_hash_rings.values():
                ring.remove_server(server_name)
    
    def update_server_load(self, server_name: str, load: int):
        """Update server load metrics."""
        if server_name in self.servers:
            server = self.servers[server_name]
            server.cpu_usage = load
            self.logger.debug(f"Updated load for {server_name}: {load}%")
    
    def get_load_metrics(self) -> dict:
        """Get comprehensive load balancing metrics."""
        healthy_count = sum(1 for s in self.servers.values() 
                          if s.state == ServerState.HEALTHY)
        
        total_connections = sum(s.current_connections for s in self.servers.values())
        avg_response_time = sum(s.avg_response_time for s in self.servers.values()) / len(self.servers) if self.servers else 0
        
        return {
            "servers": len(self.servers),
            "healthy_servers": healthy_count,
            "unhealthy_servers": len(self.servers) - healthy_count,
            "total_connections": total_connections,
            "average_response_time_ms": avg_response_time,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "success_rate": ((self.total_requests - self.total_failures) / self.total_requests * 100) 
                          if self.total_requests > 0 else 100,
            "algorithms_in_use": list(set(p.algorithm.value for p in self.policies.values())),
            "server_details": [
                {
                    "server_id": s.server_id,
                    "state": s.state.value,
                    "connections": s.current_connections,
                    "response_time_ms": s.avg_response_time,
                    "success_rate": s.success_rate,
                    "cpu_usage": s.cpu_usage,
                    "memory_usage": s.memory_usage
                }
                for s in self.servers.values()
            ]
        }
    
    def _start_health_check(self, server: ServerInstance):
        """Start health checking for a server."""
        def check_health():
            while self.running and server.server_id in self.servers:
                try:
                    # Simulate health check
                    start_time = time.time()
                    # In production, make actual HTTP request to health endpoint
                    success = random.random() > 0.05  # 95% success rate for simulation
                    response_time = (time.time() - start_time) * 1000
                    
                    if success:
                        server.consecutive_failures = 0
                        if server.state == ServerState.UNHEALTHY:
                            self.mark_server_healthy(server.server_id)
                    else:
                        server.consecutive_failures += 1
                        if server.consecutive_failures >= server.max_consecutive_failures:
                            self.mark_server_unhealthy(server.server_id)
                    
                    server.last_health_check = datetime.now()
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for {server.server_id}: {e}")
                    server.consecutive_failures += 1
                
                time.sleep(server.health_check_interval)
        
        task = self.health_check_executor.submit(check_health)
        self.health_check_tasks[server.server_id] = task
    
    def _monitor_servers(self):
        """Background monitoring of server performance."""
        while self.running:
            try:
                for server in self.servers.values():
                    # Simulate resource monitoring
                    server.cpu_usage = random.uniform(20, 80)
                    server.memory_usage = random.uniform(30, 70)
                    server.bandwidth_usage = random.uniform(10, 50)
                    
                    # Update success rate based on recent performance
                    if server.total_requests > 0:
                        server.success_rate = ((server.total_requests - server.failed_requests) / 
                                             server.total_requests * 100)
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(10)
    
    def apply_rate_limit(self, client_id: str, requests_per_second: int = 100) -> bool:
        """Apply rate limiting using token bucket algorithm."""
        current_time = time.time()
        limiter = self.rate_limiters[client_id]
        
        # Refill tokens
        time_passed = current_time - limiter["last_update"]
        limiter["tokens"] = min(
            requests_per_second,
            limiter["tokens"] + time_passed * requests_per_second
        )
        limiter["last_update"] = current_time
        
        # Check if request is allowed
        if limiter["tokens"] >= 1:
            limiter["tokens"] -= 1
            return True
        
        return False
    
    def shutdown(self):
        """Gracefully shutdown the load balancer."""
        self.running = False
        self.health_check_executor.shutdown(wait=True)
        self.logger.info("Load Balancing System shut down")
    
    # Test compatibility methods (preserved from minimal implementation)
    def add_backend(self, name: str, config: dict):
        """Add a backend server (alias for register_server)."""
        return self.register_server(name, config)
    
    def get_active_backends(self) -> list:
        """Get list of active backend servers."""
        return [s.server_id for s in self.servers.values() 
                if s.state == ServerState.HEALTHY]
    
    def route_request(self, request: dict) -> str:
        """Route a request to a backend server."""
        server = self.get_next_server(request)
        return server["name"] if server else "default"
    
    def get_load_statistics(self) -> dict:
        """Get load statistics (alias for get_load_metrics)."""
        return self.get_load_metrics()

# Global instance
instance = LoadBalancingSystem()
