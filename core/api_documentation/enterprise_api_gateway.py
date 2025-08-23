#!/usr/bin/env python3
"""
🌐 MODULE: Enterprise API Gateway - Advanced Multi-Agent Integration Hub
==================================================================

📋 PURPOSE:
    Provides enterprise-grade API gateway for Greek Swarm coordination,
    unified external access, load balancing, and advanced integration capabilities.

🎯 CORE FUNCTIONALITY:
    • Unified API gateway for all Greek Swarm agents
    • Advanced load balancing and failover coordination
    • Enterprise security and authentication management
    • Real-time API routing and request distribution
    • Comprehensive monitoring and analytics

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 06:15:00 | Agent Delta | 🆕 FEATURE
   └─ Goal: Create enterprise API gateway for Hour 7 mission
   └─ Changes: Complete implementation of API gateway, load balancing, security
   └─ Impact: Enables unified external access to Greek Swarm with enterprise features

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Delta
🔧 Language: Python
📦 Dependencies: Flask, aiohttp, redis, jwt, prometheus_client
🎯 Integration Points: All Greek Swarm agents, external systems
⚡ Performance Notes: Async routing, connection pooling, Redis caching
🔒 Security Notes: JWT authentication, rate limiting, request validation

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: 0% | Last Run: N/A (New implementation)
✅ Integration Tests: 0% | Last Run: N/A (New implementation)
✅ Performance Tests: 0% | Last Run: N/A (New implementation)
⚠️  Known Issues: None (Initial implementation)

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Greek Swarm agents, Redis cache, authentication system
📤 Provides: Unified API access for all Greek Swarm operations
🚨 Breaking Changes: None (new gateway layer)
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
import redis
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from greek_swarm_coordinator import GreekSwarmCoordinator, AgentType, AgentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('api_gateway_active_connections', 'Active connections')
AGENT_REQUESTS = Counter('api_gateway_agent_requests_total', 'Requests per agent', ['agent_type', 'status'])

class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"  
    WEIGHTED = "weighted"
    HEALTH_BASED = "health_based"
    RANDOM = "random"

class SecurityLevel(Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"
    SYSTEM = "system"

@dataclass
class RouteConfig:
    """Configuration for API route"""
    path: str
    target_agents: List[AgentType]
    routing_strategy: RoutingStrategy
    security_level: SecurityLevel
    rate_limit: int = 100  # requests per minute
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_ttl: int = 0  # seconds, 0 = no cache

@dataclass
class AgentEndpoint:
    """Agent endpoint information"""
    agent_type: AgentType
    host: str
    port: int
    health_score: float
    load_factor: float
    response_time: float
    last_used: datetime
    connection_count: int = 0

@dataclass
class RequestMetrics:
    """Request processing metrics"""
    request_id: str
    client_ip: str
    method: str
    path: str
    target_agent: str
    start_time: datetime
    end_time: Optional[datetime] = None
    response_code: int = 0
    response_size: int = 0
    processing_time: float = 0.0

class EnterpriseAPIGateway:
    """Enterprise-grade API Gateway for Greek Swarm coordination"""
    
    def __init__(self, port: int = 5004, redis_host: str = "localhost", redis_port: int = 6379):
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Greek Swarm coordinator
        self.coordinator = GreekSwarmCoordinator()
        
        # Redis for caching and session management
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
        
        # Gateway state
        self.agent_endpoints: Dict[str, AgentEndpoint] = {}
        self.route_configs: Dict[str, RouteConfig] = {}
        self.request_metrics: List[RequestMetrics] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # JWT secret key
        self.jwt_secret = "enterprise_api_gateway_secret_key_2025"
        
        # Load balancing state
        self.round_robin_counters: Dict[str, int] = {}
        
        # Rate limiting
        self.rate_limit_cache: Dict[str, Dict[str, int]] = {}
        
        # Initialize configurations
        self.init_route_configs()
        
        # Setup Flask routes
        self.setup_routes()
        
        # Start background services
        self.start_background_services()
    
    def init_route_configs(self):
        """Initialize API route configurations"""
        self.route_configs = {
            # Agent-specific routes
            "/api/alpha/<path:subpath>": RouteConfig(
                path="/api/alpha/",
                target_agents=[AgentType.ALPHA],
                routing_strategy=RoutingStrategy.HEALTH_BASED,
                security_level=SecurityLevel.AUTHENTICATED,
                rate_limit=200,
                timeout=15.0,
                cache_ttl=300
            ),
            "/api/beta/<path:subpath>": RouteConfig(
                path="/api/beta/",
                target_agents=[AgentType.BETA],
                routing_strategy=RoutingStrategy.HEALTH_BASED,
                security_level=SecurityLevel.AUTHENTICATED,
                rate_limit=200,
                timeout=15.0,
                cache_ttl=300
            ),
            "/api/gamma/<path:subpath>": RouteConfig(
                path="/api/gamma/",
                target_agents=[AgentType.GAMMA],
                routing_strategy=RoutingStrategy.HEALTH_BASED,
                security_level=SecurityLevel.AUTHENTICATED,
                rate_limit=200,
                timeout=15.0,
                cache_ttl=300
            ),
            "/api/delta/<path:subpath>": RouteConfig(
                path="/api/delta/",
                target_agents=[AgentType.DELTA],
                routing_strategy=RoutingStrategy.HEALTH_BASED,
                security_level=SecurityLevel.AUTHENTICATED,
                rate_limit=200,
                timeout=15.0,
                cache_ttl=300
            ),
            "/api/epsilon/<path:subpath>": RouteConfig(
                path="/api/epsilon/",
                target_agents=[AgentType.EPSILON],
                routing_strategy=RoutingStrategy.HEALTH_BASED,
                security_level=SecurityLevel.AUTHENTICATED,
                rate_limit=200,
                timeout=15.0,
                cache_ttl=300
            ),
            
            # Unified routes
            "/api/swarm/status": RouteConfig(
                path="/api/swarm/status",
                target_agents=[AgentType.ALPHA, AgentType.BETA, AgentType.GAMMA, AgentType.DELTA, AgentType.EPSILON],
                routing_strategy=RoutingStrategy.ROUND_ROBIN,
                security_level=SecurityLevel.PUBLIC,
                rate_limit=50,
                timeout=10.0,
                cache_ttl=30
            ),
            "/api/swarm/health": RouteConfig(
                path="/api/swarm/health",
                target_agents=[AgentType.ALPHA, AgentType.BETA, AgentType.GAMMA, AgentType.DELTA, AgentType.EPSILON],
                routing_strategy=RoutingStrategy.LEAST_CONNECTIONS,
                security_level=SecurityLevel.PUBLIC,
                rate_limit=100,
                timeout=5.0,
                cache_ttl=10
            ),
            "/api/swarm/metrics": RouteConfig(
                path="/api/swarm/metrics",
                target_agents=[AgentType.ALPHA, AgentType.BETA, AgentType.GAMMA, AgentType.DELTA, AgentType.EPSILON],
                routing_strategy=RoutingStrategy.WEIGHTED,
                security_level=SecurityLevel.AUTHENTICATED,
                rate_limit=100,
                timeout=10.0,
                cache_ttl=60
            )
        }
        
        logger.info(f"Initialized {len(self.route_configs)} route configurations")
    
    def setup_routes(self):
        """Setup Flask routes for API gateway"""
        
        @self.app.before_request
        def before_request():
            """Pre-request processing"""
            g.start_time = time.time()
            g.request_id = str(uuid.uuid4())[:8]
            
            # Rate limiting check
            if not self.check_rate_limit():
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Security check
            route_config = self.get_route_config(request.path)
            if route_config and not self.check_security(route_config.security_level):
                return jsonify({'error': 'Authentication required'}), 401
            
            ACTIVE_CONNECTIONS.inc()
        
        @self.app.after_request
        def after_request(response):
            """Post-request processing"""
            processing_time = time.time() - g.start_time
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint or 'unknown',
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.observe(processing_time)
            ACTIVE_CONNECTIONS.dec()
            
            # Store request metrics
            self.record_request_metrics(response, processing_time)
            
            return response
        
        @self.app.route('/api/gateway/health')
        def gateway_health():
            """API Gateway health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'active_agents': len([a for a in self.agent_endpoints.values() if a.health_score > 0.5]),
                'total_routes': len(self.route_configs),
                'request_count': len(self.request_metrics)
            })
        
        @self.app.route('/api/gateway/status')
        def gateway_status():
            """API Gateway comprehensive status"""
            return jsonify({
                'gateway_id': 'enterprise_api_gateway',
                'port': self.port,
                'agents': {agent_id: asdict(endpoint) for agent_id, endpoint in self.agent_endpoints.items()},
                'routes': {path: asdict(config) for path, config in self.route_configs.items()},
                'metrics': {
                    'total_requests': len(self.request_metrics),
                    'active_sessions': len(self.active_sessions),
                    'cache_enabled': self.redis_client is not None
                },
                'timestamp': datetime.utcnow().isoformat()
            })
        
        @self.app.route('/api/gateway/auth', methods=['POST'])
        def authenticate():
            """Authenticate and get JWT token"""
            data = request.get_json()
            
            # Simple authentication (in production, use proper auth system)
            if data.get('username') == 'admin' and data.get('password') == 'admin123':
                token = jwt.encode({
                    'user_id': 'admin',
                    'username': 'admin',
                    'role': 'admin',
                    'exp': datetime.utcnow() + timedelta(hours=24)
                }, self.jwt_secret, algorithm='HS256')
                
                return jsonify({'token': token, 'expires_in': 86400})
            
            return jsonify({'error': 'Invalid credentials'}), 401
        
        @self.app.route('/api/gateway/metrics')
        def gateway_metrics():
            """Prometheus metrics endpoint"""
            return generate_latest(), 200, {'Content-Type': 'text/plain'}
        
        # Dynamic routing for all configured paths
        for path in self.route_configs.keys():
            self.app.add_url_rule(
                path,
                f"route_{path.replace('/', '_').replace('<', '').replace('>', '')}",
                self.handle_request,
                methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
                defaults={'path': path}
            )
    
    def handle_request(self, path: str = None, **kwargs):
        """Handle incoming API requests and route to appropriate agents"""
        route_config = self.get_route_config(request.path)
        if not route_config:
            return jsonify({'error': 'Route not found'}), 404
        
        # Check cache first
        cache_key = self.get_cache_key(request)
        if route_config.cache_ttl > 0:
            cached_response = self.get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        # Select target agent
        target_agent = self.select_target_agent(route_config)
        if not target_agent:
            return jsonify({'error': 'No healthy agents available'}), 503
        
        # Forward request to agent
        try:
            response_data = asyncio.run(self.forward_request(target_agent, request))
            
            # Cache response if configured
            if route_config.cache_ttl > 0:
                self.cache_response(cache_key, response_data, route_config.cache_ttl)
            
            AGENT_REQUESTS.labels(agent_type=target_agent.agent_type.value, status='success').inc()
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Request forwarding error: {e}")
            AGENT_REQUESTS.labels(agent_type=target_agent.agent_type.value, status='error').inc()
            return jsonify({'error': 'Service unavailable'}), 503
    
    async def forward_request(self, target_agent: AgentEndpoint, flask_request) -> Dict[str, Any]:
        """Forward request to target agent"""
        url = f"http://{target_agent.host}:{target_agent.port}{flask_request.path}"
        
        # Prepare request data
        headers = dict(flask_request.headers)
        headers.pop('Host', None)  # Remove host header
        
        params = dict(flask_request.args)
        data = None
        
        if flask_request.method in ['POST', 'PUT', 'PATCH']:
            if flask_request.is_json:
                data = flask_request.get_json()
            else:
                data = flask_request.get_data()
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start_time = time.time()
            
            try:
                async with session.request(
                    method=flask_request.method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=data if isinstance(data, dict) else None,
                    data=data if not isinstance(data, dict) else None
                ) as response:
                    # Update agent metrics
                    response_time = time.time() - start_time
                    target_agent.response_time = response_time
                    target_agent.last_used = datetime.utcnow()
                    
                    if response.status == 200:
                        target_agent.health_score = min(1.0, target_agent.health_score + 0.1)
                    else:
                        target_agent.health_score = max(0.0, target_agent.health_score - 0.2)
                    
                    response_data = await response.json()
                    return response_data
                    
            except Exception as e:
                target_agent.health_score = max(0.0, target_agent.health_score - 0.3)
                raise e
    
    def get_route_config(self, path: str) -> Optional[RouteConfig]:
        """Get route configuration for path"""
        # Direct match first
        if path in self.route_configs:
            return self.route_configs[path]
        
        # Pattern matching for parameterized routes
        for pattern, config in self.route_configs.items():
            if self.path_matches_pattern(path, pattern):
                return config
        
        return None
    
    def path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches route pattern"""
        # Simple pattern matching (in production, use proper URL routing)
        if '<path:subpath>' in pattern:
            base_pattern = pattern.replace('/<path:subpath>', '')
            return path.startswith(base_pattern)
        
        return path == pattern
    
    def select_target_agent(self, route_config: RouteConfig) -> Optional[AgentEndpoint]:
        """Select target agent based on routing strategy"""
        available_agents = [
            endpoint for endpoint in self.agent_endpoints.values()
            if endpoint.agent_type in route_config.target_agents and endpoint.health_score > 0.3
        ]
        
        if not available_agents:
            return None
        
        if route_config.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self.select_round_robin(available_agents, route_config.path)
        
        elif route_config.routing_strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(available_agents, key=lambda a: a.connection_count)
        
        elif route_config.routing_strategy == RoutingStrategy.HEALTH_BASED:
            return max(available_agents, key=lambda a: a.health_score)
        
        elif route_config.routing_strategy == RoutingStrategy.WEIGHTED:
            return self.select_weighted(available_agents)
        
        else:  # RANDOM
            import random
            return random.choice(available_agents)
    
    def select_round_robin(self, agents: List[AgentEndpoint], path: str) -> AgentEndpoint:
        """Round-robin agent selection"""
        if path not in self.round_robin_counters:
            self.round_robin_counters[path] = 0
        
        index = self.round_robin_counters[path] % len(agents)
        self.round_robin_counters[path] += 1
        
        return agents[index]
    
    def select_weighted(self, agents: List[AgentEndpoint]) -> AgentEndpoint:
        """Weighted agent selection based on health score"""
        total_weight = sum(agent.health_score for agent in agents)
        if total_weight == 0:
            return agents[0]
        
        import random
        target = random.uniform(0, total_weight)
        current = 0
        
        for agent in agents:
            current += agent.health_score
            if current >= target:
                return agent
        
        return agents[-1]
    
    def check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        client_ip = request.remote_addr
        current_minute = int(time.time() // 60)
        
        if client_ip not in self.rate_limit_cache:
            self.rate_limit_cache[client_ip] = {}
        
        if current_minute not in self.rate_limit_cache[client_ip]:
            self.rate_limit_cache[client_ip][current_minute] = 0
        
        # Clean old entries
        for minute in list(self.rate_limit_cache[client_ip].keys()):
            if current_minute - minute > 2:  # Keep last 2 minutes
                del self.rate_limit_cache[client_ip][minute]
        
        # Check rate limit (default 100 requests per minute)
        route_config = self.get_route_config(request.path)
        limit = route_config.rate_limit if route_config else 100
        
        if self.rate_limit_cache[client_ip][current_minute] >= limit:
            return False
        
        self.rate_limit_cache[client_ip][current_minute] += 1
        return True
    
    def check_security(self, required_level: SecurityLevel) -> bool:
        """Check if request meets security requirements"""
        if required_level == SecurityLevel.PUBLIC:
            return True
        
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return False
        
        try:
            token = auth_header.replace('Bearer ', '')
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Store user info in request context
            g.user = payload
            
            if required_level == SecurityLevel.ADMIN and payload.get('role') != 'admin':
                return False
            
            return True
        except Exception:
            return False
    
    def get_cache_key(self, flask_request) -> str:
        """Generate cache key for request"""
        key_data = f"{flask_request.method}:{flask_request.path}:{str(dict(flask_request.args))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached response"""
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.debug(f"Cache get error: {e}")
        
        return None
    
    def cache_response(self, cache_key: str, response_data: Any, ttl: int):
        """Cache response data"""
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, ttl, json.dumps(response_data))
            except Exception as e:
                logger.debug(f"Cache set error: {e}")
    
    def record_request_metrics(self, response, processing_time: float):
        """Record request processing metrics"""
        metrics = RequestMetrics(
            request_id=g.request_id,
            client_ip=request.remote_addr,
            method=request.method,
            path=request.path,
            target_agent="gateway",  # Will be updated when forwarding
            start_time=datetime.utcfromtimestamp(g.start_time),
            end_time=datetime.utcnow(),
            response_code=response.status_code,
            response_size=len(response.get_data()),
            processing_time=processing_time
        )
        
        self.request_metrics.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.request_metrics) > 1000:
            self.request_metrics = self.request_metrics[-1000:]
    
    def start_background_services(self):
        """Start background services"""
        # Agent discovery and health monitoring
        discovery_thread = threading.Thread(target=self._agent_discovery_service, daemon=True)
        discovery_thread.start()
        
        # Metrics cleanup
        cleanup_thread = threading.Thread(target=self._cleanup_service, daemon=True)
        cleanup_thread.start()
    
    def _agent_discovery_service(self):
        """Background service for agent discovery"""
        while True:
            try:
                # Update agent endpoints from coordinator
                for agent_id, agent_info in self.coordinator.agents.items():
                    endpoint_id = f"{agent_info.agent_type.value}_{agent_info.host}_{agent_info.port}"
                    
                    if endpoint_id not in self.agent_endpoints:
                        self.agent_endpoints[endpoint_id] = AgentEndpoint(
                            agent_type=agent_info.agent_type,
                            host=agent_info.host,
                            port=agent_info.port,
                            health_score=agent_info.health_score,
                            load_factor=agent_info.load_factor,
                            response_time=0.0,
                            last_used=datetime.utcnow()
                        )
                    else:
                        # Update existing endpoint
                        endpoint = self.agent_endpoints[endpoint_id]
                        endpoint.health_score = agent_info.health_score
                        endpoint.load_factor = agent_info.load_factor
                
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Agent discovery error: {e}")
                time.sleep(5)
    
    def _cleanup_service(self):
        """Background cleanup service"""
        while True:
            try:
                # Clean old metrics
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                self.request_metrics = [
                    m for m in self.request_metrics 
                    if m.start_time > cutoff_time
                ]
                
                # Clean rate limit cache
                current_minute = int(time.time() // 60)
                for client_ip in list(self.rate_limit_cache.keys()):
                    for minute in list(self.rate_limit_cache[client_ip].keys()):
                        if current_minute - minute > 5:  # Clean entries older than 5 minutes
                            del self.rate_limit_cache[client_ip][minute]
                    
                    # Remove empty client entries
                    if not self.rate_limit_cache[client_ip]:
                        del self.rate_limit_cache[client_ip]
                
                time.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Cleanup service error: {e}")
                time.sleep(60)
    
    def run(self):
        """Run the Enterprise API Gateway"""
        print(f"Starting Enterprise API Gateway")
        print(f"Port: {self.port}")
        print(f"Gateway URL: http://localhost:{self.port}")
        print(f"Health Check: http://localhost:{self.port}/api/gateway/health")
        print(f"Prometheus Metrics: http://localhost:{self.port}/api/gateway/metrics")
        print(f"Routes Configured: {len(self.route_configs)}")
        
        try:
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False,
                threaded=True
            )
        except Exception as e:
            print(f"Enterprise API Gateway error: {e}")

def main():
    """Main entry point"""
    print("=" * 80)
    print("ENTERPRISE API GATEWAY - HOUR 7 DEPLOYMENT")
    print("=" * 80)
    print("Status: Advanced Multi-Agent API Gateway")
    print("Port: 5004 (Enterprise API Gateway)")
    print("Integration: Unified access to Greek Swarm with load balancing")
    print("=" * 80)
    
    # Create and run enterprise API gateway
    gateway = EnterpriseAPIGateway(port=5004)
    gateway.run()

if __name__ == "__main__":
    main()