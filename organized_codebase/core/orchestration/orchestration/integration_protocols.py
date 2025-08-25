"""
Integration Protocols
====================

Integration protocols for orchestration system integration with
external services, legacy systems, and enterprise infrastructure.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, Tuple, Union
from collections import defaultdict
import logging


class IntegrationType(Enum):
    """Types of system integration."""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    WEBHOOK = "webhook"
    RPC = "rpc"
    SOCKET = "socket"
    LEGACY_SYSTEM = "legacy_system"
    CLOUD_SERVICE = "cloud_service"


class ServiceProtocol(Enum):
    """Service communication protocols."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    AMQP = "amqp"
    MQTT = "mqtt"
    KAFKA = "kafka"
    REDIS = "redis"


class IntegrationPattern(Enum):
    """Integration patterns."""
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    FIRE_AND_FORGET = "fire_and_forget"
    SAGA = "saga"
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    RETRY = "retry"
    TIMEOUT = "timeout"


@dataclass
class ServiceEndpoint:
    """External service endpoint configuration."""
    service_id: str
    service_name: str
    endpoint_url: str
    protocol: ServiceProtocol
    integration_type: IntegrationType
    authentication: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_attempts: int = 3
    circuit_breaker_config: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationRequest:
    """Integration request data."""
    request_id: str
    service_id: str
    operation: str
    payload: Any
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    callback_url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrationResponse:
    """Integration response data."""
    response_id: str
    request_id: str
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrationMetrics:
    """Metrics for integration performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    circuit_breaker_trips: int = 0
    retry_attempts: int = 0
    rate_limit_hits: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class IntegrationProtocol(ABC):
    """Abstract base class for integration protocols."""
    
    def __init__(
        self,
        protocol_name: str,
        service_endpoint: ServiceEndpoint
    ):
        self.protocol_name = protocol_name
        self.service_endpoint = service_endpoint
        self.is_connected = False
        self.metrics = IntegrationMetrics()
        self.circuit_breaker_state = "closed"  # closed, open, half_open
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.rate_limiter_tokens = 0
        self.rate_limiter_last_refill = datetime.now()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to external service."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from external service."""
        pass
    
    @abstractmethod
    async def send_request(self, request: IntegrationRequest) -> IntegrationResponse:
        """Send request to external service."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check on service."""
        pass
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler."""
        self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit integration event."""
        for handler in self.event_handlers[event_type]:
            try:
                handler(event_type, event_data)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")


class ServiceProtocolImpl(IntegrationProtocol):
    """
    Service protocol implementation for external service integration.
    
    Implements service integration with circuit breaker, retry logic,
    rate limiting, and comprehensive error handling.
    """
    
    def __init__(
        self,
        protocol_name: str = "service_protocol",
        service_endpoint: Optional[ServiceEndpoint] = None
    ):
        if not service_endpoint:
            service_endpoint = ServiceEndpoint(
                service_id="default_service",
                service_name="Default Service",
                endpoint_url="http://localhost:8080",
                protocol=ServiceProtocol.HTTP,
                integration_type=IntegrationType.REST_API
            )
        
        super().__init__(protocol_name, service_endpoint)
        
        # Circuit breaker configuration
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = timedelta(seconds=60)
        self.circuit_breaker_recovery_timeout = timedelta(seconds=30)
        
        # Rate limiting configuration
        self.rate_limit_window = timedelta(seconds=60)
        self.rate_limit_max_requests = service_endpoint.rate_limit or 100
        
        # Connection pool
        self.connection_pool: List[Any] = []
        self.max_connections = 10
        
        # Request tracking
        self.active_requests: Dict[str, IntegrationRequest] = {}
        self.request_history: List[IntegrationResponse] = []
        self.max_history_size = 1000
    
    async def connect(self) -> bool:
        """Connect to external service."""
        try:
            # Initialize connection pool
            await self._initialize_connection_pool()
            
            # Perform initial health check
            health_ok = await self.health_check()
            if not health_ok:
                self.logger.warning(f"Health check failed for {self.service_endpoint.service_name}")
            
            self.is_connected = True
            self.logger.info(f"Connected to service: {self.service_endpoint.service_name}")
            
            # Start monitoring
            asyncio.create_task(self._monitor_service())
            
            self._emit_event("service_connected", {
                "service_id": self.service_endpoint.service_id,
                "service_name": self.service_endpoint.service_name
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to service: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from external service."""
        try:
            self.is_connected = False
            
            # Complete active requests
            await self._complete_active_requests()
            
            # Close connection pool
            await self._close_connection_pool()
            
            self.logger.info(f"Disconnected from service: {self.service_endpoint.service_name}")
            
            self._emit_event("service_disconnected", {
                "service_id": self.service_endpoint.service_id
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from service: {e}")
            return False
    
    async def send_request(self, request: IntegrationRequest) -> IntegrationResponse:
        """Send request to external service with resilience patterns."""
        request_start_time = datetime.now()
        
        try:
            # Check circuit breaker
            if not await self._check_circuit_breaker():
                return self._create_error_response(
                    request, 503, "Circuit breaker is open"
                )
            
            # Check rate limiting
            if not await self._check_rate_limit():
                self.metrics.rate_limit_hits += 1
                return self._create_error_response(
                    request, 429, "Rate limit exceeded"
                )
            
            # Track active request
            self.active_requests[request.request_id] = request
            
            # Execute request with retry logic
            response = await self._execute_with_retry(request)
            
            # Update metrics
            execution_time = (datetime.now() - request_start_time).total_seconds()
            response.execution_time = execution_time
            
            await self._update_metrics(response)
            
            # Update circuit breaker
            await self._update_circuit_breaker(response)
            
            # Store in history
            self._add_to_history(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error sending request: {e}")
            return self._create_error_response(request, 500, str(e))
            
        finally:
            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    async def health_check(self) -> bool:
        """Perform health check on service."""
        try:
            health_request = IntegrationRequest(
                request_id=f"health_{uuid.uuid4().hex[:8]}",
                service_id=self.service_endpoint.service_id,
                operation="health_check",
                payload={},
                timeout=10
            )
            
            response = await self._execute_request(health_request)
            
            return response.status_code < 400
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def bulk_request(self, requests: List[IntegrationRequest]) -> List[IntegrationResponse]:
        """Send multiple requests efficiently."""
        responses = []
        
        # Execute requests in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def execute_single_request(req):
            async with semaphore:
                return await self.send_request(req)
        
        tasks = [execute_single_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                responses[i] = self._create_error_response(
                    requests[i], 500, str(response)
                )
        
        return responses
    
    async def stream_request(self, request: IntegrationRequest) -> AsyncIterable[Dict[str, Any]]:
        """Send streaming request to service."""
        try:
            # This is a placeholder for streaming implementation
            # In real implementation, would handle actual streaming protocols
            
            yield {"status": "streaming_started", "request_id": request.request_id}
            
            # Simulate streaming data
            for i in range(5):
                await asyncio.sleep(0.5)
                yield {
                    "chunk_id": i,
                    "data": f"Stream data chunk {i}",
                    "timestamp": datetime.now().isoformat()
                }
            
            yield {"status": "streaming_completed", "request_id": request.request_id}
            
        except Exception as e:
            yield {"status": "streaming_error", "error": str(e)}
    
    async def subscribe_to_events(self, subscription_config: Dict[str, Any]) -> bool:
        """Subscribe to service events."""
        try:
            subscription_id = subscription_config.get("subscription_id", f"sub_{uuid.uuid4().hex[:8]}")
            event_types = subscription_config.get("event_types", [])
            
            self.logger.info(f"Subscribed to events: {event_types} with ID: {subscription_id}")
            
            # Start event listener
            asyncio.create_task(self._event_listener(subscription_id, event_types))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to events: {e}")
            return False
    
    async def _execute_with_retry(self, request: IntegrationRequest) -> IntegrationResponse:
        """Execute request with retry logic."""
        max_retries = request.retry_policy.get("max_retries", self.service_endpoint.retry_attempts)
        retry_delay = request.retry_policy.get("delay", 1.0)
        backoff_multiplier = request.retry_policy.get("backoff_multiplier", 2.0)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self._execute_request(request)
                
                # Check if response indicates a retryable error
                if self._is_retryable_error(response) and attempt < max_retries:
                    self.metrics.retry_attempts += 1
                    
                    self.logger.warning(
                        f"Retryable error on attempt {attempt + 1}, retrying in {retry_delay}s"
                    )
                    
                    await asyncio.sleep(retry_delay)
                    retry_delay *= backoff_multiplier
                    continue
                
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    self.metrics.retry_attempts += 1
                    
                    self.logger.warning(
                        f"Request failed on attempt {attempt + 1}: {e}, retrying in {retry_delay}s"
                    )
                    
                    await asyncio.sleep(retry_delay)
                    retry_delay *= backoff_multiplier
                else:
                    break
        
        # All retries exhausted
        error_message = str(last_exception) if last_exception else "Max retries exceeded"
        return self._create_error_response(request, 500, error_message)
    
    async def _execute_request(self, request: IntegrationRequest) -> IntegrationResponse:
        """Execute actual request to service."""
        # This is a simulation of actual service call
        # In real implementation, would use appropriate HTTP client, etc.
        
        start_time = datetime.now()
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        # Simulate various response scenarios
        import random
        
        # 90% success rate simulation
        if random.random() < 0.9:
            response_data = {
                "operation": request.operation,
                "result": f"Success for {request.operation}",
                "timestamp": datetime.now().isoformat()
            }
            
            response = IntegrationResponse(
                response_id=f"resp_{uuid.uuid4().hex[:8]}",
                request_id=request.request_id,
                status_code=200,
                data=response_data,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        else:
            # Simulate error
            response = IntegrationResponse(
                response_id=f"resp_{uuid.uuid4().hex[:8]}",
                request_id=request.request_id,
                status_code=500,
                data=None,
                error_message="Simulated service error",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        
        return response
    
    async def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state."""
        current_time = datetime.now()
        
        if self.circuit_breaker_state == "open":
            # Check if recovery timeout has passed
            if (self.circuit_breaker_last_failure and
                current_time - self.circuit_breaker_last_failure > self.circuit_breaker_recovery_timeout):
                
                self.circuit_breaker_state = "half_open"
                self.logger.info("Circuit breaker moved to half-open state")
                return True
            
            return False
        
        return True
    
    async def _check_rate_limit(self) -> bool:
        """Check rate limiting."""
        current_time = datetime.now()
        
        # Refill tokens if needed
        if current_time - self.rate_limiter_last_refill > self.rate_limit_window:
            self.rate_limiter_tokens = self.rate_limit_max_requests
            self.rate_limiter_last_refill = current_time
        
        # Check if tokens available
        if self.rate_limiter_tokens > 0:
            self.rate_limiter_tokens -= 1
            return True
        
        return False
    
    async def _update_circuit_breaker(self, response: IntegrationResponse):
        """Update circuit breaker based on response."""
        if response.status_code >= 500:
            # Server error
            self.circuit_breaker_failures += 1
            self.circuit_breaker_last_failure = datetime.now()
            
            if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                if self.circuit_breaker_state != "open":
                    self.circuit_breaker_state = "open"
                    self.metrics.circuit_breaker_trips += 1
                    
                    self.logger.warning("Circuit breaker opened due to failures")
                    self._emit_event("circuit_breaker_opened", {
                        "service_id": self.service_endpoint.service_id,
                        "failure_count": self.circuit_breaker_failures
                    })
        else:
            # Success or client error
            if self.circuit_breaker_state == "half_open":
                # Recovery successful
                self.circuit_breaker_state = "closed"
                self.circuit_breaker_failures = 0
                
                self.logger.info("Circuit breaker closed - service recovered")
                self._emit_event("circuit_breaker_closed", {
                    "service_id": self.service_endpoint.service_id
                })
            elif self.circuit_breaker_state == "closed":
                # Reset failure counter on success
                self.circuit_breaker_failures = 0
    
    async def _update_metrics(self, response: IntegrationResponse):
        """Update integration metrics."""
        self.metrics.total_requests += 1
        
        if response.status_code < 400:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        total_time = (self.metrics.average_response_time * (self.metrics.total_requests - 1) + 
                     response.execution_time)
        self.metrics.average_response_time = total_time / self.metrics.total_requests
        
        # Update error rate
        self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
        
        # Update throughput (requests per second)
        if self.metrics.total_requests > 1:
            time_window = (datetime.now() - self.metrics.last_updated).total_seconds()
            if time_window > 0:
                self.metrics.throughput = 1.0 / time_window
        
        self.metrics.last_updated = datetime.now()
    
    def _is_retryable_error(self, response: IntegrationResponse) -> bool:
        """Check if error is retryable."""
        # Retry on server errors and timeouts
        return response.status_code >= 500 or response.status_code == 408
    
    def _create_error_response(
        self,
        request: IntegrationRequest,
        status_code: int,
        error_message: str
    ) -> IntegrationResponse:
        """Create error response."""
        return IntegrationResponse(
            response_id=f"error_{uuid.uuid4().hex[:8]}",
            request_id=request.request_id,
            status_code=status_code,
            data=None,
            error_message=error_message
        )
    
    def _add_to_history(self, response: IntegrationResponse):
        """Add response to history."""
        self.request_history.append(response)
        
        # Maintain history size limit
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]
    
    async def _initialize_connection_pool(self):
        """Initialize connection pool."""
        # Placeholder for actual connection pool initialization
        self.connection_pool = [f"connection_{i}" for i in range(self.max_connections)]
        self.logger.info(f"Initialized connection pool with {self.max_connections} connections")
    
    async def _close_connection_pool(self):
        """Close connection pool."""
        self.connection_pool.clear()
        self.logger.info("Closed connection pool")
    
    async def _complete_active_requests(self):
        """Complete or cancel active requests."""
        for request_id, request in list(self.active_requests.items()):
            self.logger.info(f"Cancelling active request: {request_id}")
            # In real implementation, would properly cancel ongoing requests
        
        self.active_requests.clear()
    
    async def _monitor_service(self):
        """Monitor service health and performance."""
        while self.is_connected:
            try:
                # Perform periodic health check
                health_ok = await self.health_check()
                
                if not health_ok:
                    self.logger.warning(f"Service health check failed: {self.service_endpoint.service_name}")
                    self._emit_event("service_unhealthy", {
                        "service_id": self.service_endpoint.service_id
                    })
                
                # Monitor circuit breaker recovery
                if self.circuit_breaker_state == "open":
                    await self._check_circuit_breaker()
                
                await asyncio.sleep(30.0)  # Health check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in service monitoring: {e}")
                await asyncio.sleep(30.0)
    
    async def _event_listener(self, subscription_id: str, event_types: List[str]):
        """Listen for service events."""
        self.logger.info(f"Started event listener for subscription: {subscription_id}")
        
        try:
            while self.is_connected:
                # Simulate receiving events
                await asyncio.sleep(10.0)
                
                # Simulate event
                event_data = {
                    "event_type": "service_event",
                    "subscription_id": subscription_id,
                    "data": {"message": "Simulated service event"},
                    "timestamp": datetime.now().isoformat()
                }
                
                self._emit_event("service_event_received", event_data)
                
        except Exception as e:
            self.logger.error(f"Error in event listener: {e}")
        finally:
            self.logger.info(f"Event listener stopped for subscription: {subscription_id}")


class LegacySystemProtocol(IntegrationProtocol):
    """
    Legacy system integration protocol.
    
    Implements integration with legacy systems using adapters,
    protocol translation, and data transformation.
    """
    
    def __init__(
        self,
        protocol_name: str = "legacy_system_protocol",
        service_endpoint: Optional[ServiceEndpoint] = None
    ):
        if not service_endpoint:
            service_endpoint = ServiceEndpoint(
                service_id="legacy_system",
                service_name="Legacy System",
                endpoint_url="tcp://localhost:9999",
                protocol=ServiceProtocol.TCP,
                integration_type=IntegrationType.LEGACY_SYSTEM
            )
        
        super().__init__(protocol_name, service_endpoint)
        
        self.protocol_adapters: Dict[str, Callable] = {}
        self.data_transformers: Dict[str, Callable] = {}
        self.legacy_connections: List[Any] = []
        self.connection_timeout = 30.0
    
    async def connect(self) -> bool:
        """Connect to legacy system."""
        try:
            # Initialize protocol adapters
            await self._initialize_adapters()
            
            # Establish legacy connection
            connection = await self._establish_legacy_connection()
            if connection:
                self.legacy_connections.append(connection)
                self.is_connected = True
                
                self.logger.info(f"Connected to legacy system: {self.service_endpoint.service_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to connect to legacy system: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from legacy system."""
        try:
            for connection in self.legacy_connections:
                await self._close_legacy_connection(connection)
            
            self.legacy_connections.clear()
            self.is_connected = False
            
            self.logger.info(f"Disconnected from legacy system: {self.service_endpoint.service_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from legacy system: {e}")
            return False
    
    async def send_request(self, request: IntegrationRequest) -> IntegrationResponse:
        """Send request to legacy system with protocol adaptation."""
        try:
            # Transform request for legacy system
            legacy_request = await self._transform_request(request)
            
            # Send via appropriate adapter
            adapter = self._get_protocol_adapter(request.operation)
            legacy_response = await adapter(legacy_request)
            
            # Transform response back
            response = await self._transform_response(request, legacy_response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error sending legacy request: {e}")
            return self._create_error_response(request, 500, str(e))
    
    async def health_check(self) -> bool:
        """Check legacy system health."""
        try:
            # Send simple ping to legacy system
            ping_request = IntegrationRequest(
                request_id=f"ping_{uuid.uuid4().hex[:8]}",
                service_id=self.service_endpoint.service_id,
                operation="ping",
                payload={}
            )
            
            response = await self.send_request(ping_request)
            return response.status_code == 200
            
        except Exception:
            return False
    
    def register_protocol_adapter(self, operation: str, adapter: Callable):
        """Register protocol adapter for specific operation."""
        self.protocol_adapters[operation] = adapter
        self.logger.info(f"Registered protocol adapter for operation: {operation}")
    
    def register_data_transformer(self, data_type: str, transformer: Callable):
        """Register data transformer for specific data type."""
        self.data_transformers[data_type] = transformer
        self.logger.info(f"Registered data transformer for type: {data_type}")
    
    async def _initialize_adapters(self):
        """Initialize protocol adapters."""
        # Register default adapters
        self.register_protocol_adapter("default", self._default_adapter)
        self.register_protocol_adapter("ping", self._ping_adapter)
        self.register_protocol_adapter("query", self._query_adapter)
        self.register_protocol_adapter("update", self._update_adapter)
        
        # Register default transformers
        self.register_data_transformer("default", self._default_transformer)
        self.register_data_transformer("xml", self._xml_transformer)
        self.register_data_transformer("fixed_width", self._fixed_width_transformer)
    
    async def _establish_legacy_connection(self) -> Any:
        """Establish connection to legacy system."""
        # Simulate legacy connection establishment
        await asyncio.sleep(0.5)
        return f"legacy_connection_{uuid.uuid4().hex[:8]}"
    
    async def _close_legacy_connection(self, connection: Any):
        """Close legacy connection."""
        # Simulate connection closure
        self.logger.info(f"Closing legacy connection: {connection}")
    
    async def _transform_request(self, request: IntegrationRequest) -> Dict[str, Any]:
        """Transform request for legacy system."""
        # Get appropriate transformer
        data_type = request.headers.get("data_type", "default")
        transformer = self.data_transformers.get(data_type, self.data_transformers["default"])
        
        # Transform payload
        transformed_payload = await transformer(request.payload, "request")
        
        legacy_request = {
            "operation": request.operation,
            "payload": transformed_payload,
            "timestamp": request.timestamp.isoformat(),
            "correlation_id": request.correlation_id
        }
        
        return legacy_request
    
    async def _transform_response(
        self,
        original_request: IntegrationRequest,
        legacy_response: Dict[str, Any]
    ) -> IntegrationResponse:
        """Transform legacy response back to standard format."""
        # Get appropriate transformer
        data_type = original_request.headers.get("data_type", "default")
        transformer = self.data_transformers.get(data_type, self.data_transformers["default"])
        
        # Transform payload
        transformed_data = await transformer(legacy_response.get("data"), "response")
        
        response = IntegrationResponse(
            response_id=f"legacy_resp_{uuid.uuid4().hex[:8]}",
            request_id=original_request.request_id,
            status_code=legacy_response.get("status_code", 200),
            data=transformed_data,
            error_message=legacy_response.get("error")
        )
        
        return response
    
    def _get_protocol_adapter(self, operation: str) -> Callable:
        """Get protocol adapter for operation."""
        return self.protocol_adapters.get(operation, self.protocol_adapters["default"])
    
    async def _default_adapter(self, legacy_request: Dict[str, Any]) -> Dict[str, Any]:
        """Default protocol adapter."""
        # Simulate legacy system interaction
        await asyncio.sleep(0.2)
        
        return {
            "status_code": 200,
            "data": f"Legacy response for {legacy_request['operation']}",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _ping_adapter(self, legacy_request: Dict[str, Any]) -> Dict[str, Any]:
        """Ping adapter for health checks."""
        await asyncio.sleep(0.1)
        
        return {
            "status_code": 200,
            "data": "PONG",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _query_adapter(self, legacy_request: Dict[str, Any]) -> Dict[str, Any]:
        """Query adapter for data retrieval."""
        await asyncio.sleep(0.3)
        
        return {
            "status_code": 200,
            "data": {
                "records": [
                    {"id": 1, "name": "Record 1"},
                    {"id": 2, "name": "Record 2"}
                ],
                "count": 2
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _update_adapter(self, legacy_request: Dict[str, Any]) -> Dict[str, Any]:
        """Update adapter for data modification."""
        await asyncio.sleep(0.4)
        
        return {
            "status_code": 200,
            "data": {
                "updated": True,
                "affected_records": 1
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _default_transformer(self, data: Any, direction: str) -> Any:
        """Default data transformer."""
        return data
    
    async def _xml_transformer(self, data: Any, direction: str) -> Any:
        """XML data transformer."""
        if direction == "request":
            # Convert to XML format for legacy system
            return f"<request>{json.dumps(data)}</request>"
        else:
            # Convert from XML format
            return {"xml_data": data, "parsed": True}
    
    async def _fixed_width_transformer(self, data: Any, direction: str) -> Any:
        """Fixed-width data transformer."""
        if direction == "request":
            # Convert to fixed-width format
            if isinstance(data, dict):
                return "|".join(f"{str(v):<10}" for v in data.values())
            return str(data)
        else:
            # Parse fixed-width format
            return {"fixed_width_data": data, "parsed": True}


# Export key classes
__all__ = [
    'IntegrationType',
    'ServiceProtocol',
    'IntegrationPattern',
    'ServiceEndpoint',
    'IntegrationRequest',
    'IntegrationResponse',
    'IntegrationMetrics',
    'IntegrationProtocol',
    'ServiceProtocolImpl',
    'LegacySystemProtocol'
]