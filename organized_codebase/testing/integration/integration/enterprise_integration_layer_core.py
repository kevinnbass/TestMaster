"""
Enterprise Integration Layer
============================
"""Core Module - Split from enterprise_integration_layer.py"""


import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import hashlib
import base64


# ============================================================================
# CORE INTEGRATION TYPES
# ============================================================================


class ProtocolType(Enum):
    """Supported integration protocols"""
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    GRAPHQL = "graphql"
    MESSAGE_QUEUE = "message_queue"
    EVENT_STREAM = "event_stream"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    SOAP = "soap"
    TCP_SOCKET = "tcp_socket"


class DataFormat(Enum):
    """Supported data formats"""
    JSON = "json"
    XML = "xml"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    CSV = "csv"
    YAML = "yaml"
    BINARY = "binary"
    TEXT = "text"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class IntegrationPattern(Enum):
    """Enterprise integration patterns"""
    REQUEST_REPLY = "request_reply"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    MESSAGE_ROUTING = "message_routing"
    MESSAGE_TRANSFORMATION = "message_transformation"
    MESSAGE_AGGREGATION = "message_aggregation"
    MESSAGE_SPLITTING = "message_splitting"
    CONTENT_ENRICHMENT = "content_enrichment"
    DEAD_LETTER_QUEUE = "dead_letter_queue"
    SAGA = "saga"
    CQRS = "cqrs"


@dataclass
class IntegrationEndpoint:
    """External system endpoint configuration"""
    endpoint_id: str = field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:8]}")
    name: str = ""
    protocol: ProtocolType = ProtocolType.HTTP_REST
    connection_string: str = ""
    data_format: DataFormat = DataFormat.JSON
    authentication: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_config: Dict[str, Any] = field(default_factory=dict)
    circuit_breaker: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[Dict[str, int]] = None
    health_check: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class EnterpriseMessage:
    """Enterprise message with comprehensive metadata"""
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    message_type: str = ""
    source_system: str = ""
    target_system: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_seconds: Optional[int] = None
    
    # Content
    headers: Dict[str, Any] = field(default_factory=dict)
    payload: Any = None
    data_format: DataFormat = DataFormat.JSON
    content_encoding: str = "utf-8"
    
    # Routing
    routing_key: Optional[str] = None
    reply_to: Optional[str] = None
    
    # Processing
    retry_count: int = 0
    max_retries: int = 3
    processing_errors: List[str] = field(default_factory=list)
    
    # Audit trail
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)


@dataclass
class TransformationRule:
    """Data transformation rule definition"""
    rule_id: str = field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")
    name: str = ""
    source_format: DataFormat = DataFormat.JSON
    target_format: DataFormat = DataFormat.JSON
    transformation_type: str = "mapping"  # mapping, aggregation, split, enrich
    rules: Dict[str, Any] = field(default_factory=dict)
    validation_schema: Optional[Dict[str, Any]] = None
    enabled: bool = True


# ============================================================================
# PROTOCOL ADAPTERS
# ============================================================================

class ProtocolAdapterRegistry:
    """Registry and factory for protocol adapters"""
    
    def __init__(self):
        self.logger = logging.getLogger("protocol_adapter_registry")
        
        # Adapter implementations
        self.adapters: Dict[ProtocolType, Callable] = {
            ProtocolType.HTTP_REST: self._create_http_adapter,
            ProtocolType.WEBSOCKET: self._create_websocket_adapter,
            ProtocolType.MESSAGE_QUEUE: self._create_mq_adapter,
            ProtocolType.EVENT_STREAM: self._create_stream_adapter,
            ProtocolType.DATABASE: self._create_db_adapter,
            ProtocolType.FILE_SYSTEM: self._create_fs_adapter
        }
        
        # Active connections
        self.active_connections: Dict[str, Any] = {}
        self.connection_lock = threading.Lock()
        
        self.logger.info("Protocol adapter registry initialized")
    
    async def get_adapter(self, endpoint: IntegrationEndpoint) -> 'ProtocolAdapter':
        """Get or create protocol adapter for endpoint"""
        adapter_key = f"{endpoint.protocol.value}:{endpoint.endpoint_id}"
        
        with self.connection_lock:
            if adapter_key in self.active_connections:
                return self.active_connections[adapter_key]
            
            # Create new adapter
            if endpoint.protocol in self.adapters:
                adapter = self.adapters[endpoint.protocol](endpoint)
                self.active_connections[adapter_key] = adapter
                
                # Initialize connection
                await adapter.connect()
                
                return adapter
            else:
                raise Exception(f"Unsupported protocol: {endpoint.protocol}")
    
    def _create_http_adapter(self, endpoint: IntegrationEndpoint) -> 'HTTPAdapter':
        """Create HTTP REST adapter"""
        return HTTPAdapter(endpoint)
    
    def _create_websocket_adapter(self, endpoint: IntegrationEndpoint) -> 'WebSocketAdapter':
        """Create WebSocket adapter"""
        return WebSocketAdapter(endpoint)
    
    def _create_mq_adapter(self, endpoint: IntegrationEndpoint) -> 'MessageQueueAdapter':
        """Create Message Queue adapter"""
        return MessageQueueAdapter(endpoint)
    
    def _create_stream_adapter(self, endpoint: IntegrationEndpoint) -> 'EventStreamAdapter':
        """Create Event Stream adapter"""
        return EventStreamAdapter(endpoint)
    
    def _create_db_adapter(self, endpoint: IntegrationEndpoint) -> 'DatabaseAdapter':
        """Create Database adapter"""
        return DatabaseAdapter(endpoint)
    
    def _create_fs_adapter(self, endpoint: IntegrationEndpoint) -> 'FileSystemAdapter':
        """Create File System adapter"""
        return FileSystemAdapter(endpoint)


class ProtocolAdapter:
    """Base protocol adapter interface"""
    
    def __init__(self, endpoint: IntegrationEndpoint):
        self.endpoint = endpoint
        self.logger = logging.getLogger(f"adapter_{endpoint.protocol.value}")
        self.connected = False
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "avg_response_time": 0.0
        }
    
    async def connect(self) -> bool:
        """Establish connection to endpoint"""
        self.logger.info(f"Connecting to {self.endpoint.name}")
        # Mock connection
        await asyncio.sleep(0.1)
        self.connected = True
        return True
    
    async def disconnect(self):
        """Close connection to endpoint"""
        self.logger.info(f"Disconnecting from {self.endpoint.name}")
        self.connected = False
    
    async def send_message(self, message: EnterpriseMessage) -> Dict[str, Any]:
        """Send message through adapter"""
        if not self.connected:
            raise Exception("Adapter not connected")
        
        start_time = time.time()
        
        try:
            # Mock message sending
            await asyncio.sleep(0.05)
            
            response = {
                "message_id": message.message_id,
                "status": "sent",
                "timestamp": datetime.now().isoformat(),
                "endpoint": self.endpoint.name
            }
            
            # Update metrics
            self.metrics["messages_sent"] += 1
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
            return response
            
        except Exception as e:
            self.metrics["errors"] += 1
            raise e
    
    async def receive_message(self, timeout: Optional[int] = None) -> Optional[EnterpriseMessage]:
        """Receive message through adapter"""
        if not self.connected:
            raise Exception("Adapter not connected")
        
        # Mock message receiving
        await asyncio.sleep(0.1)
        
        # Simulate receiving a message occasionally
        if time.time() % 10 < 1:  # 10% chance
            message = EnterpriseMessage(
                message_type="incoming",
                source_system=self.endpoint.name,
                payload={"data": "mock_incoming_data"},
                timestamp=datetime.now()
            )
            
            self.metrics["messages_received"] += 1
            return message
        
        return None
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time metric"""
        current_avg = self.metrics["avg_response_time"]
        total_sent = self.metrics["messages_sent"]
        
        if total_sent > 0:
            self.metrics["avg_response_time"] = (
                (current_avg * (total_sent - 1) + response_time_ms) / total_sent
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get adapter health status"""
        return {
            "connected": self.connected,
            "endpoint": self.endpoint.name,
            "protocol": self.endpoint.protocol.value,
            "metrics": self.metrics.copy()
        }


class HTTPAdapter(ProtocolAdapter):
    """HTTP REST protocol adapter"""
    
    async def send_message(self, message: EnterpriseMessage) -> Dict[str, Any]:
        """Send HTTP request"""
        # Mock HTTP request
        self.logger.info(f"Sending HTTP request to {self.endpoint.connection_string}")
        
        # Simulate HTTP call
        await asyncio.sleep(0.1)
        
        return {
            "status_code": 200,
            "response_body": {"status": "success"},
            "headers": {"content-type": "application/json"}
        }


class MessageQueueAdapter(ProtocolAdapter):
    """Message Queue protocol adapter"""
    
    def __init__(self, endpoint: IntegrationEndpoint):
        super().__init__(endpoint)
        self.message_queue: queue.Queue = queue.Queue()
    
    async def send_message(self, message: EnterpriseMessage) -> Dict[str, Any]:
        """Send message to queue"""
        self.logger.info(f"Sending message to queue {self.endpoint.name}")
        
        # Add to local queue (mock)
        self.message_queue.put(message)
        
        return {"queue_position": self.message_queue.qsize()}


class WebSocketAdapter(ProtocolAdapter):
    """WebSocket protocol adapter"""
    
    async def send_message(self, message: EnterpriseMessage) -> Dict[str, Any]:
        """Send WebSocket message"""
        self.logger.info(f"Sending WebSocket message to {self.endpoint.connection_string}")
        return {"sent": True, "connection_id": "ws_123"}


class EventStreamAdapter(ProtocolAdapter):
    """Event Stream protocol adapter"""
    
    async def send_message(self, message: EnterpriseMessage) -> Dict[str, Any]:
        """Publish to event stream"""
        self.logger.info(f"Publishing to event stream {self.endpoint.name}")
        return {"stream_position": time.time(), "partition": 0}


class DatabaseAdapter(ProtocolAdapter):
    """Database protocol adapter"""
    
    async def send_message(self, message: EnterpriseMessage) -> Dict[str, Any]:
        """Execute database operation"""
        self.logger.info(f"Executing database operation on {self.endpoint.name}")
        return {"rows_affected": 1, "execution_time": 0.05}


class FileSystemAdapter(ProtocolAdapter):
    """File System protocol adapter"""
    
    async def send_message(self, message: EnterpriseMessage) -> Dict[str, Any]:
        """Write to file system"""
        self.logger.info(f"Writing to file system {self.endpoint.name}")
        return {"file_path": "/mock/path/file.json", "bytes_written": 1024}


# ============================================================================
# DATA TRANSFORMATION ENGINE
# ============================================================================

class DataTransformationEngine:
    """Advanced data transformation with multiple strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger("data_transformation_engine")
        
        # Transformation rules
        self.transformation_rules: Dict[str, TransformationRule] = {}
        
        # Format converters
        self.format_converters = {
            (DataFormat.JSON, DataFormat.XML): self._json_to_xml,
            (DataFormat.XML, DataFormat.JSON): self._xml_to_json,
            (DataFormat.JSON, DataFormat.CSV): self._json_to_csv,
            (DataFormat.CSV, DataFormat.JSON): self._csv_to_json,
            (DataFormat.JSON, DataFormat.YAML): self._json_to_yaml,
            (DataFormat.YAML, DataFormat.JSON): self._yaml_to_json
        }
        
        # Transformation functions
        self.transformers = {
            "mapping": self._apply_field_mapping,
            "aggregation": self._apply_aggregation,
            "split": self._apply_split,
            "enrich": self._apply_enrichment,
            "filter": self._apply_filter,
            "sort": self._apply_sort
        }
        
        self.logger.info("Data transformation engine initialized")
    
    def register_transformation_rule(self, rule: TransformationRule) -> bool:
        """Register transformation rule"""
        try:
            self.transformation_rules[rule.rule_id] = rule
            self.logger.info(f"Registered transformation rule: {rule.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register transformation rule: {e}")
            return False
    
    async def transform_message(self, message: EnterpriseMessage, 
                               rule_id: str) -> EnterpriseMessage:
        """Transform message using specified rule"""
        try:
            if rule_id not in self.transformation_rules:
                raise Exception(f"Transformation rule not found: {rule_id}")
            
            rule = self.transformation_rules[rule_id]
            
            if not rule.enabled:
                self.logger.warning(f"Transformation rule disabled: {rule_id}")
                return message
            
            # Create transformed message
            transformed_message = EnterpriseMessage(
                correlation_id=message.correlation_id,
                causation_id=message.message_id,
                message_type=message.message_type,
                source_system=message.source_system,
                target_system=message.target_system,
                headers=message.headers.copy(),
                data_format=rule.target_format,
                priority=message.priority
            )
            
            # Apply format conversion
            if rule.source_format != rule.target_format:
                converted_payload = await self._convert_format(
                    message.payload, rule.source_format, rule.target_format
                )
            else:
                converted_payload = message.payload
            
            # Apply transformation logic
            if rule.transformation_type in self.transformers:
                transformed_payload = await self.transformers[rule.transformation_type](
                    converted_payload, rule.rules
                )
            else:
                transformed_payload = converted_payload
            
            # Validate if schema provided
            if rule.validation_schema:
                validation_result = self._validate_data(transformed_payload, rule.validation_schema)
                if not validation_result["valid"]:
                    raise Exception(f"Validation failed: {validation_result['errors']}")
            
            transformed_message.payload = transformed_payload
            
            # Update audit trail
            transformed_message.transformations.append(rule_id)
            transformed_message.processing_history.append({
                "step": "transformation",
                "rule_id": rule_id,
                "rule_name": rule.name,
                "timestamp": datetime.now().isoformat()
            })
            
            return transformed_message
            
        except Exception as e:
            self.logger.error(f"Message transformation failed: {e}")
            message.processing_errors.append(str(e))
            return message
    
    async def _convert_format(self, data: Any, source_format: DataFormat, 
                             target_format: DataFormat) -> Any:
        """Convert data between formats"""
        converter_key = (source_format, target_format)
        
        if converter_key in self.format_converters:
            return await self.format_converters[converter_key](data)
        else:
            # If no direct converter, try via JSON
            if source_format != DataFormat.JSON:
                intermediate = await self._convert_format(data, source_format, DataFormat.JSON)
                return await self._convert_format(intermediate, DataFormat.JSON, target_format)
            else:
                raise Exception(f"No converter available: {source_format} -> {target_format}")
    
    async def _json_to_xml(self, data: Any) -> str:
        """Convert JSON to XML"""
        # Mock conversion
        return f"<data>{json.dumps(data)}</data>"
    
    async def _xml_to_json(self, data: str) -> Any:
        """Convert XML to JSON"""
        # Mock conversion
        return {"xml_data": data}
    
    async def _json_to_csv(self, data: Any) -> str:
        """Convert JSON to CSV"""
        # Mock conversion
        if isinstance(data, list) and data and isinstance(data[0], dict):
            headers = list(data[0].keys())
            csv_content = ",".join(headers) + "\n"
            for row in data:
                csv_content += ",".join(str(row.get(h, "")) for h in headers) + "\n"
            return csv_content
        return str(data)
    
    async def _csv_to_json(self, data: str) -> List[Dict[str, Any]]:
        """Convert CSV to JSON"""
        # Mock conversion
        lines = data.strip().split('\n')
        if len(lines) < 2:
            return []
        
        headers = lines[0].split(',')
        result = []
        for line in lines[1:]:
            values = line.split(',')
            row = {headers[i]: values[i] if i < len(values) else "" for i in range(len(headers))}
            result.append(row)
        
        return result
    
    async def _json_to_yaml(self, data: Any) -> str:
        """Convert JSON to YAML"""
        # Mock conversion
        return f"data: {json.dumps(data)}"
    
    async def _yaml_to_json(self, data: str) -> Any:
        """Convert YAML to JSON"""
        # Mock conversion
        return {"yaml_content": data}
    
    async def _apply_field_mapping(self, data: Any, rules: Dict[str, Any]) -> Any:
        """Apply field mapping transformation"""
        if not isinstance(data, dict):
            return data
        
        mapping = rules.get("field_mapping", {})
        result = {}
        
        for source_field, target_field in mapping.items():
            if source_field in data:
                result[target_field] = data[source_field]
        
        # Include unmapped fields if specified
        if rules.get("include_unmapped", False):
            for key, value in data.items():
                if key not in mapping:
                    result[key] = value
        
        return result
    
    async def _apply_aggregation(self, data: Any, rules: Dict[str, Any]) -> Any:
        """Apply aggregation transformation"""
        if not isinstance(data, list):
            return data
        
        aggregation_field = rules.get("field")
        aggregation_type = rules.get("type", "sum")
        
        if not aggregation_field:
            return data
        
        values = [item.get(aggregation_field, 0) for item in data if isinstance(item, dict)]
        
        if aggregation_type == "sum":
            result = sum(values)
        elif aggregation_type == "avg":
            result = sum(values) / len(values) if values else 0