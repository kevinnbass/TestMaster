"""
Enterprise Integration Layer
============================
"""Processing Module - Split from enterprise_integration_layer.py"""


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


        elif aggregation_type == "count":
            result = len(values)
        elif aggregation_type == "max":
            result = max(values) if values else None
        elif aggregation_type == "min":
            result = min(values) if values else None
        else:
            result = values
        
        return {
            "aggregation_type": aggregation_type,
            "field": aggregation_field,
            "result": result,
            "count": len(values)
        }
    
    async def _apply_split(self, data: Any, rules: Dict[str, Any]) -> List[Any]:
        """Apply split transformation"""
        split_field = rules.get("field")
        split_size = rules.get("size", 1)
        
        if isinstance(data, list):
            # Split list into chunks
            return [data[i:i+split_size] for i in range(0, len(data), split_size)]
        elif isinstance(data, dict) and split_field:
            # Split by field value
            field_value = data.get(split_field, "")
            if isinstance(field_value, str):
                delimiter = rules.get("delimiter", ",")
                parts = field_value.split(delimiter)
                return [{"part": part.strip(), "index": i} for i, part in enumerate(parts)]
        
        return [data]
    
    async def _apply_enrichment(self, data: Any, rules: Dict[str, Any]) -> Any:
        """Apply enrichment transformation"""
        if not isinstance(data, dict):
            return data
        
        enrichment_data = rules.get("enrichment", {})
        result = data.copy()
        result.update(enrichment_data)
        
        # Add metadata
        result["_enriched_at"] = datetime.now().isoformat()
        result["_enrichment_source"] = "transformation_engine"
        
        return result
    
    async def _apply_filter(self, data: Any, rules: Dict[str, Any]) -> Any:
        """Apply filter transformation"""
        if isinstance(data, list):
            filter_field = rules.get("field")
            filter_value = rules.get("value")
            filter_operator = rules.get("operator", "equals")
            
            if not filter_field:
                return data
            
            filtered = []
            for item in data:
                if isinstance(item, dict) and filter_field in item:
                    item_value = item[filter_field]
                    
                    if filter_operator == "equals" and item_value == filter_value:
                        filtered.append(item)
                    elif filter_operator == "contains" and filter_value in str(item_value):
                        filtered.append(item)
                    elif filter_operator == "greater_than" and item_value > filter_value:
                        filtered.append(item)
                    elif filter_operator == "less_than" and item_value < filter_value:
                        filtered.append(item)
            
            return filtered
        
        return data
    
    async def _apply_sort(self, data: Any, rules: Dict[str, Any]) -> Any:
        """Apply sort transformation"""
        if isinstance(data, list):
            sort_field = rules.get("field")
            sort_order = rules.get("order", "asc")
            
            if sort_field:
                reverse = sort_order == "desc"
                data.sort(key=lambda x: x.get(sort_field, "") if isinstance(x, dict) else x, 
                         reverse=reverse)
        
        return data
    
    def _validate_data(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against schema"""
        # Basic validation implementation
        errors = []
        
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                errors.append(f"Expected object, got {type(data).__name__}")
            elif expected_type == "array" and not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
        
        if "required" in schema and isinstance(data, dict):
            for field in schema["required"]:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


# ============================================================================
# ENTERPRISE INTEGRATION LAYER
# ============================================================================

class EnterpriseIntegrationLayer:
    """
    Ultimate enterprise integration layer providing comprehensive connectivity,
    transformation, and monitoring across all systems and protocols.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("enterprise_integration_layer")
        
        # Core components
        self.protocol_registry = ProtocolAdapterRegistry()
        self.transformation_engine = DataTransformationEngine()
        
        # Endpoint management
        self.endpoints: Dict[str, IntegrationEndpoint] = {}
        self.endpoint_adapters: Dict[str, ProtocolAdapter] = {}
        
        # Message processing
        self.message_queues: Dict[str, queue.Queue] = {}
        self.processing_workers: Dict[str, ThreadPoolExecutor] = {}
        
        # Enterprise Service Bus
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.dead_letter_queue: queue.Queue = queue.Queue()
        
        # Monitoring and metrics
        self.integration_metrics = {
            "messages_processed": 0,
            "transformations_applied": 0,
            "endpoints_connected": 0,
            "processing_errors": 0,
            "average_processing_time": 0.0
        }
        
        # Enterprise patterns
        self.sagas: Dict[str, Dict[str, Any]] = {}
        self.event_subscriptions: Dict[str, List[str]] = {}
        
        # Initialize default configurations
        self._initialize_default_endpoints()
        self._initialize_default_transformations()
        
        self.logger.info("Enterprise integration layer initialized")
    
    def _initialize_default_endpoints(self):
        """Initialize default integration endpoints"""
        default_endpoints = [
            IntegrationEndpoint(
                name="intelligence_analytics",
                protocol=ProtocolType.HTTP_REST,
                connection_string="http://localhost:8080/api/analytics",
                data_format=DataFormat.JSON,
                timeout_seconds=30,
                health_check={"path": "/health", "interval": 60}
            ),
            IntegrationEndpoint(
                name="event_streaming",
                protocol=ProtocolType.EVENT_STREAM,
                connection_string="kafka://localhost:9092/events",
                data_format=DataFormat.JSON,
                timeout_seconds=10
            ),
            IntegrationEndpoint(
                name="coordination_services",
                protocol=ProtocolType.MESSAGE_QUEUE,
                connection_string="amqp://localhost:5672/coordination",
                data_format=DataFormat.JSON,
                timeout_seconds=15
            ),
            IntegrationEndpoint(
                name="external_database",
                protocol=ProtocolType.DATABASE,
                connection_string="postgresql://localhost:5432/enterprise",
                data_format=DataFormat.JSON,
                timeout_seconds=20
            ),
            IntegrationEndpoint(
                name="file_storage",
                protocol=ProtocolType.FILE_SYSTEM,
                connection_string="/enterprise/shared/storage",
                data_format=DataFormat.JSON,
                timeout_seconds=5
            )
        ]
        
        for endpoint in default_endpoints:
            self.register_endpoint(endpoint)
    
    def _initialize_default_transformations(self):
        """Initialize default transformation rules"""
        default_rules = [
            TransformationRule(
                name="Analytics Data Normalization",
                source_format=DataFormat.JSON,
                target_format=DataFormat.JSON,
                transformation_type="mapping",
                rules={
                    "field_mapping": {
                        "timestamp": "event_time",
                        "data": "payload",
                        "source": "origin_system"
                    },
                    "include_unmapped": True
                }
            ),
            TransformationRule(
                name="Event Aggregation",
                source_format=DataFormat.JSON,
                target_format=DataFormat.JSON,
                transformation_type="aggregation",
                rules={
                    "field": "value",
                    "type": "sum"
                }
            ),
            TransformationRule(
                name="CSV to JSON Conversion",
                source_format=DataFormat.CSV,
                target_format=DataFormat.JSON,
                transformation_type="mapping",
                rules={}
            )
        ]
        
        for rule in default_rules:
            self.transformation_engine.register_transformation_rule(rule)
    
    def register_endpoint(self, endpoint: IntegrationEndpoint) -> bool:
        """Register integration endpoint"""
        try:
            self.endpoints[endpoint.endpoint_id] = endpoint
            self.logger.info(f"Registered endpoint: {endpoint.name} ({endpoint.protocol.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register endpoint: {e}")
            return False
    
    async def connect_endpoint(self, endpoint_id: str) -> bool:
        """Connect to registered endpoint"""
        try:
            if endpoint_id not in self.endpoints:
                raise Exception(f"Endpoint not found: {endpoint_id}")
            
            endpoint = self.endpoints[endpoint_id]
            adapter = await self.protocol_registry.get_adapter(endpoint)
            
            self.endpoint_adapters[endpoint_id] = adapter
            self.integration_metrics["endpoints_connected"] += 1
            
            self.logger.info(f"Connected to endpoint: {endpoint.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to endpoint {endpoint_id}: {e}")
            return False
    
    async def send_message(self, endpoint_id: str, message: EnterpriseMessage,
                          transformation_rule_id: Optional[str] = None) -> Dict[str, Any]:
        """Send message through integration layer"""
        start_time = time.time()
        
        try:
            # Get endpoint adapter
            if endpoint_id not in self.endpoint_adapters:
                await self.connect_endpoint(endpoint_id)
            
            adapter = self.endpoint_adapters[endpoint_id]
            
            # Apply transformation if specified
            processed_message = message
            if transformation_rule_id:
                processed_message = await self.transformation_engine.transform_message(
                    message, transformation_rule_id
                )
                self.integration_metrics["transformations_applied"] += 1
            
            # Send through adapter
            response = await adapter.send_message(processed_message)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_metrics(processing_time, True)
            
            # Add to processing history
            processed_message.processing_history.append({
                "step": "send",
                "endpoint_id": endpoint_id,
                "timestamp": datetime.now().isoformat(),
                "response": response
            })
            
            return {
                "success": True,
                "message_id": processed_message.message_id,
                "endpoint": endpoint_id,
                "response": response,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_metrics(processing_time, False)
            
            # Add to dead letter queue for retry
            message.processing_errors.append(str(e))
            message.retry_count += 1
            
            if message.retry_count <= message.max_retries:
                self.dead_letter_queue.put((endpoint_id, message, transformation_rule_id))
            
            self.logger.error(f"Message sending failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message_id": message.message_id,
                "processing_time_ms": processing_time
            }
    
    async def publish_event(self, event_type: str, data: Any, 
                           routing_key: Optional[str] = None) -> Dict[str, Any]:
        """Publish event to enterprise service bus"""
        try:
            message = EnterpriseMessage(
                message_type=event_type,
                source_system="integration_layer",
                payload=data,
                routing_key=routing_key or event_type,
                priority=MessagePriority.NORMAL
            )
            
            # Notify all subscribers
            subscribers = self.event_subscriptions.get(event_type, [])
            results = []
            
            for subscriber_endpoint in subscribers:
                result = await self.send_message(subscriber_endpoint, message)
                results.append(result)
            
            return {
                "event_type": event_type,
                "message_id": message.message_id,
                "subscribers_notified": len(subscribers),
                "delivery_results": results
            }
            
        except Exception as e:
            self.logger.error(f"Event publishing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def subscribe_to_event(self, event_type: str, endpoint_id: str) -> bool:
        """Subscribe endpoint to event type"""
        try:
            if event_type not in self.event_subscriptions:
                self.event_subscriptions[event_type] = []
            
            if endpoint_id not in self.event_subscriptions[event_type]:
                self.event_subscriptions[event_type].append(endpoint_id)
                self.logger.info(f"Subscribed {endpoint_id} to {event_type}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Event subscription failed: {e}")
            return False
    
    async def start_saga(self, saga_id: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Start distributed saga transaction"""
        try:
            saga = {
                "saga_id": saga_id,
                "steps": steps,
                "current_step": 0,
                "status": "running",
                "compensation_steps": [],
                "start_time": datetime.now(),
                "context": {}
            }
            
            self.sagas[saga_id] = saga
            
            # Execute first step
            result = await self._execute_saga_step(saga_id, 0)
            
            return {
                "saga_id": saga_id,
                "status": saga["status"],
                "current_step": saga["current_step"],
                "step_result": result
            }
            
        except Exception as e:
            self.logger.error(f"Saga start failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_saga_step(self, saga_id: str, step_index: int) -> Dict[str, Any]:
        """Execute single saga step"""
        saga = self.sagas[saga_id]
        
        if step_index >= len(saga["steps"]):
            saga["status"] = "completed"
            return {"status": "saga_completed"}
        
        step = saga["steps"][step_index]
        
        try:
            # Create message for step
            message = EnterpriseMessage(
                message_type=step.get("message_type", "saga_step"),
                source_system="integration_layer",
                payload=step.get("payload", {}),
                correlation_id=saga_id
            )
            
            # Send message
            result = await self.send_message(
                step["endpoint_id"], 
                message, 
                step.get("transformation_rule_id")
            )
            
            if result["success"]:
                # Step successful, move to next
                saga["current_step"] = step_index + 1
                saga["context"][f"step_{step_index}_result"] = result
                
                # Execute next step
                if saga["current_step"] < len(saga["steps"]):
                    return await self._execute_saga_step(saga_id, saga["current_step"])
                else:
                    saga["status"] = "completed"
                    return {"status": "saga_completed", "result": result}
            else:
                # Step failed, start compensation
                saga["status"] = "compensating"
                await self._compensate_saga(saga_id, step_index - 1)
                return {"status": "saga_failed", "error": result.get("error")}
                
        except Exception as e:
            saga["status"] = "failed"
            self.logger.error(f"Saga step execution failed: {e}")
            return {"status": "saga_failed", "error": str(e)}
    
    async def _compensate_saga(self, saga_id: str, from_step: int):
        """Execute saga compensation steps"""
        saga = self.sagas[saga_id]
        
        # Execute compensation in reverse order
        for step_index in range(from_step, -1, -1):
            step = saga["steps"][step_index]
            compensation = step.get("compensation")
            
            if compensation:
                try:
                    # Execute compensation step
                    message = EnterpriseMessage(
                        message_type=compensation.get("message_type", "saga_compensation"),
                        source_system="integration_layer",
                        payload=compensation.get("payload", {}),
                        correlation_id=saga_id
                    )
                    
                    await self.send_message(
                        compensation["endpoint_id"], 
                        message,
                        compensation.get("transformation_rule_id")
                    )
                    
                except Exception as e:
                    self.logger.error(f"Saga compensation failed for step {step_index}: {e}")
        
        saga["status"] = "compensated"
    
    def _update_processing_metrics(self, processing_time_ms: float, success: bool):
        """Update integration processing metrics"""
        self.integration_metrics["messages_processed"] += 1
        
        if not success:
            self.integration_metrics["processing_errors"] += 1
        
        # Update average processing time
        total_messages = self.integration_metrics["messages_processed"]
        current_avg = self.integration_metrics["average_processing_time"]
        
        self.integration_metrics["average_processing_time"] = (
            (current_avg * (total_messages - 1) + processing_time_ms) / total_messages
        )
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        # Collect adapter statuses
        adapter_statuses = {}
        for endpoint_id, adapter in self.endpoint_adapters.items():
            adapter_statuses[endpoint_id] = adapter.get_health_status()
        
        return {
            "integration_metrics": self.integration_metrics.copy(),
            "registered_endpoints": len(self.endpoints),
            "connected_endpoints": len(self.endpoint_adapters),
            "active_sagas": len([s for s in self.sagas.values() if s["status"] == "running"]),
            "event_subscriptions": len(self.event_subscriptions),
            "dead_letter_queue_size": self.dead_letter_queue.qsize(),
            "adapter_statuses": adapter_statuses,
            "transformation_rules": len(self.transformation_engine.transformation_rules)
        }
    
    def get_endpoint_documentation(self) -> Dict[str, Any]:
        """Get documentation for all endpoints"""
        docs = {}
        
        for endpoint_id, endpoint in self.endpoints.items():
            docs[endpoint_id] = {
                "name": endpoint.name,
                "protocol": endpoint.protocol.value,
                "connection_string": endpoint.connection_string,
                "data_format": endpoint.data_format.value,
                "timeout_seconds": endpoint.timeout_seconds,
                "enabled": endpoint.enabled,
                "metadata": endpoint.metadata
            }
        
        return docs


# ============================================================================
# GLOBAL INTEGRATION INSTANCE
# ============================================================================

# Global instance for enterprise integration
enterprise_integration = EnterpriseIntegrationLayer()

# Export for external use
__all__ = [
    'ProtocolType',
    'DataFormat',
    'MessagePriority',
    'IntegrationPattern',
    'IntegrationEndpoint',
    'EnterpriseMessage',
    'TransformationRule',
    'ProtocolAdapterRegistry',
    'ProtocolAdapter',
    'DataTransformationEngine',
    'EnterpriseIntegrationLayer',
    'enterprise_integration'
]