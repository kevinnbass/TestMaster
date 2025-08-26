"""
Adapter Abstractions
===================

Integration adapter abstractions for connecting to external systems
with protocol translation, data transformation, and error handling.

Author: Agent E - Infrastructure Consolidation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import json

from .integration_base import (
    IntegrationBase, IntegrationConfiguration, IntegrationContext,
    IntegrationStatus, IntegrationMetrics
)
from .service_abstractions import (
    ServiceBase, ServiceRequest, ServiceResponse, ServiceCredentials
)

T = TypeVar('T')
R = TypeVar('R')
S = TypeVar('S')  # Source type
D = TypeVar('D')  # Destination type


class AdapterType(Enum):
    """Adapter type enumeration."""
    PROTOCOL_ADAPTER = "protocol_adapter"
    DATA_ADAPTER = "data_adapter"
    FORMAT_ADAPTER = "format_adapter"
    SYSTEM_ADAPTER = "system_adapter"
    LEGACY_ADAPTER = "legacy_adapter"
    CLOUD_ADAPTER = "cloud_adapter"
    DATABASE_ADAPTER = "database_adapter"
    MESSAGE_ADAPTER = "message_adapter"
    FILE_ADAPTER = "file_adapter"
    API_ADAPTER = "api_adapter"


class TransformationType(Enum):
    """Data transformation type."""
    MAPPING = "mapping"
    CONVERSION = "conversion"
    ENRICHMENT = "enrichment"
    FILTERING = "filtering"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    ENCRYPTION = "encryption"
    COMPRESSION = "compression"
    CUSTOM = "custom"


class AdapterDirection(Enum):
    """Adapter data flow direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class TransformationRule:
    """Data transformation rule definition."""
    
    name: str
    transformation_type: TransformationType
    source_field: str
    target_field: str
    
    # Transformation function or mapping
    transform_function: Optional[Callable[[Any], Any]] = None
    value_mapping: Optional[Dict[Any, Any]] = None
    
    # Validation rules
    required: bool = False
    default_value: Optional[Any] = None
    validation_pattern: Optional[str] = None
    
    # Conditional transformation
    condition: Optional[str] = None
    condition_function: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    def apply_transformation(self, source_data: Dict[str, Any]) -> Any:
        """Apply transformation rule to source data."""
        # Check condition if specified
        if self.condition_function and not self.condition_function(source_data):
            return None
        
        # Get source value
        source_value = source_data.get(self.source_field)
        
        if source_value is None:
            if self.required:
                raise ValueError(f"Required field '{self.source_field}' is missing")
            return self.default_value
        
        # Apply transformation
        if self.transform_function:
            return self.transform_function(source_value)
        elif self.value_mapping:
            return self.value_mapping.get(source_value, source_value)
        else:
            return source_value


@dataclass
class AdapterConfiguration(IntegrationConfiguration):
    """Adapter-specific configuration."""
    
    # Adapter type and direction
    adapter_type: AdapterType = AdapterType.SYSTEM_ADAPTER
    direction: AdapterDirection = AdapterDirection.BIDIRECTIONAL
    
    # Source and target system information
    source_system: str = ""
    target_system: str = ""
    source_protocol: str = ""
    target_protocol: str = ""
    
    # Transformation settings
    transformation_enabled: bool = True
    validation_enabled: bool = True
    error_handling_mode: str = "strict"  # strict, lenient, ignore
    
    # Mapping and schema settings
    schema_validation_enabled: bool = True
    auto_mapping_enabled: bool = False
    preserve_original_data: bool = True
    
    # Performance settings
    batch_processing_enabled: bool = False
    batch_size: int = 100
    parallel_processing: bool = False
    max_parallel_workers: int = 4
    
    # Caching settings
    cache_enabled: bool = False
    cache_ttl: int = 300  # seconds
    cache_size: int = 1000
    
    # Monitoring settings
    transformation_logging: bool = True
    performance_monitoring: bool = True
    error_tracking: bool = True


@dataclass
class AdapterMetrics(IntegrationMetrics):
    """Adapter-specific metrics."""
    
    # Transformation metrics
    transformations_applied: int = 0
    transformation_errors: int = 0
    transformation_time: float = 0.0
    
    # Data metrics
    records_processed: int = 0
    records_transformed: int = 0
    records_failed: int = 0
    data_volume_bytes: int = 0
    
    # Performance metrics
    average_transformation_time: float = 0.0
    throughput_records_per_second: float = 0.0
    
    # Cache metrics (if enabled)
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size_current: int = 0
    
    def update_transformation_metrics(self, processing_time: float, success: bool = True):
        """Update transformation metrics."""
        self.transformations_applied += 1
        if success:
            self.records_transformed += 1
        else:
            self.transformation_errors += 1
            self.records_failed += 1
        
        # Update average transformation time
        if self.transformations_applied == 1:
            self.average_transformation_time = processing_time
        else:
            self.average_transformation_time = (
                (self.average_transformation_time * (self.transformations_applied - 1) + 
                 processing_time) / self.transformations_applied
            )
        
        # Update throughput
        if self.transformation_time > 0:
            self.throughput_records_per_second = self.records_processed / self.transformation_time


class AdapterBase(IntegrationBase[T, R], Generic[S, D]):
    """
    Abstract base class for all adapter implementations.
    
    Provides unified interface for data transformation, protocol conversion,
    and system integration with comprehensive error handling and monitoring.
    """
    
    def __init__(self, config: AdapterConfiguration):
        super().__init__(config)
        self.adapter_config = config
        self.adapter_metrics = AdapterMetrics()
        self.transformation_rules: List[TransformationRule] = []
        self._transformation_cache: Dict[str, Any] = {}
    
    @property
    def adapter_type(self) -> AdapterType:
        """Get adapter type."""
        return self.adapter_config.adapter_type
    
    @property
    def direction(self) -> AdapterDirection:
        """Get adapter direction."""
        return self.adapter_config.direction
    
    # Abstract methods - must be implemented by subclasses
    
    @abstractmethod
    async def transform_inbound(self, source_data: S, context: IntegrationContext) -> D:
        """Transform data from source system format to target format."""
        pass
    
    @abstractmethod
    async def transform_outbound(self, target_data: D, context: IntegrationContext) -> S:
        """Transform data from target format to source system format."""
        pass
    
    @abstractmethod
    async def validate_source_data(self, data: S) -> bool:
        """Validate source data format and content."""
        pass
    
    @abstractmethod
    async def validate_target_data(self, data: D) -> bool:
        """Validate target data format and content."""
        pass
    
    # Transformation rule management
    
    def add_transformation_rule(self, rule: TransformationRule):
        """Add transformation rule."""
        self.transformation_rules.append(rule)
    
    def remove_transformation_rule(self, rule_name: str):
        """Remove transformation rule by name."""
        self.transformation_rules = [
            rule for rule in self.transformation_rules 
            if rule.name != rule_name
        ]
    
    def get_transformation_rule(self, rule_name: str) -> Optional[TransformationRule]:
        """Get transformation rule by name."""
        for rule in self.transformation_rules:
            if rule.name == rule_name:
                return rule
        return None
    
    def clear_transformation_rules(self):
        """Clear all transformation rules."""
        self.transformation_rules.clear()
    
    # Data transformation implementation
    
    async def apply_transformation_rules(self, 
                                       source_data: Dict[str, Any],
                                       direction: str = "inbound") -> Dict[str, Any]:
        """Apply transformation rules to data."""
        start_time = datetime.now()
        transformed_data = {}
        
        try:
            for rule in self.transformation_rules:
                try:
                    # Apply transformation rule
                    transformed_value = rule.apply_transformation(source_data)
                    if transformed_value is not None:
                        transformed_data[rule.target_field] = transformed_value
                        
                except Exception as e:
                    if self.adapter_config.error_handling_mode == "strict":
                        raise Exception(f"Transformation rule '{rule.name}' failed: {e}")
                    elif self.adapter_config.error_handling_mode == "lenient":
                        # Log error but continue
                        await self._emit_event("transformation_rule_error", {
                            "rule_name": rule.name,
                            "error": str(e)
                        })
                    # In "ignore" mode, silently skip failed transformations
            
            # Preserve original data if configured
            if self.adapter_config.preserve_original_data:
                transformed_data["_original"] = source_data
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.adapter_metrics.update_transformation_metrics(processing_time, True)
            
            return transformed_data
            
        except Exception as e:
            # Update error metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.adapter_metrics.update_transformation_metrics(processing_time, False)
            raise
    
    # Integration implementation
    
    async def initialize(self) -> bool:
        """Initialize adapter."""
        try:
            # Validate configuration
            if not self.adapter_config.source_system:
                raise ValueError("Source system must be specified")
            
            if not self.adapter_config.target_system:
                raise ValueError("Target system must be specified")
            
            # Initialize transformation cache if enabled
            if self.adapter_config.cache_enabled:
                self._transformation_cache = {}
            
            # Perform adapter-specific initialization
            return await self._initialize_adapter()
            
        except Exception as e:
            await self._emit_event("adapter_initialization_error", {"error": str(e)})
            return False
    
    async def connect(self) -> bool:
        """Establish adapter connections."""
        try:
            # Connect to source and target systems
            source_connected = await self._connect_source()
            target_connected = await self._connect_target()
            
            if source_connected and target_connected:
                self.status = IntegrationStatus.CONNECTED
                return True
            else:
                return False
                
        except Exception as e:
            await self._emit_event("adapter_connection_error", {"error": str(e)})
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect adapter."""
        try:
            # Disconnect from source and target systems
            await self._disconnect_source()
            await self._disconnect_target()
            
            self.status = IntegrationStatus.DISCONNECTED
            return True
            
        except Exception as e:
            await self._emit_event("adapter_disconnection_error", {"error": str(e)})
            return False
    
    async def send_request(self, request: T, context: IntegrationContext) -> R:
        """Process adapter request."""
        try:
            start_time = datetime.now()
            
            # Determine transformation direction
            if self.direction == AdapterDirection.INBOUND:
                result = await self._process_inbound_request(request, context)
            elif self.direction == AdapterDirection.OUTBOUND:
                result = await self._process_outbound_request(request, context)
            else:  # BIDIRECTIONAL
                result = await self._process_bidirectional_request(request, context)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.adapter_metrics.transformation_time += processing_time
            self.adapter_metrics.records_processed += 1
            
            return result
            
        except Exception as e:
            await self._emit_event("adapter_request_error", {
                "error": str(e),
                "context": context.integration_id
            })
            raise
    
    async def receive_response(self, context: IntegrationContext) -> R:
        """Receive adapter response."""
        # Default implementation for synchronous adapters
        raise NotImplementedError("Async response handling not implemented")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform adapter health check."""
        try:
            health_info = {
                "adapter_type": self.adapter_type.value,
                "direction": self.direction.value,
                "source_system": self.adapter_config.source_system,
                "target_system": self.adapter_config.target_system,
                "transformation_rules": len(self.transformation_rules),
                "metrics": {
                    "transformations_applied": self.adapter_metrics.transformations_applied,
                    "transformation_errors": self.adapter_metrics.transformation_errors,
                    "success_rate": (
                        (self.adapter_metrics.transformations_applied - 
                         self.adapter_metrics.transformation_errors) /
                        max(1, self.adapter_metrics.transformations_applied)
                    ),
                    "average_transformation_time": self.adapter_metrics.average_transformation_time
                },
                "healthy": self.is_connected
            }
            
            # Add adapter-specific health information
            adapter_health = await self._perform_adapter_health_check()
            health_info.update(adapter_health)
            
            return health_info
            
        except Exception as e:
            return {
                "adapter_type": self.adapter_type.value,
                "status": "error",
                "error": str(e),
                "healthy": False
            }
    
    # Abstract methods for subclass implementation
    
    async def _initialize_adapter(self) -> bool:
        """Adapter-specific initialization."""
        return True
    
    async def _connect_source(self) -> bool:
        """Connect to source system."""
        return True
    
    async def _connect_target(self) -> bool:
        """Connect to target system."""
        return True
    
    async def _disconnect_source(self):
        """Disconnect from source system."""
        pass
    
    async def _disconnect_target(self):
        """Disconnect from target system."""
        pass
    
    async def _process_inbound_request(self, request: T, context: IntegrationContext) -> R:
        """Process inbound transformation request."""
        raise NotImplementedError("Inbound processing not implemented")
    
    async def _process_outbound_request(self, request: T, context: IntegrationContext) -> R:
        """Process outbound transformation request."""
        raise NotImplementedError("Outbound processing not implemented")
    
    async def _process_bidirectional_request(self, request: T, context: IntegrationContext) -> R:
        """Process bidirectional transformation request."""
        raise NotImplementedError("Bidirectional processing not implemented")
    
    async def _perform_adapter_health_check(self) -> Dict[str, Any]:
        """Perform adapter-specific health check."""
        return {"adapter_specific_health": "ok"}
    
    # Convenience methods
    
    async def transform_data(self, 
                            data: Union[S, D], 
                            direction: str,
                            context: Optional[IntegrationContext] = None) -> Union[D, S]:
        """Transform data in specified direction."""
        if not context:
            context = IntegrationContext(
                integration_id=self.integration_id,
                session_id=f"transform_{int(datetime.now().timestamp())}"
            )
        
        if direction == "inbound":
            return await self.transform_inbound(data, context)
        elif direction == "outbound":
            return await self.transform_outbound(data, context)
        else:
            raise ValueError(f"Invalid transformation direction: {direction}")
    
    async def batch_transform(self, 
                             data_list: List[Union[S, D]], 
                             direction: str,
                             context: Optional[IntegrationContext] = None) -> List[Union[D, S]]:
        """Transform multiple data items."""
        results = []
        
        if self.adapter_config.parallel_processing:
            # Parallel processing
            tasks = []
            for data in data_list:
                task = self.transform_data(data, direction, context)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential processing
            for data in data_list:
                try:
                    result = await self.transform_data(data, direction, context)
                    results.append(result)
                except Exception as e:
                    if self.adapter_config.error_handling_mode == "strict":
                        raise
                    else:
                        results.append(e)
        
        return results
    
    def get_adapter_metrics(self) -> AdapterMetrics:
        """Get adapter-specific metrics."""
        return self.adapter_metrics
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"type={self.adapter_type.value}, "
                f"direction={self.direction.value}, "
                f"source={self.adapter_config.source_system}, "
                f"target={self.adapter_config.target_system})")


# Adapter registry for managing adapter instances
class AdapterRegistry:
    """Registry for managing adapter instances."""
    
    def __init__(self):
        self._adapters: Dict[str, AdapterBase] = {}
        self._adapter_types: Dict[str, type] = {}
    
    def register_adapter_type(self, adapter_type: str, adapter_class: type):
        """Register adapter class for type."""
        self._adapter_types[adapter_type] = adapter_class
    
    def create_adapter(self, 
                      adapter_id: str,
                      adapter_type: str,
                      config: AdapterConfiguration) -> AdapterBase:
        """Create and register adapter instance."""
        if adapter_type not in self._adapter_types:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        adapter_class = self._adapter_types[adapter_type]
        adapter = adapter_class(config)
        
        self._adapters[adapter_id] = adapter
        return adapter
    
    def get_adapter(self, adapter_id: str) -> Optional[AdapterBase]:
        """Get adapter instance by ID."""
        return self._adapters.get(adapter_id)
    
    def remove_adapter(self, adapter_id: str):
        """Remove adapter from registry."""
        if adapter_id in self._adapters:
            del self._adapters[adapter_id]
    
    def list_adapters(self) -> List[str]:
        """List registered adapter IDs."""
        return list(self._adapters.keys())
    
    def get_adapters_by_type(self, adapter_type: AdapterType) -> List[AdapterBase]:
        """Get adapters by type."""
        return [adapter for adapter in self._adapters.values() 
                if adapter.adapter_type == adapter_type]
    
    async def start_all_adapters(self) -> Dict[str, bool]:
        """Start all registered adapters."""
        results = {}
        for adapter_id, adapter in self._adapters.items():
            results[adapter_id] = await adapter.start()
        return results
    
    async def stop_all_adapters(self) -> Dict[str, bool]:
        """Stop all registered adapters."""
        results = {}
        for adapter_id, adapter in self._adapters.items():
            results[adapter_id] = await adapter.stop()
        return results


__all__ = [
    'AdapterType',
    'TransformationType',
    'AdapterDirection',
    'TransformationRule',
    'AdapterConfiguration',
    'AdapterMetrics',
    'AdapterBase',
    'AdapterRegistry'
]