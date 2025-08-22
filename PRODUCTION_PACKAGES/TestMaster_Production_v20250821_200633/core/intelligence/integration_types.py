"""
Integration Types and Data Structures
====================================

Core type definitions and data structures for the Intelligence Integration Master.
Provides enterprise-grade type safety for intelligence system integration,
coordination, and unified operation management with advanced integration patterns.

This module contains all Enum definitions and dataclass structures used throughout
the intelligence integration system, implementing advanced coordination patterns.

Author: Agent A - PHASE 4: Hours 300-400
Created: 2025-08-22
Module: integration_types.py (140 lines)
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class IntelligenceSystemType(Enum):
    """Types of intelligence systems for systematic classification"""
    ANALYTICS = "analytics"
    ML_ORCHESTRATION = "ml_orchestration"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    PATTERN_RECOGNITION = "pattern_recognition"
    AUTONOMOUS_GOVERNANCE = "autonomous_governance"
    CODE_UNDERSTANDING = "code_understanding"
    ARCHITECTURE_INTELLIGENCE = "architecture_intelligence"
    ORCHESTRATION = "orchestration"
    COORDINATION = "coordination"
    META_INTELLIGENCE = "meta_intelligence"
    TEMPORAL_INTELLIGENCE = "temporal_intelligence"
    PRESCRIPTIVE_INTELLIGENCE = "prescriptive_intelligence"


class IntegrationStatus(Enum):
    """Status of system integration for tracking"""
    NOT_INTEGRATED = "not_integrated"
    INTEGRATING = "integrating"
    INTEGRATED = "integrated"
    OPTIMIZING = "optimizing"
    FAILED = "failed"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"
    UPGRADING = "upgrading"


class OperationPriority(Enum):
    """Priority levels for intelligence operations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class HealthStatus(Enum):
    """Health status indicators for system monitoring"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class CommunicationProtocol(Enum):
    """Communication protocols for system interoperability"""
    ASYNC_PYTHON = "async_python"
    REST_API = "rest_api"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    WEBSOCKET = "websocket"
    DIRECT_CALL = "direct_call"


@dataclass
class IntelligenceSystemInfo:
    """Comprehensive information about an intelligence system"""
    system_id: str
    name: str
    type: IntelligenceSystemType
    version: str
    capabilities: List[str]
    status: IntegrationStatus
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    last_health_check: datetime = field(default_factory=datetime.now)
    integration_score: float = 0.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    communication_protocol: CommunicationProtocol = CommunicationProtocol.ASYNC_PYTHON


@dataclass
class IntelligenceOperation:
    """Represents an intelligence operation request with comprehensive metadata"""
    operation_id: str
    operation_type: str
    target_systems: List[str]
    parameters: Dict[str, Any]
    priority: OperationPriority
    timeout: int = 300  # seconds
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expected_execution_time: float = 0.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class OperationResult:
    """Comprehensive result of an intelligence operation"""
    operation_id: str
    success: bool
    results: Dict[str, Any]
    execution_time: float
    systems_used: List[str]
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0


@dataclass
class SystemHealthMetrics:
    """Comprehensive health metrics for an intelligence system"""
    system_id: str
    cpu_usage: float
    memory_usage: float
    response_time: float
    throughput: float
    error_rate: float
    availability: float
    last_updated: datetime = field(default_factory=datetime.now)
    health_status: HealthStatus = HealthStatus.UNKNOWN
    warning_indicators: List[str] = field(default_factory=list)
    performance_trend: str = "stable"  # stable, improving, degrading


@dataclass
class IntegrationConfiguration:
    """Configuration for system integration parameters"""
    system_id: str
    communication_protocol: CommunicationProtocol
    data_format: str = "json"
    compression: str = "gzip"
    encryption: str = "aes256"
    timeout: int = 30
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    batch_support: bool = False
    streaming_support: bool = False
    caching_enabled: bool = True
    monitoring_enabled: bool = True


@dataclass
class SystemCapability:
    """Represents a specific capability provided by an intelligence system"""
    capability_id: str
    name: str
    description: str
    provider_system: str
    input_types: List[str]
    output_types: List[str]
    performance_characteristics: Dict[str, float] = field(default_factory=dict)
    reliability_score: float = 0.0
    complexity_level: str = "medium"  # low, medium, high, expert


@dataclass
class ResourceAllocation:
    """Resource allocation for intelligence systems"""
    system_id: str
    allocated_cpu: float
    allocated_memory: float
    allocated_storage: float
    allocated_network: float
    priority_weight: float = 1.0
    scaling_policy: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrationMetrics:
    """Comprehensive metrics for integration performance"""
    total_registered_systems: int
    integrated_systems: int
    failed_integrations: int
    integration_rate: float
    capability_coverage: int
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    performance_summary: Dict[str, float] = field(default_factory=dict)
    health_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationStatistics:
    """Statistics for intelligence operations"""
    total_operations: int
    successful_operations: int
    failed_operations: int
    success_rate: float
    average_execution_time: float
    peak_operations_per_minute: int
    resource_efficiency: float
    system_utilization: Dict[str, float] = field(default_factory=dict)


# Export all integration types
__all__ = [
    'IntelligenceSystemType',
    'IntegrationStatus', 
    'OperationPriority',
    'HealthStatus',
    'CommunicationProtocol',
    'IntelligenceSystemInfo',
    'IntelligenceOperation',
    'OperationResult',
    'SystemHealthMetrics',
    'IntegrationConfiguration',
    'SystemCapability',
    'ResourceAllocation',
    'IntegrationMetrics',
    'OperationStatistics'
]