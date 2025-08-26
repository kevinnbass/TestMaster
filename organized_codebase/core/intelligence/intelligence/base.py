"""
TestMaster Integration Base Structures
======================================

Shared data structures for the integration hub components.
These are extracted to avoid circular imports between components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class IntegrationStatus(Enum):
    """Integration connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected" 
    CONNECTING = "connecting"
    ERROR = "error"
    DEGRADED = "degraded"


class IntegrationType(Enum):
    """Types of system integrations."""
    API_GATEWAY = "api_gateway"
    DATABASE_SYNC = "database_sync"
    EVENT_STREAM = "event_stream"
    FILE_TRANSFER = "file_transfer"
    REAL_TIME_SYNC = "real_time_sync"
    BATCH_PROCESSING = "batch_processing"
    AUTHENTICATION = "authentication"
    MONITORING = "monitoring"


@dataclass
class IntegrationEndpoint:
    """Enhanced integration endpoint with comprehensive capabilities."""
    
    # Core endpoint data
    endpoint_id: str
    name: str
    url: str
    integration_type: IntegrationType
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    
    # Connection configuration
    authentication: Dict[str, Any] = field(default_factory=dict)
    timeout_settings: Dict[str, float] = field(default_factory=dict)
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    response_times: List[float] = field(default_factory=list)
    error_rates: Dict[str, float] = field(default_factory=dict)
    throughput_metrics: Dict[str, float] = field(default_factory=dict)
    availability_percentage: float = 100.0
    
    # Advanced features
    correlation_tracking: bool = True
    performance_profiling: bool = True
    automatic_failover: bool = False
    circuit_breaker_enabled: bool = True
    
    # Real-time capabilities
    websocket_enabled: bool = False
    event_streaming: bool = False
    real_time_sync: bool = False
    
    # Monitoring and analytics
    health_check_url: Optional[str] = None
    metrics_collection_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Last activity tracking
    last_successful_connection: Optional[datetime] = None
    last_error: Optional[str] = None
    last_health_check: Optional[datetime] = None


@dataclass
class CrossSystemAnalysis:
    """Comprehensive cross-system analysis results."""
    
    # Analysis metadata
    analysis_id: str
    timestamp: datetime
    systems_analyzed: List[str]
    analysis_duration: float
    
    # Correlation analysis
    system_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_correlations: Dict[str, float] = field(default_factory=dict)
    error_correlations: Dict[str, List[str]] = field(default_factory=dict)
    
    # Performance analysis
    cross_system_latency: Dict[str, float] = field(default_factory=dict)
    bottleneck_analysis: Dict[str, str] = field(default_factory=dict)
    resource_contention: List[str] = field(default_factory=list)
    
    # Health and availability
    system_health_scores: Dict[str, float] = field(default_factory=dict)
    availability_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fault_tolerance_assessment: Dict[str, str] = field(default_factory=dict)
    
    # Data flow analysis
    data_flow_patterns: Dict[str, List[str]] = field(default_factory=dict)
    data_consistency_checks: Dict[str, bool] = field(default_factory=dict)
    synchronization_status: Dict[str, str] = field(default_factory=dict)
    
    # Optimization recommendations
    optimization_opportunities: List[str] = field(default_factory=list)
    scaling_recommendations: Dict[str, str] = field(default_factory=dict)
    integration_improvements: List[str] = field(default_factory=list)
    
    # Predictive insights
    predicted_failures: Dict[str, float] = field(default_factory=dict)
    capacity_forecasts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    maintenance_recommendations: List[str] = field(default_factory=list)


@dataclass
class IntegrationEvent:
    """Enhanced integration event with comprehensive tracking."""
    
    # Event basics
    event_id: str
    timestamp: datetime
    source_system: str
    target_system: Optional[str]
    event_type: str
    
    # Event data
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    # Processing status
    status: str = "pending"  # pending, processing, completed, failed
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Routing and delivery
    routing_rules: List[str] = field(default_factory=list)
    delivery_attempts: List[datetime] = field(default_factory=list)
    acknowledgments: Dict[str, datetime] = field(default_factory=dict)
    
    # Performance tracking
    latency_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)


# Export all base structures
__all__ = [
    'IntegrationStatus',
    'IntegrationType',
    'IntegrationEndpoint',
    'CrossSystemAnalysis',
    'IntegrationEvent'
]