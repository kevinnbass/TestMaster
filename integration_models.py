"""
Integration Models and Data Structures
Extracted from advanced_system_integration.py for modularization

Contains all data classes, enums, and type definitions for the integration system.
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class IntegrationType(Enum):
    """Integration type enumeration"""
    API = "api"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    CACHE = "cache"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ServiceHealth:
    """Service health status data structure"""
    service_name: str
    status: ServiceStatus
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    uptime_percentage: float = 0.0
    dependency_status: Dict[str, bool] = None


@dataclass
class IntegrationEndpoint:
    """Integration endpoint configuration"""
    name: str
    type: IntegrationType
    url: str
    method: str = "GET"
    headers: Dict[str, str] = None
    timeout_seconds: int = 30
    retry_count: int = 3
    health_check_path: str = "/health"
    required_dependencies: List[str] = None


@dataclass
class SystemMetrics:
    """System-wide integration metrics"""
    total_services: int
    healthy_services: int
    degraded_services: int
    unhealthy_services: int
    average_response_time: float
    uptime_percentage: float
    integration_score: float
    last_updated: datetime