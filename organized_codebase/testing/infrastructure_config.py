"""
Infrastructure Configuration Module
===================================

Infrastructure and deployment configuration settings.
Modularized from testmaster_config.py and unified_config.py.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from .data_models import ConfigBase, DeploymentMode, ServiceType


@dataclass
class DeploymentConfig(ConfigBase):
    """Deployment configuration."""
    
    # Deployment Settings
    mode: DeploymentMode = DeploymentMode.DEVELOPMENT
    auto_deploy: bool = False
    rolling_update: bool = True
    blue_green_deployment: bool = False
    
    # Container Settings
    use_containers: bool = True
    container_registry: str = "docker.io"
    container_image: str = "testmaster:latest"
    container_port: int = 8080
    
    # Kubernetes Settings
    kubernetes_enabled: bool = False
    namespace: str = "default"
    replicas: int = 3
    resource_limits: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "1000m",
        "memory": "2Gi"
    })
    resource_requests: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "500m",
        "memory": "1Gi"
    })
    
    # Health Checks
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    liveness_probe_path: str = "/alive"
    startup_probe_delay: int = 30
    
    # Auto-scaling
    autoscaling_enabled: bool = False
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    def validate(self) -> List[str]:
        """Validate deployment configuration."""
        errors = []
        
        if self.container_port <= 0 or self.container_port > 65535:
            errors.append("Container port must be valid")
        
        if self.kubernetes_enabled:
            if self.replicas <= 0:
                errors.append("Replicas must be positive")
            
            if self.autoscaling_enabled:
                if self.min_replicas > self.max_replicas:
                    errors.append("Min replicas cannot exceed max replicas")
                
                if self.target_cpu_utilization <= 0 or self.target_cpu_utilization > 100:
                    errors.append("Target CPU utilization must be between 1 and 100")
        
        return errors


@dataclass
class DatabaseConfig(ConfigBase):
    """Database configuration."""
    
    # Connection Settings
    engine: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "testmaster"
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Connection Pool
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Performance
    echo_sql: bool = False
    async_enabled: bool = True
    connection_timeout: int = 10
    command_timeout: int = 30
    
    # Backup Settings
    backup_enabled: bool = True
    backup_frequency_hours: int = 24
    backup_retention_days: int = 30
    backup_path: Path = Path("backups")
    
    # Migration Settings
    auto_migrate: bool = False
    migration_path: Path = Path("migrations")
    
    def validate(self) -> List[str]:
        """Validate database configuration."""
        errors = []
        
        if self.port <= 0 or self.port > 65535:
            errors.append("Database port must be valid")
        
        if self.pool_size <= 0:
            errors.append("Pool size must be positive")
        
        if self.backup_frequency_hours <= 0:
            errors.append("Backup frequency must be positive")
        
        if self.backup_retention_days <= 0:
            errors.append("Backup retention must be positive")
        
        return errors


@dataclass
class CacheConfig(ConfigBase):
    """Cache configuration."""
    
    # Cache Settings
    enabled: bool = True
    backend: str = "redis"
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    
    # Cache Behavior
    default_ttl: int = 3600
    max_entries: int = 10000
    eviction_policy: str = "lru"
    
    # Performance
    connection_pool_size: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    
    # Persistence
    persist_to_disk: bool = True
    persistence_path: Path = Path("cache")
    snapshot_frequency: int = 300
    
    def validate(self) -> List[str]:
        """Validate cache configuration."""
        errors = []
        
        if self.port <= 0 or self.port > 65535:
            errors.append("Cache port must be valid")
        
        if self.default_ttl < 0:
            errors.append("Default TTL cannot be negative")
        
        if self.max_entries <= 0:
            errors.append("Max entries must be positive")
        
        valid_policies = ["lru", "lfu", "fifo", "random"]
        if self.eviction_policy not in valid_policies:
            errors.append(f"Eviction policy must be one of {valid_policies}")
        
        return errors


@dataclass
class QueueConfig(ConfigBase):
    """Message queue configuration."""
    
    # Queue Settings
    enabled: bool = True
    broker: str = "redis"
    broker_url: str = "redis://localhost:6379/1"
    result_backend: str = "redis://localhost:6379/2"
    
    # Worker Settings
    worker_concurrency: int = 4
    worker_prefetch_multiplier: int = 4
    task_time_limit: int = 300
    task_soft_time_limit: int = 240
    
    # Queue Behavior
    task_acks_late: bool = True
    task_reject_on_worker_lost: bool = True
    task_track_started: bool = True
    result_expires: int = 3600
    
    # Priority Queues
    priority_queues: Dict[str, int] = field(default_factory=lambda: {
        "high": 10,
        "medium": 5,
        "low": 1
    })
    
    def validate(self) -> List[str]:
        """Validate queue configuration."""
        errors = []
        
        if self.worker_concurrency <= 0:
            errors.append("Worker concurrency must be positive")
        
        if self.task_time_limit <= 0:
            errors.append("Task time limit must be positive")
        
        if self.task_soft_time_limit >= self.task_time_limit:
            errors.append("Soft time limit must be less than hard time limit")
        
        return errors


@dataclass
class StorageConfig(ConfigBase):
    """Storage configuration."""
    
    # Storage Settings
    storage_backend: str = "local"
    storage_path: Path = Path("storage")
    
    # Cloud Storage
    cloud_provider: Optional[str] = None  # aws, gcp, azure
    bucket_name: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: str = "us-east-1"
    
    # Storage Behavior
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".json", ".yaml", ".txt", ".log", ".csv"
    ])
    
    # Retention
    retention_days: int = 90
    archive_old_files: bool = True
    compression_enabled: bool = True
    
    def validate(self) -> List[str]:
        """Validate storage configuration."""
        errors = []
        
        if self.max_file_size_mb <= 0:
            errors.append("Max file size must be positive")
        
        if self.retention_days <= 0:
            errors.append("Retention days must be positive")
        
        if self.cloud_provider:
            if not self.bucket_name:
                errors.append("Bucket name required for cloud storage")
            
            if not self.access_key or not self.secret_key:
                errors.append("Access credentials required for cloud storage")
        
        return errors


@dataclass
class NetworkConfig(ConfigBase):
    """Network configuration."""
    
    # Network Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # SSL/TLS
    ssl_enabled: bool = False
    ssl_cert_path: Optional[Path] = None
    ssl_key_path: Optional[Path] = None
    ssl_ca_path: Optional[Path] = None
    
    # Proxy Settings
    behind_proxy: bool = False
    proxy_headers: List[str] = field(default_factory=lambda: [
        "X-Forwarded-For",
        "X-Forwarded-Proto",
        "X-Forwarded-Host"
    ])
    trusted_proxies: List[str] = field(default_factory=list)
    
    # Timeouts
    request_timeout: int = 60
    keepalive_timeout: int = 5
    
    def validate(self) -> List[str]:
        """Validate network configuration."""
        errors = []
        
        if self.port <= 0 or self.port > 65535:
            errors.append("Port must be valid")
        
        if self.workers <= 0:
            errors.append("Workers must be positive")
        
        if self.ssl_enabled:
            if not self.ssl_cert_path or not self.ssl_key_path:
                errors.append("SSL cert and key paths required when SSL enabled")
        
        return errors


@dataclass
class InfrastructureConfig(ConfigBase):
    """Combined infrastructure configuration."""
    
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Service Discovery
    service_discovery_enabled: bool = False
    service_registry: str = "consul"
    service_name: str = "testmaster"
    
    # Load Balancing
    load_balancer_enabled: bool = False
    load_balancer_algorithm: str = "round_robin"
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60
    
    def validate(self) -> List[str]:
        """Validate all infrastructure configurations."""
        errors = []
        errors.extend(self.deployment.validate())
        errors.extend(self.database.validate())
        errors.extend(self.cache.validate())
        errors.extend(self.queue.validate())
        errors.extend(self.storage.validate())
        errors.extend(self.network.validate())
        
        if self.failure_threshold <= 0:
            errors.append("Failure threshold must be positive")
        
        if self.recovery_timeout <= 0:
            errors.append("Recovery timeout must be positive")
        
        return errors


__all__ = [
    'DeploymentConfig',
    'DatabaseConfig',
    'CacheConfig',
    'QueueConfig',
    'StorageConfig',
    'NetworkConfig',
    'InfrastructureConfig'
]