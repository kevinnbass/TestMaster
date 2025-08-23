"""
Dashboard Data Models - Core data structures for the Enhanced Linkage Dashboard

This module provides comprehensive data models and structures for the web dashboard
system, including linkage analysis results, system health metrics, performance data,
and real-time monitoring structures. Designed for enterprise-scale dashboard systems
with advanced analytics and visualization capabilities.

Enterprise Features:
- Comprehensive data models for linkage analysis and system monitoring
- Real-time metrics collection and aggregation structures
- Performance monitoring with advanced analytics support
- Security status tracking and vulnerability management
- Module health monitoring with dependency analysis
- Quality metrics with trend analysis and reporting

Key Components:
- LinkageAnalysisResult: Complete linkage analysis data structures
- SystemHealthMetrics: System performance and health monitoring
- DashboardConfiguration: Dashboard settings and preferences
- LiveDataStream: Real-time data streaming and updates
- SecurityMetrics: Security monitoring and threat detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
from enum import Enum
from pathlib import Path
import json


class FileConnectionStatus(Enum):
    """File connection status classification."""
    ORPHANED = "orphaned"          # No incoming or outgoing connections
    HANGING = "hanging"            # Only outgoing connections
    MARGINAL = "marginal"          # Few connections (1-3)
    WELL_CONNECTED = "well_connected"  # Many connections (4+)
    CRITICAL_HUB = "critical_hub"  # Central node with many connections


class SystemHealthStatus(Enum):
    """System health status levels."""
    EXCELLENT = "excellent"        # 90-100% health
    GOOD = "good"                 # 70-89% health
    WARNING = "warning"           # 50-69% health
    CRITICAL = "critical"         # 30-49% health
    FAILING = "failing"           # 0-29% health


class SecurityLevel(Enum):
    """Security assessment levels."""
    SECURE = "secure"             # No security issues
    LOW_RISK = "low_risk"         # Minor security concerns
    MEDIUM_RISK = "medium_risk"   # Moderate security issues
    HIGH_RISK = "high_risk"       # Significant security vulnerabilities
    CRITICAL_RISK = "critical_risk"  # Critical security threats


class ModuleStatus(Enum):
    """Module operational status."""
    ACTIVE = "active"             # Module is operational
    INACTIVE = "inactive"         # Module is not running
    ERROR = "error"               # Module has errors
    UNKNOWN = "unknown"           # Module status unknown


@dataclass
class FileMetrics:
    """Metrics for individual files in the system."""
    file_path: str
    file_size: int
    lines_of_code: int
    complexity_score: float
    last_modified: datetime
    import_count: int = 0
    export_count: int = 0
    dependency_count: int = 0
    connection_status: FileConnectionStatus = FileConnectionStatus.ORPHANED
    security_score: float = 100.0
    quality_score: float = 0.0
    test_coverage: float = 0.0
    performance_score: float = 0.0


@dataclass
class LinkageConnection:
    """Represents a connection between two files."""
    source_file: str
    target_file: str
    connection_type: str  # import, call, inheritance, etc.
    strength: float       # Connection strength (0.0 - 1.0)
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkageAnalysisResult:
    """Complete results of linkage analysis."""
    analysis_timestamp: datetime
    total_files: int
    total_codebase_files: int
    analysis_coverage: str
    
    # File categorization
    orphaned_files: List[FileMetrics] = field(default_factory=list)
    hanging_files: List[FileMetrics] = field(default_factory=list)
    marginal_files: List[FileMetrics] = field(default_factory=list)
    well_connected_files: List[FileMetrics] = field(default_factory=list)
    critical_hub_files: List[FileMetrics] = field(default_factory=list)
    
    # Connection analysis
    connections: List[LinkageConnection] = field(default_factory=list)
    connection_density: float = 0.0
    average_connections_per_file: float = 0.0
    
    # Quality metrics
    overall_connectivity_score: float = 0.0
    modularity_score: float = 0.0
    dependency_health_score: float = 0.0
    
    # Performance metrics
    analysis_duration_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'total_files': self.total_files,
            'total_codebase_files': self.total_codebase_files,
            'analysis_coverage': self.analysis_coverage,
            'orphaned_files': [
                {
                    'file_path': f.file_path,
                    'file_size': f.file_size,
                    'lines_of_code': f.lines_of_code,
                    'complexity_score': f.complexity_score,
                    'last_modified': f.last_modified.isoformat(),
                    'connection_status': f.connection_status.value
                }
                for f in self.orphaned_files
            ],
            'hanging_files': [
                {
                    'file_path': f.file_path,
                    'connection_status': f.connection_status.value,
                    'dependency_count': f.dependency_count
                }
                for f in self.hanging_files
            ],
            'marginal_files': [
                {
                    'file_path': f.file_path,
                    'connection_status': f.connection_status.value,
                    'quality_score': f.quality_score
                }
                for f in self.marginal_files
            ],
            'well_connected_files': [
                {
                    'file_path': f.file_path,
                    'connection_status': f.connection_status.value,
                    'performance_score': f.performance_score
                }
                for f in self.well_connected_files
            ],
            'connections': [
                {
                    'source': c.source_file,
                    'target': c.target_file,
                    'type': c.connection_type,
                    'strength': c.strength
                }
                for c in self.connections
            ],
            'connection_density': self.connection_density,
            'overall_connectivity_score': self.overall_connectivity_score,
            'analysis_duration_seconds': self.analysis_duration_seconds
        }


@dataclass
class SystemHealthMetrics:
    """Comprehensive system health metrics."""
    timestamp: datetime
    overall_health_score: float
    health_status: SystemHealthStatus
    
    # CPU and Memory
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Application metrics
    active_modules: int = 0
    failed_modules: int = 0
    total_modules: int = 0
    
    # Performance metrics
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate_percent: float = 0.0
    
    # Quality metrics
    code_quality_score: float = 0.0
    test_coverage_percent: float = 0.0
    technical_debt_score: float = 0.0
    
    # Security metrics
    security_score: float = 100.0
    vulnerability_count: int = 0
    security_level: SecurityLevel = SecurityLevel.SECURE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_health_score': self.overall_health_score,
            'health_status': self.health_status.value,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'disk_usage_percent': self.disk_usage_percent,
            'active_modules': self.active_modules,
            'failed_modules': self.failed_modules,
            'total_modules': self.total_modules,
            'response_time_ms': self.response_time_ms,
            'throughput_rps': self.throughput_rps,
            'error_rate_percent': self.error_rate_percent,
            'code_quality_score': self.code_quality_score,
            'test_coverage_percent': self.test_coverage_percent,
            'security_score': self.security_score,
            'vulnerability_count': self.vulnerability_count,
            'security_level': self.security_level.value
        }


@dataclass
class ModuleHealthStatus:
    """Health status for individual modules."""
    module_name: str
    status: ModuleStatus
    health_score: float
    last_check: datetime
    
    # Module-specific metrics
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Functionality metrics
    operations_per_second: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    
    # Issues
    current_issues: List[str] = field(default_factory=list)
    resolved_issues: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    timestamp: datetime
    
    # Response times
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    transactions_per_second: float = 0.0
    data_throughput_mbps: float = 0.0
    
    # Resource utilization
    cpu_cores_used: float = 0.0
    memory_gb_used: float = 0.0
    disk_io_mbps: float = 0.0
    network_io_mbps: float = 0.0
    
    # Quality metrics
    error_rate: float = 0.0
    success_rate: float = 100.0
    availability_percent: float = 100.0


@dataclass
class SecurityMetrics:
    """Security monitoring and assessment metrics."""
    timestamp: datetime
    overall_security_score: float
    security_level: SecurityLevel
    
    # Vulnerability tracking
    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    medium_vulnerabilities: int = 0
    low_vulnerabilities: int = 0
    
    # Security checks
    authentication_failures: int = 0
    authorization_failures: int = 0
    suspicious_activities: int = 0
    
    # Compliance
    compliance_score: float = 100.0
    policy_violations: int = 0
    
    # Threat detection
    threats_detected: int = 0
    threats_blocked: int = 0
    false_positives: int = 0


@dataclass
class QualityMetrics:
    """Code quality and testing metrics."""
    timestamp: datetime
    
    # Code quality
    overall_quality_score: float = 0.0
    maintainability_index: float = 0.0
    complexity_score: float = 0.0
    duplication_percentage: float = 0.0
    
    # Testing metrics
    test_coverage_percentage: float = 0.0
    unit_test_count: int = 0
    integration_test_count: int = 0
    failed_test_count: int = 0
    
    # Technical debt
    technical_debt_ratio: float = 0.0
    code_smells: int = 0
    bugs: int = 0
    vulnerabilities: int = 0
    
    # Documentation
    documentation_coverage: float = 0.0
    outdated_documentation_count: int = 0


@dataclass
class DashboardConfiguration:
    """Dashboard configuration and preferences."""
    dashboard_title: str = "Enhanced Linkage Dashboard"
    refresh_interval_seconds: int = 30
    max_files_analyzed: Optional[int] = None
    base_directory: str = "../TestMaster"
    
    # Display preferences
    show_orphaned_files: bool = True
    show_hanging_files: bool = True
    show_performance_metrics: bool = True
    show_security_status: bool = True
    
    # Analysis settings
    enable_deep_analysis: bool = False
    enable_performance_monitoring: bool = True
    enable_security_scanning: bool = True
    
    # Thresholds
    health_warning_threshold: float = 70.0
    health_critical_threshold: float = 50.0
    performance_warning_threshold: float = 1000.0  # ms
    
    # Real-time settings
    enable_real_time_updates: bool = True
    websocket_enabled: bool = True
    auto_refresh: bool = True


@dataclass
class LiveDataStream:
    """Real-time data streaming structure."""
    stream_id: str
    timestamp: datetime
    data_type: str  # linkage, health, performance, security, quality
    
    # Data payload
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source: str = "dashboard_system"
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON transmission."""
        return {
            'stream_id': self.stream_id,
            'timestamp': self.timestamp.isoformat(),
            'data_type': self.data_type,
            'data': self.data,
            'source': self.source,
            'priority': self.priority,
            'tags': self.tags
        }


# Factory Functions

def create_file_metrics(file_path: str, **kwargs) -> FileMetrics:
    """
    Create file metrics with default values.
    
    Args:
        file_path: Path to the file
        **kwargs: Additional file metrics parameters
        
    Returns:
        Configured FileMetrics instance
    """
    path_obj = Path(file_path)
    
    defaults = {
        'file_size': path_obj.stat().st_size if path_obj.exists() else 0,
        'lines_of_code': 0,
        'complexity_score': 0.0,
        'last_modified': datetime.fromtimestamp(path_obj.stat().st_mtime) if path_obj.exists() else datetime.now()
    }
    
    defaults.update(kwargs)
    
    return FileMetrics(
        file_path=file_path,
        **defaults
    )


def create_linkage_analysis_result(base_directory: str = "../TestMaster") -> LinkageAnalysisResult:
    """
    Create empty linkage analysis result structure.
    
    Args:
        base_directory: Base directory for analysis
        
    Returns:
        Empty LinkageAnalysisResult instance
    """
    return LinkageAnalysisResult(
        analysis_timestamp=datetime.now(),
        total_files=0,
        total_codebase_files=0,
        analysis_coverage="0/0"
    )


def create_system_health_metrics() -> SystemHealthMetrics:
    """
    Create system health metrics with default values.
    
    Returns:
        SystemHealthMetrics instance with defaults
    """
    return SystemHealthMetrics(
        timestamp=datetime.now(),
        overall_health_score=100.0,
        health_status=SystemHealthStatus.EXCELLENT
    )


def create_dashboard_config(**kwargs) -> DashboardConfiguration:
    """
    Create dashboard configuration with custom settings.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        DashboardConfiguration instance
    """
    return DashboardConfiguration(**kwargs)


def create_live_data_stream(data_type: str, data: Dict[str, Any], **kwargs) -> LiveDataStream:
    """
    Create live data stream for real-time updates.
    
    Args:
        data_type: Type of data (linkage, health, performance, etc.)
        data: Data payload
        **kwargs: Additional stream parameters
        
    Returns:
        LiveDataStream instance
    """
    import uuid
    
    defaults = {
        'stream_id': str(uuid.uuid4()),
        'timestamp': datetime.now()
    }
    
    defaults.update(kwargs)
    
    return LiveDataStream(
        data_type=data_type,
        data=data,
        **defaults
    )


# Utility Functions

def calculate_health_status(health_score: float) -> SystemHealthStatus:
    """
    Calculate health status based on score.
    
    Args:
        health_score: Health score (0-100)
        
    Returns:
        SystemHealthStatus enum value
    """
    if health_score >= 90:
        return SystemHealthStatus.EXCELLENT
    elif health_score >= 70:
        return SystemHealthStatus.GOOD
    elif health_score >= 50:
        return SystemHealthStatus.WARNING
    elif health_score >= 30:
        return SystemHealthStatus.CRITICAL
    else:
        return SystemHealthStatus.FAILING


def calculate_security_level(security_score: float, vulnerability_count: int) -> SecurityLevel:
    """
    Calculate security level based on score and vulnerabilities.
    
    Args:
        security_score: Security score (0-100)
        vulnerability_count: Number of vulnerabilities
        
    Returns:
        SecurityLevel enum value
    """
    if security_score >= 95 and vulnerability_count == 0:
        return SecurityLevel.SECURE
    elif security_score >= 80 and vulnerability_count <= 2:
        return SecurityLevel.LOW_RISK
    elif security_score >= 60 and vulnerability_count <= 5:
        return SecurityLevel.MEDIUM_RISK
    elif security_score >= 40 and vulnerability_count <= 10:
        return SecurityLevel.HIGH_RISK
    else:
        return SecurityLevel.CRITICAL_RISK


def categorize_file_connection_status(import_count: int, export_count: int, dependency_count: int) -> FileConnectionStatus:
    """
    Categorize file connection status based on metrics.
    
    Args:
        import_count: Number of imports
        export_count: Number of exports
        dependency_count: Number of dependencies
        
    Returns:
        FileConnectionStatus enum value
    """
    total_connections = import_count + export_count + dependency_count
    
    if total_connections == 0:
        return FileConnectionStatus.ORPHANED
    elif export_count > 0 and import_count == 0:
        return FileConnectionStatus.HANGING
    elif total_connections <= 3:
        return FileConnectionStatus.MARGINAL
    elif total_connections >= 10:
        return FileConnectionStatus.CRITICAL_HUB
    else:
        return FileConnectionStatus.WELL_CONNECTED


def aggregate_metrics(metrics_list: List[Union[SystemHealthMetrics, PerformanceMetrics, SecurityMetrics]]) -> Dict[str, float]:
    """
    Aggregate multiple metric objects into summary statistics.
    
    Args:
        metrics_list: List of metric objects
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not metrics_list:
        return {}
    
    # Extract numeric fields for aggregation
    aggregated = {}
    
    for metrics in metrics_list:
        metrics_dict = metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics.__dict__
        
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)
    
    # Calculate statistics
    result = {}
    for key, values in aggregated.items():
        result[f"{key}_avg"] = sum(values) / len(values)
        result[f"{key}_min"] = min(values)
        result[f"{key}_max"] = max(values)
        result[f"{key}_count"] = len(values)
    
    return result


# Constants

DEFAULT_THRESHOLDS = {
    'health_excellent': 90.0,
    'health_good': 70.0,
    'health_warning': 50.0,
    'health_critical': 30.0,
    'performance_excellent': 100.0,  # ms
    'performance_good': 500.0,
    'performance_warning': 1000.0,
    'performance_critical': 2000.0,
    'security_secure': 95.0,
    'security_low_risk': 80.0,
    'security_medium_risk': 60.0,
    'security_high_risk': 40.0
}

DASHBOARD_COLORS = {
    'excellent': '#28a745',
    'good': '#17a2b8',
    'warning': '#ffc107',
    'critical': '#dc3545',
    'failing': '#6c757d',
    'secure': '#28a745',
    'low_risk': '#17a2b8',
    'medium_risk': '#ffc107',
    'high_risk': '#fd7e14',
    'critical_risk': '#dc3545'
}

# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Dashboard Models Team'
__description__ = 'Comprehensive data models for Enhanced Linkage Dashboard system'