"""
Monitoring Configuration Module
==============================

Monitoring, metrics, and observability configuration settings.
Modularized from testmaster_config.py and unified_config.py.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from .data_models import ConfigBase


@dataclass
class MonitoringConfig(ConfigBase):
    """Monitoring configuration."""
    
    # Core Monitoring
    enabled: bool = True
    continuous_mode: bool = False
    interval_minutes: int = 120
    idle_threshold_minutes: int = 10
    
    # Watch Configuration
    watch_directories: List[str] = field(default_factory=list)
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", "*.log", ".pytest_cache"
    ])
    watch_file_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".yaml", ".yml", ".json", ".toml"
    ])
    
    # Metrics Collection
    collect_performance_metrics: bool = True
    collect_error_metrics: bool = True
    collect_usage_metrics: bool = True
    metrics_retention_days: int = 30
    
    # Alerting
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.05,  # 5% error rate
        "response_time_ms": 1000,  # 1 second
        "memory_usage_mb": 2048,  # 2GB
        "cpu_usage_percent": 80,  # 80%
        "test_failure_rate": 0.10  # 10%
    })
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_rotation_size_mb: int = 100
    log_retention_count: int = 10
    
    # Health Checks
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 60
    health_check_timeout_seconds: int = 10
    
    def validate(self) -> List[str]:
        """Validate monitoring configuration."""
        errors = []
        
        if self.interval_minutes <= 0:
            errors.append("Monitoring interval must be positive")
        
        if self.idle_threshold_minutes < 0:
            errors.append("Idle threshold cannot be negative")
        
        if self.metrics_retention_days <= 0:
            errors.append("Metrics retention must be positive")
        
        # Validate alert thresholds
        for metric, threshold in self.alert_thresholds.items():
            if threshold < 0:
                errors.append(f"Alert threshold for {metric} cannot be negative")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Invalid log level. Must be one of {valid_log_levels}")
        
        return errors


@dataclass
class ReportingConfig(ConfigBase):
    """Reporting configuration."""
    
    # Report Generation
    generate_html_reports: bool = True
    generate_json_reports: bool = True
    generate_xml_reports: bool = False
    generate_markdown_reports: bool = True
    
    # Report Content
    include_coverage_metrics: bool = True
    include_performance_metrics: bool = True
    include_quality_metrics: bool = True
    include_trend_analysis: bool = True
    include_recommendations: bool = True
    
    # Report Storage
    reports_directory: Path = Path("reports")
    archive_reports: bool = True
    archive_after_days: int = 7
    max_report_storage_gb: float = 10.0
    
    # Report Distribution
    email_reports: bool = False
    email_recipients: List[str] = field(default_factory=list)
    upload_to_dashboard: bool = True
    dashboard_url: Optional[str] = None
    
    # Visualization
    generate_charts: bool = True
    chart_types: List[str] = field(default_factory=lambda: [
        "coverage_trend", "test_duration", "failure_rate", "complexity"
    ])
    
    def validate(self) -> List[str]:
        """Validate reporting configuration."""
        errors = []
        
        if self.archive_after_days <= 0:
            errors.append("Archive after days must be positive")
        
        if self.max_report_storage_gb <= 0:
            errors.append("Max report storage must be positive")
        
        if self.email_reports and not self.email_recipients:
            errors.append("Email recipients required when email reports enabled")
        
        return errors


@dataclass
class MetricsConfig(ConfigBase):
    """Metrics collection and analysis configuration."""
    
    # Metrics Types
    track_code_metrics: bool = True
    track_test_metrics: bool = True
    track_performance_metrics: bool = True
    track_quality_metrics: bool = True
    
    # Code Metrics
    code_metrics: Dict[str, bool] = field(default_factory=lambda: {
        "lines_of_code": True,
        "cyclomatic_complexity": True,
        "maintainability_index": True,
        "technical_debt": True,
        "code_duplication": True,
        "dependencies": True
    })
    
    # Test Metrics
    test_metrics: Dict[str, bool] = field(default_factory=lambda: {
        "test_count": True,
        "test_coverage": True,
        "test_duration": True,
        "test_success_rate": True,
        "flaky_tests": True,
        "test_maintainability": True
    })
    
    # Performance Metrics
    performance_metrics: Dict[str, bool] = field(default_factory=lambda: {
        "execution_time": True,
        "memory_usage": True,
        "cpu_usage": True,
        "io_operations": True,
        "network_calls": True,
        "database_queries": True
    })
    
    # Analysis Settings
    analysis_interval_hours: int = 24
    trend_analysis_window_days: int = 30
    anomaly_detection_enabled: bool = True
    baseline_comparison_enabled: bool = True
    
    # Storage
    metrics_database_path: Path = Path("metrics.db")
    export_to_prometheus: bool = False
    prometheus_port: int = 9090
    
    def validate(self) -> List[str]:
        """Validate metrics configuration."""
        errors = []
        
        if self.analysis_interval_hours <= 0:
            errors.append("Analysis interval must be positive")
        
        if self.trend_analysis_window_days <= 0:
            errors.append("Trend analysis window must be positive")
        
        if self.export_to_prometheus and (self.prometheus_port <= 0 or self.prometheus_port > 65535):
            errors.append("Invalid Prometheus port")
        
        return errors


@dataclass
class ObservabilityConfig(ConfigBase):
    """Combined observability configuration."""
    
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # Distributed Tracing
    tracing_enabled: bool = False
    tracing_backend: str = "jaeger"
    tracing_endpoint: Optional[str] = None
    
    # APM Integration
    apm_enabled: bool = False
    apm_service_name: str = "testmaster"
    apm_environment: str = "development"
    
    def validate(self) -> List[str]:
        """Validate all observability configurations."""
        errors = []
        errors.extend(self.monitoring.validate())
        errors.extend(self.reporting.validate())
        errors.extend(self.metrics.validate())
        
        if self.tracing_enabled and not self.tracing_endpoint:
            errors.append("Tracing endpoint required when tracing enabled")
        
        return errors


__all__ = [
    'MonitoringConfig',
    'ReportingConfig',
    'MetricsConfig',
    'ObservabilityConfig'
]