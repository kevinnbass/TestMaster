"""
TestMaster Performance Report Generator

Comprehensive reporting system for performance metrics, telemetry data,
and system insights. Generates detailed reports across all TestMaster
components with visual analytics and trend analysis.

Features:
- Multi-format report generation (HTML, PDF, JSON, CSV)
- Performance dashboards with visual charts
- Trend analysis and anomaly detection
- Component-specific and system-wide reports
- Automated report scheduling and distribution
- Integration with all TestMaster monitoring systems
"""

from .report_generator import (
    ReportGenerator, ReportConfig, ReportFormat, ReportType,
    get_report_generator, generate_performance_report
)
from .dashboard_builder import (
    DashboardBuilder, ChartType, DashboardConfig, DashboardSection,
    get_dashboard_builder, build_performance_dashboard
)
from .metrics_analyzer import (
    MetricsAnalyzer, AnalysisType, TrendAnalysis, AnomalyDetection,
    get_metrics_analyzer, analyze_performance_metrics
)
from .report_scheduler import (
    ReportScheduler, ScheduleConfig, ReportDelivery, DeliveryMethod,
    get_report_scheduler, schedule_automated_reports
)
from .data_collector import (
    DataCollector, DataSource, MetricAggregation, TimeRange,
    get_data_collector, collect_system_metrics
)

__all__ = [
    # Core report generation
    'ReportGenerator',
    'ReportConfig',
    'ReportFormat',
    'ReportType',
    'get_report_generator',
    'generate_performance_report',
    
    # Dashboard building
    'DashboardBuilder',
    'ChartType',
    'DashboardConfig',
    'DashboardSection',
    'get_dashboard_builder',
    'build_performance_dashboard',
    
    # Metrics analysis
    'MetricsAnalyzer',
    'AnalysisType',
    'TrendAnalysis',
    'AnomalyDetection',
    'get_metrics_analyzer',
    'analyze_performance_metrics',
    
    # Report scheduling
    'ReportScheduler',
    'ScheduleConfig',
    'ReportDelivery',
    'DeliveryMethod',
    'get_report_scheduler',
    'schedule_automated_reports',
    
    # Data collection
    'DataCollector',
    'DataSource',
    'MetricAggregation',
    'TimeRange',
    'get_data_collector',
    'collect_system_metrics',
    
    # Utilities
    'is_reporting_enabled',
    'configure_reporting',
    'shutdown_reporting'
]

def is_reporting_enabled() -> bool:
    """Check if performance reporting is enabled."""
    from ..core.feature_flags import FeatureFlags
    return FeatureFlags.is_enabled('layer3_orchestration', 'performance_reporting')

def configure_reporting(output_dir: str = "reports",
                       enable_scheduling: bool = True,
                       enable_analytics: bool = True):
    """Configure performance reporting system."""
    if not is_reporting_enabled():
        print("Performance reporting is disabled")
        return
    
    # Configure report generator
    generator = get_report_generator()
    generator.configure(output_directory=output_dir)
    
    # Configure data collector
    collector = get_data_collector()
    collector.start_collection()
    
    # Configure analytics
    if enable_analytics:
        analyzer = get_metrics_analyzer()
        analyzer.start_analysis()
    
    # Configure scheduling
    if enable_scheduling:
        scheduler = get_report_scheduler()
        scheduler.start_scheduler()
    
    print(f"Performance reporting configured (output: {output_dir}, scheduling: {enable_scheduling})")

def shutdown_reporting():
    """Shutdown all reporting components."""
    try:
        # Shutdown in reverse order of dependencies
        scheduler = get_report_scheduler()
        scheduler.shutdown()
        
        analyzer = get_metrics_analyzer()
        analyzer.shutdown()
        
        collector = get_data_collector()
        collector.stop_collection()
        
        generator = get_report_generator()
        generator.shutdown()
        
        print("Performance reporting shutdown completed")
    except Exception as e:
        print(f"Error during reporting shutdown: {e}")

# Initialize reporting if enabled
if is_reporting_enabled():
    configure_reporting()
