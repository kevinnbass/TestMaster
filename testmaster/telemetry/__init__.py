"""
TestMaster Advanced Telemetry System

Comprehensive telemetry and performance monitoring infrastructure inspired by
PraisonAI's telemetry patterns, providing:

- Anonymous usage tracking with privacy-first design
- Advanced performance monitoring and analysis
- Function execution flow visualization
- System health and bottleneck identification
- Real-time metrics collection and reporting
- Integration with external APM tools

Telemetry can be disabled via environment variables:
- TESTMASTER_TELEMETRY_DISABLED=true
- TESTMASTER_DISABLE_TELEMETRY=true
- DO_NOT_TRACK=true

Performance monitoring can be optimized via:
- TESTMASTER_PERFORMANCE_DISABLED=true (disables monitoring overhead)
- TESTMASTER_FLOW_ANALYSIS_ENABLED=true (enables expensive flow analysis)
"""

import os
import atexit
from typing import Optional, Dict, Any, List

# Import core telemetry components
from .telemetry_collector import TelemetryCollector, get_telemetry_collector
from .performance_monitor import (
    AdvancedPerformanceMonitor, get_performance_monitor,
    monitor_execution, track_operation, monitor_performance
)
from .flow_analyzer import (
    ExecutionFlowAnalyzer, get_flow_analyzer,
    analyze_execution_flow, visualize_flow, detect_bottlenecks
)
from .system_profiler import (
    SystemProfiler, get_system_profiler,
    profile_system, get_system_metrics, monitor_resources
)
from .telemetry_dashboard import (
    TelemetryDashboard, get_telemetry_dashboard,
    create_telemetry_report, export_telemetry_data
)

__all__ = [
    # Core telemetry
    'TelemetryCollector',
    'get_telemetry_collector',
    'enable_telemetry',
    'disable_telemetry',
    'is_telemetry_enabled',
    
    # Performance monitoring
    'AdvancedPerformanceMonitor',
    'get_performance_monitor',
    'monitor_execution',
    'track_operation',
    'monitor_performance',
    
    # Flow analysis
    'ExecutionFlowAnalyzer',
    'get_flow_analyzer',
    'analyze_execution_flow',
    'visualize_flow',
    'detect_bottlenecks',
    
    # System profiling
    'SystemProfiler',
    'get_system_profiler',
    'profile_system',
    'get_system_metrics',
    'monitor_resources',
    
    # Telemetry dashboard
    'TelemetryDashboard',
    'get_telemetry_dashboard',
    'create_telemetry_report',
    'export_telemetry_data',
    
    # Utilities
    'cleanup_telemetry',
    'get_telemetry_status'
]

# Global instances
_telemetry_enabled = None
_atexit_registered = False

def _is_telemetry_disabled() -> bool:
    """Check if telemetry is disabled via environment variables."""
    return any([
        os.environ.get('TESTMASTER_TELEMETRY_DISABLED', '').lower() in ('true', '1', 'yes'),
        os.environ.get('TESTMASTER_DISABLE_TELEMETRY', '').lower() in ('true', '1', 'yes'),
        os.environ.get('DO_NOT_TRACK', '').lower() in ('true', '1', 'yes'),
    ])

def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled."""
    global _telemetry_enabled
    if _telemetry_enabled is None:
        _telemetry_enabled = not _is_telemetry_disabled()
    return _telemetry_enabled

def enable_telemetry():
    """Enable telemetry system."""
    global _telemetry_enabled
    if not _is_telemetry_disabled():
        _telemetry_enabled = True
        _ensure_atexit_handler()
        print("TestMaster telemetry enabled")

def disable_telemetry():
    """Disable telemetry system."""
    global _telemetry_enabled
    _telemetry_enabled = False
    cleanup_telemetry()
    print("TestMaster telemetry disabled")

def _ensure_atexit_handler():
    """Ensure atexit handler is registered for proper cleanup."""
    global _atexit_registered
    if _atexit_registered:
        return
    
    if is_telemetry_enabled():
        atexit.register(cleanup_telemetry)
        _atexit_registered = True

def cleanup_telemetry():
    """Clean up all telemetry resources."""
    try:
        # Clean up components in reverse order of dependency
        dashboard = get_telemetry_dashboard()
        if dashboard and hasattr(dashboard, 'shutdown'):
            dashboard.shutdown()
        
        profiler = get_system_profiler()
        if profiler and hasattr(profiler, 'stop_monitoring'):
            profiler.stop_monitoring()
        
        flow_analyzer = get_flow_analyzer()
        if flow_analyzer and hasattr(flow_analyzer, 'clear_data'):
            flow_analyzer.clear_data()
        
        perf_monitor = get_performance_monitor()
        if perf_monitor and hasattr(perf_monitor, 'shutdown'):
            perf_monitor.shutdown()
        
        collector = get_telemetry_collector()
        if collector and hasattr(collector, 'shutdown'):
            collector.shutdown()
        
    except Exception:
        # Silently handle cleanup errors
        pass

def get_telemetry_status() -> Dict[str, Any]:
    """Get comprehensive telemetry system status."""
    if not is_telemetry_enabled():
        return {
            "enabled": False,
            "reason": "Telemetry disabled via environment variables"
        }
    
    status = {
        "enabled": True,
        "components": {}
    }
    
    try:
        # Check component statuses
        collector = get_telemetry_collector()
        status["components"]["collector"] = {
            "active": collector is not None,
            "events_collected": getattr(collector, 'events_collected', 0) if collector else 0
        }
        
        perf_monitor = get_performance_monitor()
        status["components"]["performance_monitor"] = {
            "active": perf_monitor is not None,
            "operations_tracked": getattr(perf_monitor, 'operations_tracked', 0) if perf_monitor else 0
        }
        
        flow_analyzer = get_flow_analyzer()
        status["components"]["flow_analyzer"] = {
            "active": flow_analyzer is not None,
            "flows_analyzed": getattr(flow_analyzer, 'flows_analyzed', 0) if flow_analyzer else 0
        }
        
        system_profiler = get_system_profiler()
        status["components"]["system_profiler"] = {
            "active": system_profiler is not None,
            "monitoring": getattr(system_profiler, 'is_monitoring', False) if system_profiler else False
        }
        
        dashboard = get_telemetry_dashboard()
        status["components"]["dashboard"] = {
            "active": dashboard is not None,
            "reports_generated": getattr(dashboard, 'reports_generated', 0) if dashboard else 0
        }
        
    except Exception as e:
        status["error"] = str(e)
    
    return status

# Initialize telemetry system if enabled
if is_telemetry_enabled():
    _ensure_atexit_handler()