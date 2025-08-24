"""
System Profiler for TestMaster

Comprehensive system resource monitoring and profiling
with real-time metrics collection and analysis.
"""

import time
import threading
import platform
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
import json

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from .telemetry_collector import get_telemetry_collector

@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_used_percent: float
    disk_free_gb: float
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None
    process_count: int = 0
    thread_count: int = 0
    file_descriptors: int = 0
    python_memory_mb: float = 0.0

@dataclass
class ProcessMetrics:
    """Process-specific metrics."""
    timestamp: datetime
    pid: int
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    num_threads: int
    num_fds: int = 0
    status: str = "unknown"
    create_time: float = 0.0

@dataclass
class ResourceAlert:
    """Resource usage alert."""
    alert_id: str
    timestamp: datetime
    resource_type: str  # cpu, memory, disk, etc.
    current_value: float
    threshold: float
    severity: str  # info, warning, critical
    message: str
    resolved: bool = False

class SystemProfiler:
    """
    Comprehensive system profiler for TestMaster.
    
    Monitors:
    - CPU usage and load
    - Memory usage and availability
    - Disk usage and I/O
    - Network activity
    - Process and thread counts
    - Python-specific metrics
    """
    
    def __init__(self, collection_interval: float = 10.0, max_metrics: int = 1000):
        """
        Initialize system profiler.
        
        Args:
            collection_interval: Metrics collection interval in seconds
            max_metrics: Maximum metrics to keep in memory
        """
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system')
        
        if not self.enabled:
            return
        
        self.collection_interval = collection_interval
        self.max_metrics = max_metrics
        
        # Data storage
        self.system_metrics: deque = deque(maxlen=max_metrics)
        self.process_metrics: deque = deque(maxlen=max_metrics)
        self.alerts: List[ResourceAlert] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # System monitoring capabilities
        self.psutil_available = self._check_psutil()
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_percent": {"warning": 85.0, "critical": 95.0},
            "disk_used_percent": {"warning": 85.0, "critical": 95.0}
        }
        
        # Integrations
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        self.telemetry = get_telemetry_collector()
        
        # Statistics
        self.metrics_collected = 0
        self.alerts_generated = 0
        
        print("System profiler initialized")
        print(f"   Collection interval: {self.collection_interval}s")
        print(f"   PSUtil available: {self.psutil_available}")
    
    def _check_psutil(self) -> bool:
        """Check if psutil is available for system monitoring."""
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def start_monitoring(self):
        """Start system monitoring."""
        if not self.enabled or self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.shutdown_event.clear()
        
        def monitor_worker():
            while not self.shutdown_event.is_set():
                try:
                    # Collect metrics
                    self._collect_system_metrics()
                    self._collect_process_metrics()
                    
                    # Check for alerts
                    self._check_resource_alerts()
                    
                    # Wait for next collection
                    if self.shutdown_event.wait(timeout=self.collection_interval):
                        break
                    
                except Exception as e:
                    # Log error and continue
                    print(f"System profiler error: {e}")
                    if self.shutdown_event.wait(timeout=5.0):
                        break
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        print(f"System monitoring started (interval: {self.collection_interval}s)")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        if not self.enabled or not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.shutdown_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        print("System monitoring stopped")
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        if not self.psutil_available:
            # Fallback to basic metrics
            self._collect_basic_metrics()
            return
        
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = round(memory.used / 1024 / 1024, 2)
            memory_available_mb = round(memory.available / 1024 / 1024, 2)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used_percent = (disk.used / disk.total) * 100
            disk_free_gb = round(disk.free / 1024 / 1024 / 1024, 2)
            
            # Network metrics (if available)
            network_bytes_sent = None
            network_bytes_recv = None
            try:
                network = psutil.net_io_counters()
                network_bytes_sent = network.bytes_sent
                network_bytes_recv = network.bytes_recv
            except:
                pass
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Thread count (sum across all processes)
            thread_count = 0
            try:
                for proc in psutil.process_iter(['num_threads']):
                    try:
                        thread_count += proc.info['num_threads'] or 0
                    except:
                        pass
            except:
                pass
            
            # Python process memory
            python_memory_mb = 0.0
            try:
                current_process = psutil.Process()
                python_memory_mb = round(current_process.memory_info().rss / 1024 / 1024, 2)
            except:
                pass
            
            # Create metrics snapshot
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=round(cpu_percent, 2),
                memory_percent=round(memory_percent, 2),
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_used_percent=round(disk_used_percent, 2),
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                thread_count=thread_count,
                python_memory_mb=python_memory_mb
            )
            
            # Store metrics
            self.system_metrics.append(metrics)
            self.metrics_collected += 1
            
            # Send to telemetry
            self.telemetry.record_event(
                event_type="system_metrics",
                component="system_profiler",
                operation="collect_metrics",
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_used_percent": disk_used_percent,
                    "process_count": process_count
                }
            )
            
            # Update shared state
            if self.shared_state:
                self.shared_state.set("system_cpu_percent", cpu_percent)
                self.shared_state.set("system_memory_percent", memory_percent)
                self.shared_state.set("system_metrics_collected", self.metrics_collected)
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    def _collect_basic_metrics(self):
        """Collect basic metrics without psutil."""
        # Very basic metrics using os module
        try:
            cpu_count = os.cpu_count() or 1
            
            # Create minimal metrics
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,  # Can't get without psutil
                memory_percent=0.0,  # Can't get without psutil
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_used_percent=0.0,
                disk_free_gb=0.0,
                process_count=0,
                thread_count=0,
                python_memory_mb=0.0
            )
            
            self.system_metrics.append(metrics)
            self.metrics_collected += 1
            
        except Exception as e:
            print(f"Error collecting basic metrics: {e}")
    
    def _collect_process_metrics(self):
        """Collect current process metrics."""
        if not self.psutil_available:
            return
        
        try:
            import psutil
            
            current_process = psutil.Process()
            
            # Get process info
            process_info = current_process.as_dict([
                'pid', 'cpu_percent', 'memory_percent', 'memory_info',
                'num_threads', 'status', 'create_time'
            ])
            
            # Create process metrics
            metrics = ProcessMetrics(
                timestamp=datetime.now(),
                pid=process_info.get('pid', 0),
                cpu_percent=round(process_info.get('cpu_percent', 0.0), 2),
                memory_percent=round(process_info.get('memory_percent', 0.0), 2),
                memory_rss_mb=round(process_info.get('memory_info', {}).get('rss', 0) / 1024 / 1024, 2),
                memory_vms_mb=round(process_info.get('memory_info', {}).get('vms', 0) / 1024 / 1024, 2),
                num_threads=process_info.get('num_threads', 0),
                status=process_info.get('status', 'unknown'),
                create_time=process_info.get('create_time', 0.0)
            )
            
            # Try to get file descriptors (Unix only)
            try:
                metrics.num_fds = current_process.num_fds()
            except:
                metrics.num_fds = 0
            
            self.process_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error collecting process metrics: {e}")
    
    def _check_resource_alerts(self):
        """Check for resource usage alerts."""
        if not self.system_metrics:
            return
        
        latest_metrics = self.system_metrics[-1]
        current_time = datetime.now()
        
        # Check CPU usage
        self._check_threshold_alert(
            "cpu_percent",
            latest_metrics.cpu_percent,
            self.alert_thresholds["cpu_percent"],
            current_time
        )
        
        # Check memory usage
        self._check_threshold_alert(
            "memory_percent",
            latest_metrics.memory_percent,
            self.alert_thresholds["memory_percent"],
            current_time
        )
        
        # Check disk usage
        self._check_threshold_alert(
            "disk_used_percent",
            latest_metrics.disk_used_percent,
            self.alert_thresholds["disk_used_percent"],
            current_time
        )
    
    def _check_threshold_alert(self, resource_type: str, current_value: float,
                             thresholds: Dict[str, float], timestamp: datetime):
        """Check if a metric exceeds thresholds and generate alerts."""
        if current_value >= thresholds["critical"]:
            severity = "critical"
            threshold = thresholds["critical"]
        elif current_value >= thresholds["warning"]:
            severity = "warning"
            threshold = thresholds["warning"]
        else:
            return  # No alert needed
        
        # Check if we already have an unresolved alert for this resource
        existing_alert = next(
            (alert for alert in self.alerts
             if alert.resource_type == resource_type and not alert.resolved),
            None
        )
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.timestamp = timestamp
        else:
            # Create new alert
            alert = ResourceAlert(
                alert_id=f"{resource_type}_{int(timestamp.timestamp())}",
                timestamp=timestamp,
                resource_type=resource_type,
                current_value=current_value,
                threshold=threshold,
                severity=severity,
                message=f"{resource_type.replace('_', ' ').title()} usage is {current_value:.1f}% (threshold: {threshold:.1f}%)"
            )
            
            self.alerts.append(alert)
            self.alerts_generated += 1
            
            # Send alert to telemetry
            self.telemetry.record_event(
                event_type="resource_alert",
                component="system_profiler",
                operation="threshold_exceeded",
                metadata={
                    "resource_type": resource_type,
                    "current_value": current_value,
                    "threshold": threshold,
                    "severity": severity
                }
            )
            
            print(f"Resource alert: {alert.message}")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        if not self.enabled or not self.system_metrics:
            return None
        
        return self.system_metrics[-1]
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get system metrics history for the specified timeframe."""
        if not self.enabled:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metrics for metrics in self.system_metrics
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_resource_trends(self, resource: str, hours: int = 24) -> Dict[str, Any]:
        """Get trends for a specific resource."""
        if not self.enabled:
            return {}
        
        history = self.get_metrics_history(hours)
        if not history:
            return {"error": "No metrics available"}
        
        values = []
        if resource == "cpu_percent":
            values = [m.cpu_percent for m in history]
        elif resource == "memory_percent":
            values = [m.memory_percent for m in history]
        elif resource == "disk_used_percent":
            values = [m.disk_used_percent for m in history]
        else:
            return {"error": f"Unknown resource: {resource}"}
        
        if not values:
            return {"error": "No data for resource"}
        
        return {
            "resource": resource,
            "timeframe_hours": hours,
            "samples": len(values),
            "current": values[-1],
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "trend": "increasing" if values[-1] > values[0] else "decreasing"
        }
    
    def get_active_alerts(self) -> List[ResourceAlert]:
        """Get all active (unresolved) alerts."""
        if not self.enabled:
            return []
        
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if not self.enabled:
            return
        
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                break
    
    def get_profiler_statistics(self) -> Dict[str, Any]:
        """Get system profiler statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        active_alerts = len(self.get_active_alerts())
        current_metrics = self.get_current_metrics()
        
        stats = {
            "enabled": True,
            "is_monitoring": self.is_monitoring,
            "collection_interval": self.collection_interval,
            "metrics_collected": self.metrics_collected,
            "alerts_generated": self.alerts_generated,
            "active_alerts": active_alerts,
            "psutil_available": self.psutil_available
        }
        
        if current_metrics:
            stats["current_metrics"] = {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "disk_used_percent": current_metrics.disk_used_percent,
                "python_memory_mb": current_metrics.python_memory_mb
            }
        
        return stats
    
    def export_metrics(self, format: str = "json", hours: int = 24) -> str:
        """Export metrics data."""
        if not self.enabled:
            return "{}" if format == "json" else ""
        
        history = self.get_metrics_history(hours)
        
        if format == "json":
            # Convert to serializable format
            export_data = []
            for metrics in history:
                metrics_dict = asdict(metrics)
                metrics_dict['timestamp'] = metrics.timestamp.isoformat()
                export_data.append(metrics_dict)
            
            return json.dumps({
                "export_timestamp": datetime.now().isoformat(),
                "timeframe_hours": hours,
                "total_metrics": len(export_data),
                "metrics": export_data
            }, indent=2)
        
        return str([asdict(m) for m in history])
    
    def clear_metrics(self):
        """Clear all collected metrics and alerts."""
        if not self.enabled:
            return
        
        self.system_metrics.clear()
        self.process_metrics.clear()
        self.alerts.clear()
        self.metrics_collected = 0
        self.alerts_generated = 0
    
    def shutdown(self):
        """Shutdown system profiler."""
        if not self.enabled:
            return
        
        self.stop_monitoring()
        print(f"System profiler shutdown - collected {self.metrics_collected} metrics")

# Global instance
_system_profiler: Optional[SystemProfiler] = None

def get_system_profiler() -> SystemProfiler:
    """Get the global system profiler instance."""
    global _system_profiler
    if _system_profiler is None:
        _system_profiler = SystemProfiler()
    return _system_profiler

# Convenience functions
def profile_system() -> Optional[SystemMetrics]:
    """Get current system metrics."""
    profiler = get_system_profiler()
    return profiler.get_current_metrics()

def get_system_metrics() -> Dict[str, Any]:
    """Get system profiler statistics."""
    profiler = get_system_profiler()
    return profiler.get_profiler_statistics()

def monitor_resources(start: bool = True):
    """Start or stop resource monitoring."""
    profiler = get_system_profiler()
    if start:
        profiler.start_monitoring()
    else:
        profiler.stop_monitoring()