#!/usr/bin/env python3
"""
Real-time Frontend Monitor - Atomic Component
Real-time monitoring for frontend dashboards
Agent Z - STEELCLAD Frontend Atomization
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict


class MonitoringState(Enum):
    """Monitoring system states"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class MetricType(Enum):
    """Types of metrics to monitor"""
    PERFORMANCE = "performance"
    HEALTH = "health"
    USAGE = "usage"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class MonitoringMetric:
    """Monitoring metric data"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    unit: str = ""
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'unit': self.unit,
            'tags': self.tags or []
        }


class RealtimeFrontendMonitor:
    """
    Real-time frontend monitoring component
    Monitors and reports dashboard metrics to frontend
    """
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.state = MonitoringState.STOPPED
        
        # Metric storage
        self.current_metrics: Dict[str, MonitoringMetric] = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=100))
        
        # Monitoring configuration
        self.monitored_metrics = {
            'response_time': MetricType.PERFORMANCE,
            'active_users': MetricType.USAGE,
            'error_rate': MetricType.ERROR,
            'cpu_usage': MetricType.HEALTH,
            'memory_usage': MetricType.HEALTH
        }
        
        # Thresholds for alerts
        self.thresholds = {
            'response_time': {'warning': 100, 'critical': 200},
            'error_rate': {'warning': 0.05, 'critical': 0.10},
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95}
        }
        
        # Callbacks for metric updates
        self.update_callbacks: List[Callable] = []
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.monitoring_stats = {
            'cycles_completed': 0,
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'avg_cycle_time': 0.0,
            'start_time': None
        }
    
    def start_monitoring(self) -> bool:
        """Start real-time monitoring"""
        if self.state != MonitoringState.STOPPED:
            return False
        
        self.state = MonitoringState.STARTING
        self.stop_event.clear()
        self.monitoring_stats['start_time'] = datetime.now()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.state = MonitoringState.RUNNING
        return True
    
    def stop_monitoring(self) -> bool:
        """Stop real-time monitoring"""
        if self.state not in [MonitoringState.RUNNING, MonitoringState.PAUSED]:
            return False
        
        self.state = MonitoringState.STOPPED
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        return True
    
    def pause_monitoring(self):
        """Pause monitoring"""
        if self.state == MonitoringState.RUNNING:
            self.state = MonitoringState.PAUSED
    
    def resume_monitoring(self):
        """Resume monitoring"""
        if self.state == MonitoringState.PAUSED:
            self.state = MonitoringState.RUNNING
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            if self.state == MonitoringState.RUNNING:
                cycle_start = time.time()
                
                try:
                    # Collect metrics
                    metrics = self._collect_metrics()
                    
                    # Update current metrics
                    for metric in metrics:
                        self.current_metrics[metric.name] = metric
                        self.metric_history[metric.name].append(metric)
                    
                    # Check thresholds
                    alerts = self._check_thresholds(metrics)
                    
                    # Notify callbacks
                    self._notify_updates(metrics, alerts)
                    
                    # Update stats
                    cycle_time = time.time() - cycle_start
                    self._update_monitoring_stats(cycle_time, len(metrics))
                    
                except Exception:
                    self.state = MonitoringState.ERROR
            
            # Wait for next cycle
            self.stop_event.wait(self.update_interval)
    
    def _collect_metrics(self) -> List[MonitoringMetric]:
        """Collect current metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # Simulate metric collection (would be replaced with actual collection)
        for metric_name, metric_type in self.monitored_metrics.items():
            if metric_name == 'response_time':
                value = 45.0  # ms
                unit = 'ms'
            elif metric_name == 'active_users':
                value = 10.0
                unit = 'users'
            elif metric_name == 'error_rate':
                value = 0.02
                unit = '%'
            elif metric_name == 'cpu_usage':
                value = 35.0
                unit = '%'
            elif metric_name == 'memory_usage':
                value = 62.0
                unit = '%'
            else:
                value = 0.0
                unit = ''
            
            metric = MonitoringMetric(
                name=metric_name,
                value=value,
                metric_type=metric_type,
                timestamp=timestamp,
                unit=unit
            )
            
            metrics.append(metric)
        
        self.monitoring_stats['metrics_collected'] += len(metrics)
        
        return metrics
    
    def _check_thresholds(self, metrics: List[MonitoringMetric]) -> List[Dict[str, Any]]:
        """Check metrics against thresholds"""
        alerts = []
        
        for metric in metrics:
            if metric.name in self.thresholds:
                threshold = self.thresholds[metric.name]
                
                if metric.value >= threshold.get('critical', float('inf')):
                    alerts.append({
                        'metric': metric.name,
                        'value': metric.value,
                        'threshold': threshold['critical'],
                        'severity': 'critical',
                        'timestamp': metric.timestamp.isoformat()
                    })
                    self.monitoring_stats['alerts_triggered'] += 1
                    
                elif metric.value >= threshold.get('warning', float('inf')):
                    alerts.append({
                        'metric': metric.name,
                        'value': metric.value,
                        'threshold': threshold['warning'],
                        'severity': 'warning',
                        'timestamp': metric.timestamp.isoformat()
                    })
                    self.monitoring_stats['alerts_triggered'] += 1
        
        return alerts
    
    def _notify_updates(self, metrics: List[MonitoringMetric], alerts: List[Dict[str, Any]]):
        """Notify callbacks of metric updates"""
        update_data = {
            'metrics': [m.to_dict() for m in metrics],
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
        
        for callback in self.update_callbacks:
            try:
                callback(update_data)
            except Exception:
                pass
    
    def _update_monitoring_stats(self, cycle_time: float, metrics_count: int):
        """Update monitoring statistics"""
        self.monitoring_stats['cycles_completed'] += 1
        
        # Update average cycle time
        cycles = self.monitoring_stats['cycles_completed']
        current_avg = self.monitoring_stats['avg_cycle_time']
        self.monitoring_stats['avg_cycle_time'] = (
            (current_avg * (cycles - 1) + cycle_time) / cycles
        )
    
    def add_update_callback(self, callback: Callable):
        """Add callback for metric updates"""
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable):
        """Remove update callback"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def add_custom_metric(self, name: str, metric_type: MetricType = MetricType.CUSTOM):
        """Add custom metric to monitor"""
        self.monitored_metrics[name] = metric_type
    
    def set_threshold(self, metric_name: str, warning: float = None, critical: float = None):
        """Set threshold for metric"""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        
        if warning is not None:
            self.thresholds[metric_name]['warning'] = warning
        if critical is not None:
            self.thresholds[metric_name]['critical'] = critical
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        return {
            name: {
                'value': metric.value,
                'unit': metric.unit,
                'type': metric.metric_type.value,
                'timestamp': metric.timestamp.isoformat()
            }
            for name, metric in self.current_metrics.items()
        }
    
    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metric history"""
        if metric_name not in self.metric_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        history = self.metric_history[metric_name]
        
        return [
            m.to_dict() for m in history
            if m.timestamp >= cutoff_time
        ]
    
    def get_metric_statistics(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a metric"""
        if metric_name not in self.metric_history:
            return {}
        
        history = self.metric_history[metric_name]
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'current': values[-1] if values else 0,
            'count': len(values)
        }
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        uptime = None
        if self.monitoring_stats['start_time']:
            uptime = (datetime.now() - self.monitoring_stats['start_time']).total_seconds()
        
        return {
            'state': self.state.value,
            'uptime_seconds': uptime,
            'cycles_completed': self.monitoring_stats['cycles_completed'],
            'metrics_collected': self.monitoring_stats['metrics_collected'],
            'alerts_triggered': self.monitoring_stats['alerts_triggered'],
            'avg_cycle_time_ms': self.monitoring_stats['avg_cycle_time'] * 1000,
            'update_interval_seconds': self.update_interval,
            'monitored_metrics_count': len(self.monitored_metrics),
            'active_callbacks': len(self.update_callbacks)
        }
    
    def export_metrics(self, format_type: str = 'json') -> Any:
        """Export current metrics in specified format"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.get_current_metrics(),
            'statistics': {
                name: self.get_metric_statistics(name)
                for name in self.monitored_metrics.keys()
            },
            'monitoring_status': self.get_monitoring_status()
        }
        
        if format_type == 'json':
            return data
        else:
            # Could support other formats
            return data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return {
            'state': self.state.value,
            'monitored_metrics': len(self.monitored_metrics),
            'thresholds_configured': len(self.thresholds),
            'cycles_completed': self.monitoring_stats['cycles_completed'],
            'metrics_collected': self.monitoring_stats['metrics_collected'],
            'alerts_triggered': self.monitoring_stats['alerts_triggered'],
            'latency_target_met': self.monitoring_stats['avg_cycle_time'] * 1000 < 50
        }