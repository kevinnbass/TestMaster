#!/usr/bin/env python3
"""
Performance Monitoring Module
Agent D Hour 5 - Modularized System Performance Tracking

Handles system performance monitoring and resource optimization
following STEELCLAD Anti-Regression Modularization Protocol.
"""

import asyncio
import datetime
import json
import logging
import os
import psutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

@dataclass
class PerformanceMetrics:
    """System performance metrics snapshot"""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_io_bytes_sent: int
    network_io_bytes_recv: int
    active_processes: int
    load_average: float
    uptime_seconds: float
    
    # Security-specific metrics
    security_scan_load: float = 0.0
    threat_detection_rate: float = 0.0
    response_time_avg_ms: float = 0.0
    correlation_engine_load: float = 0.0

@dataclass
class PerformanceAlert:
    """Performance threshold alert"""
    timestamp: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    description: str
    suggested_actions: List[str]

class PerformanceThresholds:
    """Performance threshold definitions"""
    
    def __init__(self):
        self.thresholds = {
            'cpu_usage_percent': {
                'warning': 70.0,
                'critical': 85.0,
                'emergency': 95.0
            },
            'memory_usage_percent': {
                'warning': 75.0,
                'critical': 90.0,
                'emergency': 95.0
            },
            'disk_usage_percent': {
                'warning': 80.0,
                'critical': 90.0,
                'emergency': 95.0
            },
            'response_time_avg_ms': {
                'warning': 1000.0,
                'critical': 2000.0,
                'emergency': 5000.0
            },
            'security_scan_load': {
                'warning': 15.0,
                'critical': 25.0,
                'emergency': 40.0
            }
        }
    
    def check_threshold(self, metric_name: str, value: float) -> Optional[str]:
        """Check if metric value exceeds threshold"""
        if metric_name not in self.thresholds:
            return None
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds.get('emergency', float('inf')):
            return 'emergency'
        elif value >= thresholds.get('critical', float('inf')):
            return 'critical'
        elif value >= thresholds.get('warning', float('inf')):
            return 'warning'
        
        return None

class SystemResourceMonitor:
    """Core system resource monitoring"""
    
    def __init__(self):
        """Initialize system resource monitor"""
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.boot_time = psutil.boot_time()
        
        # Resource tracking
        self.cpu_percent_history = deque(maxlen=60)  # Last 60 readings
        self.memory_history = deque(maxlen=60)
        self.disk_history = deque(maxlen=60)
        self.network_history = deque(maxlen=60)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            self.cpu_percent_history.append(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_history.append(memory.percent)
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            disk_free_gb = disk_usage.free / (1024**3)
            self.disk_history.append(disk_percent)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.network_history.append((net_io.bytes_sent, net_io.bytes_recv))
            
            # Process metrics
            active_processes = len(psutil.pids())
            
            # Load average (Windows approximation using CPU)
            load_avg = sum(self.cpu_percent_history) / len(self.cpu_percent_history) if self.cpu_percent_history else 0.0
            
            # Uptime
            uptime = time.time() - self.boot_time
            
            return PerformanceMetrics(
                timestamp=datetime.datetime.now().isoformat(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                memory_available_mb=memory.available / (1024**2),
                disk_usage_percent=disk_percent,
                disk_free_gb=disk_free_gb,
                network_io_bytes_sent=net_io.bytes_sent,
                network_io_bytes_recv=net_io.bytes_recv,
                active_processes=active_processes,
                load_average=load_avg,
                uptime_seconds=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics on error
            return PerformanceMetrics(
                timestamp=datetime.datetime.now().isoformat(),
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_io_bytes_sent=0,
                network_io_bytes_recv=0,
                active_processes=0,
                load_average=0.0,
                uptime_seconds=0.0
            )
    
    def get_process_metrics(self, process_name: str) -> Dict[str, Any]:
        """Get metrics for specific process"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                if process_name.lower() in proc.info['name'].lower():
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
            
            return {
                'process_name': process_name,
                'matching_processes': processes,
                'total_processes': len(processes),
                'total_cpu_percent': sum(p['cpu_percent'] or 0 for p in processes),
                'total_memory_percent': sum(p['memory_percent'] or 0 for p in processes)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting process metrics for {process_name}: {e}")
            return {'process_name': process_name, 'error': str(e)}
    
    def get_trend_analysis(self, minutes: int = 10) -> Dict[str, Any]:
        """Get performance trend analysis"""
        try:
            # Calculate trends from history
            cpu_trend = self._calculate_trend(list(self.cpu_percent_history)[-minutes:])
            memory_trend = self._calculate_trend(list(self.memory_history)[-minutes:])
            disk_trend = self._calculate_trend(list(self.disk_history)[-minutes:])
            
            return {
                'analysis_period_minutes': minutes,
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'disk_trend': disk_trend,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trends: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a list of values"""
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'correlation': 0.0}
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0.0
        
        # Determine trend direction
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        # Calculate correlation coefficient
        if denominator > 0 and len(set(values)) > 1:
            y_variance = sum((values[i] - y_mean) ** 2 for i in range(n))
            correlation = abs(numerator) / (denominator * y_variance) ** 0.5 if y_variance > 0 else 0.0
        else:
            correlation = 0.0
        
        return {
            'trend': trend,
            'slope': slope,
            'correlation': min(correlation, 1.0),
            'current_value': values[-1],
            'average_value': y_mean,
            'min_value': min(values),
            'max_value': max(values)
        }

class SecurityPerformanceTracker:
    """Track performance impact of security operations"""
    
    def __init__(self):
        """Initialize security performance tracker"""
        self.logger = logging.getLogger(__name__)
        self.security_operations = {}
        self.baseline_metrics = None
        
        # Operation tracking
        self.scan_operations = deque(maxlen=100)
        self.correlation_operations = deque(maxlen=100)
        self.response_operations = deque(maxlen=100)
    
    def start_operation(self, operation_type: str, operation_id: str) -> str:
        """Start tracking a security operation"""
        start_time = datetime.datetime.now()
        
        operation_key = f"{operation_type}_{operation_id}"
        self.security_operations[operation_key] = {
            'type': operation_type,
            'id': operation_id,
            'start_time': start_time,
            'start_cpu': psutil.cpu_percent(),
            'start_memory': psutil.virtual_memory().percent,
            'completed': False
        }
        
        return operation_key
    
    def end_operation(self, operation_key: str) -> Dict[str, Any]:
        """End tracking a security operation and calculate impact"""
        if operation_key not in self.security_operations:
            return {}
        
        operation = self.security_operations[operation_key]
        end_time = datetime.datetime.now()
        
        duration = (end_time - operation['start_time']).total_seconds()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        
        impact = {
            'operation_type': operation['type'],
            'operation_id': operation['id'],
            'duration_seconds': duration,
            'cpu_impact': end_cpu - operation['start_cpu'],
            'memory_impact': end_memory - operation['start_memory'],
            'start_time': operation['start_time'].isoformat(),
            'end_time': end_time.isoformat()
        }
        
        # Store in appropriate queue
        if operation['type'] == 'scan':
            self.scan_operations.append(impact)
        elif operation['type'] == 'correlation':
            self.correlation_operations.append(impact)
        elif operation['type'] == 'response':
            self.response_operations.append(impact)
        
        # Mark as completed
        operation['completed'] = True
        operation['impact'] = impact
        
        return impact
    
    def get_security_load_metrics(self) -> Dict[str, Any]:
        """Calculate security system load metrics"""
        try:
            current_time = datetime.datetime.now()
            recent_cutoff = current_time - datetime.timedelta(minutes=5)
            
            # Calculate recent operation rates
            recent_scans = [op for op in self.scan_operations 
                          if datetime.datetime.fromisoformat(op['end_time']) > recent_cutoff]
            recent_correlations = [op for op in self.correlation_operations 
                                 if datetime.datetime.fromisoformat(op['end_time']) > recent_cutoff]
            recent_responses = [op for op in self.response_operations 
                              if datetime.datetime.fromisoformat(op['end_time']) > recent_cutoff]
            
            # Calculate average impacts
            avg_scan_cpu = sum(op.get('cpu_impact', 0) for op in recent_scans) / max(len(recent_scans), 1)
            avg_correlation_cpu = sum(op.get('cpu_impact', 0) for op in recent_correlations) / max(len(recent_correlations), 1)
            avg_response_cpu = sum(op.get('cpu_impact', 0) for op in recent_responses) / max(len(recent_responses), 1)
            
            # Calculate operation rates (operations per minute)
            scan_rate = len(recent_scans) * 12  # 5-minute window * 12 = per hour
            correlation_rate = len(recent_correlations) * 12
            response_rate = len(recent_responses) * 12
            
            # Calculate total security load
            total_security_load = avg_scan_cpu + avg_correlation_cpu + avg_response_cpu
            
            # Average response times
            avg_scan_time = sum(op.get('duration_seconds', 0) for op in recent_scans) / max(len(recent_scans), 1)
            avg_correlation_time = sum(op.get('duration_seconds', 0) for op in recent_correlations) / max(len(recent_correlations), 1)
            avg_response_time = sum(op.get('duration_seconds', 0) for op in recent_responses) / max(len(recent_responses), 1)
            
            return {
                'total_security_load_percent': min(total_security_load, 100.0),
                'scan_load_percent': min(avg_scan_cpu, 100.0),
                'correlation_load_percent': min(avg_correlation_cpu, 100.0),
                'response_load_percent': min(avg_response_cpu, 100.0),
                'scan_rate_per_hour': scan_rate,
                'correlation_rate_per_hour': correlation_rate,
                'response_rate_per_hour': response_rate,
                'avg_scan_time_seconds': avg_scan_time,
                'avg_correlation_time_seconds': avg_correlation_time,
                'avg_response_time_seconds': avg_response_time,
                'timestamp': current_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating security load metrics: {e}")
            return {}

class PerformanceMonitor:
    """Main performance monitoring coordinator"""
    
    def __init__(self, monitoring_interval: int = 30):
        """Initialize performance monitor"""
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.resource_monitor = SystemResourceMonitor()
        self.security_tracker = SecurityPerformanceTracker()
        self.thresholds = PerformanceThresholds()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.alerts_history = deque(maxlen=100)
        
        # Storage
        self.storage_path = Path(__file__).parent.parent / "performance_data"
        self.storage_path.mkdir(exist_ok=True)
        
        # Callbacks
        self.alert_callbacks = []
    
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info(f"Performance monitoring started with {self.monitoring_interval}s interval")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get current metrics
                system_metrics = self.resource_monitor.get_current_metrics()
                security_metrics = self.security_tracker.get_security_load_metrics()
                
                # Combine metrics
                combined_metrics = asdict(system_metrics)
                combined_metrics.update(security_metrics)
                
                # Store metrics
                self.metrics_history.append(combined_metrics)
                
                # Check thresholds and generate alerts
                await self._check_thresholds(combined_metrics)
                
                # Save metrics to disk periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 readings
                    self._save_metrics_to_disk()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check performance thresholds and generate alerts"""
        alerts = []
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                severity = self.thresholds.check_threshold(metric_name, value)
                if severity:
                    alert = PerformanceAlert(
                        timestamp=datetime.datetime.now().isoformat(),
                        metric_name=metric_name,
                        current_value=value,
                        threshold_value=self.thresholds.thresholds[metric_name][severity],
                        severity=severity,
                        description=f"{metric_name} is {value:.2f}, exceeding {severity} threshold",
                        suggested_actions=self._get_suggested_actions(metric_name, severity)
                    )
                    alerts.append(alert)
        
        # Store alerts
        for alert in alerts:
            self.alerts_history.append(asdict(alert))
            self.logger.warning(f"Performance alert: {alert.description}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def _get_suggested_actions(self, metric_name: str, severity: str) -> List[str]:
        """Get suggested actions for performance alerts"""
        actions = {
            'cpu_usage_percent': [
                "Check for high CPU processes",
                "Consider reducing security scan frequency",
                "Optimize correlation algorithms",
                "Scale horizontally if possible"
            ],
            'memory_usage_percent': [
                "Clear security event caches",
                "Reduce correlation window size",
                "Check for memory leaks",
                "Restart monitoring services"
            ],
            'disk_usage_percent': [
                "Clean up old security logs",
                "Compress quarantined files",
                "Archive old event data",
                "Monitor disk space growth"
            ],
            'response_time_avg_ms': [
                "Optimize database queries",
                "Reduce correlation complexity",
                "Check network connectivity",
                "Scale response handling"
            ],
            'security_scan_load': [
                "Reduce scan frequency",
                "Optimize scan algorithms",
                "Implement scan throttling",
                "Schedule scans during off-peak hours"
            ]
        }
        
        return actions.get(metric_name, ["Review system performance", "Consider optimization"])
    
    def _save_metrics_to_disk(self):
        """Save metrics history to disk"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = self.storage_path / f"metrics_{timestamp}.json"
            
            # Save last 100 metrics
            recent_metrics = list(self.metrics_history)[-100:]
            
            with open(metrics_file, 'w') as f:
                json.dump(recent_metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics to disk: {e}")
    
    def add_alert_callback(self, callback):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance summary"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        trend_analysis = self.resource_monitor.get_trend_analysis(10)
        
        return {
            'current_metrics': latest_metrics,
            'trend_analysis': trend_analysis,
            'recent_alerts': list(self.alerts_history)[-10:],
            'monitoring_status': {
                'active': self.monitoring_active,
                'interval_seconds': self.monitoring_interval,
                'metrics_collected': len(self.metrics_history),
                'alerts_generated': len(self.alerts_history)
            }
        }
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            
            # Filter metrics by time
            recent_metrics = []
            for metric in self.metrics_history:
                metric_time = datetime.datetime.fromisoformat(metric['timestamp'])
                if metric_time > cutoff_time:
                    recent_metrics.append(metric)
            
            if not recent_metrics:
                return {'error': 'No metrics available for specified time period'}
            
            # Calculate statistics
            cpu_values = [m['cpu_usage_percent'] for m in recent_metrics]
            memory_values = [m['memory_usage_percent'] for m in recent_metrics]
            
            report = {
                'report_period_hours': hours,
                'metrics_count': len(recent_metrics),
                'cpu_statistics': {
                    'average': sum(cpu_values) / len(cpu_values),
                    'minimum': min(cpu_values),
                    'maximum': max(cpu_values),
                    'current': cpu_values[-1] if cpu_values else 0
                },
                'memory_statistics': {
                    'average': sum(memory_values) / len(memory_values),
                    'minimum': min(memory_values),
                    'maximum': max(memory_values),
                    'current': memory_values[-1] if memory_values else 0
                },
                'security_impact': self.security_tracker.get_security_load_metrics(),
                'alerts_summary': {
                    'total_alerts': len(self.alerts_history),
                    'critical_alerts': len([a for a in self.alerts_history if a['severity'] == 'critical']),
                    'warning_alerts': len([a for a in self.alerts_history if a['severity'] == 'warning'])
                },
                'generated_at': datetime.datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}