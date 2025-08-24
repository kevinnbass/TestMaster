#!/usr/bin/env python3
"""
Enhanced Real-Time Analytics System
====================================
Adds real metrics collection and enhanced analytics capabilities.
"""

import psutil
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeAnalyticsCollector:
    """Collects real system metrics and test execution data."""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 data points
        self.test_results = deque(maxlen=100)      # Keep last 100 test results
        self.system_events = deque(maxlen=500)     # Keep last 500 events
        self.collection_interval = 0.1             # 100ms intervals
        self.running = False
        self.thread = None
        
        # Performance baselines
        self.baselines = {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'disk_threshold': 90.0,
            'response_time_threshold': 1000  # ms
        }
        
        # Analytics cache
        self.analytics_cache = {
            'last_update': None,
            'summary': {},
            'trends': {},
            'alerts': []
        }
    
    def start_collection(self):
        """Start real-time metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        logger.info("Real-time analytics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Real-time analytics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            start_time = time.time()
            
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                self._analyze_metrics(metrics)
                self._check_alerts(metrics)
            except Exception as e:
                logger.error(f"Collection error: {e}")
            
            # Maintain collection interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.collection_interval - elapsed)
            time.sleep(sleep_time)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'frequency': cpu_freq.current if cpu_freq else 0,
                    'cores': cpu_count,
                    'per_core': psutil.cpu_percent(percpu=True)
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'swap_percent': swap.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv,
                    'errors': network.errin + network.errout
                },
                'process': {
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent(),
                    'num_threads': process.num_threads(),
                    'open_files': len(process.open_files())
                }
            }
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def _analyze_metrics(self, metrics: Dict[str, Any]):
        """Analyze metrics for trends and patterns."""
        if len(self.metrics_history) < 10:
            return
        
        # Calculate moving averages
        recent_metrics = list(self.metrics_history)[-10:]
        
        cpu_avg = sum(m.get('cpu', {}).get('percent', 0) for m in recent_metrics) / 10
        mem_avg = sum(m.get('memory', {}).get('percent', 0) for m in recent_metrics) / 10
        
        # Detect trends
        if len(self.metrics_history) >= 20:
            older_metrics = list(self.metrics_history)[-20:-10]
            old_cpu_avg = sum(m.get('cpu', {}).get('percent', 0) for m in older_metrics) / 10
            old_mem_avg = sum(m.get('memory', {}).get('percent', 0) for m in older_metrics) / 10
            
            cpu_trend = 'increasing' if cpu_avg > old_cpu_avg + 5 else 'decreasing' if cpu_avg < old_cpu_avg - 5 else 'stable'
            mem_trend = 'increasing' if mem_avg > old_mem_avg + 5 else 'decreasing' if mem_avg < old_mem_avg - 5 else 'stable'
        else:
            cpu_trend = 'stable'
            mem_trend = 'stable'
        
        self.analytics_cache['trends'] = {
            'cpu': {'average': cpu_avg, 'trend': cpu_trend},
            'memory': {'average': mem_avg, 'trend': mem_trend}
        }
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions."""
        alerts = []
        
        # CPU alert
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        if cpu_percent > self.baselines['cpu_threshold']:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning' if cpu_percent < 90 else 'critical',
                'message': f"CPU usage high: {cpu_percent:.1f}%",
                'timestamp': metrics['timestamp']
            })
        
        # Memory alert
        mem_percent = metrics.get('memory', {}).get('percent', 0)
        if mem_percent > self.baselines['memory_threshold']:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning' if mem_percent < 95 else 'critical',
                'message': f"Memory usage high: {mem_percent:.1f}%",
                'timestamp': metrics['timestamp']
            })
        
        # Disk alert
        disk_percent = metrics.get('disk', {}).get('percent', 0)
        if disk_percent > self.baselines['disk_threshold']:
            alerts.append({
                'type': 'disk_high',
                'severity': 'warning',
                'message': f"Disk usage high: {disk_percent:.1f}%",
                'timestamp': metrics['timestamp']
            })
        
        # Add new alerts to cache
        for alert in alerts:
            self.analytics_cache['alerts'].append(alert)
            logger.warning(f"Alert: {alert['message']}")
        
        # Keep only recent alerts (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.analytics_cache['alerts'] = [
            a for a in self.analytics_cache['alerts']
            if datetime.fromisoformat(a['timestamp']) > cutoff_time
        ]
    
    def get_real_time_data(self) -> Dict[str, Any]:
        """Get current real-time data for dashboard."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest = self.metrics_history[-1]
        
        # Get last 30 seconds of data for charts
        chart_data = []
        for metric in list(self.metrics_history)[-300:]:  # 30 seconds at 100ms intervals
            chart_data.append({
                'timestamp': metric.get('timestamp'),
                'cpu': metric.get('cpu', {}).get('percent', 0),
                'memory': metric.get('memory', {}).get('percent', 0),
                'disk': metric.get('disk', {}).get('percent', 0)
            })
        
        return {
            'status': 'active',
            'current': latest,
            'chart_data': chart_data,
            'trends': self.analytics_cache.get('trends', {}),
            'alerts': self.analytics_cache.get('alerts', [])[-10:],  # Last 10 alerts
            'statistics': self._calculate_statistics()
        }
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if len(self.metrics_history) < 10:
            return {}
        
        recent = list(self.metrics_history)[-100:]  # Last 10 seconds
        
        cpu_values = [m.get('cpu', {}).get('percent', 0) for m in recent]
        mem_values = [m.get('memory', {}).get('percent', 0) for m in recent]
        
        return {
            'cpu': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory': {
                'min': min(mem_values),
                'max': max(mem_values),
                'avg': sum(mem_values) / len(mem_values),
                'current': mem_values[-1] if mem_values else 0
            },
            'uptime_seconds': len(self.metrics_history) * self.collection_interval,
            'data_points': len(self.metrics_history)
        }
    
    def record_test_result(self, test_name: str, status: str, duration: float, details: Optional[Dict] = None):
        """Record a test execution result."""
        result = {
            'test_name': test_name,
            'status': status,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.test_results.append(result)
        
        # Log event
        self.system_events.append({
            'type': 'test_execution',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Test result: {test_name} - {status} ({duration:.2f}s)")
    
    def get_test_analytics(self) -> Dict[str, Any]:
        """Get test execution analytics."""
        if not self.test_results:
            return {'status': 'no_tests'}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t['status'] == 'passed')
        failed_tests = sum(1 for t in self.test_results if t['status'] == 'failed')
        
        durations = [t['duration'] for t in self.test_results]
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'average_duration': sum(durations) / len(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'recent_tests': list(self.test_results)[-10:],
            'test_trend': self._analyze_test_trend()
        }
    
    def _analyze_test_trend(self) -> str:
        """Analyze test execution trend."""
        if len(self.test_results) < 20:
            return 'insufficient_data'
        
        recent = list(self.test_results)[-10:]
        older = list(self.test_results)[-20:-10]
        
        recent_pass_rate = sum(1 for t in recent if t['status'] == 'passed') / 10 * 100
        older_pass_rate = sum(1 for t in older if t['status'] == 'passed') / 10 * 100
        
        if recent_pass_rate > older_pass_rate + 10:
            return 'improving'
        elif recent_pass_rate < older_pass_rate - 10:
            return 'degrading'
        else:
            return 'stable'
    
    def export_analytics(self, filepath: str):
        """Export analytics data to file."""
        data = {
            'export_time': datetime.now().isoformat(),
            'real_time_data': self.get_real_time_data(),
            'test_analytics': self.get_test_analytics(),
            'system_events': list(self.system_events)[-100:],
            'configuration': {
                'collection_interval': self.collection_interval,
                'baselines': self.baselines
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Analytics exported to {filepath}")


def integrate_with_dashboard():
    """Integrate analytics with existing dashboard."""
    print("Integrating enhanced analytics with dashboard...")
    
    # Create global analytics instance
    analytics_file = Path('dashboard/dashboard_core/real_time_analytics.py')
    
    analytics_file.write_text('''"""
Real-Time Analytics Integration
================================
Enhanced analytics for dashboard.
"""

from enhance_analytics import RealTimeAnalyticsCollector

# Global analytics instance
analytics_collector = RealTimeAnalyticsCollector()

def get_analytics_collector():
    """Get the global analytics collector instance."""
    return analytics_collector

def start_analytics():
    """Start analytics collection."""
    analytics_collector.start_collection()

def stop_analytics():
    """Stop analytics collection."""
    analytics_collector.stop_collection()

def get_real_time_metrics():
    """Get real-time metrics for dashboard."""
    return analytics_collector.get_real_time_data()

def get_test_metrics():
    """Get test execution metrics."""
    return analytics_collector.get_test_analytics()

# Auto-start on import
start_analytics()
''', encoding='utf-8')
    
    print(f"  Created: {analytics_file}")
    
    # Update server to use real analytics
    print("  Analytics integration complete!")
    print("  - Real-time system metrics collection")
    print("  - Test execution tracking")
    print("  - Performance trend analysis")
    print("  - Alert monitoring")
    print("  - Statistics calculation")


def main():
    """Demo the analytics system."""
    print("Enhanced Analytics System Demo")
    print("="*60)
    
    # Create collector
    collector = RealTimeAnalyticsCollector()
    
    # Start collection
    collector.start_collection()
    print("Started real-time collection...")
    
    # Let it collect for a few seconds
    time.sleep(3)
    
    # Get real-time data
    data = collector.get_real_time_data()
    
    print("\nReal-Time Analytics:")
    print(f"  Status: {data['status']}")
    if 'statistics' in data:
        stats = data['statistics']
        print(f"  CPU: {stats.get('cpu', {}).get('current', 0):.1f}% (avg: {stats.get('cpu', {}).get('avg', 0):.1f}%)")
        print(f"  Memory: {stats.get('memory', {}).get('current', 0):.1f}% (avg: {stats.get('memory', {}).get('avg', 0):.1f}%)")
        print(f"  Data points: {stats.get('data_points', 0)}")
    
    # Simulate some test results
    collector.record_test_result("test_login", "passed", 1.2)
    collector.record_test_result("test_api", "passed", 0.8)
    collector.record_test_result("test_database", "failed", 2.5, {"error": "connection timeout"})
    
    # Get test analytics
    test_data = collector.get_test_analytics()
    print(f"\nTest Analytics:")
    print(f"  Total tests: {test_data.get('total_tests', 0)}")
    print(f"  Pass rate: {test_data.get('pass_rate', 0):.1f}%")
    print(f"  Average duration: {test_data.get('average_duration', 0):.2f}s")
    
    # Check for alerts
    if data.get('alerts'):
        print(f"\nAlerts:")
        for alert in data['alerts'][-3:]:
            print(f"  [{alert['severity']}] {alert['message']}")
    
    # Export analytics
    collector.export_analytics('analytics_export.json')
    
    # Stop collection
    collector.stop_collection()
    print("\nCollection stopped.")
    
    # Integrate with dashboard
    integrate_with_dashboard()
    
    print("\n[SUCCESS] Enhanced analytics system ready!")


if __name__ == "__main__":
    main()