"""
System Monitor with Security Integration

Provides system monitoring capabilities with security metrics integration.
"""

import time
import threading
import logging
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    System monitor that integrates security metrics with real-time monitoring.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize system monitor with security integration.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread = None
        
        # Metrics storage
        self.metrics = {
            'security': deque(maxlen=100),
            'performance': deque(maxlen=100),
            'system': deque(maxlen=100)
        }
        
        # Security metrics
        self.security_metrics = {
            'vulnerabilities_detected': 0,
            'threats_blocked': 0,
            'compliance_score': 100.0,
            'last_scan_time': None
        }
        
        logger.info("System Monitor initialized with security integration")
    
    def start(self):
        """Start monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                timestamp = datetime.now()
                
                # Collect security metrics
                security_data = self._collect_security_metrics()
                self.metrics['security'].append({
                    'timestamp': timestamp,
                    'data': security_data
                })
                
                # Collect performance metrics
                performance_data = self._collect_performance_metrics()
                self.metrics['performance'].append({
                    'timestamp': timestamp,
                    'data': performance_data
                })
                
                # Collect system metrics
                system_data = self._collect_system_metrics()
                self.metrics['system'].append({
                    'timestamp': timestamp,
                    'data': system_data
                })
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def _collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics"""
        try:
            # Import security modules if available
            from ...core.intelligence.security.security_dashboard import SecurityDashboard
            dashboard = SecurityDashboard()
            
            status = dashboard.get_current_status()
            
            return {
                'threat_level': status.get('threat_level', 'low'),
                'active_scans': status.get('active_scans', 0),
                'vulnerabilities': self.security_metrics['vulnerabilities_detected'],
                'compliance_score': self.security_metrics['compliance_score']
            }
        except ImportError:
            # Fallback if security modules not available
            return self.security_metrics
        except Exception as e:
            logger.error(f"Failed to collect security metrics: {e}")
            return self.security_metrics
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_usage': 0
            }
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        try:
            import psutil
            
            return {
                'processes': len(psutil.pids()),
                'threads': threading.active_count(),
                'uptime': time.time()
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {
                'processes': 0,
                'threads': threading.active_count(),
                'uptime': time.time()
            }
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics"""
        latest = {}
        
        for metric_type, data in self.metrics.items():
            if data:
                latest[metric_type] = data[-1]
            else:
                latest[metric_type] = None
        
        return latest
    
    def get_metrics_history(self, metric_type: str = 'security', 
                           limit: int = 50) -> List[Dict[str, Any]]:
        """Get metrics history"""
        if metric_type in self.metrics:
            return list(self.metrics[metric_type])[-limit:]
        return []
    
    def update_security_metrics(self, updates: Dict[str, Any]):
        """
        Update security metrics from external sources.
        
        Args:
            updates: Dictionary of metric updates
        """
        for key, value in updates.items():
            if key in self.security_metrics:
                self.security_metrics[key] = value
        
        logger.info(f"Security metrics updated: {updates.keys()}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""
        return {
            'current_metrics': self.get_latest_metrics(),
            'security_summary': self.security_metrics,
            'history': {
                'security': self.get_metrics_history('security', 20),
                'performance': self.get_metrics_history('performance', 20)
            },
            'status': 'monitoring' if self.running else 'stopped'
        }