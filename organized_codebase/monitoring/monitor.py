"""
Real-time Monitoring Core Module
================================

Core monitoring engine for real-time performance data collection.
This module is CRITICAL for the 100ms performance chart updates.

Author: TestMaster Team
"""

# Import SystemMonitor for backward compatibility
from .system_monitor import SystemMonitor

import time
import threading
import psutil
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class RealTimeMonitor:
    """
    Real-time system monitoring engine.
    
    Collects performance metrics every 100ms and maintains
    circular buffers for efficient data access.
    """
    
    def __init__(self, max_history_points: int = 300, collection_interval: float = 0.1):
        """
        Initialize the real-time monitor.
        
        Args:
            max_history_points: Maximum points to keep in history (default: 300 for 30s at 100ms)
            collection_interval: Collection interval in seconds (default: 0.1 for 100ms)
        """
        self.max_history_points = max_history_points
        self.collection_interval = collection_interval
        self.running = False
        
        # Per-codebase performance history
        self.performance_history: Dict[str, Dict[str, deque]] = {}
        
        # Thread management
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Last collection time for timing
        self.last_collection_time = 0
        
        # Process tracking for per-codebase metrics
        self.codebase_processes: Dict[str, List[int]] = {}
        
        # Thread lock for data safety
        self.data_lock = threading.Lock()
        
        logger.info(f"RealTimeMonitor initialized: {max_history_points} points, {collection_interval}s interval")
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop - runs every 100ms."""
        logger.debug("Starting monitoring loop")
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            try:
                # Collect current metrics
                self._collect_system_metrics()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Calculate sleep time to maintain interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.collection_interval - elapsed)
            
            if sleep_time > 0:
                self.stop_event.wait(sleep_time)
        
        logger.debug("Monitoring loop stopped")
    
    def _collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        current_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            network = psutil.net_io_counters()
            
            # Calculate network rate (KB/s)
            network_rate = 0
            if hasattr(self, '_last_network_bytes') and hasattr(self, '_last_network_time'):
                time_diff = current_time - self._last_network_time
                if time_diff > 0:
                    bytes_diff = (network.bytes_sent + network.bytes_recv) - self._last_network_bytes
                    network_rate = (bytes_diff / time_diff) / 1024  # Convert to KB/s
            
            # Store for next calculation
            self._last_network_bytes = network.bytes_sent + network.bytes_recv
            self._last_network_time = current_time
            
            # Update metrics for all known codebases
            with self.data_lock:
                # Default codebase
                self._update_codebase_metrics('/testmaster', {
                    'cpu_usage': cpu_percent,
                    'memory_usage_mb': memory.used / (1024 * 1024),
                    'network_kb_s': network_rate,
                    'timestamp': current_time
                })
                
                # Update any other registered codebases with same data
                for codebase in list(self.performance_history.keys()):
                    if codebase != '/testmaster':
                        self._update_codebase_metrics(codebase, {
                            'cpu_usage': cpu_percent,
                            'memory_usage_mb': memory.used / (1024 * 1024),
                            'network_kb_s': network_rate,
                            'timestamp': current_time
                        })
            
            self.last_collection_time = current_time
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _update_codebase_metrics(self, codebase: str, metrics: Dict[str, float]) -> None:
        """Update metrics for a specific codebase."""
        if codebase not in self.performance_history:
            self.performance_history[codebase] = {
                'cpu_usage': deque(maxlen=self.max_history_points),
                'memory_usage_mb': deque(maxlen=self.max_history_points),
                'network_kb_s': deque(maxlen=self.max_history_points),
                'timestamps': deque(maxlen=self.max_history_points)
            }
        
        # Add new data points
        history = self.performance_history[codebase]
        history['cpu_usage'].append(metrics['cpu_usage'])
        history['memory_usage_mb'].append(metrics['memory_usage_mb'])
        history['network_kb_s'].append(metrics['network_kb_s'])
        history['timestamps'].append(metrics['timestamp'])
    
    def get_current_metrics(self, codebase: str = '/testmaster') -> Optional[Dict[str, List[float]]]:
        """
        Get current metrics for a codebase.
        
        Args:
            codebase: Codebase identifier
            
        Returns:
            Dictionary with current metric arrays for charts
        """
        with self.data_lock:
            if codebase not in self.performance_history:
                logger.warning(f"No metrics available for codebase: {codebase}")
                # Initialize if needed
                self._initialize_codebase(codebase)
                return None
            
            history = self.performance_history[codebase]
            
            # Return latest data as arrays (for chart compatibility)
            return {
                'cpu_usage': list(history['cpu_usage']),
                'memory_usage_mb': list(history['memory_usage_mb']),
                'network_kb_s': list(history['network_kb_s']),
                'timestamps': [datetime.fromtimestamp(t).isoformat() for t in history['timestamps']]
            }
    
    def _initialize_codebase(self, codebase: str) -> None:
        """Initialize empty history for a codebase."""
        logger.info(f"Initializing new codebase monitoring: {codebase}")
        self.performance_history[codebase] = {
            'cpu_usage': deque(maxlen=self.max_history_points),
            'memory_usage_mb': deque(maxlen=self.max_history_points),
            'network_kb_s': deque(maxlen=self.max_history_points),
            'timestamps': deque(maxlen=self.max_history_points)
        }
    
    def get_metrics_history(self, codebase: str = '/testmaster', hours: int = 1) -> Dict[str, Any]:
        """
        Get historical metrics for a codebase.
        
        Args:
            codebase: Codebase identifier
            hours: Hours of history to return
            
        Returns:
            Historical metrics data
        """
        with self.data_lock:
            if codebase not in self.performance_history:
                return {
                    'cpu_usage': [],
                    'memory_usage_mb': [],
                    'network_kb_s': [],
                    'timestamps': []
                }
            
            history = self.performance_history[codebase]
            
            # Filter by time if needed
            cutoff_time = time.time() - (hours * 3600)
            
            filtered_data = {
                'cpu_usage': [],
                'memory_usage_mb': [],
                'network_kb_s': [],
                'timestamps': []
            }
            
            for i, timestamp in enumerate(history['timestamps']):
                if timestamp >= cutoff_time:
                    filtered_data['cpu_usage'].append(history['cpu_usage'][i])
                    filtered_data['memory_usage_mb'].append(history['memory_usage_mb'][i])
                    filtered_data['network_kb_s'].append(history['network_kb_s'][i])
                    filtered_data['timestamps'].append(datetime.fromtimestamp(timestamp).isoformat())
            
            return filtered_data
    
    def get_performance_summary(self, codebase: str = '/testmaster') -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            codebase: Codebase identifier
            
        Returns:
            Summary statistics
        """
        with self.data_lock:
            if codebase not in self.performance_history:
                return {
                    'avg_cpu': 0,
                    'max_cpu': 0,
                    'avg_memory': 0,
                    'max_memory': 0,
                    'avg_network': 0,
                    'max_network': 0
                }
            
            history = self.performance_history[codebase]
            
            if not history['cpu_usage']:
                return {
                    'avg_cpu': 0,
                    'max_cpu': 0,
                    'avg_memory': 0,
                    'max_memory': 0,
                    'avg_network': 0,
                    'max_network': 0
                }
            
            return {
                'avg_cpu': sum(history['cpu_usage']) / len(history['cpu_usage']),
                'max_cpu': max(history['cpu_usage']),
                'avg_memory': sum(history['memory_usage_mb']) / len(history['memory_usage_mb']),
                'max_memory': max(history['memory_usage_mb']),
                'avg_network': sum(history['network_kb_s']) / len(history['network_kb_s']),
                'max_network': max(history['network_kb_s']),
                'data_points': len(history['cpu_usage']),
                'collection_time_span': len(history['cpu_usage']) * self.collection_interval
            }
    
    def register_codebase(self, codebase: str) -> None:
        """Register a new codebase for monitoring."""
        with self.data_lock:
            if codebase not in self.performance_history:
                self._initialize_codebase(codebase)
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status information."""
        return {
            'running': self.running,
            'collection_interval': self.collection_interval,
            'max_history_points': self.max_history_points,
            'monitored_codebases': list(self.performance_history.keys()),
            'last_collection': self.last_collection_time,
            'thread_alive': self.monitor_thread.is_alive() if self.monitor_thread else False
        }