"""
Resource Intelligence Monitor
============================

Real-time resource monitoring with anomaly detection and health scoring.
Extracted from intelligent_resource_allocator.py for enterprise modular architecture.

Agent D Implementation - Hour 10-11: Revolutionary Intelligence Modularization
"""

import logging
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque, defaultdict

from .data_models import LoadBalancingMetrics, ResourceAllocationMetrics


class ResourceMonitor:
    """Advanced resource monitoring system with anomaly detection"""
    
    def __init__(self, monitoring_interval: float = 10.0,
                 anomaly_detection_window: int = 50,
                 health_check_interval: float = 30.0):
        self.monitoring_interval = monitoring_interval
        self.anomaly_detection_window = anomaly_detection_window
        self.health_check_interval = health_check_interval
        
        # Monitoring data
        self.resource_metrics = {}  # resource_type -> deque of metrics
        self.framework_health = {}  # framework_id -> LoadBalancingMetrics
        self.anomaly_history = {}  # resource_type -> deque of anomaly events
        self.performance_baselines = {}  # resource_type -> baseline metrics
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.health_check_thread = None
        self.callbacks = {'anomaly': [], 'health_change': [], 'performance_degradation': []}
        
        # Anomaly detection parameters
        self.anomaly_threshold_multiplier = 2.0  # Z-score threshold
        self.performance_degradation_threshold = 0.3  # 30% performance drop
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def start_monitoring(self):
        """Start resource monitoring threads"""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start health check thread
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
        
        self.logger.info("Resource monitoring stopped")
    
    def update_resource_metrics(self, resource_type: str, metrics: Dict[str, float]):
        """Update metrics for a resource type"""
        if resource_type not in self.resource_metrics:
            self.resource_metrics[resource_type] = deque(maxlen=self.anomaly_detection_window * 2)
        
        # Add timestamp to metrics
        timestamped_metrics = {**metrics, 'timestamp': time.time()}
        self.resource_metrics[resource_type].append(timestamped_metrics)
        
        # Check for anomalies
        self._check_for_anomalies(resource_type, metrics)
        
        # Update performance baselines
        self._update_performance_baselines(resource_type, metrics)
        
        self.logger.debug(f"Updated metrics for {resource_type}: {metrics}")
    
    def update_framework_health(self, framework_id: str, health_metrics: LoadBalancingMetrics):
        """Update health metrics for a framework"""
        previous_health = self.framework_health.get(framework_id)
        self.framework_health[framework_id] = health_metrics
        
        # Check for significant health changes
        if previous_health:
            health_change = abs(health_metrics.health_score - previous_health.health_score)
            if health_change > 0.2:  # 20% health change threshold
                self._trigger_health_change_callback(framework_id, previous_health, health_metrics)
        
        self.logger.debug(f"Updated health for {framework_id}: score={health_metrics.health_score:.2f}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Check for performance degradation
                self._check_performance_degradation()
                
                # Clean old data
                self._cleanup_old_data()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _health_check_loop(self):
        """Health check loop for frameworks"""
        while self.is_monitoring:
            try:
                # Perform health checks
                self._perform_health_checks()
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_for_anomalies(self, resource_type: str, current_metrics: Dict[str, float]):
        """Check for anomalies using z-score analysis"""
        try:
            if len(self.resource_metrics[resource_type]) < 10:
                return  # Need more data for anomaly detection
            
            # Get recent metrics for baseline
            recent_metrics = list(self.resource_metrics[resource_type])[-self.anomaly_detection_window:]
            
            for metric_name, current_value in current_metrics.items():
                if metric_name == 'timestamp':
                    continue
                
                # Extract historical values for this metric
                historical_values = [m.get(metric_name, 0) for m in recent_metrics if metric_name in m]
                
                if len(historical_values) < 5:
                    continue
                
                # Calculate z-score
                mean_val = np.mean(historical_values)
                std_val = np.std(historical_values)
                
                if std_val > 0:
                    z_score = abs(current_value - mean_val) / std_val
                    
                    if z_score > self.anomaly_threshold_multiplier:
                        # Anomaly detected
                        anomaly_event = {
                            'timestamp': datetime.now(),
                            'resource_type': resource_type,
                            'metric_name': metric_name,
                            'current_value': current_value,
                            'expected_value': mean_val,
                            'z_score': z_score,
                            'severity': 'high' if z_score > 3.0 else 'medium'
                        }
                        
                        self._record_anomaly(resource_type, anomaly_event)
                        self._trigger_anomaly_callback(anomaly_event)
                        
        except Exception as e:
            self.logger.error(f"Error checking anomalies for {resource_type}: {e}")
    
    def _record_anomaly(self, resource_type: str, anomaly_event: Dict):
        """Record anomaly event"""
        if resource_type not in self.anomaly_history:
            self.anomaly_history[resource_type] = deque(maxlen=100)
        
        self.anomaly_history[resource_type].append(anomaly_event)
        
        self.logger.warning(f"Anomaly detected in {resource_type}: "
                          f"{anomaly_event['metric_name']} = {anomaly_event['current_value']:.2f} "
                          f"(z-score: {anomaly_event['z_score']:.2f})")
    
    def _update_performance_baselines(self, resource_type: str, metrics: Dict[str, float]):
        """Update performance baselines for resource type"""
        if resource_type not in self.performance_baselines:
            self.performance_baselines[resource_type] = {}
        
        # Update baselines with exponential moving average
        alpha = 0.1  # Smoothing factor
        
        for metric_name, current_value in metrics.items():
            if metric_name == 'timestamp':
                continue
            
            if metric_name in self.performance_baselines[resource_type]:
                # Update existing baseline
                baseline = self.performance_baselines[resource_type][metric_name]
                self.performance_baselines[resource_type][metric_name] = (
                    alpha * current_value + (1 - alpha) * baseline
                )
            else:
                # Initialize baseline
                self.performance_baselines[resource_type][metric_name] = current_value
    
    def _check_performance_degradation(self):
        """Check for performance degradation across all resources"""
        try:
            for resource_type, metrics_deque in self.resource_metrics.items():
                if len(metrics_deque) < 10:
                    continue
                
                recent_metrics = list(metrics_deque)[-5:]  # Last 5 data points
                older_metrics = list(metrics_deque)[-15:-5]  # 10 data points before that
                
                if len(older_metrics) < 5:
                    continue
                
                # Check key performance metrics
                performance_metrics = ['response_time', 'error_rate', 'utilization']
                
                for metric_name in performance_metrics:
                    recent_values = [m.get(metric_name, 0) for m in recent_metrics if metric_name in m]
                    older_values = [m.get(metric_name, 0) for m in older_metrics if metric_name in m]
                    
                    if len(recent_values) < 3 or len(older_values) < 3:
                        continue
                    
                    recent_avg = np.mean(recent_values)
                    older_avg = np.mean(older_values)
                    
                    # Check for degradation (depends on metric type)
                    if metric_name in ['response_time', 'error_rate', 'utilization']:
                        # Higher is worse
                        if older_avg > 0:
                            degradation_ratio = (recent_avg - older_avg) / older_avg
                            if degradation_ratio > self.performance_degradation_threshold:
                                self._trigger_performance_degradation_callback(
                                    resource_type, metric_name, degradation_ratio, recent_avg, older_avg
                                )
                    
        except Exception as e:
            self.logger.error(f"Error checking performance degradation: {e}")
    
    def _perform_health_checks(self):
        """Perform health checks on all frameworks"""
        try:
            current_time = time.time()
            
            for framework_id, health_metrics in self.framework_health.items():
                # Check if health metrics are stale
                if hasattr(health_metrics, 'last_updated'):
                    time_since_update = current_time - health_metrics.last_updated
                    if time_since_update > 300:  # 5 minutes
                        self.logger.warning(f"Stale health metrics for framework {framework_id}")
                
                # Check critical health thresholds
                if health_metrics.health_score < 0.3:
                    self.logger.warning(f"Framework {framework_id} has low health score: "
                                      f"{health_metrics.health_score:.2f}")
                
                if health_metrics.error_rate > 0.1:
                    self.logger.warning(f"Framework {framework_id} has high error rate: "
                                      f"{health_metrics.error_rate:.2%}")
                
                if health_metrics.utilization > 0.95:
                    self.logger.warning(f"Framework {framework_id} is over-utilized: "
                                      f"{health_metrics.utilization:.2%}")
                    
        except Exception as e:
            self.logger.error(f"Error performing health checks: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            current_time = time.time()
            retention_period = 3600 * 24  # 24 hours
            
            # Clean resource metrics
            for resource_type, metrics_deque in self.resource_metrics.items():
                while (metrics_deque and 
                       current_time - metrics_deque[0].get('timestamp', current_time) > retention_period):
                    metrics_deque.popleft()
            
            # Clean anomaly history
            for resource_type, anomaly_deque in self.anomaly_history.items():
                while (anomaly_deque and 
                       (current_time - anomaly_deque[0]['timestamp'].timestamp()) > retention_period):
                    anomaly_deque.popleft()
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for monitoring events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            self.logger.info(f"Registered callback for {event_type} events")
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _trigger_anomaly_callback(self, anomaly_event: Dict):
        """Trigger anomaly callbacks"""
        for callback in self.callbacks['anomaly']:
            try:
                callback(anomaly_event)
            except Exception as e:
                self.logger.error(f"Error in anomaly callback: {e}")
    
    def _trigger_health_change_callback(self, framework_id: str, 
                                      previous_health: LoadBalancingMetrics,
                                      current_health: LoadBalancingMetrics):
        """Trigger health change callbacks"""
        health_change_event = {
            'framework_id': framework_id,
            'previous_health_score': previous_health.health_score,
            'current_health_score': current_health.health_score,
            'change': current_health.health_score - previous_health.health_score,
            'timestamp': datetime.now()
        }
        
        for callback in self.callbacks['health_change']:
            try:
                callback(health_change_event)
            except Exception as e:
                self.logger.error(f"Error in health change callback: {e}")
    
    def _trigger_performance_degradation_callback(self, resource_type: str, metric_name: str,
                                                degradation_ratio: float, recent_avg: float,
                                                older_avg: float):
        """Trigger performance degradation callbacks"""
        degradation_event = {
            'resource_type': resource_type,
            'metric_name': metric_name,
            'degradation_ratio': degradation_ratio,
            'recent_average': recent_avg,
            'baseline_average': older_avg,
            'severity': 'high' if degradation_ratio > 0.5 else 'medium',
            'timestamp': datetime.now()
        }
        
        for callback in self.callbacks['performance_degradation']:
            try:
                callback(degradation_event)
            except Exception as e:
                self.logger.error(f"Error in performance degradation callback: {e}")
    
    def get_monitoring_summary(self) -> Dict:
        """Get comprehensive monitoring summary"""
        summary = {
            'monitoring_status': 'active' if self.is_monitoring else 'inactive',
            'monitored_resources': len(self.resource_metrics),
            'monitored_frameworks': len(self.framework_health),
            'total_anomalies': sum(len(anomalies) for anomalies in self.anomaly_history.values()),
            'resource_health': {},
            'framework_health': {},
            'recent_anomalies': []
        }
        
        # Resource health summary
        for resource_type, metrics_deque in self.resource_metrics.items():
            if metrics_deque:
                latest_metrics = metrics_deque[-1]
                summary['resource_health'][resource_type] = {
                    'last_updated': datetime.fromtimestamp(latest_metrics.get('timestamp', 0)),
                    'data_points': len(metrics_deque),
                    'latest_metrics': {k: v for k, v in latest_metrics.items() if k != 'timestamp'}
                }
        
        # Framework health summary
        for framework_id, health_metrics in self.framework_health.items():
            summary['framework_health'][framework_id] = {
                'health_score': health_metrics.health_score,
                'utilization': health_metrics.utilization,
                'response_time': health_metrics.response_time,
                'error_rate': health_metrics.error_rate,
                'status': 'healthy' if health_metrics.health_score > 0.7 else 'degraded'
            }
        
        # Recent anomalies (last 10)
        all_anomalies = []
        for resource_type, anomaly_deque in self.anomaly_history.items():
            all_anomalies.extend(list(anomaly_deque))
        
        # Sort by timestamp and get most recent
        all_anomalies.sort(key=lambda x: x['timestamp'], reverse=True)
        summary['recent_anomalies'] = all_anomalies[:10]
        
        return summary


def create_resource_monitor(monitoring_interval: float = 10.0,
                          anomaly_detection_window: int = 50,
                          health_check_interval: float = 30.0) -> ResourceMonitor:
    """Factory function to create resource monitor"""
    return ResourceMonitor(
        monitoring_interval=monitoring_interval,
        anomaly_detection_window=anomaly_detection_window,
        health_check_interval=health_check_interval
    )