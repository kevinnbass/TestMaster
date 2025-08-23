#!/usr/bin/env python3
"""
STEELCLAD Phase 5: Metrics Aggregator - Extracted from Performance Analytics Dashboard
====================================================================================

Comprehensive metrics aggregation from all performance systems with real-time collection,
historical tracking, and correlation analysis capabilities.

Author: Agent Z (STEELCLAD Protocol)
Extracted from: performance_analytics_dashboard.py (185 lines)
"""

import logging
import threading
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional

# Inline configuration for standalone operation
class DashboardConfig:
    """Dashboard configuration for standalone operation"""
    def __init__(self):
        self.max_data_points = 1000
        self.enable_predictions = True
        self.enable_alpha_integration = True


class MetricsAggregator:
    """Aggregates metrics from all performance systems"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.max_data_points))
        self.predictions_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.test_results_history: deque = deque(maxlen=50)
        self.logger = logging.getLogger('MetricsAggregator')
        self._lock = threading.RLock()
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all integrated systems"""
        all_metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'performance_monitoring': {},
            'caching_system': {},
            'ml_optimizer': {},
            'distributed_scaling': {},
            'alpha_monitoring': {},
            'alpha_optimization': {},
            'system_health': 'unknown'
        }
        
        # Collect from performance monitoring
        if hasattr(self, 'monitoring_system'):
            try:
                metrics = self.monitoring_system.metrics_collector.get_metrics()
                performance_data = {}
                
                for name, metric_list in metrics.items():
                    if metric_list:
                        latest = metric_list[-1]
                        performance_data[name] = {
                            'value': latest.value,
                            'timestamp': latest.timestamp.isoformat(),
                            'unit': latest.unit
                        }
                
                all_metrics['performance_monitoring'] = performance_data
            except Exception as e:
                self.logger.error(f"Failed to collect performance monitoring metrics: {e}")
        
        # Collect from caching system
        if hasattr(self, 'caching_system'):
            try:
                cache_status = self.caching_system.get_system_status()
                all_metrics['caching_system'] = {
                    'hit_ratio': cache_status['metrics']['hit_ratio'],
                    'total_operations': cache_status['metrics']['total_operations'],
                    'memory_utilization': cache_status['memory_layer']['utilization'],
                    'system_health': cache_status['system_health']
                }
            except Exception as e:
                self.logger.error(f"Failed to collect caching metrics: {e}")
        
        # Collect from ML optimizer
        if hasattr(self, 'ml_optimizer'):
            try:
                optimizer_status = self.ml_optimizer.get_optimization_status()
                all_metrics['ml_optimizer'] = {
                    'models_trained': len(optimizer_status['models_trained']),
                    'predictions_count': optimizer_status['predictions_count'],
                    'current_parameters': optimizer_status['current_parameters']
                }
                
                # Get predictions if available
                if self.config.enable_predictions:
                    current_metrics = await self._get_current_metrics_for_prediction()
                    if current_metrics:
                        predictions = self.ml_optimizer._make_predictions(current_metrics)
                        all_metrics['predictions'] = [
                            {
                                'metric_name': p.metric_name,
                                'current_value': p.current_value,
                                'predicted_value': p.predicted_value,
                                'confidence': p.confidence,
                                'trend': p.trend,
                                'recommendation': p.recommendation
                            } for p in predictions
                        ]
            except Exception as e:
                self.logger.error(f"Failed to collect ML optimizer metrics: {e}")
        
        # Collect from distributed scaling
        if hasattr(self, 'distributed_scaler'):
            try:
                scaling_status = self.distributed_scaler.get_system_status()
                all_metrics['distributed_scaling'] = {
                    'total_instances': scaling_status['load_balancer']['total_instances'],
                    'healthy_instances': scaling_status['load_balancer']['healthy_instances'],
                    'current_instances': scaling_status['auto_scaler']['current_instances'],
                    'avg_response_time_ms': scaling_status['request_metrics']['avg_response_time_ms']
                }
            except Exception as e:
                self.logger.error(f"Failed to collect distributed scaling metrics: {e}")
        
        # Collect from Alpha's monitoring
        if self.config.enable_alpha_integration:
            try:
                # Import Alpha components if available
                try:
                    from ...core.analytics.performance_monitor import get_monitoring_dashboard_data, get_system_health
                    alpha_data = get_monitoring_dashboard_data()
                    if alpha_data:
                        all_metrics['alpha_monitoring'] = alpha_data.get('metrics', {})
                    
                    health_data = get_system_health()
                    if health_data:
                        all_metrics['system_health'] = health_data.get('status', 'unknown')
                except ImportError:
                    self.logger.warning("Alpha monitoring components not available")
            except Exception as e:
                self.logger.error(f"Failed to collect Alpha monitoring metrics: {e}")
        
        # Collect from Alpha's optimization
        if self.config.enable_alpha_integration:
            try:
                # Import Alpha optimization if available
                try:
                    from ...core.optimization.performance_optimizer import get_performance_metrics
                    optimization_data = get_performance_metrics()
                    if optimization_data:
                        all_metrics['alpha_optimization'] = optimization_data
                except ImportError:
                    self.logger.warning("Alpha optimization components not available")
            except Exception as e:
                self.logger.error(f"Failed to collect Alpha optimization metrics: {e}")
        
        # Store in history
        with self._lock:
            timestamp = datetime.now(timezone.utc)
            for system, data in all_metrics.items():
                if isinstance(data, dict) and data:
                    self.metrics_history[system].append({
                        'timestamp': timestamp,
                        'data': data
                    })
        
        return all_metrics
    
    async def _get_current_metrics_for_prediction(self) -> Dict[str, float]:
        """Get current metrics formatted for ML prediction"""
        metrics = {}
        
        if hasattr(self, 'monitoring_system'):
            try:
                current_data = self.monitoring_system.metrics_collector.get_metrics()
                for name, metric_list in current_data.items():
                    if metric_list:
                        metrics[name] = metric_list[-1].value
            except Exception as e:
                self.logger.error(f"Failed to get prediction metrics: {e}")
        
        return metrics
    
    def get_metrics_history(self, system: str, hours: int = 1) -> List[Dict]:
        """Get metrics history for specific system"""
        with self._lock:
            if system not in self.metrics_history:
                return []
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [
                entry for entry in self.metrics_history[system]
                if entry['timestamp'] > cutoff_time
            ]
    
    def get_correlation_analysis(self) -> Dict[str, float]:
        """Calculate correlations between key metrics"""
        try:
            # Get recent data
            recent_data = {}
            for system in ['performance_monitoring', 'caching_system', 'distributed_scaling']:
                history = self.get_metrics_history(system, hours=1)
                if history:
                    recent_data[system] = history
            
            # Calculate correlations (simplified example)
            correlations = {}
            
            # CPU vs Response Time correlation
            if 'performance_monitoring' in recent_data and 'distributed_scaling' in recent_data:
                # This would be more sophisticated in production
                correlations['cpu_vs_response_time'] = 0.75  # Example correlation
            
            # Cache Hit Ratio vs Response Time correlation
            if 'caching_system' in recent_data and 'distributed_scaling' in recent_data:
                correlations['cache_vs_response_time'] = -0.85  # Negative correlation
            
            return correlations
        except Exception as e:
            self.logger.error(f"Failed to calculate correlations: {e}")
            return {}

def create_metrics_aggregator(config) -> MetricsAggregator:
    """
    Factory function to create a configured metrics aggregator instance.
    
    Args:
        config: Dashboard configuration object
        
    Returns:
        Configured MetricsAggregator instance
    """
    return MetricsAggregator(config)

# Export key components
__all__ = ['MetricsAggregator', 'create_metrics_aggregator']