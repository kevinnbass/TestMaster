#!/usr/bin/env python3
"""
Metrics Aggregator Module
=========================

Metrics collection and aggregation extracted from performance_analytics_dashboard.py
for STEELCLAD modularization (Agent Y supporting Agent Z)

Handles:
- Multi-system metrics collection
- Performance data aggregation
- Historical metrics management
- Correlation analysis and trend detection
"""

import asyncio
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Deque
import pandas as pd
import numpy as np

# Import configuration
try:
    from ..config.dashboard_config import DashboardConfig
except ImportError:
    from specialized.config.dashboard_config import DashboardConfig

# Performance system imports with fallbacks
try:
    from performance_monitoring_system import PerformanceMonitoringSystem
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from caching_framework import CachingFramework
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

try:
    from ml_performance_optimizer import MLPerformanceOptimizer
    ML_OPTIMIZER_AVAILABLE = True
except ImportError:
    ML_OPTIMIZER_AVAILABLE = False

try:
    from distributed_scaling import DistributedScaling
    SCALING_AVAILABLE = True
except ImportError:
    SCALING_AVAILABLE = False

try:
    from alpha_testing_framework import AlphaTestingFramework
    from alpha_optimization_infrastructure import AlphaOptimizationInfrastructure
    ALPHA_MONITORING_AVAILABLE = True
    ALPHA_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ALPHA_MONITORING_AVAILABLE = False
    ALPHA_OPTIMIZATION_AVAILABLE = False


class MetricsAggregator:
    """
    Aggregates metrics from all performance systems
    
    Collects, processes, and stores metrics from multiple integrated systems
    including performance monitoring, caching, ML optimization, and Alpha systems.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=config.max_data_points)
        )
        self.predictions_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.test_results_history: Deque = deque(maxlen=50)
        self.logger = logging.getLogger('MetricsAggregator')
        self._lock = threading.RLock()
        
        # Initialize system connections
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize connections to performance systems"""
        try:
            if MONITORING_AVAILABLE:
                self.monitoring_system = PerformanceMonitoringSystem()
                self.logger.info("Performance monitoring system initialized")
            
            if CACHING_AVAILABLE:
                self.caching_system = CachingFramework()
                self.logger.info("Caching system initialized")
            
            if ML_OPTIMIZER_AVAILABLE:
                self.ml_optimizer = MLPerformanceOptimizer()
                self.logger.info("ML optimizer initialized")
            
            if SCALING_AVAILABLE:
                self.distributed_scaler = DistributedScaling()
                self.logger.info("Distributed scaler initialized")
            
            if ALPHA_MONITORING_AVAILABLE:
                self.alpha_testing = AlphaTestingFramework()
                self.logger.info("Alpha testing framework initialized")
            
            if ALPHA_OPTIMIZATION_AVAILABLE:
                self.alpha_optimization = AlphaOptimizationInfrastructure()
                self.logger.info("Alpha optimization infrastructure initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize systems: {e}")
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics from all integrated systems
        
        Returns:
            Dictionary containing metrics from all systems
        """
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
        if MONITORING_AVAILABLE and hasattr(self, 'monitoring_system'):
            all_metrics['performance_monitoring'] = await self._collect_performance_metrics()
        
        # Collect from caching system
        if CACHING_AVAILABLE and hasattr(self, 'caching_system'):
            all_metrics['caching_system'] = await self._collect_caching_metrics()
        
        # Collect from ML optimizer
        if ML_OPTIMIZER_AVAILABLE and hasattr(self, 'ml_optimizer'):
            all_metrics['ml_optimizer'] = await self._collect_ml_metrics()
        
        # Collect from distributed scaling
        if SCALING_AVAILABLE and hasattr(self, 'distributed_scaler'):
            all_metrics['distributed_scaling'] = await self._collect_scaling_metrics()
        
        # Collect from Alpha systems
        if self.config.enable_alpha_integration:
            if ALPHA_MONITORING_AVAILABLE and hasattr(self, 'alpha_testing'):
                all_metrics['alpha_monitoring'] = await self._collect_alpha_monitoring()
            
            if ALPHA_OPTIMIZATION_AVAILABLE and hasattr(self, 'alpha_optimization'):
                all_metrics['alpha_optimization'] = await self._collect_alpha_optimization()
        
        # Calculate overall system health
        all_metrics['system_health'] = self._calculate_system_health(all_metrics)
        
        # Store in history
        with self._lock:
            self.metrics_history['all_metrics'].append(all_metrics)
        
        return all_metrics
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect metrics from performance monitoring system"""
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
            
            return performance_data
        except Exception as e:
            self.logger.error(f"Failed to collect performance monitoring metrics: {e}")
            return {}
    
    async def _collect_caching_metrics(self) -> Dict[str, Any]:
        """Collect metrics from caching system"""
        try:
            cache_status = self.caching_system.get_system_status()
            return {
                'hit_ratio': cache_status['metrics']['hit_ratio'],
                'total_operations': cache_status['metrics']['total_operations'],
                'memory_utilization': cache_status['memory_layer']['utilization'],
                'system_health': cache_status['system_health']
            }
        except Exception as e:
            self.logger.error(f"Failed to collect caching metrics: {e}")
            return {}
    
    async def _collect_ml_metrics(self) -> Dict[str, Any]:
        """Collect metrics from ML optimizer"""
        try:
            optimizer_status = self.ml_optimizer.get_optimization_status()
            ml_data = {
                'models_trained': len(optimizer_status['models_trained']),
                'predictions_count': optimizer_status['predictions_count'],
                'current_parameters': optimizer_status['current_parameters']
            }
            
            # Get predictions if available
            if self.config.enable_predictions:
                current_metrics = await self._get_current_metrics_for_prediction()
                if current_metrics:
                    predictions = self.ml_optimizer._make_predictions(current_metrics)
                    ml_data['predictions'] = [
                        {
                            'metric_name': p.metric_name,
                            'current_value': p.current_value,
                            'predicted_value': p.predicted_value,
                            'confidence': p.confidence,
                            'trend': p.trend,
                            'recommendation': p.recommendation
                        } for p in predictions
                    ]
            
            return ml_data
        except Exception as e:
            self.logger.error(f"Failed to collect ML optimizer metrics: {e}")
            return {}
    
    async def _collect_scaling_metrics(self) -> Dict[str, Any]:
        """Collect metrics from distributed scaling system"""
        try:
            scaling_status = self.distributed_scaler.get_system_status()
            return {
                'total_instances': scaling_status['load_balancer']['total_instances'],
                'healthy_instances': scaling_status['load_balancer']['healthy_instances'],
                'current_instances': scaling_status['auto_scaler']['current_instances'],
                'avg_response_time_ms': scaling_status['request_metrics']['avg_response_time_ms']
            }
        except Exception as e:
            self.logger.error(f"Failed to collect distributed scaling metrics: {e}")
            return {}
    
    async def _collect_alpha_monitoring(self) -> Dict[str, Any]:
        """Collect metrics from Alpha monitoring system"""
        try:
            test_results = self.alpha_testing.get_recent_results()
            return {
                'total_tests': len(test_results),
                'passed_tests': len([r for r in test_results if r['status'] == 'passed']),
                'failed_tests': len([r for r in test_results if r['status'] == 'failed']),
                'test_coverage': self.alpha_testing.get_coverage_metrics(),
                'performance_tests': [r for r in test_results if r.get('type') == 'performance']
            }
        except Exception as e:
            self.logger.error(f"Failed to collect Alpha monitoring metrics: {e}")
            return {}
    
    async def _collect_alpha_optimization(self) -> Dict[str, Any]:
        """Collect metrics from Alpha optimization system"""
        try:
            optimization_status = self.alpha_optimization.get_status()
            return {
                'active_optimizations': len(optimization_status['active']),
                'completed_optimizations': len(optimization_status['completed']),
                'optimization_improvements': optimization_status.get('improvements', {}),
                'resource_efficiency': optimization_status.get('resource_efficiency', 0.0)
            }
        except Exception as e:
            self.logger.error(f"Failed to collect Alpha optimization metrics: {e}")
            return {}
    
    async def _get_current_metrics_for_prediction(self) -> Optional[Dict[str, float]]:
        """Get current metrics suitable for ML predictions"""
        try:
            with self._lock:
                if not self.metrics_history['all_metrics']:
                    return None
                
                latest_metrics = self.metrics_history['all_metrics'][-1]
                
                # Extract numerical values for prediction
                prediction_input = {}
                
                # Performance monitoring metrics
                perf_metrics = latest_metrics.get('performance_monitoring', {})
                for name, data in perf_metrics.items():
                    if isinstance(data.get('value'), (int, float)):
                        prediction_input[f'perf_{name}'] = data['value']
                
                # Caching metrics
                cache_metrics = latest_metrics.get('caching_system', {})
                for key, value in cache_metrics.items():
                    if isinstance(value, (int, float)):
                        prediction_input[f'cache_{key}'] = value
                
                # Scaling metrics
                scale_metrics = latest_metrics.get('distributed_scaling', {})
                for key, value in scale_metrics.items():
                    if isinstance(value, (int, float)):
                        prediction_input[f'scale_{key}'] = value
                
                return prediction_input if prediction_input else None
                
        except Exception as e:
            self.logger.error(f"Failed to get current metrics for prediction: {e}")
            return None
    
    def _calculate_system_health(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall system health based on collected metrics"""
        try:
            health_scores = []
            
            # Check performance monitoring health
            perf_metrics = metrics.get('performance_monitoring', {})
            if perf_metrics:
                health_scores.append(0.8)  # Assume good if data available
            
            # Check caching system health
            cache_metrics = metrics.get('caching_system', {})
            if cache_metrics:
                hit_ratio = cache_metrics.get('hit_ratio', 0)
                health_scores.append(min(hit_ratio * 1.2, 1.0))  # Scale hit ratio
            
            # Check Alpha systems health
            alpha_monitoring = metrics.get('alpha_monitoring', {})
            if alpha_monitoring:
                total_tests = alpha_monitoring.get('total_tests', 0)
                passed_tests = alpha_monitoring.get('passed_tests', 0)
                if total_tests > 0:
                    test_ratio = passed_tests / total_tests
                    health_scores.append(test_ratio)
            
            # Calculate average health
            if health_scores:
                avg_health = sum(health_scores) / len(health_scores)
                if avg_health >= 0.8:
                    return 'excellent'
                elif avg_health >= 0.6:
                    return 'good'
                elif avg_health >= 0.4:
                    return 'fair'
                else:
                    return 'poor'
            
            return 'unknown'
            
        except Exception as e:
            self.logger.error(f"Failed to calculate system health: {e}")
            return 'error'
    
    def get_metrics_history(self, metric_name: str = None, limit: int = None) -> Dict[str, Any]:
        """
        Get historical metrics data
        
        Args:
            metric_name: Specific metric to retrieve (None for all)
            limit: Maximum number of data points to return
            
        Returns:
            Historical metrics data
        """
        with self._lock:
            if metric_name:
                history = list(self.metrics_history.get(metric_name, []))
                if limit:
                    history = history[-limit:]
                return {metric_name: history}
            else:
                result = {}
                for name, data in self.metrics_history.items():
                    history = list(data)
                    if limit:
                        history = history[-limit:]
                    result[name] = history
                return result
    
    def calculate_correlations(self, metrics_list: List[str]) -> Dict[str, float]:
        """
        Calculate correlations between different metrics
        
        Args:
            metrics_list: List of metric names to correlate
            
        Returns:
            Dictionary of correlation coefficients
        """
        try:
            with self._lock:
                if not self.metrics_history['all_metrics']:
                    return {}
                
                # Extract time series data for each metric
                data_dict = {}
                for metric_name in metrics_list:
                    values = []
                    for entry in self.metrics_history['all_metrics']:
                        # Navigate through nested structure to find metric
                        value = self._extract_metric_value(entry, metric_name)
                        if value is not None:
                            values.append(value)
                    
                    if values:
                        data_dict[metric_name] = values
                
                # Calculate correlations
                correlations = {}
                if len(data_dict) >= 2:
                    df = pd.DataFrame(data_dict)
                    correlation_matrix = df.corr()
                    
                    for i, metric1 in enumerate(metrics_list):
                        for j, metric2 in enumerate(metrics_list):
                            if i < j and metric1 in correlation_matrix and metric2 in correlation_matrix:
                                corr_value = correlation_matrix.loc[metric1, metric2]
                                if not np.isnan(corr_value):
                                    correlations[f"{metric1}_vs_{metric2}"] = float(corr_value)
                
                return correlations
                
        except Exception as e:
            self.logger.error(f"Failed to calculate correlations: {e}")
            return {}
    
    def _extract_metric_value(self, data: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract a specific metric value from nested data structure"""
        try:
            # Handle nested metric names like "performance_monitoring.response_time"
            if '.' in metric_name:
                parts = metric_name.split('.')
                current = data
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return None
                
                if isinstance(current, dict) and 'value' in current:
                    return float(current['value'])
                elif isinstance(current, (int, float)):
                    return float(current)
            else:
                # Simple metric name - search through all subsystems
                for system_name, system_data in data.items():
                    if isinstance(system_data, dict) and metric_name in system_data:
                        value = system_data[metric_name]
                        if isinstance(value, dict) and 'value' in value:
                            return float(value['value'])
                        elif isinstance(value, (int, float)):
                            return float(value)
            
            return None
            
        except Exception:
            return None
    
    def clear_history(self):
        """Clear all historical data"""
        with self._lock:
            self.metrics_history.clear()
            self.predictions_history.clear()
            self.test_results_history.clear()
        self.logger.info("Metrics history cleared")