"""
Analytics Performance Optimization Engine
========================================

Advanced performance optimization system for analytics components with
automatic tuning, resource optimization, and intelligent scaling.

Author: TestMaster Team
"""

import logging
import time
import threading
import queue
import gc
import sys
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    CACHE = "cache"
    NETWORK = "network"
    ALGORITHM = "algorithm"

class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class OptimizationRule:
    """Performance optimization rule."""
    rule_id: str
    optimization_type: OptimizationType
    condition: Callable[[], bool]
    action: Callable[[], None]
    priority: int
    cooldown_seconds: int
    max_applications: int
    description: str

@dataclass
class PerformanceMetric:
    """Performance measurement point."""
    metric_name: str
    value: float
    timestamp: datetime
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    unit: str = ""
    component: str = "system"

@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    rule_id: str
    timestamp: datetime
    success: bool
    improvement_percent: float
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    error: Optional[str] = None

class AnalyticsPerformanceOptimizer:
    """
    Advanced performance optimization engine for analytics system.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.MODERATE,
                 monitoring_interval: int = 60):
        """
        Initialize analytics performance optimizer.
        
        Args:
            optimization_level: Level of optimization aggressiveness
            monitoring_interval: Interval between optimization checks
        """
        self.optimization_level = optimization_level
        self.monitoring_interval = monitoring_interval
        
        # Optimization tracking
        self.optimization_rules = {}
        self.performance_metrics = defaultdict(deque)
        self.optimization_history = deque(maxlen=1000)
        self.rule_applications = defaultdict(int)
        self.rule_last_applied = {}
        
        # Performance baselines
        self.performance_baselines = {}
        self.baseline_windows = {
            'cpu_usage': 300,      # 5 minutes
            'memory_usage': 600,   # 10 minutes
            'response_time': 180,  # 3 minutes
            'throughput': 300      # 5 minutes
        }
        
        # Threading
        self.optimizer_active = False
        self.optimization_thread = None
        self.metrics_thread = None
        
        # Analytics components references
        self.analytics_aggregator = None
        self.cache_systems = []
        self.data_stores = []
        self.processing_pipelines = []
        
        # Optimization statistics
        self.optimizer_stats = {
            'optimizations_applied': 0,
            'performance_improvements': 0,
            'total_cpu_saved_percent': 0,
            'total_memory_saved_mb': 0,
            'average_response_time_improvement': 0,
            'start_time': datetime.now()
        }
        
        # Setup default optimization rules
        self._setup_default_rules()
        
        logger.info(f"Analytics Performance Optimizer initialized: {optimization_level.value}")
    
    def start_optimization(self):
        """Start performance optimization monitoring."""
        if self.optimizer_active:
            return
        
        self.optimizer_active = True
        
        # Start optimization threads
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        
        self.optimization_thread.start()
        self.metrics_thread.start()
        
        logger.info("Analytics performance optimization started")
    
    def stop_optimization(self):
        """Stop performance optimization."""
        self.optimizer_active = False
        
        # Wait for threads to finish
        for thread in [self.optimization_thread, self.metrics_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("Analytics performance optimization stopped")
    
    def register_analytics_component(self, component_name: str, component_instance: Any):
        """Register an analytics component for optimization."""
        if component_name == 'aggregator':
            self.analytics_aggregator = component_instance
        elif 'cache' in component_name.lower():
            self.cache_systems.append(component_instance)
        elif 'store' in component_name.lower() or 'database' in component_name.lower():
            self.data_stores.append(component_instance)
        elif 'pipeline' in component_name.lower():
            self.processing_pipelines.append(component_instance)
        
        logger.info(f"Registered component for optimization: {component_name}")
    
    def add_optimization_rule(self, rule: OptimizationRule):
        """Add a custom optimization rule."""
        self.optimization_rules[rule.rule_id] = rule
        logger.info(f"Added optimization rule: {rule.rule_id}")
    
    def record_performance_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        self.performance_metrics[metric.metric_name].append(metric)
        
        # Keep only recent metrics
        cutoff_time = datetime.now() - timedelta(hours=2)
        while (self.performance_metrics[metric.metric_name] and 
               self.performance_metrics[metric.metric_name][0].timestamp < cutoff_time):
            self.performance_metrics[metric.metric_name].popleft()
        
        # Update baseline if needed
        self._update_performance_baseline(metric)
    
    def optimize_component(self, component_name: str) -> List[OptimizationResult]:
        """Manually trigger optimization for a specific component."""
        results = []
        
        for rule_id, rule in self.optimization_rules.items():
            if component_name.lower() in rule.description.lower():
                if self._can_apply_rule(rule):
                    result = self._apply_optimization_rule(rule)
                    if result:
                        results.append(result)
        
        return results
    
    def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        # Analyze current metrics
        current_metrics = self._collect_current_metrics()
        
        # Memory optimization recommendations
        memory_usage = current_metrics.get('memory_usage_percent', 0)
        if memory_usage > 80:
            recommendations.append({
                'type': 'memory',
                'priority': 'high',
                'description': f'Memory usage is {memory_usage:.1f}%. Consider cache cleanup or garbage collection.',
                'estimated_improvement': '10-20% memory reduction',
                'action': 'memory_cleanup'
            })
        
        # CPU optimization recommendations
        cpu_usage = current_metrics.get('cpu_usage_percent', 0)
        if cpu_usage > 75:
            recommendations.append({
                'type': 'cpu',
                'priority': 'high',
                'description': f'CPU usage is {cpu_usage:.1f}%. Consider algorithm optimization or parallel processing.',
                'estimated_improvement': '15-30% CPU reduction',
                'action': 'algorithm_optimization'
            })
        
        # Response time recommendations
        avg_response_time = current_metrics.get('average_response_time_ms', 0)
        if avg_response_time > 5000:  # 5 seconds
            recommendations.append({
                'type': 'response_time',
                'priority': 'medium',
                'description': f'Average response time is {avg_response_time:.0f}ms. Consider caching improvements.',
                'estimated_improvement': '20-40% response time reduction',
                'action': 'cache_optimization'
            })
        
        # Cache efficiency recommendations
        cache_hit_rate = current_metrics.get('cache_hit_rate', 100)
        if cache_hit_rate < 70:
            recommendations.append({
                'type': 'cache',
                'priority': 'medium',
                'description': f'Cache hit rate is {cache_hit_rate:.1f}%. Consider cache tuning or size adjustment.',
                'estimated_improvement': '30-50% performance improvement',
                'action': 'cache_tuning'
            })
        
        return recommendations
    
    def _setup_default_rules(self):
        """Setup default optimization rules."""
        
        # Memory optimization rule
        def memory_pressure_condition():
            return psutil.virtual_memory().percent > 85
        
        def memory_cleanup_action():
            # Force garbage collection
            gc.collect()
            
            # Clean up caches if available
            for cache in self.cache_systems:
                if hasattr(cache, 'cleanup_expired'):
                    cache.cleanup_expired()
                elif hasattr(cache, 'clear_old_entries'):
                    cache.clear_old_entries()
            
            logger.info("Applied memory cleanup optimization")
        
        memory_rule = OptimizationRule(
            rule_id="memory_cleanup",
            optimization_type=OptimizationType.MEMORY,
            condition=memory_pressure_condition,
            action=memory_cleanup_action,
            priority=8,
            cooldown_seconds=300,  # 5 minutes
            max_applications=10,
            description="Clean up memory when usage exceeds 85%"
        )
        
        # CPU optimization rule
        def cpu_pressure_condition():
            return psutil.cpu_percent(interval=1) > 80
        
        def cpu_optimization_action():
            # Reduce background processing if possible
            if self.analytics_aggregator and hasattr(self.analytics_aggregator, 'cache_ttl'):
                # Increase cache TTL to reduce computation
                self.analytics_aggregator.cache_ttl = min(self.analytics_aggregator.cache_ttl * 1.5, 300)
            
            # Reduce processing pipeline workers if available
            for pipeline in self.processing_pipelines:
                if hasattr(pipeline, 'reduce_workers'):
                    pipeline.reduce_workers()
            
            logger.info("Applied CPU optimization")
        
        cpu_rule = OptimizationRule(
            rule_id="cpu_optimization",
            optimization_type=OptimizationType.CPU,
            condition=cpu_pressure_condition,
            action=cpu_optimization_action,
            priority=7,
            cooldown_seconds=180,  # 3 minutes
            max_applications=5,
            description="Optimize CPU usage when load exceeds 80%"
        )
        
        # Cache optimization rule
        def cache_miss_condition():
            # Check cache hit rates across cache systems
            for cache in self.cache_systems:
                if hasattr(cache, 'get_hit_rate'):
                    hit_rate = cache.get_hit_rate()
                    if hit_rate < 0.6:  # Less than 60% hit rate
                        return True
            return False
        
        def cache_optimization_action():
            # Optimize cache configurations
            for cache in self.cache_systems:
                if hasattr(cache, 'optimize_configuration'):
                    cache.optimize_configuration()
                elif hasattr(cache, 'increase_size'):
                    cache.increase_size(factor=1.2)
            
            logger.info("Applied cache optimization")
        
        cache_rule = OptimizationRule(
            rule_id="cache_optimization",
            optimization_type=OptimizationType.CACHE,
            condition=cache_miss_condition,
            action=cache_optimization_action,
            priority=6,
            cooldown_seconds=600,  # 10 minutes
            max_applications=3,
            description="Optimize cache when hit rate is low"
        )
        
        # IO optimization rule
        def io_pressure_condition():
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    # Simple heuristic: high read/write operations
                    return (disk_io.read_count + disk_io.write_count) > 10000
            except:
                pass
            return False
        
        def io_optimization_action():
            # Optimize data store operations
            for store in self.data_stores:
                if hasattr(store, 'optimize_io'):
                    store.optimize_io()
                elif hasattr(store, 'enable_batch_mode'):
                    store.enable_batch_mode()
            
            logger.info("Applied IO optimization")
        
        io_rule = OptimizationRule(
            rule_id="io_optimization",
            optimization_type=OptimizationType.IO,
            condition=io_pressure_condition,
            action=io_optimization_action,
            priority=5,
            cooldown_seconds=240,  # 4 minutes
            max_applications=5,
            description="Optimize IO when disk activity is high"
        )
        
        # Add rules based on optimization level
        if self.optimization_level in [OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE]:
            self.optimization_rules[memory_rule.rule_id] = memory_rule
            self.optimization_rules[cache_rule.rule_id] = cache_rule
        
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            self.optimization_rules[cpu_rule.rule_id] = cpu_rule
            self.optimization_rules[io_rule.rule_id] = io_rule
    
    def _optimization_loop(self):
        """Main optimization monitoring loop."""
        while self.optimizer_active:
            try:
                time.sleep(self.monitoring_interval)
                
                # Check all optimization rules
                for rule_id, rule in self.optimization_rules.items():
                    if self._can_apply_rule(rule):
                        if rule.condition():
                            result = self._apply_optimization_rule(rule)
                            if result and result.success:
                                self.optimizer_stats['optimizations_applied'] += 1
                                if result.improvement_percent > 0:
                                    self.optimizer_stats['performance_improvements'] += 1
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
    
    def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self.optimizer_active:
            try:
                time.sleep(30)  # Collect metrics every 30 seconds
                
                # Collect system metrics
                current_metrics = self._collect_current_metrics()
                
                for metric_name, value in current_metrics.items():
                    if isinstance(value, (int, float)):
                        metric = PerformanceMetric(
                            metric_name=metric_name,
                            value=value,
                            timestamp=datetime.now(),
                            component="system"
                        )
                        self.record_performance_metric(metric)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        metrics = {}
        
        try:
            # System metrics
            metrics['cpu_usage_percent'] = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            metrics['memory_usage_percent'] = memory.percent
            metrics['memory_available_mb'] = memory.available / (1024 * 1024)
            
            # Process-specific metrics
            process = psutil.Process()
            metrics['process_memory_mb'] = process.memory_info().rss / (1024 * 1024)
            metrics['process_cpu_percent'] = process.cpu_percent()
            
            # Analytics-specific metrics
            if self.analytics_aggregator:
                start_time = time.time()
                # Test response time with a lightweight operation
                try:
                    test_result = self.analytics_aggregator._get_system_metrics()
                    response_time = (time.time() - start_time) * 1000
                    metrics['analytics_response_time_ms'] = response_time
                except:
                    pass
            
            # Cache metrics
            cache_hit_rates = []
            for cache in self.cache_systems:
                if hasattr(cache, 'get_hit_rate'):
                    hit_rate = cache.get_hit_rate()
                    cache_hit_rates.append(hit_rate)
                elif hasattr(cache, 'get_cache_stats'):
                    stats = cache.get_cache_stats()
                    if 'hit_rate' in stats:
                        cache_hit_rates.append(stats['hit_rate'])
            
            if cache_hit_rates:
                metrics['cache_hit_rate'] = sum(cache_hit_rates) / len(cache_hit_rates)
            
        except Exception as e:
            logger.warning(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _can_apply_rule(self, rule: OptimizationRule) -> bool:
        """Check if a rule can be applied."""
        # Check max applications
        if self.rule_applications[rule.rule_id] >= rule.max_applications:
            return False
        
        # Check cooldown
        last_applied = self.rule_last_applied.get(rule.rule_id)
        if last_applied:
            time_since_last = (datetime.now() - last_applied).total_seconds()
            if time_since_last < rule.cooldown_seconds:
                return False
        
        return True
    
    def _apply_optimization_rule(self, rule: OptimizationRule) -> Optional[OptimizationResult]:
        """Apply an optimization rule and measure results."""
        try:
            # Collect before metrics
            before_metrics = self._collect_current_metrics()
            
            # Apply the optimization
            rule.action()
            
            # Wait a moment for effects to take place
            time.sleep(2)
            
            # Collect after metrics
            after_metrics = self._collect_current_metrics()
            
            # Calculate improvement
            improvement_percent = self._calculate_improvement(before_metrics, after_metrics, rule.optimization_type)
            
            # Record the application
            self.rule_applications[rule.rule_id] += 1
            self.rule_last_applied[rule.rule_id] = datetime.now()
            
            result = OptimizationResult(
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                success=True,
                improvement_percent=improvement_percent,
                before_metrics=before_metrics,
                after_metrics=after_metrics
            )
            
            self.optimization_history.append(result)
            
            # Update statistics
            if improvement_percent > 0:
                if rule.optimization_type == OptimizationType.CPU:
                    cpu_before = before_metrics.get('cpu_usage_percent', 0)
                    cpu_after = after_metrics.get('cpu_usage_percent', 0)
                    self.optimizer_stats['total_cpu_saved_percent'] += max(0, cpu_before - cpu_after)
                
                elif rule.optimization_type == OptimizationType.MEMORY:
                    mem_before = before_metrics.get('memory_usage_percent', 0)
                    mem_after = after_metrics.get('memory_usage_percent', 0)
                    memory_saved_percent = max(0, mem_before - mem_after)
                    total_memory = psutil.virtual_memory().total / (1024 * 1024)
                    self.optimizer_stats['total_memory_saved_mb'] += (memory_saved_percent / 100) * total_memory
            
            logger.info(f"Applied optimization rule {rule.rule_id}: {improvement_percent:.1f}% improvement")
            return result
            
        except Exception as e:
            result = OptimizationResult(
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                success=False,
                improvement_percent=0,
                before_metrics=before_metrics if 'before_metrics' in locals() else {},
                after_metrics={},
                error=str(e)
            )
            
            self.optimization_history.append(result)
            logger.error(f"Optimization rule {rule.rule_id} failed: {e}")
            return result
    
    def _calculate_improvement(self, before: Dict[str, Any], after: Dict[str, Any], 
                             optimization_type: OptimizationType) -> float:
        """Calculate performance improvement percentage."""
        try:
            if optimization_type == OptimizationType.MEMORY:
                before_val = before.get('memory_usage_percent', 0)
                after_val = after.get('memory_usage_percent', 0)
                if before_val > 0:
                    return max(0, (before_val - after_val) / before_val * 100)
            
            elif optimization_type == OptimizationType.CPU:
                before_val = before.get('cpu_usage_percent', 0)
                after_val = after.get('cpu_usage_percent', 0)
                if before_val > 0:
                    return max(0, (before_val - after_val) / before_val * 100)
            
            elif optimization_type == OptimizationType.CACHE:
                before_val = before.get('cache_hit_rate', 0)
                after_val = after.get('cache_hit_rate', 0)
                if before_val < after_val:
                    return (after_val - before_val) * 100  # Absolute improvement in hit rate
            
            # For other types, use response time as general metric
            before_val = before.get('analytics_response_time_ms', 0)
            after_val = after.get('analytics_response_time_ms', 0)
            if before_val > 0 and after_val < before_val:
                return (before_val - after_val) / before_val * 100
                
        except Exception as e:
            logger.warning(f"Error calculating improvement: {e}")
        
        return 0
    
    def _update_performance_baseline(self, metric: PerformanceMetric):
        """Update performance baseline for a metric."""
        metric_name = metric.metric_name
        
        # Get recent metrics for baseline calculation
        window_size = self.baseline_windows.get(metric_name, 300)
        cutoff_time = datetime.now() - timedelta(seconds=window_size)
        
        recent_metrics = [m for m in self.performance_metrics[metric_name] 
                         if m.timestamp >= cutoff_time]
        
        if len(recent_metrics) >= 5:  # Need at least 5 points for baseline
            values = [m.value for m in recent_metrics]
            self.performance_baselines[metric_name] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': math.sqrt(sum((x - sum(values)/len(values))**2 for x in values) / len(values)),
                'last_updated': datetime.now()
            }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization system summary."""
        uptime = (datetime.now() - self.optimizer_stats['start_time']).total_seconds()
        
        # Recent optimizations (last hour)
        recent_optimizations = [opt for opt in self.optimization_history 
                              if (datetime.now() - opt.timestamp).total_seconds() < 3600]
        
        successful_optimizations = [opt for opt in recent_optimizations if opt.success]
        
        return {
            'optimization_level': self.optimization_level.value,
            'optimizer_active': self.optimizer_active,
            'uptime_seconds': uptime,
            'registered_components': {
                'cache_systems': len(self.cache_systems),
                'data_stores': len(self.data_stores),
                'processing_pipelines': len(self.processing_pipelines),
                'has_aggregator': self.analytics_aggregator is not None
            },
            'optimization_rules': {
                'total_rules': len(self.optimization_rules),
                'rules_by_type': {
                    opt_type.value: len([r for r in self.optimization_rules.values() 
                                       if r.optimization_type == opt_type])
                    for opt_type in OptimizationType
                }
            },
            'statistics': self.optimizer_stats.copy(),
            'recent_activity': {
                'optimizations_last_hour': len(recent_optimizations),
                'successful_optimizations_last_hour': len(successful_optimizations),
                'average_improvement_percent': (
                    sum(opt.improvement_percent for opt in successful_optimizations) / 
                    len(successful_optimizations) if successful_optimizations else 0
                )
            },
            'current_metrics': self._collect_current_metrics(),
            'performance_baselines': self.performance_baselines
        }
    
    def get_optimization_history(self, hours: int = 24) -> List[OptimizationResult]:
        """Get optimization history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [opt for opt in self.optimization_history if opt.timestamp >= cutoff_time]
    
    def shutdown(self):
        """Shutdown optimization engine."""
        self.stop_optimization()
        logger.info("Analytics Performance Optimizer shutdown")