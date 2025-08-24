"""
Unified Performance Hub - AGENT B Hour 19-21 Enhancement
========================================================

Consolidates all performance optimization systems into a unified hub providing:
- Enterprise performance monitoring with real-time metrics
- ML-powered optimization recommendations 
- Performance optimization engine integration
- Comprehensive resource usage tracking
- Intelligent alerting and trend analysis

This replaces scattered performance files with a centralized performance management system.
"""

import time
import threading
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

from .enterprise_performance_monitor import EnterprisePerformanceMonitor, PerformanceAlert
from .performance_optimization_engine import *
from ..ml.advanced.performance_optimizer import *
from ..ml.advanced.performance_ml_engine import PerformanceMLEngine
from ..ml.advanced.performance_execution_manager import PerformanceExecutionManager

logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good" 
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


class OptimizationType(Enum):
    """Types of performance optimizations."""
    CPU_OPTIMIZATION = "cpu_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    IO_OPTIMIZATION = "io_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    CACHE_OPTIMIZATION = "cache_optimization"
    DATABASE_OPTIMIZATION = "database_optimization"


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    optimization_id: str
    optimization_type: OptimizationType
    priority: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    estimated_improvement: float  # percentage
    implementation_effort: str  # "low", "medium", "high"
    resources_required: List[str]
    code_changes: Optional[str] = None
    testing_required: bool = True


@dataclass
class PerformanceProfile:
    """Comprehensive performance profile."""
    profile_id: str
    system_name: str
    timestamp: datetime
    overall_score: float
    level: PerformanceLevel
    metrics: List[PerformanceMetric]
    bottlenecks: List[str]
    recommendations: List[OptimizationRecommendation]
    resource_usage: Dict[str, float]


class UnifiedPerformanceHub:
    """
    Unified Performance Hub - Central performance management system.
    
    Integrates enterprise monitoring, ML optimization, and performance 
    engineering into a single comprehensive performance management platform.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified performance hub."""
        self.config = config or {}
        
        # Initialize core components
        self.enterprise_monitor = EnterprisePerformanceMonitor(
            monitor_interval=self.config.get('monitor_interval', 30),
            history_size=self.config.get('history_size', 5000),
            enable_detailed_tracking=True
        )
        
        # Initialize ML components with error handling
        try:
            self.ml_engine = PerformanceMLEngine(self.config)
            self.execution_manager = PerformanceExecutionManager(self.config)
        except Exception as e:
            logger.warning(f"ML components not available: {e}")
            self.ml_engine = None
            self.execution_manager = None
        
        # Performance data storage
        self._performance_history = deque(maxlen=10000)
        self._metrics_buffer = deque(maxlen=50000) 
        self._profiles = {}
        self._optimization_history = deque(maxlen=1000)
        
        # Real-time monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self._performance_callbacks = []
        
        # Performance baselines and targets
        self._performance_baselines = self._initialize_baselines()
        self._performance_targets = self._initialize_targets()
        
        # Optimization engine
        self._optimization_engine = self._initialize_optimization_engine()
        
        # Start monitoring if configured
        if self.config.get('auto_start', False):
            self.start_monitoring()
    
    def _initialize_baselines(self) -> Dict[str, float]:
        """Initialize performance baselines."""
        return {
            'cpu_usage': 20.0,
            'memory_usage': 30.0,
            'response_time': 0.5,
            'throughput': 100.0,
            'error_rate': 0.01,
            'thread_count': 10,
            'io_wait': 0.1
        }
    
    def _initialize_targets(self) -> Dict[str, float]:
        """Initialize performance targets."""
        return {
            'cpu_usage': 15.0,
            'memory_usage': 25.0,
            'response_time': 0.3,
            'throughput': 150.0,
            'error_rate': 0.005,
            'thread_count': 8,
            'io_wait': 0.05
        }
    
    def _initialize_optimization_engine(self):
        """Initialize the optimization engine with error handling."""
        try:
            # Try to initialize from the split modules
            from .performance_optimization_engine_modules.performance_optimization_engine_core import PerformanceOptimizationCore
            return PerformanceOptimizationCore(self.config)
        except ImportError:
            logger.warning("Performance optimization engine not available, using simplified version")
            return None
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            
            # Start enterprise monitor
            self.enterprise_monitor.start_monitoring()
            
            # Start unified monitoring thread
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            
            logger.info("Unified performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        self._monitoring_active = False
        
        # Stop enterprise monitor
        if hasattr(self.enterprise_monitor, 'stop_monitoring'):
            self.enterprise_monitor.stop_monitoring()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logger.info("Unified performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect comprehensive metrics
                metrics = self._collect_comprehensive_metrics()
                
                # Store metrics
                for metric in metrics:
                    self._metrics_buffer.append(metric)
                
                # Analyze performance
                profile = self._analyze_performance(metrics)
                if profile:
                    self._performance_history.append(profile)
                    
                    # Trigger callbacks
                    for callback in self._performance_callbacks:
                        try:
                            callback(profile)
                        except Exception as e:
                            logger.error(f"Performance callback failed: {e}")
                
                # Check for optimization opportunities
                self._check_optimization_opportunities(metrics)
                
                time.sleep(self.config.get('monitoring_interval', 30))
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _collect_comprehensive_metrics(self) -> List[PerformanceMetric]:
        """Collect comprehensive performance metrics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics.extend([
                PerformanceMetric("cpu_usage", cpu_percent, "percent", timestamp, "system"),
                PerformanceMetric("memory_usage", memory.percent, "percent", timestamp, "system"),
                PerformanceMetric("memory_available", memory.available / (1024**3), "GB", timestamp, "system"),
                PerformanceMetric("disk_usage", disk.percent, "percent", timestamp, "system"),
                PerformanceMetric("disk_free", disk.free / (1024**3), "GB", timestamp, "system")
            ])
            
            # Process-specific metrics
            process = psutil.Process()
            proc_memory = process.memory_info()
            
            metrics.extend([
                PerformanceMetric("process_memory_rss", proc_memory.rss / (1024**2), "MB", timestamp, "process"),
                PerformanceMetric("process_memory_vms", proc_memory.vms / (1024**2), "MB", timestamp, "process"),
                PerformanceMetric("process_cpu", process.cpu_percent(), "percent", timestamp, "process"),
                PerformanceMetric("thread_count", process.num_threads(), "count", timestamp, "process")
            ])
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                metrics.extend([
                    PerformanceMetric("network_bytes_sent", network.bytes_sent, "bytes", timestamp, "network"),
                    PerformanceMetric("network_bytes_recv", network.bytes_recv, "bytes", timestamp, "network"),
                ])
            except:
                pass
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        # Application-specific metrics from enterprise monitor
        if hasattr(self.enterprise_monitor, 'get_current_metrics'):
            try:
                app_metrics = self.enterprise_monitor.get_current_metrics()
                for name, value in app_metrics.items():
                    metrics.append(PerformanceMetric(name, value, "units", timestamp, "application"))
            except:
                pass
        
        return metrics
    
    def _analyze_performance(self, metrics: List[PerformanceMetric]) -> Optional[PerformanceProfile]:
        """Analyze performance metrics and create profile."""
        if not metrics:
            return None
        
        # Calculate overall performance score
        score_components = []
        resource_usage = {}
        bottlenecks = []
        
        for metric in metrics:
            resource_usage[metric.name] = metric.value
            
            # Score individual metrics
            if metric.name in self._performance_baselines:
                baseline = self._performance_baselines[metric.name]
                target = self._performance_targets[metric.name]
                
                # Calculate performance score for this metric
                if metric.name in ['cpu_usage', 'memory_usage', 'error_rate', 'io_wait']:
                    # Lower is better
                    if metric.value <= target:
                        component_score = 100.0
                    elif metric.value <= baseline:
                        component_score = 80.0 - ((metric.value - target) / (baseline - target)) * 30
                    else:
                        component_score = max(0, 50.0 - ((metric.value - baseline) / baseline) * 50)
                else:
                    # Higher is better
                    if metric.value >= target:
                        component_score = 100.0
                    elif metric.value >= baseline:
                        component_score = 80.0 + ((metric.value - baseline) / (target - baseline)) * 20
                    else:
                        component_score = max(0, 50.0 * (metric.value / baseline))
                
                score_components.append(component_score)
                
                # Identify bottlenecks
                if component_score < 60:
                    bottlenecks.append(f"{metric.name}: {metric.value:.2f} {metric.unit}")
        
        # Calculate overall score
        overall_score = statistics.mean(score_components) if score_components else 50.0
        
        # Determine performance level
        if overall_score >= 90:
            level = PerformanceLevel.EXCELLENT
        elif overall_score >= 80:
            level = PerformanceLevel.GOOD
        elif overall_score >= 60:
            level = PerformanceLevel.AVERAGE
        elif overall_score >= 40:
            level = PerformanceLevel.POOR
        else:
            level = PerformanceLevel.CRITICAL
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(metrics, bottlenecks)
        
        return PerformanceProfile(
            profile_id=f"profile_{int(time.time() * 1000000)}",
            system_name=self.config.get('system_name', 'testmaster'),
            timestamp=datetime.now(),
            overall_score=overall_score,
            level=level,
            metrics=metrics,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            resource_usage=resource_usage
        )
    
    def _generate_optimization_recommendations(self, metrics: List[PerformanceMetric], 
                                              bottlenecks: List[str]) -> List[OptimizationRecommendation]:
        """Generate intelligent optimization recommendations."""
        recommendations = []
        
        # Analyze metrics for optimization opportunities
        metric_dict = {m.name: m.value for m in metrics}
        
        # CPU optimization recommendations
        if metric_dict.get('cpu_usage', 0) > 70:
            recommendations.append(OptimizationRecommendation(
                optimization_id=f"cpu_opt_{int(time.time())}",
                optimization_type=OptimizationType.CPU_OPTIMIZATION,
                priority="high",
                title="High CPU Usage Detected",
                description="Implement CPU optimization techniques to reduce processor load",
                estimated_improvement=25.0,
                implementation_effort="medium",
                resources_required=["profiling_tools", "optimization_engineer"],
                code_changes="Optimize hot code paths, implement caching, reduce computational complexity",
                testing_required=True
            ))
        
        # Memory optimization recommendations
        if metric_dict.get('memory_usage', 0) > 75:
            recommendations.append(OptimizationRecommendation(
                optimization_id=f"mem_opt_{int(time.time())}",
                optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
                priority="high",
                title="High Memory Usage Detected",
                description="Implement memory optimization to reduce RAM consumption",
                estimated_improvement=30.0,
                implementation_effort="medium",
                resources_required=["memory_profiler", "optimization_engineer"],
                code_changes="Implement object pooling, optimize data structures, fix memory leaks",
                testing_required=True
            ))
        
        # Algorithm optimization for slow response times
        if any('response_time' in metric.name and metric.value > 2.0 for metric in metrics):
            recommendations.append(OptimizationRecommendation(
                optimization_id=f"algo_opt_{int(time.time())}",
                optimization_type=OptimizationType.ALGORITHM_OPTIMIZATION,
                priority="medium",
                title="Slow Response Time Detected",
                description="Optimize algorithms and data access patterns for better response times",
                estimated_improvement=40.0,
                implementation_effort="high",
                resources_required=["algorithm_expert", "performance_testing"],
                code_changes="Implement better algorithms, optimize database queries, add caching layers",
                testing_required=True
            ))
        
        # Use ML engine for advanced recommendations if available
        if self.ml_engine:
            try:
                ml_recommendations = self.ml_engine.generate_optimization_recommendations(metrics)
                recommendations.extend(ml_recommendations)
            except Exception as e:
                logger.warning(f"ML recommendations failed: {e}")
        
        return recommendations
    
    def _check_optimization_opportunities(self, metrics: List[PerformanceMetric]):
        """Check for immediate optimization opportunities."""
        metric_dict = {m.name: m.value for m in metrics}
        
        # Critical performance issues requiring immediate action
        critical_issues = []
        
        if metric_dict.get('cpu_usage', 0) > 95:
            critical_issues.append("CPU usage critical - immediate intervention required")
        
        if metric_dict.get('memory_usage', 0) > 90:
            critical_issues.append("Memory usage critical - risk of OOM errors")
        
        if metric_dict.get('error_rate', 0) > 0.1:
            critical_issues.append("High error rate detected - system stability at risk")
        
        # Log critical issues
        for issue in critical_issues:
            logger.critical(f"PERFORMANCE CRITICAL: {issue}")
    
    def measure_operation_performance(self, operation_name: str, operation_func: Callable, 
                                     *args, **kwargs) -> Dict[str, Any]:
        """Measure the performance of a specific operation."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        performance_data = {
            'operation_name': operation_name,
            'duration': end_time - start_time,
            'memory_delta': (end_memory - start_memory) / (1024 * 1024),  # MB
            'success': success,
            'error': error,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        # Record with enterprise monitor
        if hasattr(self.enterprise_monitor, 'record_operation'):
            self.enterprise_monitor.record_operation(operation_name, performance_data)
        
        return performance_data
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_metrics = self._collect_comprehensive_metrics()
        latest_profile = self._performance_history[-1] if self._performance_history else None
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self._monitoring_active,
            'total_profiles': len(self._performance_history),
            'total_metrics': len(self._metrics_buffer),
            'current_performance': {
                'overall_score': latest_profile.overall_score if latest_profile else 0,
                'level': latest_profile.level.value if latest_profile else 'unknown',
                'bottlenecks': latest_profile.bottlenecks if latest_profile else []
            },
            'system_health': {
                'cpu_usage': next((m.value for m in current_metrics if m.name == 'cpu_usage'), 0),
                'memory_usage': next((m.value for m in current_metrics if m.name == 'memory_usage'), 0),
                'thread_count': next((m.value for m in current_metrics if m.name == 'thread_count'), 0)
            },
            'optimization_opportunities': len(latest_profile.recommendations) if latest_profile else 0
        }
        
        return summary
    
    def get_optimization_recommendations(self, limit: int = 10) -> List[OptimizationRecommendation]:
        """Get top optimization recommendations."""
        if not self._performance_history:
            return []
        
        latest_profile = self._performance_history[-1]
        recommendations = latest_profile.recommendations
        
        # Sort by priority and estimated improvement
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        sorted_recommendations = sorted(
            recommendations,
            key=lambda r: (priority_order.get(r.priority, 0), r.estimated_improvement),
            reverse=True
        )
        
        return sorted_recommendations[:limit]
    
    def export_performance_data(self, format: str = 'json') -> Union[str, Dict]:
        """Export performance data in specified format."""
        data = {
            'summary': self.get_performance_summary(),
            'recent_profiles': [
                {
                    'profile_id': p.profile_id,
                    'timestamp': p.timestamp.isoformat(),
                    'overall_score': p.overall_score,
                    'level': p.level.value,
                    'bottlenecks': p.bottlenecks,
                    'resource_usage': p.resource_usage
                }
                for p in list(self._performance_history)[-10:]
            ],
            'optimization_recommendations': [
                {
                    'optimization_id': r.optimization_id,
                    'type': r.optimization_type.value,
                    'priority': r.priority,
                    'title': r.title,
                    'estimated_improvement': r.estimated_improvement
                }
                for r in self.get_optimization_recommendations()
            ]
        }
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            return data
    
    def register_performance_callback(self, callback: Callable[[PerformanceProfile], None]):
        """Register callback for performance profile updates."""
        self._performance_callbacks.append(callback)
    
    def __repr__(self) -> str:
        """String representation."""
        latest_profile = self._performance_history[-1] if self._performance_history else None
        score = f"{latest_profile.overall_score:.1f}" if latest_profile else "N/A"
        return f"UnifiedPerformanceHub(monitoring={self._monitoring_active}, score={score})"


# Export main class
__all__ = ['UnifiedPerformanceHub', 'PerformanceLevel', 'OptimizationType', 
           'PerformanceMetric', 'OptimizationRecommendation', 'PerformanceProfile']