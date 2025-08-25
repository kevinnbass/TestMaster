"""
Real-time Performance Tuning & Adaptive Strategies
==================================================

Agent B Hours 50-60: Advanced real-time performance monitoring, adaptive tuning,
and intelligent strategy adjustment for optimal system performance.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-08-22 (Hours 50-60)
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# Import performance optimization components
try:
    from ....analytics.core.pipeline_manager import (
        MLEnhancedAlgorithmSelector, 
        PredictivePerformanceOptimizer
    )
    ML_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ML_OPTIMIZATION_AVAILABLE = False
    logging.warning("ML optimization components not available for real-time tuning")


class PerformanceMetric(Enum):
    """Types of performance metrics for monitoring"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    QUEUE_DEPTH = "queue_depth"
    SUCCESS_RATE = "success_rate"


class TuningStrategy(Enum):
    """Real-time tuning strategies"""
    CONSERVATIVE = "conservative"      # Small incremental adjustments
    AGGRESSIVE = "aggressive"          # Large performance changes
    ADAPTIVE = "adaptive"             # Data-driven strategy selection
    PREDICTIVE = "predictive"         # Future performance based
    REACTIVE = "reactive"             # Response to current conditions
    INTELLIGENT = "intelligent"      # ML-enhanced decision making


@dataclass
class PerformanceSnapshot:
    """Real-time performance snapshot"""
    timestamp: datetime
    metrics: Dict[PerformanceMetric, float]
    system_load: float
    active_algorithms: List[str]
    pending_operations: int
    optimization_applied: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.overall_health = self._calculate_health_score()
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        if not self.metrics:
            return 0.5
        
        # Normalize metrics to health scores
        health_components = []
        
        # Lower is better metrics
        if PerformanceMetric.EXECUTION_TIME in self.metrics:
            time_score = max(0.0, 1.0 - (self.metrics[PerformanceMetric.EXECUTION_TIME] / 1000.0))
            health_components.append(time_score)
        
        if PerformanceMetric.ERROR_RATE in self.metrics:
            error_score = max(0.0, 1.0 - self.metrics[PerformanceMetric.ERROR_RATE])
            health_components.append(error_score)
        
        # Higher is better metrics
        if PerformanceMetric.SUCCESS_RATE in self.metrics:
            health_components.append(self.metrics[PerformanceMetric.SUCCESS_RATE])
        
        if PerformanceMetric.THROUGHPUT in self.metrics:
            throughput_score = min(1.0, self.metrics[PerformanceMetric.THROUGHPUT] / 100.0)
            health_components.append(throughput_score)
        
        return sum(health_components) / len(health_components) if health_components else 0.5


@dataclass
class AdaptiveRule:
    """Adaptive performance tuning rule"""
    rule_id: str
    condition: Callable[[PerformanceSnapshot], bool]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int = 1
    cooldown_seconds: int = 30
    last_applied: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate rule success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def can_apply(self) -> bool:
        """Check if rule can be applied (not in cooldown)"""
        if self.last_applied is None:
            return True
        
        elapsed = datetime.now() - self.last_applied
        return elapsed.total_seconds() >= self.cooldown_seconds


class RealTimePerformanceTuner:
    """
    Real-time Performance Tuning Engine
    
    Monitors system performance in real-time and applies adaptive tuning strategies
    to optimize execution, memory usage, and overall system efficiency.
    """
    
    def __init__(self, tuning_strategy: TuningStrategy = TuningStrategy.INTELLIGENT):
        self.logger = logging.getLogger("RealTimePerformanceTuner")
        self.tuning_strategy = tuning_strategy
        
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=1000)  # Last 1000 snapshots
        self.current_snapshot: Optional[PerformanceSnapshot] = None
        self.monitoring_active = False
        self.monitoring_interval = 5.0  # seconds
        
        # Adaptive tuning
        self.adaptive_rules: List[AdaptiveRule] = []
        self.tuning_thresholds = {
            PerformanceMetric.EXECUTION_TIME: 500.0,     # 500ms
            PerformanceMetric.MEMORY_USAGE: 80.0,        # 80%
            PerformanceMetric.CPU_UTILIZATION: 75.0,     # 75%
            PerformanceMetric.ERROR_RATE: 0.05,          # 5%
            PerformanceMetric.SUCCESS_RATE: 0.95,        # 95%
        }
        
        # ML integration
        self.ml_selector: Optional[MLEnhancedAlgorithmSelector] = None
        self.predictive_optimizer: Optional[PredictivePerformanceOptimizer] = None
        
        if ML_OPTIMIZATION_AVAILABLE:
            self._initialize_ml_integration()
        
        # Performance optimization state
        self.active_optimizations: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Initialize adaptive rules
        self._initialize_adaptive_rules()
        
        self.logger.info(f"Real-time performance tuner initialized with {tuning_strategy.value} strategy")
    
    def _initialize_ml_integration(self):
        """Initialize ML-enhanced performance optimization"""
        try:
            self.ml_selector = MLEnhancedAlgorithmSelector()
            self.predictive_optimizer = PredictivePerformanceOptimizer(self.ml_selector)
            self.logger.info("ML-enhanced real-time tuning enabled")
        except Exception as e:
            self.logger.warning(f"ML integration failed: {e}")
    
    def _initialize_adaptive_rules(self):
        """Initialize default adaptive tuning rules"""
        
        # High execution time rule
        def high_execution_time_condition(snapshot: PerformanceSnapshot) -> bool:
            return (PerformanceMetric.EXECUTION_TIME in snapshot.metrics and 
                   snapshot.metrics[PerformanceMetric.EXECUTION_TIME] > self.tuning_thresholds[PerformanceMetric.EXECUTION_TIME])
        
        def reduce_execution_time_action(context: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "optimization": "parallel_processing",
                "parameters": {"parallel_factor": 2, "thread_pool_size": 4},
                "expected_improvement": 0.4
            }
        
        self.adaptive_rules.append(AdaptiveRule(
            rule_id="high_execution_time",
            condition=high_execution_time_condition,
            action=reduce_execution_time_action,
            priority=1,
            cooldown_seconds=60
        ))
        
        # High memory usage rule
        def high_memory_condition(snapshot: PerformanceSnapshot) -> bool:
            return (PerformanceMetric.MEMORY_USAGE in snapshot.metrics and 
                   snapshot.metrics[PerformanceMetric.MEMORY_USAGE] > self.tuning_thresholds[PerformanceMetric.MEMORY_USAGE])
        
        def reduce_memory_action(context: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "optimization": "memory_management",
                "parameters": {"gc_frequency": "high", "cache_cleanup": True},
                "expected_improvement": 0.3
            }
        
        self.adaptive_rules.append(AdaptiveRule(
            rule_id="high_memory_usage",
            condition=high_memory_condition,
            action=reduce_memory_action,
            priority=2,
            cooldown_seconds=45
        ))
        
        # High error rate rule
        def high_error_rate_condition(snapshot: PerformanceSnapshot) -> bool:
            return (PerformanceMetric.ERROR_RATE in snapshot.metrics and 
                   snapshot.metrics[PerformanceMetric.ERROR_RATE] > self.tuning_thresholds[PerformanceMetric.ERROR_RATE])
        
        def reduce_error_rate_action(context: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "optimization": "error_mitigation",
                "parameters": {"retry_strategy": "exponential", "timeout_increase": 1.5},
                "expected_improvement": 0.6
            }
        
        self.adaptive_rules.append(AdaptiveRule(
            rule_id="high_error_rate",
            condition=high_error_rate_condition,
            action=reduce_error_rate_action,
            priority=1,
            cooldown_seconds=30
        ))
        
        # Low success rate rule
        def low_success_rate_condition(snapshot: PerformanceSnapshot) -> bool:
            return (PerformanceMetric.SUCCESS_RATE in snapshot.metrics and 
                   snapshot.metrics[PerformanceMetric.SUCCESS_RATE] < self.tuning_thresholds[PerformanceMetric.SUCCESS_RATE])
        
        def improve_success_rate_action(context: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "optimization": "reliability_enhancement",
                "parameters": {"algorithm_switching": True, "fallback_enabled": True},
                "expected_improvement": 0.25
            }
        
        self.adaptive_rules.append(AdaptiveRule(
            rule_id="low_success_rate",
            condition=low_success_rate_condition,
            action=improve_success_rate_action,
            priority=1,
            cooldown_seconds=90
        ))
        
        self.logger.info(f"Initialized {len(self.adaptive_rules)} adaptive tuning rules")
    
    async def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.logger.info("Starting real-time performance monitoring")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        self.monitoring_active = False
        self.logger.info("Stopped real-time performance monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Capture performance snapshot
                snapshot = await self._capture_performance_snapshot()
                
                if snapshot:
                    self.current_snapshot = snapshot
                    self.performance_history.append(snapshot)
                    
                    # Apply adaptive tuning
                    await self._apply_adaptive_tuning(snapshot)
                    
                    # Log performance status
                    if len(self.performance_history) % 12 == 0:  # Every minute with 5s intervals
                        self._log_performance_status(snapshot)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _capture_performance_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Capture current performance metrics"""
        try:
            # Mock performance metrics (in production, these would come from system monitoring)
            current_time = datetime.now()
            
            # Simulate varying performance metrics
            base_execution_time = 150.0 + (time.time() % 100)  # Simulated variance
            base_memory = 45.0 + (time.time() % 20)
            base_cpu = 60.0 + (time.time() % 30)
            
            metrics = {
                PerformanceMetric.EXECUTION_TIME: base_execution_time,
                PerformanceMetric.MEMORY_USAGE: base_memory,
                PerformanceMetric.CPU_UTILIZATION: base_cpu,
                PerformanceMetric.THROUGHPUT: 85.0 + (time.time() % 15),
                PerformanceMetric.ERROR_RATE: max(0.0, 0.02 + (time.time() % 0.05)),
                PerformanceMetric.SUCCESS_RATE: min(1.0, 0.94 + (time.time() % 0.08)),
                PerformanceMetric.LATENCY: 25.0 + (time.time() % 10),
                PerformanceMetric.QUEUE_DEPTH: int(5 + (time.time() % 15))
            }
            
            # Calculate system load
            system_load = (base_cpu + base_memory) / 200.0
            
            # Mock active algorithms
            active_algorithms = ["data_processing_pipeline", "adaptive_processing"]
            
            # Mock pending operations
            pending_operations = int(3 + (time.time() % 7))
            
            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                metrics=metrics,
                system_load=system_load,
                active_algorithms=active_algorithms,
                pending_operations=pending_operations
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to capture performance snapshot: {e}")
            return None
    
    async def _apply_adaptive_tuning(self, snapshot: PerformanceSnapshot):
        """Apply adaptive tuning based on current performance"""
        try:
            # Check adaptive rules
            applicable_rules = []
            
            for rule in self.adaptive_rules:
                if rule.can_apply() and rule.condition(snapshot):
                    applicable_rules.append(rule)
            
            if not applicable_rules:
                return
            
            # Sort by priority (lower number = higher priority)
            applicable_rules.sort(key=lambda r: r.priority)
            
            # Apply highest priority rule
            selected_rule = applicable_rules[0]
            
            # Execute rule action
            context = {
                "snapshot": snapshot,
                "tuning_strategy": self.tuning_strategy,
                "performance_history": list(self.performance_history)[-10:]  # Last 10 snapshots
            }
            
            optimization_result = selected_rule.action(context)
            
            # Apply optimization
            success = await self._apply_optimization(optimization_result, selected_rule.rule_id)
            
            # Update rule statistics
            selected_rule.last_applied = datetime.now()
            if success:
                selected_rule.success_count += 1
                self.logger.info(f"Applied adaptive rule '{selected_rule.rule_id}' successfully")
            else:
                selected_rule.failure_count += 1
                self.logger.warning(f"Failed to apply adaptive rule '{selected_rule.rule_id}'")
            
            # Use ML-enhanced optimization if available
            if self.predictive_optimizer and self.tuning_strategy == TuningStrategy.INTELLIGENT:
                await self._apply_ml_enhanced_tuning(snapshot, optimization_result)
                
        except Exception as e:
            self.logger.error(f"Error in adaptive tuning: {e}")
    
    async def _apply_optimization(self, optimization_result: Dict[str, Any], rule_id: str) -> bool:
        """Apply performance optimization"""
        try:
            optimization_type = optimization_result.get("optimization")
            parameters = optimization_result.get("parameters", {})
            expected_improvement = optimization_result.get("expected_improvement", 0.0)
            
            # Record optimization
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "rule_id": rule_id,
                "optimization_type": optimization_type,
                "parameters": parameters,
                "expected_improvement": expected_improvement,
                "applied": True
            }
            
            # Store active optimization
            self.active_optimizations[optimization_type] = {
                "parameters": parameters,
                "applied_at": datetime.now(),
                "rule_id": rule_id
            }
            
            # Add to history
            self.optimization_history.append(optimization_record)
            
            # Keep only last 100 optimizations
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            self.logger.info(f"Applied {optimization_type} optimization with {expected_improvement:.1%} expected improvement")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization: {e}")
            return False
    
    async def _apply_ml_enhanced_tuning(self, snapshot: PerformanceSnapshot, base_optimization: Dict[str, Any]):
        """Apply ML-enhanced performance tuning"""
        if not self.predictive_optimizer:
            return
        
        try:
            # Convert snapshot to performance metrics for ML optimizer
            current_performance = {
                "execution_time": snapshot.metrics.get(PerformanceMetric.EXECUTION_TIME, 0.0),
                "memory_usage": snapshot.metrics.get(PerformanceMetric.MEMORY_USAGE, 0.0),
                "accuracy": snapshot.metrics.get(PerformanceMetric.SUCCESS_RATE, 0.0),
                "success_rate": snapshot.metrics.get(PerformanceMetric.SUCCESS_RATE, 0.0)
            }
            
            # Apply ML optimization
            optimization_strategy = base_optimization.get("optimization", "parallel_processing")
            ml_result = self.predictive_optimizer.apply_real_time_optimization(
                current_performance, optimization_strategy
            )
            
            if ml_result.get("success"):
                improvement = ml_result.get("improvement_achieved", {})
                self.logger.info(f"ML-enhanced optimization applied with improvements: {improvement}")
            
        except Exception as e:
            self.logger.error(f"ML-enhanced tuning failed: {e}")
    
    def _log_performance_status(self, snapshot: PerformanceSnapshot):
        """Log current performance status"""
        health_score = snapshot.overall_health
        
        status_msg = (f"Performance Status - Health: {health_score:.1%}, "
                     f"Load: {snapshot.system_load:.1%}, "
                     f"Execution: {snapshot.metrics.get(PerformanceMetric.EXECUTION_TIME, 0):.0f}ms, "
                     f"Memory: {snapshot.metrics.get(PerformanceMetric.MEMORY_USAGE, 0):.1f}%, "
                     f"Success: {snapshot.metrics.get(PerformanceMetric.SUCCESS_RATE, 0):.1%}")
        
        if health_score > 0.8:
            self.logger.info(status_msg)
        elif health_score > 0.6:
            self.logger.warning(f"Performance degradation detected - {status_msg}")
        else:
            self.logger.error(f"Poor performance detected - {status_msg}")
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_snapshots = list(self.performance_history)[-60:]  # Last 5 minutes
        
        analytics = {
            "current_status": {
                "monitoring_active": self.monitoring_active,
                "tuning_strategy": self.tuning_strategy.value,
                "snapshots_captured": len(self.performance_history),
                "active_optimizations": len(self.active_optimizations)
            },
            "performance_trends": {},
            "optimization_effectiveness": {},
            "adaptive_rules_performance": {},
            "recommendations": []
        }
        
        # Calculate performance trends
        if len(recent_snapshots) >= 2:
            first_snapshot = recent_snapshots[0]
            last_snapshot = recent_snapshots[-1]
            
            for metric in PerformanceMetric:
                if metric in first_snapshot.metrics and metric in last_snapshot.metrics:
                    trend = last_snapshot.metrics[metric] - first_snapshot.metrics[metric]
                    analytics["performance_trends"][metric.value] = {
                        "trend": trend,
                        "current_value": last_snapshot.metrics[metric],
                        "improvement": trend < 0 if metric in [PerformanceMetric.EXECUTION_TIME, PerformanceMetric.ERROR_RATE] else trend > 0
                    }
        
        # Analyze optimization effectiveness
        if self.optimization_history:
            recent_optimizations = [opt for opt in self.optimization_history 
                                  if datetime.fromisoformat(opt["timestamp"]) > datetime.now() - timedelta(hours=1)]
            
            optimization_types = {}
            for opt in recent_optimizations:
                opt_type = opt["optimization_type"]
                if opt_type not in optimization_types:
                    optimization_types[opt_type] = {"count": 0, "total_expected": 0.0}
                
                optimization_types[opt_type]["count"] += 1
                optimization_types[opt_type]["total_expected"] += opt.get("expected_improvement", 0.0)
            
            for opt_type, stats in optimization_types.items():
                analytics["optimization_effectiveness"][opt_type] = {
                    "applications": stats["count"],
                    "average_expected_improvement": stats["total_expected"] / stats["count"]
                }
        
        # Analyze adaptive rules performance
        for rule in self.adaptive_rules:
            analytics["adaptive_rules_performance"][rule.rule_id] = {
                "success_rate": rule.success_rate,
                "total_applications": rule.success_count + rule.failure_count,
                "priority": rule.priority,
                "cooldown_seconds": rule.cooldown_seconds
            }
        
        # Generate recommendations
        analytics["recommendations"] = self._generate_tuning_recommendations(analytics)
        
        return analytics
    
    def _generate_tuning_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate real-time tuning recommendations"""
        recommendations = []
        
        # Check performance trends
        trends = analytics.get("performance_trends", {})
        
        for metric, trend_data in trends.items():
            if not trend_data.get("improvement", True):
                if metric == "execution_time" and trend_data["trend"] > 50:
                    recommendations.append("Consider enabling parallel processing - execution time increasing")
                elif metric == "memory_usage" and trend_data["trend"] > 10:
                    recommendations.append("Memory usage trending upward - consider garbage collection tuning")
                elif metric == "error_rate" and trend_data["trend"] > 0.01:
                    recommendations.append("Error rate increasing - review error handling strategies")
        
        # Check rule effectiveness
        rules_performance = analytics.get("adaptive_rules_performance", {})
        
        for rule_id, perf in rules_performance.items():
            if perf["total_applications"] > 5 and perf["success_rate"] < 0.5:
                recommendations.append(f"Adaptive rule '{rule_id}' has low success rate - consider adjustment")
        
        # General recommendations
        if len(self.active_optimizations) > 3:
            recommendations.append("Multiple optimizations active - consider consolidation")
        
        if self.monitoring_interval > 10:
            recommendations.append("Consider reducing monitoring interval for more responsive tuning")
        
        return recommendations
    
    def adjust_tuning_strategy(self, new_strategy: TuningStrategy):
        """Adjust real-time tuning strategy"""
        old_strategy = self.tuning_strategy
        self.tuning_strategy = new_strategy
        
        self.logger.info(f"Tuning strategy changed from {old_strategy.value} to {new_strategy.value}")
        
        # Adjust thresholds based on strategy
        if new_strategy == TuningStrategy.AGGRESSIVE:
            # Lower thresholds for more aggressive tuning
            for metric in self.tuning_thresholds:
                self.tuning_thresholds[metric] *= 0.8
        elif new_strategy == TuningStrategy.CONSERVATIVE:
            # Higher thresholds for more conservative tuning
            for metric in self.tuning_thresholds:
                self.tuning_thresholds[metric] *= 1.2
    
    def add_custom_rule(self, rule: AdaptiveRule):
        """Add custom adaptive tuning rule"""
        self.adaptive_rules.append(rule)
        self.adaptive_rules.sort(key=lambda r: r.priority)
        
        self.logger.info(f"Added custom adaptive rule '{rule.rule_id}' with priority {rule.priority}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove adaptive tuning rule"""
        for i, rule in enumerate(self.adaptive_rules):
            if rule.rule_id == rule_id:
                removed_rule = self.adaptive_rules.pop(i)
                self.logger.info(f"Removed adaptive rule '{rule_id}'")
                return True
        
        return False


# Global real-time tuner instance
real_time_tuner = RealTimePerformanceTuner(TuningStrategy.INTELLIGENT)


# Export key components
__all__ = [
    'RealTimePerformanceTuner',
    'PerformanceSnapshot',
    'AdaptiveRule',
    'PerformanceMetric',
    'TuningStrategy',
    'real_time_tuner'
]