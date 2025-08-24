"""
Performance Optimization Engine
===============================
"""Processing Module - Split from performance_optimization_engine.py"""


import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
import psutil
import collections


# ============================================================================
# PERFORMANCE MONITORING TYPES
# ============================================================================


        # Optimization state
        self.optimization_strategy = OptimizationStrategy.ADAPTIVE
        self.auto_optimization_enabled = True
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Performance baselines
        self.performance_baselines = {
            "api_response_time_ms": 100.0,
            "cpu_usage_percent": 50.0,
            "memory_usage_percent": 60.0,
            "error_rate_percent": 1.0,
            "throughput_rps": 50.0
        }
        
        # Optimization rules
        self.optimization_rules = {
            "cpu_scaling": {
                "trigger_threshold": 80.0,
                "scale_factor": 1.5,
                "cooldown_minutes": 15
            },
            "memory_optimization": {
                "trigger_threshold": 85.0,
                "cleanup_threshold": 90.0,
                "cache_reduction_factor": 0.8
            },
            "response_time_optimization": {
                "trigger_threshold": 1000.0,
                "cache_increase_factor": 1.2,
                "timeout_reduction_factor": 0.9
            }
        }
        
        # Monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        self.logger.info("Performance optimization engine initialized")
    
    async def start_optimization(self):
        """Start performance optimization monitoring"""
        try:
            # Start metrics collection
            await self.metrics_collector.start_collection()
            
            # Start optimization monitoring
            self.monitoring_tasks["optimization"] = asyncio.create_task(self._optimization_loop())
            
            # Start alert monitoring
            self.monitoring_tasks["alerts"] = asyncio.create_task(self._alert_monitoring_loop())
            
            self.logger.info("Performance optimization started")
            
        except Exception as e:
            self.logger.error(f"Failed to start optimization: {e}")
            raise e
    
    async def stop_optimization(self):
        """Stop performance optimization"""
        try:
            # Stop monitoring tasks
            for task_name, task in self.monitoring_tasks.items():
                try:
                    task.cancel()
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            self.monitoring_tasks.clear()
            
            # Stop metrics collection
            await self.metrics_collector.stop_collection()
            
            self.logger.info("Performance optimization stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop optimization: {e}")
    
    async def _optimization_loop(self):
        """Main optimization monitoring loop"""
        while True:
            try:
                # Perform performance analysis
                analysis_result = await self.analyzer.analyze_system_performance()
                
                # Apply optimizations if enabled
                if self.auto_optimization_enabled:
                    optimization_actions = await self._apply_optimizations(analysis_result)
                    
                    if optimization_actions:
                        self.optimization_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "analysis": analysis_result,
                            "actions": optimization_actions
                        })
                        
                        # Keep only last 100 optimization records
                        if len(self.optimization_history) > 100:
                            self.optimization_history = self.optimization_history[-100:]
                
                # Wait for next optimization cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_monitoring_loop(self):
        """Monitor for performance alerts"""
        while True:
            try:
                current_metrics = self.metrics_collector.get_all_metrics()
                
                # Check for critical alerts
                alerts = self._check_performance_alerts(current_metrics)
                
                for alert in alerts:
                    await self._handle_performance_alert(alert)
                
                # Wait for next alert check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(30)
    
    def _check_performance_alerts(self, metrics: Dict[str, PerformanceMetric]) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        
        # CPU usage alert
        cpu_metric = metrics.get("system.cpu.usage_percent")
        if cpu_metric and cpu_metric.value > 90:
            alerts.append({
                "type": "cpu_critical",
                "severity": AlertSeverity.CRITICAL,
                "message": f"CPU usage critical: {cpu_metric.value:.1f}%",
                "metric": "system.cpu.usage_percent",
                "value": cpu_metric.value,
                "threshold": 90
            })
        
        # Memory usage alert
        memory_metric = metrics.get("system.memory.usage_percent")
        if memory_metric and memory_metric.value > 95:
            alerts.append({
                "type": "memory_critical",
                "severity": AlertSeverity.CRITICAL,
                "message": f"Memory usage critical: {memory_metric.value:.1f}%",
                "metric": "system.memory.usage_percent",
                "value": memory_metric.value,
                "threshold": 95
            })
        
        # Error rate alert
        error_rate = metrics.get("api_gateway.error_rate_percent")
        if error_rate and error_rate.value > 10:
            alerts.append({
                "type": "error_rate_critical",
                "severity": AlertSeverity.CRITICAL,
                "message": f"Error rate critical: {error_rate.value:.1f}%",
                "metric": "api_gateway.error_rate_percent",
                "value": error_rate.value,
                "threshold": 10
            })
        
        return alerts
    
    async def _handle_performance_alert(self, alert: Dict[str, Any]):
        """Handle performance alert"""
        self.logger.warning(f"Performance alert: {alert['message']}")
        
        # Apply immediate optimizations based on alert type
        alert_type = alert.get("type", "")
        
        if alert_type == "cpu_critical":
            await self._apply_cpu_optimization()
        elif alert_type == "memory_critical":
            await self._apply_memory_optimization()
        elif alert_type == "error_rate_critical":
            await self._apply_error_rate_optimization()
    
    async def _apply_optimizations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply optimization actions based on analysis"""
        actions = []
        
        bottlenecks = analysis_result.get("bottlenecks_detected", [])
        
        for bottleneck in bottlenecks:
            bottleneck_type = bottleneck.get("type", "")
            severity = bottleneck.get("severity", "warning")
            
            if severity in ["critical", "warning"]:
                if bottleneck_type == "high_cpu":
                    action = await self._apply_cpu_optimization()
                    if action:
                        actions.append(action)
                
                elif bottleneck_type == "high_memory":
                    action = await self._apply_memory_optimization()
                    if action:
                        actions.append(action)
                
                elif bottleneck_type == "slow_response":
                    action = await self._apply_response_time_optimization()
                    if action:
                        actions.append(action)
        
        return actions
    
    async def _apply_cpu_optimization(self) -> Optional[Dict[str, Any]]:
        """Apply CPU optimization"""
        try:
            # Mock CPU optimization actions
            optimization_action = {
                "type": "cpu_optimization",
                "action": "reduce_processing_intensity",
                "description": "Reduced processing intensity to lower CPU usage",
                "timestamp": datetime.now().isoformat(),
                "expected_impact": "10-15% CPU reduction"
            }
            
            self.logger.info("Applied CPU optimization")
            return optimization_action
            
        except Exception as e:
            self.logger.error(f"CPU optimization failed: {e}")
            return None
    
    async def _apply_memory_optimization(self) -> Optional[Dict[str, Any]]:
        """Apply memory optimization"""
        try:
            # Mock memory optimization actions
            optimization_action = {
                "type": "memory_optimization",
                "action": "cache_cleanup",
                "description": "Performed cache cleanup to free memory",
                "timestamp": datetime.now().isoformat(),
                "expected_impact": "15-20% memory reduction"
            }
            
            self.logger.info("Applied memory optimization")
            return optimization_action
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return None
    
    async def _apply_response_time_optimization(self) -> Optional[Dict[str, Any]]:
        """Apply response time optimization"""
        try:
            # Mock response time optimization actions
            optimization_action = {
                "type": "response_time_optimization",
                "action": "increase_cache_size",
                "description": "Increased cache size to improve response times",
                "timestamp": datetime.now().isoformat(),
                "expected_impact": "20-30% response time improvement"
            }
            
            self.logger.info("Applied response time optimization")
            return optimization_action
            
        except Exception as e:
            self.logger.error(f"Response time optimization failed: {e}")
            return None
    
    async def _apply_error_rate_optimization(self) -> Optional[Dict[str, Any]]:
        """Apply error rate optimization"""
        try:
            # Mock error rate optimization actions
            optimization_action = {
                "type": "error_rate_optimization",
                "action": "increase_timeout_values",
                "description": "Increased timeout values to reduce timeout errors",
                "timestamp": datetime.now().isoformat(),
                "expected_impact": "5-10% error rate reduction"
            }
            
            self.logger.info("Applied error rate optimization")
            return optimization_action
            
        except Exception as e:
            self.logger.error(f"Error rate optimization failed: {e}")
            return None
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        try:
            # Get current metrics
            current_metrics = self.metrics_collector.get_all_metrics()
            
            # Get latest analysis
            analysis_result = await self.analyzer.analyze_system_performance()
            
            # Calculate performance scores
            performance_scores = self._calculate_performance_scores(current_metrics)
            
            return {
                "optimization_engine_status": "running" if self.monitoring_tasks else "stopped",
                "auto_optimization_enabled": self.auto_optimization_enabled,
                "optimization_strategy": self.optimization_strategy.value,
                "current_metrics": {name: metric.value for name, metric in current_metrics.items()},
                "performance_scores": performance_scores,
                "latest_analysis": analysis_result,
                "optimization_history_count": len(self.optimization_history),
                "active_monitoring_tasks": len(self.monitoring_tasks)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization status: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_scores(self, metrics: Dict[str, PerformanceMetric]) -> Dict[str, float]:
        """Calculate performance scores for key metrics"""
        scores = {}
        
        # CPU score (inverse relationship - lower usage = higher score)
        cpu_metric = metrics.get("system.cpu.usage_percent")
        if cpu_metric:
            scores["cpu_score"] = max(0, 100 - cpu_metric.value)
        
        # Memory score
        memory_metric = metrics.get("system.memory.usage_percent")
        if memory_metric:
            scores["memory_score"] = max(0, 100 - memory_metric.value)
        
        # Response time score
        response_time = metrics.get("api_gateway.response_time_ms")
        if response_time:
            baseline = self.performance_baselines["api_response_time_ms"]
            scores["response_time_score"] = max(0, 100 - ((response_time.value / baseline) * 50))
        
        # Error rate score
        error_rate = metrics.get("api_gateway.error_rate_percent")
        if error_rate:
            scores["error_rate_score"] = max(0, 100 - (error_rate.value * 10))
        
        # Overall score
        if scores:
            scores["overall_score"] = statistics.mean(scores.values())
        
        return scores


# ============================================================================
# GLOBAL OPTIMIZATION ENGINE INSTANCE
# ============================================================================

# Global instance for performance optimization
performance_optimization_engine = PerformanceOptimizationEngine()

# Export for external use
__all__ = [
    'MetricType',
    'OptimizationStrategy',
    'AlertSeverity',
    'PerformanceMetric',
    'PerformanceProfile',
    'OptimizationRecommendation',
    'RealTimeMetricsCollector',
    'PerformanceAnalyzer',
    'PerformanceOptimizationEngine',
    'performance_optimization_engine'
]