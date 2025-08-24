"""
Performance Optimization Engine
===============================
"""Analysis Module - Split from performance_optimization_engine.py"""


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


            }
            
            # Get current metrics
            current_metrics = self.metrics_collector.get_all_metrics()
            
            # Analyze each pattern
            for pattern_name, analyzer in self.analysis_patterns.items():
                try:
                    pattern_result = await analyzer(current_metrics)
                    if pattern_result:
                        analysis_results["bottlenecks_detected"].append(pattern_result)
                except Exception as e:
                    self.logger.error(f"Pattern analysis failed for {pattern_name}: {e}")
            
            # Generate performance summary
            performance_summary = self._generate_performance_summary(current_metrics)
            analysis_results["performance_summary"] = performance_summary
            
            # Calculate overall health score
            health_score = self._calculate_health_score(current_metrics, analysis_results["bottlenecks_detected"])
            analysis_results["overall_health_score"] = health_score
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                analysis_results["bottlenecks_detected"], 
                performance_summary
            )
            analysis_results["recommendations"] = recommendations
            
            # Perform trend analysis
            trend_analysis = await self._perform_trend_analysis()
            analysis_results["trend_analysis"] = trend_analysis
            
            # Record analysis duration
            analysis_results["analysis_duration_ms"] = (time.time() - start_time) * 1000
            
            self.logger.info(f"Performance analysis completed: health score {health_score:.1f}")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "analysis_duration_ms": (time.time() - start_time) * 1000
            }
    
    async def _analyze_high_cpu(self, metrics: Dict[str, PerformanceMetric]) -> Optional[Dict[str, Any]]:
        """Analyze high CPU usage patterns"""
        cpu_metric = metrics.get("system.cpu.usage_percent")
        
        if cpu_metric and cpu_metric.value > 80:
            return {
                "type": "high_cpu",
                "severity": "critical" if cpu_metric.value > 90 else "warning",
                "current_value": cpu_metric.value,
                "threshold": 80,
                "description": f"CPU usage at {cpu_metric.value:.1f}%",
                "impact": "high",
                "affected_systems": ["all"]
            }
        
        return None
    
    async def _analyze_high_memory(self, metrics: Dict[str, PerformanceMetric]) -> Optional[Dict[str, Any]]:
        """Analyze high memory usage patterns"""
        memory_metric = metrics.get("system.memory.usage_percent")
        
        if memory_metric and memory_metric.value > 85:
            return {
                "type": "high_memory",
                "severity": "critical" if memory_metric.value > 95 else "warning",
                "current_value": memory_metric.value,
                "threshold": 85,
                "description": f"Memory usage at {memory_metric.value:.1f}%",
                "impact": "high",
                "affected_systems": ["all"]
            }
        
        return None
    
    async def _analyze_slow_response(self, metrics: Dict[str, PerformanceMetric]) -> Optional[Dict[str, Any]]:
        """Analyze slow response time patterns"""
        api_response_time = metrics.get("api_gateway.response_time_ms")
        
        if api_response_time and api_response_time.value > 1000:
            return {
                "type": "slow_response",
                "severity": "critical" if api_response_time.value > 2000 else "warning",
                "current_value": api_response_time.value,
                "threshold": 1000,
                "description": f"API response time at {api_response_time.value:.1f}ms",
                "impact": "medium",
                "affected_systems": ["api_gateway"]
            }
        
        return None
    
    async def _analyze_high_error_rate(self, metrics: Dict[str, PerformanceMetric]) -> Optional[Dict[str, Any]]:
        """Analyze high error rate patterns"""
        error_rate = metrics.get("api_gateway.error_rate_percent")
        
        if error_rate and error_rate.value > 5:
            return {
                "type": "high_error_rate",
                "severity": "critical" if error_rate.value > 10 else "warning",
                "current_value": error_rate.value,
                "threshold": 5,
                "description": f"Error rate at {error_rate.value:.1f}%",
                "impact": "high",
                "affected_systems": ["api_gateway"]
            }
        
        return None
    
    async def _analyze_resource_contention(self, metrics: Dict[str, PerformanceMetric]) -> Optional[Dict[str, Any]]:
        """Analyze resource contention patterns"""
        queue_depth = metrics.get("orchestration.queue_depth")
        active_workflows = metrics.get("orchestration.workflows_active")
        
        if queue_depth and active_workflows:
            contention_ratio = queue_depth.value / max(active_workflows.value, 1)
            
            if contention_ratio > 2:
                return {
                    "type": "resource_contention",
                    "severity": "warning",
                    "current_value": contention_ratio,
                    "threshold": 2,
                    "description": f"High queue depth to active workflows ratio: {contention_ratio:.1f}",
                    "impact": "medium",
                    "affected_systems": ["orchestration"]
                }
        
        return None
    
    async def _analyze_capacity_limits(self, metrics: Dict[str, PerformanceMetric]) -> Optional[Dict[str, Any]]:
        """Analyze capacity limit patterns"""
        requests_per_second = metrics.get("api_gateway.requests_per_second")
        
        if requests_per_second and requests_per_second.value > 80:  # Assume 100 req/s is max capacity
            utilization = (requests_per_second.value / 100) * 100
            
            if utilization > 80:
                return {
                    "type": "capacity_limits",
                    "severity": "warning" if utilization < 95 else "critical",
                    "current_value": utilization,
                    "threshold": 80,
                    "description": f"API Gateway capacity utilization at {utilization:.1f}%",
                    "impact": "high",
                    "affected_systems": ["api_gateway"]
                }
        
        return None
    
    def _generate_performance_summary(self, metrics: Dict[str, PerformanceMetric]) -> Dict[str, Any]:
        """Generate performance summary from metrics"""
        summary = {
            "cpu_usage_percent": 0.0,
            "memory_usage_percent": 0.0,
            "api_response_time_ms": 0.0,
            "api_throughput_rps": 0.0,
            "error_rate_percent": 0.0,
            "active_workflows": 0,
            "messages_per_second": 0.0
        }
        
        # Extract key metrics
        if "system.cpu.usage_percent" in metrics:
            summary["cpu_usage_percent"] = metrics["system.cpu.usage_percent"].value
        
        if "system.memory.usage_percent" in metrics:
            summary["memory_usage_percent"] = metrics["system.memory.usage_percent"].value
        
        if "api_gateway.response_time_ms" in metrics:
            summary["api_response_time_ms"] = metrics["api_gateway.response_time_ms"].value
        
        if "api_gateway.requests_per_second" in metrics:
            summary["api_throughput_rps"] = metrics["api_gateway.requests_per_second"].value
        
        if "api_gateway.error_rate_percent" in metrics:
            summary["error_rate_percent"] = metrics["api_gateway.error_rate_percent"].value
        
        if "orchestration.workflows_active" in metrics:
            summary["active_workflows"] = metrics["orchestration.workflows_active"].value
        
        if "integration.messages_per_second" in metrics:
            summary["messages_per_second"] = metrics["integration.messages_per_second"].value
        
        return summary
    
    def _calculate_health_score(self, metrics: Dict[str, PerformanceMetric], 
                               bottlenecks: List[Dict[str, Any]]) -> float:
        """Calculate overall system health score"""
        base_score = 100.0
        
        # Deduct points for bottlenecks
        for bottleneck in bottlenecks:
            severity = bottleneck.get("severity", "warning")
            
            if severity == "critical":
                base_score -= 20
            elif severity == "warning":
                base_score -= 10
            else:
                base_score -= 5
        
        # Additional deductions for specific metrics
        cpu_metric = metrics.get("system.cpu.usage_percent")
        if cpu_metric:
            if cpu_metric.value > 90:
                base_score -= 15
            elif cpu_metric.value > 80:
                base_score -= 10
        
        memory_metric = metrics.get("system.memory.usage_percent")
        if memory_metric:
            if memory_metric.value > 95:
                base_score -= 15
            elif memory_metric.value > 85:
                base_score -= 10
        
        return max(0.0, base_score)
    
    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]], 
                                performance_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Recommendations based on bottlenecks
        for bottleneck in bottlenecks:
            bottleneck_type = bottleneck.get("type", "unknown")
            
            if bottleneck_type == "high_cpu":
                recommendations.append({
                    "title": "Optimize CPU Usage",
                    "description": "CPU usage is high. Consider scaling horizontally or optimizing algorithms.",
                    "priority": 8,
                    "category": "resource_optimization",
                    "actions": [
                        "Review CPU-intensive processes",
                        "Consider horizontal scaling",
                        "Optimize algorithms and queries",
                        "Implement caching strategies"
                    ]
                })
            
            elif bottleneck_type == "high_memory":
                recommendations.append({
                    "title": "Optimize Memory Usage",
                    "description": "Memory usage is high. Consider memory optimization strategies.",
                    "priority": 7,
                    "category": "resource_optimization",
                    "actions": [
                        "Review memory leaks",
                        "Optimize data structures",
                        "Implement memory caching",
                        "Consider memory scaling"
                    ]
                })
            
            elif bottleneck_type == "slow_response":
                recommendations.append({
                    "title": "Improve Response Times",
                    "description": "API response times are slow. Consider performance optimizations.",
                    "priority": 6,
                    "category": "performance_tuning",
                    "actions": [
                        "Optimize database queries",
                        "Implement response caching",
                        "Review third-party service calls",
                        "Consider CDN implementation"
                    ]
                })
            
            elif bottleneck_type == "high_error_rate":
                recommendations.append({
                    "title": "Reduce Error Rate",
                    "description": "High error rate detected. Investigate and fix issues.",
                    "priority": 9,
                    "category": "reliability",
                    "actions": [
                        "Review error logs",
                        "Implement better error handling",
                        "Add circuit breakers",
                        "Improve input validation"
                    ]
                })
        
        # General recommendations based on performance summary
        if performance_summary.get("api_throughput_rps", 0) > 80:
            recommendations.append({
                "title": "Scale API Gateway",
                "description": "API Gateway approaching capacity limits.",
                "priority": 7,
                "category": "scaling",
                "actions": [
                    "Add more API Gateway instances",
                    "Implement load balancing",
                    "Consider rate limiting adjustments",
                    "Monitor capacity trends"
                ]
            })
        
        return recommendations
    
    async def _perform_trend_analysis(self) -> Dict[str, Any]:
        """Perform trend analysis on historical data"""
        try:
            # Get historical data for key metrics
            cpu_history = self.metrics_collector.get_metric_history("system.cpu.usage_percent", 60)
            memory_history = self.metrics_collector.get_metric_history("system.memory.usage_percent", 60)
            response_time_history = self.metrics_collector.get_metric_history("api_gateway.response_time_ms", 60)
            
            trends = {}
            
            # Analyze CPU trend
            if len(cpu_history) > 10:
                cpu_values = [entry["value"] for entry in cpu_history[-10:]]
                cpu_trend = "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
                cpu_slope = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
                trends["cpu"] = {"trend": cpu_trend, "slope": cpu_slope}
            
            # Analyze memory trend
            if len(memory_history) > 10:
                memory_values = [entry["value"] for entry in memory_history[-10:]]
                memory_trend = "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
                memory_slope = (memory_values[-1] - memory_values[0]) / len(memory_values)
                trends["memory"] = {"trend": memory_trend, "slope": memory_slope}
            
            # Analyze response time trend
            if len(response_time_history) > 10:
                response_values = [entry["value"] for entry in response_time_history[-10:]]
                response_trend = "increasing" if response_values[-1] > response_values[0] else "decreasing"
                response_slope = (response_values[-1] - response_values[0]) / len(response_values)
                trends["response_time"] = {"trend": response_trend, "slope": response_slope}
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {}


# ============================================================================
# PERFORMANCE OPTIMIZATION ENGINE
# ============================================================================

class PerformanceOptimizationEngine:
    """
    Comprehensive performance optimization engine with real-time monitoring,
    analysis, and automated optimization recommendations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("performance_optimization_engine")
        
        # Core components
        self.metrics_collector = RealTimeMetricsCollector()
        self.analyzer = PerformanceAnalyzer(self.metrics_collector)
        