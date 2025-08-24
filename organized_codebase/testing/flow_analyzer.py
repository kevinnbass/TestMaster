"""
Flow Analyzer for TestMaster Execution Flow Optimizer

Analyzes execution flows to identify bottlenecks and optimization opportunities.
"""

import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import statistics

from core.feature_flags import FeatureFlags

class AnalysisType(Enum):
    """Types of flow analysis."""
    PERFORMANCE = "performance"
    BOTTLENECK = "bottleneck"
    DEPENDENCY = "dependency"
    RESOURCE_USAGE = "resource_usage"
    PARALLELIZATION = "parallelization"

@dataclass
class FlowMetric:
    """Flow performance metric."""
    name: str
    value: float
    unit: str
    baseline: float
    threshold: float
    status: str

@dataclass
class BottleneckInfo:
    """Bottleneck information."""
    location: str
    severity: str
    impact_score: float
    description: str
    recommendations: List[str]

@dataclass
class FlowAnalysis:
    """Flow analysis result."""
    workflow_id: str
    metrics: List[FlowMetric]
    bottlenecks: List[BottleneckInfo]
    efficiency_score: float
    status: str
    optimization_recommendations: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.optimization_recommendations is None:
            self.optimization_recommendations = []

class FlowAnalyzer:
    """Flow analyzer for execution optimization."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'flow_optimizer')
        self.lock = threading.RLock()
        self.analysis_history: Dict[str, List[FlowAnalysis]] = {}
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        if not self.enabled:
            return
        
        print("Flow analyzer initialized")
        print("   Analysis types: performance, bottleneck, dependency, resource_usage, parallelization")
    
    def analyze_flow(
        self,
        workflow_id: str,
        execution_data: List[Dict[str, Any]],
        include_dependencies: bool = True
    ) -> FlowAnalysis:
        """
        Analyze execution flow for optimization opportunities.
        
        Args:
            workflow_id: Workflow identifier
            execution_data: Historical execution data
            include_dependencies: Include dependency analysis
            
        Returns:
            Flow analysis with bottlenecks and recommendations
        """
        if not self.enabled:
            return FlowAnalysis(workflow_id, [], [], 0.0, "disabled")
        
        start_time = time.time()
        
        # Analyze performance metrics
        metrics = self._analyze_performance_metrics(workflow_id, execution_data)
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(workflow_id, execution_data, metrics)
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(metrics, bottlenecks)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(metrics, bottlenecks)
        
        analysis = FlowAnalysis(
            workflow_id=workflow_id,
            metrics=metrics,
            bottlenecks=bottlenecks,
            efficiency_score=efficiency_score,
            status="completed",
            optimization_recommendations=recommendations
        )
        
        # Store in history
        with self.lock:
            if workflow_id not in self.analysis_history:
                self.analysis_history[workflow_id] = []
            self.analysis_history[workflow_id].append(analysis)
        
        analysis_time = time.time() - start_time
        print(f"Flow analysis completed for {workflow_id}: {efficiency_score:.3f} efficiency in {analysis_time*1000:.1f}ms")
        
        return analysis
    
    def _analyze_performance_metrics(self, workflow_id: str, execution_data: List[Dict[str, Any]]) -> List[FlowMetric]:
        """Analyze performance metrics from execution data."""
        metrics = []
        
        if not execution_data:
            return metrics
        
        # Extract timing data
        execution_times = [item.get('execution_time', 0.0) for item in execution_data]
        wait_times = [item.get('wait_time', 0.0) for item in execution_data]
        resource_usage = [item.get('resource_usage', 0.0) for item in execution_data]
        
        # Calculate performance metrics
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
        avg_wait_time = statistics.mean(wait_times) if wait_times else 0.0
        avg_resource_usage = statistics.mean(resource_usage) if resource_usage else 0.0
        
        # Get baselines
        baselines = self.performance_baselines.get(workflow_id, {})
        
        # Create metrics
        metrics.extend([
            FlowMetric(
                name="average_execution_time",
                value=avg_execution_time,
                unit="ms",
                baseline=baselines.get("execution_time", avg_execution_time),
                threshold=500.0,
                status="pass" if avg_execution_time <= 500.0 else "fail"
            ),
            FlowMetric(
                name="average_wait_time",
                value=avg_wait_time,
                unit="ms",
                baseline=baselines.get("wait_time", avg_wait_time),
                threshold=100.0,
                status="pass" if avg_wait_time <= 100.0 else "fail"
            ),
            FlowMetric(
                name="resource_utilization",
                value=avg_resource_usage,
                unit="%",
                baseline=baselines.get("resource_usage", avg_resource_usage),
                threshold=80.0,
                status="pass" if avg_resource_usage <= 80.0 else "fail"
            ),
            FlowMetric(
                name="throughput",
                value=len(execution_data) / max(sum(execution_times), 1.0) * 1000,
                unit="tasks/sec",
                baseline=baselines.get("throughput", 1.0),
                threshold=5.0,
                status="pass" if len(execution_data) / max(sum(execution_times), 1.0) * 1000 >= 5.0 else "fail"
            )
        ])
        
        return metrics
    
    def _detect_bottlenecks(self, workflow_id: str, execution_data: List[Dict[str, Any]], metrics: List[FlowMetric]) -> List[BottleneckInfo]:
        """Detect bottlenecks in execution flow."""
        bottlenecks = []
        
        # Check for high execution time bottlenecks
        exec_time_metric = next((m for m in metrics if m.name == "average_execution_time"), None)
        if exec_time_metric and exec_time_metric.status == "fail":
            bottlenecks.append(BottleneckInfo(
                location="execution_pipeline",
                severity="high",
                impact_score=0.8,
                description=f"High execution time: {exec_time_metric.value:.1f}ms (threshold: {exec_time_metric.threshold:.1f}ms)",
                recommendations=[
                    "Optimize task processing algorithms",
                    "Consider parallel execution",
                    "Review resource allocation"
                ]
            ))
        
        # Check for wait time bottlenecks
        wait_time_metric = next((m for m in metrics if m.name == "average_wait_time"), None)
        if wait_time_metric and wait_time_metric.status == "fail":
            bottlenecks.append(BottleneckInfo(
                location="resource_queue",
                severity="medium",
                impact_score=0.6,
                description=f"High wait time: {wait_time_metric.value:.1f}ms (threshold: {wait_time_metric.threshold:.1f}ms)",
                recommendations=[
                    "Increase resource availability",
                    "Implement priority queuing",
                    "Optimize resource scheduling"
                ]
            ))
        
        # Check for resource utilization bottlenecks
        resource_metric = next((m for m in metrics if m.name == "resource_utilization"), None)
        if resource_metric and resource_metric.status == "fail":
            bottlenecks.append(BottleneckInfo(
                location="resource_pool",
                severity="high",
                impact_score=0.9,
                description=f"High resource utilization: {resource_metric.value:.1f}% (threshold: {resource_metric.threshold:.1f}%)",
                recommendations=[
                    "Scale up resource pool",
                    "Implement load balancing",
                    "Optimize resource usage patterns"
                ]
            ))
        
        # Check for throughput bottlenecks
        throughput_metric = next((m for m in metrics if m.name == "throughput"), None)
        if throughput_metric and throughput_metric.status == "fail":
            bottlenecks.append(BottleneckInfo(
                location="processing_pipeline",
                severity="high",
                impact_score=0.7,
                description=f"Low throughput: {throughput_metric.value:.2f} tasks/sec (threshold: {throughput_metric.threshold:.2f})",
                recommendations=[
                    "Implement parallel processing",
                    "Optimize task batching",
                    "Review pipeline architecture"
                ]
            ))
        
        # Detect dependency bottlenecks
        if execution_data:
            dependency_chains = self._analyze_dependency_chains(execution_data)
            if dependency_chains:
                bottlenecks.append(BottleneckInfo(
                    location="dependency_chain",
                    severity="medium",
                    impact_score=0.5,
                    description=f"Long dependency chains detected: {len(dependency_chains)} chains",
                    recommendations=[
                        "Reduce task dependencies",
                        "Implement parallel execution paths",
                        "Optimize dependency resolution"
                    ]
                ))
        
        return bottlenecks
    
    def _analyze_dependency_chains(self, execution_data: List[Dict[str, Any]]) -> List[List[str]]:
        """Analyze dependency chains in execution data."""
        # Simplified dependency chain analysis
        chains = []
        for item in execution_data:
            dependencies = item.get('dependencies', [])
            if len(dependencies) > 3:  # Consider chains longer than 3 as potential bottlenecks
                chains.append(dependencies)
        return chains
    
    def _calculate_efficiency_score(self, metrics: List[FlowMetric], bottlenecks: List[BottleneckInfo]) -> float:
        """Calculate overall efficiency score."""
        if not metrics:
            return 0.0
        
        # Base score from metrics performance
        metric_scores = []
        for metric in metrics:
            if metric.threshold > 0:
                if metric.name in ["average_execution_time", "average_wait_time", "resource_utilization"]:
                    # Lower is better
                    score = min(1.0, metric.threshold / max(metric.value, 0.1))
                else:
                    # Higher is better
                    score = min(1.0, metric.value / max(metric.threshold, 0.1))
                metric_scores.append(score)
        
        base_score = statistics.mean(metric_scores) if metric_scores else 0.5
        
        # Apply bottleneck penalties
        bottleneck_penalty = 0.0
        for bottleneck in bottlenecks:
            penalty_weight = {
                "low": 0.05,
                "medium": 0.1,
                "high": 0.15,
                "critical": 0.2
            }.get(bottleneck.severity, 0.1)
            
            bottleneck_penalty += penalty_weight * bottleneck.impact_score
        
        final_score = max(0.0, min(1.0, base_score - bottleneck_penalty))
        return final_score
    
    def _generate_optimization_recommendations(self, metrics: List[FlowMetric], bottlenecks: List[BottleneckInfo]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Add bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            recommendations.extend(bottleneck.recommendations)
        
        # Add general recommendations based on metrics
        failed_metrics = [m for m in metrics if m.status == "fail"]
        if failed_metrics:
            recommendations.extend([
                "Review overall system performance",
                "Consider implementing caching mechanisms",
                "Evaluate hardware resource requirements"
            ])
        
        # Add parallelization recommendations
        if any("execution_time" in m.name for m in failed_metrics):
            recommendations.append("Implement parallel execution strategies")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    def set_performance_baseline(self, workflow_id: str, metric_name: str, value: float):
        """Set performance baseline for a workflow metric."""
        if workflow_id not in self.performance_baselines:
            self.performance_baselines[workflow_id] = {}
        
        self.performance_baselines[workflow_id][metric_name] = value
        print(f"Performance baseline set for {workflow_id}.{metric_name}: {value}")
    
    def get_analysis_history(self, workflow_id: str) -> List[FlowAnalysis]:
        """Get analysis history for a workflow."""
        with self.lock:
            return self.analysis_history.get(workflow_id, [])
    
    def get_efficiency_trends(self, workflow_id: str) -> Dict[str, Any]:
        """Get efficiency trends for a workflow."""
        history = self.get_analysis_history(workflow_id)
        if len(history) < 2:
            return {"status": "insufficient_data"}
        
        efficiency_scores = [analysis.efficiency_score for analysis in history]
        return {
            "trend": "improving" if efficiency_scores[-1] > efficiency_scores[0] else "declining",
            "latest_score": efficiency_scores[-1],
            "average_score": statistics.mean(efficiency_scores),
            "best_score": max(efficiency_scores),
            "improvement_rate": (efficiency_scores[-1] - efficiency_scores[0]) / len(efficiency_scores)
        }

def get_flow_analyzer() -> FlowAnalyzer:
    """Get flow analyzer instance."""
    return FlowAnalyzer()