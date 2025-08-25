"""
Benchmarking Suite for TestMaster Agent QA

Performance benchmarking for agent operations.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import statistics

from core.feature_flags import FeatureFlags

class BenchmarkType(Enum):
    """Types of benchmarks."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    ACCURACY = "accuracy"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"

@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    value: float
    unit: str
    baseline: float
    threshold: float
    status: str

@dataclass
class BenchmarkResult:
    """Benchmark result."""
    agent_id: str
    metrics: List[PerformanceMetric]
    overall_score: float
    status: str
    duration_ms: float = 0.0
    iterations: int = 0
    
class BenchmarkingSuite:
    """Benchmarking suite for agent performance."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer1_test_foundation', 'agent_qa')
        self.lock = threading.RLock()
        self.benchmark_history: Dict[str, List[BenchmarkResult]] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.thresholds = self._setup_default_thresholds()
        
        if not self.enabled:
            return
        
        print("Benchmarking suite initialized")
        print(f"   Benchmark types: {[bt.value for bt in BenchmarkType]}")
    
    def _setup_default_thresholds(self) -> Dict[str, float]:
        """Setup default performance thresholds."""
        return {
            "response_time_ms": 200.0,
            "throughput_ops_sec": 100.0,
            "memory_usage_mb": 100.0,
            "cpu_utilization_percent": 80.0,
            "accuracy_percent": 95.0,
            "reliability_percent": 99.0
        }
    
    def run_benchmarks(
        self,
        agent_id: str,
        benchmark_types: List[BenchmarkType] = None,
        iterations: int = 10
    ) -> BenchmarkResult:
        """
        Run performance benchmarks for an agent.
        
        Args:
            agent_id: Agent identifier
            benchmark_types: Types of benchmarks to run
            iterations: Number of iterations for each benchmark
            
        Returns:
            Benchmark results with performance metrics
        """
        if not self.enabled:
            return BenchmarkResult(agent_id, [], 0.0, "disabled")
        
        start_time = time.time()
        
        # Use all benchmark types if none specified
        if benchmark_types is None:
            benchmark_types = list(BenchmarkType)
        
        metrics = []
        
        for benchmark_type in benchmark_types:
            metric = self._run_single_benchmark(agent_id, benchmark_type, iterations)
            metrics.append(metric)
        
        # Calculate overall score
        overall_score = self._calculate_overall_benchmark_score(metrics)
        
        # Determine status
        status = self._determine_benchmark_status(overall_score)
        
        duration_ms = (time.time() - start_time) * 1000
        
        result = BenchmarkResult(
            agent_id=agent_id,
            metrics=metrics,
            overall_score=overall_score,
            status=status,
            duration_ms=duration_ms,
            iterations=iterations
        )
        
        # Store in history
        with self.lock:
            if agent_id not in self.benchmark_history:
                self.benchmark_history[agent_id] = []
            self.benchmark_history[agent_id].append(result)
        
        print(f"Benchmarking completed for {agent_id}: {overall_score:.3f} ({status}) in {duration_ms:.1f}ms")
        
        return result
    
    def _run_single_benchmark(self, agent_id: str, benchmark_type: BenchmarkType, iterations: int) -> PerformanceMetric:
        """Run a single benchmark test."""
        measurements = []
        
        for _ in range(iterations):
            measurement = self._execute_benchmark(agent_id, benchmark_type)
            measurements.append(measurement)
        
        # Calculate statistics
        avg_value = statistics.mean(measurements)
        baseline = self._get_baseline(agent_id, benchmark_type.value)
        threshold = self.thresholds.get(f"{benchmark_type.value}_{'ms' if 'time' in benchmark_type.value else 'ops_sec' if 'throughput' in benchmark_type.value else 'mb' if 'memory' in benchmark_type.value else 'percent'}", 100.0)
        
        # Determine status
        if benchmark_type in [BenchmarkType.RESPONSE_TIME, BenchmarkType.MEMORY_USAGE, BenchmarkType.CPU_UTILIZATION]:
            # Lower is better
            status = "pass" if avg_value <= threshold else "fail"
        else:
            # Higher is better
            status = "pass" if avg_value >= threshold else "fail"
        
        unit = self._get_metric_unit(benchmark_type)
        
        return PerformanceMetric(
            name=benchmark_type.value,
            value=avg_value,
            unit=unit,
            baseline=baseline,
            threshold=threshold,
            status=status
        )
    
    def _execute_benchmark(self, agent_id: str, benchmark_type: BenchmarkType) -> float:
        """Execute a specific benchmark and return measurement."""
        if benchmark_type == BenchmarkType.RESPONSE_TIME:
            return self._benchmark_response_time(agent_id)
        elif benchmark_type == BenchmarkType.THROUGHPUT:
            return self._benchmark_throughput(agent_id)
        elif benchmark_type == BenchmarkType.MEMORY_USAGE:
            return self._benchmark_memory_usage(agent_id)
        elif benchmark_type == BenchmarkType.CPU_UTILIZATION:
            return self._benchmark_cpu_utilization(agent_id)
        elif benchmark_type == BenchmarkType.ACCURACY:
            return self._benchmark_accuracy(agent_id)
        elif benchmark_type == BenchmarkType.SCALABILITY:
            return self._benchmark_scalability(agent_id)
        elif benchmark_type == BenchmarkType.RELIABILITY:
            return self._benchmark_reliability(agent_id)
        else:
            return 0.0
    
    def _benchmark_response_time(self, agent_id: str) -> float:
        """Benchmark response time."""
        start_time = time.time()
        
        # Simulate agent operation
        time.sleep(0.05)  # 50ms simulated operation
        
        end_time = time.time()
        return (end_time - start_time) * 1000  # Return in milliseconds
    
    def _benchmark_throughput(self, agent_id: str) -> float:
        """Benchmark throughput (operations per second)."""
        operations = 0
        start_time = time.time()
        
        # Simulate operations for 1 second
        while time.time() - start_time < 0.1:  # 100ms test
            operations += 1
            time.sleep(0.001)  # Small delay per operation
        
        duration = time.time() - start_time
        return operations / duration  # Operations per second
    
    def _benchmark_memory_usage(self, agent_id: str) -> float:
        """Benchmark memory usage."""
        # Simulate memory usage measurement
        import sys
        base_size = sys.getsizeof({})
        
        # Create some test data structures
        test_data = {i: f"test_data_{i}" for i in range(1000)}
        memory_used = sys.getsizeof(test_data) / (1024 * 1024)  # Convert to MB
        
        return memory_used
    
    def _benchmark_cpu_utilization(self, agent_id: str) -> float:
        """Benchmark CPU utilization."""
        # Simulate CPU-intensive task
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < 0.01:  # 10ms test
            # Simple CPU work
            _ = sum(i * i for i in range(100))
            iterations += 1
        
        # Return simulated CPU percentage
        return min(iterations / 10.0, 100.0)  # Scale to percentage
    
    def _benchmark_accuracy(self, agent_id: str) -> float:
        """Benchmark accuracy."""
        # Simulate accuracy test
        correct_predictions = 96
        total_predictions = 100
        return (correct_predictions / total_predictions) * 100.0
    
    def _benchmark_scalability(self, agent_id: str) -> float:
        """Benchmark scalability."""
        # Simulate scalability test
        base_response_time = 50.0  # ms
        load_factor = 2.0  # 2x load
        scaled_response_time = base_response_time * (load_factor ** 0.5)  # Sub-linear scaling
        
        # Return scalability score (higher is better)
        return 100.0 / (scaled_response_time / base_response_time)
    
    def _benchmark_reliability(self, agent_id: str) -> float:
        """Benchmark reliability."""
        # Simulate reliability test
        successful_operations = 99
        total_operations = 100
        return (successful_operations / total_operations) * 100.0
    
    def _get_metric_unit(self, benchmark_type: BenchmarkType) -> str:
        """Get unit for benchmark metric."""
        units = {
            BenchmarkType.RESPONSE_TIME: "ms",
            BenchmarkType.THROUGHPUT: "ops/sec",
            BenchmarkType.MEMORY_USAGE: "MB",
            BenchmarkType.CPU_UTILIZATION: "%",
            BenchmarkType.ACCURACY: "%",
            BenchmarkType.SCALABILITY: "score",
            BenchmarkType.RELIABILITY: "%"
        }
        return units.get(benchmark_type, "units")
    
    def _get_baseline(self, agent_id: str, metric_name: str) -> float:
        """Get baseline value for metric."""
        agent_baselines = self.baselines.get(agent_id, {})
        return agent_baselines.get(metric_name, 0.0)
    
    def _calculate_overall_benchmark_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall benchmark score."""
        if not metrics:
            return 0.0
        
        # Calculate weighted score based on metric performance vs threshold
        total_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            # Calculate score as ratio of value to threshold
            if metric.name in ["response_time", "memory_usage", "cpu_utilization"]:
                # Lower is better - score is inverse
                score = min(1.0, metric.threshold / max(metric.value, 0.1))
            else:
                # Higher is better
                score = min(1.0, metric.value / max(metric.threshold, 0.1))
            
            weight = self._get_metric_weight(metric.name)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_metric_weight(self, metric_name: str) -> float:
        """Get weight for metric in overall score calculation."""
        weights = {
            "response_time": 0.25,
            "throughput": 0.2,
            "memory_usage": 0.15,
            "cpu_utilization": 0.15,
            "accuracy": 0.15,
            "scalability": 0.05,
            "reliability": 0.05
        }
        return weights.get(metric_name, 0.1)
    
    def _determine_benchmark_status(self, overall_score: float) -> str:
        """Determine benchmark status from overall score."""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
            return "satisfactory"
        elif overall_score >= 0.6:
            return "needs_improvement"
        else:
            return "poor"
    
    def set_baseline(self, agent_id: str, metric_name: str, value: float):
        """Set baseline value for an agent metric."""
        if agent_id not in self.baselines:
            self.baselines[agent_id] = {}
        self.baselines[agent_id][metric_name] = value
        print(f"Baseline set for {agent_id}.{metric_name}: {value}")
    
    def get_benchmark_history(self, agent_id: str) -> List[BenchmarkResult]:
        """Get benchmark history for an agent."""
        with self.lock:
            return self.benchmark_history.get(agent_id, [])
    
    def get_performance_trends(self, agent_id: str) -> Dict[str, Any]:
        """Get performance trends for an agent."""
        history = self.get_benchmark_history(agent_id)
        if len(history) < 2:
            return {"status": "insufficient_data"}
        
        # Analyze trends for each metric
        trends = {}
        for metric_name in [m.name for m in history[0].metrics]:
            values = []
            for result in history:
                metric = next((m for m in result.metrics if m.name == metric_name), None)
                if metric:
                    values.append(metric.value)
            
            if len(values) >= 2:
                trend = "improving" if values[-1] < values[0] and metric_name in ["response_time", "memory_usage"] else \
                       "improving" if values[-1] > values[0] and metric_name not in ["response_time", "memory_usage"] else \
                       "stable" if abs(values[-1] - values[0]) < 0.1 * values[0] else "declining"
                trends[metric_name] = trend
        
        return {
            "trends": trends,
            "latest_score": history[-1].overall_score,
            "best_score": max(h.overall_score for h in history),
            "average_score": sum(h.overall_score for h in history) / len(history)
        }

def get_benchmarking_suite() -> BenchmarkingSuite:
    """Get benchmarking suite instance."""
    return BenchmarkingSuite()