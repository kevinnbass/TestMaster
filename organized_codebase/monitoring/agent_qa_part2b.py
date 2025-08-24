            return "satisfactory"
        elif overall_score >= 0.6:
            return "needs_improvement"
        else:
            return "poor"
    
    def _calculate_percentile(self, agent_id: str, current_score: float) -> float:
        """Calculate percentile ranking for the score."""
        # Get all historical scores
        all_scores = []
        for scores in self.scoring_history.values():
            all_scores.extend([s.overall_score for s in scores])
        
        if not all_scores:
            return 50.0  # Default percentile
        
        # Calculate percentile
        scores_below = sum(1 for score in all_scores if score < current_score)
        percentile = (scores_below / len(all_scores)) * 100
        
        return percentile
    
    # =========================================================================
    # Performance Benchmarking
    # =========================================================================
    
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
        
        self._log(f"Benchmarking completed for {agent_id}: {overall_score:.3f} ({status}) in {duration_ms:.1f}ms")
        
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
        threshold = self.performance_thresholds.get(f"{benchmark_type.value}_{'ms' if 'time' in benchmark_type.value else 'ops_sec' if 'throughput' in benchmark_type.value else 'mb' if 'memory' in benchmark_type.value else 'percent'}", 100.0)
        
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
        
        # Simulate operations for 100ms
        while time.time() - start_time < 0.1:
            operations += 1
            time.sleep(0.001)  # Small delay per operation
        
        duration = time.time() - start_time
        return operations / duration  # Operations per second
    
    def _benchmark_memory_usage(self, agent_id: str) -> float:
        """Benchmark memory usage."""
        # Simulate memory usage measurement
        import sys
        
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
            
            weight = self._get_benchmark_metric_weight(metric.name)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_benchmark_metric_weight(self, metric_name: str) -> float:
        """Get weight for metric in overall benchmark score calculation."""
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
    
    # =========================================================================
    # Configuration and Management
    # =========================================================================
    
    def add_threshold(self, threshold: QualityThreshold):
        """Add custom quality threshold."""
        self.thresholds.append(threshold)
        self._log(f"Added quality threshold: {threshold.name}")
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Add callback for quality alerts."""
        self.alert_callbacks.append(callback)
    
    def add_custom_validation_rule(self, category: str, rule: ValidationRule):
        """Add custom validation rule."""
        if category not in self.validation_rules:
            self.validation_rules[category] = []
        self.validation_rules[category].append(rule)
        self._log(f"Added custom validation rule: {rule.name} to {category}")
    
    def set_baseline(self, agent_id: str, metric_name: str, value: float):
        """Set baseline value for an agent metric."""
        if agent_id not in self.baselines:
            self.baselines[agent_id] = {}
        self.baselines[agent_id][metric_name] = value
        self._log(f"Baseline set for {agent_id}.{metric_name}: {value}")
    
    def set_benchmark(self, benchmark_name: str, score: float):
        """Set a benchmark score for comparison."""
        self.benchmarks[benchmark_name] = score
        self._log(f"Benchmark set: {benchmark_name} = {score:.3f}")
    
    # =========================================================================
    # Status and History
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        with self.lock:
            total_alerts = len(self.alerts)
            unacknowledged_alerts = len([a for a in self.alerts if not a.acknowledged])
            critical_alerts = len([a for a in self.alerts if a.severity == "critical"])
            
            return {
                "enabled": self.enabled,
                "monitoring": self.monitoring,
                "monitored_agents": len(self.agent_metrics),
                "total_alerts": total_alerts,
                "unacknowledged_alerts": unacknowledged_alerts,
                "critical_alerts": critical_alerts,
                "thresholds": len(self.thresholds),
                "monitoring_interval": self.monitoring_interval,
                "validation_rules": sum(len(rules) for rules in self.validation_rules.values()),
                "benchmarks": len(self.benchmarks)
            }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status for a specific agent."""
        agent_alerts = [a for a in self.alerts if a.agent_id == agent_id]
        recent_alerts = [a for a in agent_alerts if (datetime.now() - a.timestamp).days < 1]
        
        metrics = self.agent_metrics.get(agent_id, {})
        latest_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                latest_metrics[metric_name] = values[-1]
        
        return {
            "agent_id": agent_id,
            "total_alerts": len(agent_alerts),
            "recent_alerts": len(recent_alerts),
            "latest_metrics": latest_metrics,
            "metrics_count": sum(len(values) for values in metrics.values()),
            "inspections": len(self.inspection_history.get(agent_id, [])),
            "validations": len(self.validation_history.get(agent_id, [])),
            "benchmarks": len(self.benchmark_history.get(agent_id, []))
        }
    
    def get_alerts(self, agent_id: str = None, severity: str = None, since: datetime = None) -> List[QualityAlert]:
        """Get quality alerts with optional filtering."""
        alerts = self.alerts.copy()
        
        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    self._log(f"Alert acknowledged: {alert_id}")
                    break
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.lock:
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_date]
        
        self._log(f"Cleared alerts older than {days} days")
    
    def get_inspection_history(self, agent_id: str) -> List[QualityReport]:
        """Get inspection history for an agent."""
        with self.lock:
            return self.inspection_history.get(agent_id, [])
    
    def get_validation_history(self, agent_id: str) -> List[ValidationResult]:
        """Get validation history for an agent."""
        with self.lock:
            return self.validation_history.get(agent_id, [])
    
    def get_scoring_history(self, agent_id: str) -> List[QualityScore]:
        """Get scoring history for an agent."""
        with self.lock:
            return self.scoring_history.get(agent_id, [])
    
    def get_benchmark_history(self, agent_id: str) -> List[BenchmarkResult]:
        """Get benchmark history for an agent."""
        with self.lock:
            return self.benchmark_history.get(agent_id, [])
    
    def shutdown(self):
        """Shutdown the Agent QA system."""
        self.stop_monitoring()
        self._log("Agent QA system shutdown completed")


# =============================================================================
# Factory Functions and Convenience Interface
# =============================================================================

# Global instance
_agent_qa_instance: Optional[AgentQualityAssurance] = None


def get_agent_qa(enable_monitoring: bool = True) -> AgentQualityAssurance:
    """Get or create the global Agent QA instance."""
    global _agent_qa_instance
    if _agent_qa_instance is None:
        _agent_qa_instance = AgentQualityAssurance(enable_monitoring=enable_monitoring)
    return _agent_qa_instance


def configure_agent_qa(
    similarity_threshold: float = 0.7,
    enable_benchmarking: bool = True,
    enable_monitoring: bool = True,
    alert_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Configure agent quality assurance system.
    
    Args:
        similarity_threshold: Threshold for quality similarity checks
        enable_benchmarking: Enable performance benchmarking
        enable_monitoring: Enable continuous quality monitoring
        alert_threshold: Threshold for quality alerts
        
    Returns:
        Configuration status
    """
    qa_system = get_agent_qa(enable_monitoring=enable_monitoring)
    
    config = {
        "similarity_threshold": similarity_threshold,
        "benchmarking_enabled": enable_benchmarking,
        "monitoring_enabled": enable_monitoring,
        "alert_threshold": alert_threshold
    }
    
    return {"status": "configured", "config": config}


# =============================================================================
# Convenience Functions
# =============================================================================

def inspect_agent_quality(
    agent_id: str,
    test_cases: List[Dict[str, Any]] = None,
    include_benchmarks: bool = True
) -> QualityReport:
    """Perform comprehensive quality inspection of an agent."""
    qa_system = get_agent_qa()
    return qa_system.inspect_agent(agent_id, test_cases, include_benchmarks)


def validate_agent_output(
    agent_id: str,
    output: Any,
    expected: Any = None,
    validation_rules: List[ValidationRule] = None
) -> ValidationResult:
    """Validate agent output against rules and expectations."""
    qa_system = get_agent_qa()
    return qa_system.validate_output(agent_id, output, expected, validation_rules)


def score_agent_quality(
    agent_id: str,
    quality_metrics: List[QualityMetric],
    custom_weights: Dict[str, float] = None
) -> QualityScore:
    """Calculate comprehensive quality score for an agent."""
    qa_system = get_agent_qa()
    return qa_system.calculate_score(agent_id, quality_metrics, custom_weights)


def benchmark_agent_performance(
    agent_id: str,
    benchmark_types: List[BenchmarkType] = None,
    iterations: int = 10
) -> BenchmarkResult:
    """Run performance benchmarks for an agent."""
    qa_system = get_agent_qa()
    return qa_system.run_benchmarks(agent_id, benchmark_types, iterations)


def get_quality_status() -> Dict[str, Any]:
    """Get current quality status across all agents."""
    qa_system = get_agent_qa()
    return qa_system.get_status()


def shutdown_agent_qa():
    """Shutdown agent quality assurance system."""
    global _agent_qa_instance
    if _agent_qa_instance:
        _agent_qa_instance.shutdown()
        _agent_qa_instance = None


# Convenience aliases
inspect_quality = inspect_agent_quality
validate_output = validate_agent_output
calculate_score = score_agent_quality
run_benchmarks = benchmark_agent_performance