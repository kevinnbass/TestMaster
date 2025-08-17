"""
Agent Quality Assurance for TestMaster

Comprehensive quality assurance system for agent operations including:
- Automated quality testing
- Agent validation and scoring
- Quality metrics collection
- Continuous quality monitoring
- Agent performance benchmarking
"""

from typing import Dict, Any, List, Optional, Union
from .quality_inspector import (
    QualityInspector, QualityReport, QualityMetric,
    QualityLevel, QualityCheck, get_quality_inspector
)
from .validation_engine import (
    ValidationEngine, ValidationResult, ValidationRule,
    ValidationType, get_validation_engine
)
from .scoring_system import (
    ScoringSystem, QualityScore, ScoreCategory,
    ScoreWeight, get_scoring_system
)
from .benchmarking_suite import (
    BenchmarkingSuite, BenchmarkResult, BenchmarkType,
    PerformanceMetric, get_benchmarking_suite
)
from .quality_monitor import (
    QualityMonitor, QualityAlert, AlertType,
    QualityThreshold, get_quality_monitor
)
from ..core.feature_flags import FeatureFlags

# Global instances
_quality_inspector = None
_validation_engine = None
_scoring_system = None
_benchmarking_suite = None
_quality_monitor = None

def is_agent_qa_enabled() -> bool:
    """Check if agent quality assurance is enabled."""
    return FeatureFlags.is_enabled('layer1_test_foundation', 'agent_qa')

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
    if not is_agent_qa_enabled():
        return {"status": "disabled", "reason": "agent_qa feature not enabled"}
    
    global _quality_inspector, _validation_engine, _scoring_system
    global _benchmarking_suite, _quality_monitor
    
    # Initialize components
    _quality_inspector = get_quality_inspector()
    _validation_engine = get_validation_engine()
    _scoring_system = get_scoring_system()
    
    if enable_benchmarking:
        _benchmarking_suite = get_benchmarking_suite()
    
    if enable_monitoring:
        _quality_monitor = get_quality_monitor()
        _quality_monitor.start_monitoring()
    
    config = {
        "similarity_threshold": similarity_threshold,
        "benchmarking_enabled": enable_benchmarking,
        "monitoring_enabled": enable_monitoring,
        "alert_threshold": alert_threshold
    }
    
    print(f"Agent QA configured: {config}")
    return {"status": "configured", "config": config}

def inspect_agent_quality(
    agent_id: str,
    test_cases: List[Dict[str, Any]] = None,
    include_benchmarks: bool = True
) -> QualityReport:
    """
    Perform comprehensive quality inspection of an agent.
    
    Args:
        agent_id: ID of the agent to inspect
        test_cases: Optional test cases for validation
        include_benchmarks: Whether to include benchmark results
        
    Returns:
        Quality report with detailed analysis
    """
    if not is_agent_qa_enabled():
        return QualityReport(agent_id, [], 0.0, "disabled")
    
    inspector = get_quality_inspector()
    return inspector.inspect_agent(agent_id, test_cases, include_benchmarks)

def validate_agent_output(
    agent_id: str,
    output: Any,
    expected: Any = None,
    validation_rules: List[ValidationRule] = None
) -> ValidationResult:
    """
    Validate agent output against rules and expectations.
    
    Args:
        agent_id: ID of the agent
        output: Agent output to validate
        expected: Expected output for comparison
        validation_rules: Custom validation rules
        
    Returns:
        Validation result with score and details
    """
    if not is_agent_qa_enabled():
        return ValidationResult("disabled", False, 0.0, [])
    
    engine = get_validation_engine()
    return engine.validate_output(agent_id, output, expected, validation_rules)

def score_agent_quality(
    agent_id: str,
    quality_metrics: List[QualityMetric],
    custom_weights: Dict[str, float] = None
) -> QualityScore:
    """
    Calculate comprehensive quality score for an agent.
    
    Args:
        agent_id: ID of the agent
        quality_metrics: List of quality metrics
        custom_weights: Custom weights for score calculation
        
    Returns:
        Quality score with breakdown
    """
    if not is_agent_qa_enabled():
        return QualityScore(agent_id, 0.0, {}, "disabled")
    
    scoring = get_scoring_system()
    return scoring.calculate_score(agent_id, quality_metrics, custom_weights)

def benchmark_agent_performance(
    agent_id: str,
    benchmark_types: List[BenchmarkType] = None,
    iterations: int = 10
) -> BenchmarkResult:
    """
    Run performance benchmarks for an agent.
    
    Args:
        agent_id: ID of the agent
        benchmark_types: Types of benchmarks to run
        iterations: Number of benchmark iterations
        
    Returns:
        Benchmark results with performance metrics
    """
    if not is_agent_qa_enabled():
        return BenchmarkResult(agent_id, [], 0.0, "disabled")
    
    benchmarking = get_benchmarking_suite()
    return benchmarking.run_benchmarks(agent_id, benchmark_types, iterations)

def get_quality_status() -> Dict[str, Any]:
    """
    Get current quality status across all agents.
    
    Returns:
        Quality status summary
    """
    if not is_agent_qa_enabled():
        return {"status": "disabled"}
    
    monitor = get_quality_monitor()
    return monitor.get_status() if monitor else {"status": "not_configured"}

def shutdown_agent_qa():
    """Shutdown agent quality assurance system."""
    global _quality_inspector, _validation_engine, _scoring_system
    global _benchmarking_suite, _quality_monitor
    
    if _quality_monitor:
        _quality_monitor.stop_monitoring()
    
    # Reset instances
    _quality_inspector = None
    _validation_engine = None
    _scoring_system = None
    _benchmarking_suite = None
    _quality_monitor = None
    
    print("Agent QA shutdown completed")

# Convenience aliases
inspect_quality = inspect_agent_quality
validate_output = validate_agent_output
calculate_score = score_agent_quality
run_benchmarks = benchmark_agent_performance