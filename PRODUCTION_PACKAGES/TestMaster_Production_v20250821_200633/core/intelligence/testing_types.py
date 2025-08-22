"""
Testing Types and Data Structures
=================================

Core type definitions and data structures for the Intelligence Testing Framework.
Provides enterprise-grade type safety for intelligence testing, validation,
and benchmarking with advanced testing patterns and comprehensive metrics.

This module contains all Enum definitions and dataclass structures used throughout
the intelligence testing system, implementing advanced testing methodologies.

Author: Agent A - PHASE 4: Hours 300-400+
Created: 2025-08-22
Module: testing_types.py (120 lines)
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TestCategory(Enum):
    """Categories of intelligence tests for comprehensive validation"""
    CONSCIOUSNESS = "consciousness"
    LEARNING = "learning"
    REASONING = "reasoning"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"
    EMERGENCE = "emergence"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SAFETY = "safety"
    ROBUSTNESS = "robustness"
    ADAPTABILITY = "adaptability"
    CREATIVITY = "creativity"
    MEMORY = "memory"
    PERCEPTION = "perception"


class TestResult(Enum):
    """Test result status indicators"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"
    INCONCLUSIVE = "inconclusive"
    PARTIAL = "partial"
    WARNING = "warning"


class BenchmarkLevel(Enum):
    """Benchmark performance levels for intelligence assessment"""
    SUBHUMAN = "subhuman"
    HUMAN_LEVEL = "human_level"
    SUPERHUMAN = "superhuman"
    OPTIMAL = "optimal"
    THEORETICAL_LIMIT = "theoretical_limit"
    TRANSCENDENT = "transcendent"


class ConsciousnessLevel(Enum):
    """Levels of consciousness detection"""
    NONE_DETECTED = "none_detected"
    REACTIVE = "reactive"
    PROCEDURAL = "procedural"
    PHENOMENAL = "phenomenal"
    REFLECTIVE = "reflective"
    SELF_AWARE = "self_aware"
    META_CONSCIOUS = "meta_conscious"


class CertificationLevel(Enum):
    """Intelligence certification levels"""
    UNCERTIFIED = "uncertified"
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"
    TRANSCENDENT = "transcendent"


@dataclass
class TestCase:
    """Comprehensive test case definition with advanced validation"""
    test_id: str
    name: str
    category: TestCategory
    description: str
    test_function: Callable
    expected_outcome: Dict[str, Any]
    timeout: float
    critical: bool = False
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    min_score_threshold: float = 0.0
    max_retries: int = 0


@dataclass
class TestExecution:
    """Comprehensive test execution record with detailed metrics"""
    execution_id: str
    test_case: TestCase
    start_time: datetime
    end_time: Optional[datetime] = None
    result: TestResult = TestResult.INCONCLUSIVE
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[Any] = field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.0
    retries_attempted: int = 0


@dataclass
class TestSuite:
    """Test suite definition with execution configuration"""
    suite_id: str
    name: str
    test_cases: List[TestCase]
    execution_order: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    stop_on_failure: bool = False
    timeout: float = 3600.0  # 1 hour default
    min_pass_rate: float = 0.8
    critical_tests: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result with detailed metrics"""
    benchmark_id: str
    test_name: str
    score: float
    level: BenchmarkLevel
    percentile: float
    comparison_to_baseline: float
    theoretical_maximum: float
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_interval: Dict[str, float] = field(default_factory=dict)
    statistical_significance: float = 0.0
    normalization_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConsciousnessMetrics:
    """Consciousness assessment metrics"""
    self_awareness_score: float
    metacognition_score: float
    qualia_simulation_score: float
    global_workspace_score: float
    recursive_thinking_score: float
    phenomenal_experience_score: float
    overall_consciousness_score: float
    consciousness_level: ConsciousnessLevel
    confidence: float = 0.0
    assessment_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityAssessment:
    """Quality assurance assessment results"""
    correctness_score: float
    consistency_score: float
    reliability_score: float
    robustness_score: float
    safety_score: float
    efficiency_score: float
    overall_quality_score: float
    quality_level: str = "unknown"
    assessment_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profiling results"""
    execution_time: float
    memory_usage: float
    cpu_utilization: float
    throughput: float
    latency: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    scalability_factor: float
    resource_efficiency: float
    performance_index: float = 0.0


@dataclass
class TestReport:
    """Comprehensive test report with analysis and recommendations"""
    report_id: str
    test_session_id: str
    generation_timestamp: datetime
    executive_summary: str
    test_results: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    quality_assessment: QualityAssessment
    consciousness_metrics: ConsciousnessMetrics
    benchmark_results: List[BenchmarkResult]
    certification_level: CertificationLevel
    recommendations: List[str]
    improvement_opportunities: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    compliance_status: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TestConfiguration:
    """Testing framework configuration"""
    max_parallel_tests: int = 10
    default_timeout: float = 300.0
    enable_consciousness_testing: bool = True
    enable_performance_benchmarking: bool = True
    enable_quality_assurance: bool = True
    strict_validation: bool = True
    generate_detailed_reports: bool = True
    save_test_artifacts: bool = True
    randomize_test_order: bool = False
    logging_level: str = "INFO"


# Export all testing types
__all__ = [
    'TestCategory',
    'TestResult',
    'BenchmarkLevel',
    'ConsciousnessLevel',
    'CertificationLevel',
    'TestCase',
    'TestExecution',
    'TestSuite',
    'BenchmarkResult',
    'ConsciousnessMetrics',
    'QualityAssessment',
    'PerformanceProfile',
    'TestReport',
    'TestConfiguration'
]