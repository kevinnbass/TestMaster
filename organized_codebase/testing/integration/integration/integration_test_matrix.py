"""
Integration Test Matrix for comprehensive interface testing.
Tests all module-to-module interactions and data flows.
"""

from typing import Dict, List, Any, Tuple, Type, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import inspect
import asyncio
from pathlib import Path
import json
import time
from datetime import datetime

from src.interfaces import (
    IPatternGenerator, IPatternEvaluator, IPatternOptimizer,
    ILLMProvider, IProviderFactory, IPatternDeduplicator,
    IPatternValidator, IPatternConsolidator, IPerformanceAnalyzer,
    ICostAnalyzer, IQualityAnalyzer, ICache, ILogger,
    IConfiguration, IPipeline, IPipelineOrchestrator, IExporter,
    Pattern, GenerationRequest, EvaluationMetrics, OptimizationConfig
)


class TestStatus(Enum):
    """Status of integration test."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class IntegrationTestCase:
    """Represents a single integration test case."""
    test_id: str
    name: str
    description: str
    interfaces: List[Type]
    test_function: Optional[callable] = None
    dependencies: List[str] = field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    execution_time: float = 0.0
    error_message: Optional[str] = None
    result_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of integration test execution."""
    test_id: str
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class IntegrationTestMatrix:
    """
    Comprehensive integration test matrix covering all module interfaces.
    """
    
    def __init__(self):
        self.test_cases: Dict[str, IntegrationTestCase] = {}
        self.test_results: Dict[str, TestResult] = {}
        self.interface_map: Dict[Type, Any] = {}  # Interface -> Implementation
        self.test_data: Dict[str, Any] = {}
        
        # Initialize test matrix
        self._initialize_test_matrix()
        
    def _initialize_test_matrix(self) -> None:
        """Initialize the complete test matrix."""
        
        # Provider Integration Tests
        self._create_provider_tests()
        
        # Pattern Processing Tests
        self._create_pattern_processing_tests()
        
        # Analysis Integration Tests
        self._create_analysis_tests()
        
        # Pipeline Integration Tests
        self._create_pipeline_tests()
        
        # Utility Integration Tests
        self._create_utility_tests()
        
        # Cross-Module Integration Tests
        self._create_cross_module_tests()
        
        # Data Flow Tests
        self._create_data_flow_tests()
        
        # Error Propagation Tests
        self._create_error_propagation_tests()
        
        # Performance Integration Tests
        self._create_performance_integration_tests()
        
    # ========================================================================
    # Provider Integration Tests
    # ========================================================================
    
    def _create_provider_tests(self) -> None:
        """Create tests for provider interfaces."""
        
        # Test: Provider Factory -> Provider
        self.test_cases["provider_factory_creation"] = IntegrationTestCase(
            test_id="provider_factory_creation",
            name="Provider Factory Creation",
            description="Test that provider factory creates valid providers",
            interfaces=[IProviderFactory, ILLMProvider],
            test_function=self._test_provider_factory_creation
        )
        
        # Test: Provider -> Pattern Generator
        self.test_cases["provider_to_generator"] = IntegrationTestCase(
            test_id="provider_to_generator",
            name="Provider to Generator Integration",
            description="Test provider integration with pattern generator",
            interfaces=[ILLMProvider, IPatternGenerator],
            test_function=self._test_provider_to_generator
        )
        
        # Test: Multiple Providers -> Cost Analyzer
        self.test_cases["providers_cost_tracking"] = IntegrationTestCase(
            test_id="providers_cost_tracking",
            name="Provider Cost Tracking",
            description="Test cost tracking across multiple providers",
            interfaces=[ILLMProvider, ICostAnalyzer],
            test_function=self._test_providers_cost_tracking
        )
        
    def _test_provider_factory_creation(self) -> TestResult:
        """Test provider factory creation."""
        try:
            factory = self.interface_map.get(IProviderFactory)
            if not factory:
                return TestResult("provider_factory_creation", TestStatus.SKIPPED, 0.0, "No factory implementation")
                
            start_time = time.time()
            
            # Test creating different provider types
            available_providers = factory.get_available_providers()
            
            for provider_type in available_providers[:2]:  # Test first 2
                models = factory.get_available_models(provider_type)
                if models:
                    provider = factory.create_provider(provider_type, models[0])
                    
                    # Verify provider implements interface
                    assert isinstance(provider, ILLLProvider)
                    assert provider.provider_type == provider_type
                    
            execution_time = time.time() - start_time
            
            return TestResult(
                "provider_factory_creation", 
                TestStatus.PASSED, 
                execution_time,
                details={"providers_tested": len(available_providers[:2])}
            )
            
        except Exception as e:
            return TestResult("provider_factory_creation", TestStatus.ERROR, 0.0, str(e))
            
    def _test_provider_to_generator(self) -> TestResult:
        """Test provider to generator integration."""
        try:
            provider = self.interface_map.get(ILLMProvider)
            generator = self.interface_map.get(IPatternGenerator)
            
            if not provider or not generator:
                return TestResult("provider_to_generator", TestStatus.SKIPPED, 0.0, "Missing implementations")
                
            start_time = time.time()
            
            # Create test request
            request = GenerationRequest(
                label="test_pattern",
                positive_examples=["urgent message", "critical alert"],
                negative_examples=["normal text", "regular message"],
                n_patterns=3
            )
            
            # Test generation
            patterns = generator.generate_patterns(request)
            
            # Verify results
            assert isinstance(patterns, list)
            assert len(patterns) <= request.n_patterns
            assert all(isinstance(p, Pattern) for p in patterns)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "provider_to_generator",
                TestStatus.PASSED,
                execution_time,
                details={"patterns_generated": len(patterns)}
            )
            
        except Exception as e:
            return TestResult("provider_to_generator", TestStatus.ERROR, 0.0, str(e))
            
    def _test_providers_cost_tracking(self) -> TestResult:
        """Test cost tracking across providers."""
        try:
            cost_analyzer = self.interface_map.get(ICostAnalyzer)
            if not cost_analyzer:
                return TestResult("providers_cost_tracking", TestStatus.SKIPPED, 0.0, "No cost analyzer")
                
            start_time = time.time()
            
            # Simulate API usage
            from src.interfaces import ProviderType
            cost_analyzer.track_api_usage(ProviderType.OPENROUTER, 100, 50)
            cost_analyzer.track_api_usage(ProviderType.ANTHROPIC, 200, 75)
            
            total_cost = cost_analyzer.calculate_total_cost()
            report = cost_analyzer.generate_cost_report()
            
            assert total_cost >= 0
            assert isinstance(report, dict)
            assert 'total_cost' in report
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "providers_cost_tracking",
                TestStatus.PASSED,
                execution_time,
                details={"total_cost": total_cost}
            )
            
        except Exception as e:
            return TestResult("providers_cost_tracking", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Pattern Processing Tests
    # ========================================================================
    
    def _create_pattern_processing_tests(self) -> None:
        """Create pattern processing integration tests."""
        
        # Test: Generator -> Evaluator
        self.test_cases["generator_to_evaluator"] = IntegrationTestCase(
            test_id="generator_to_evaluator",
            name="Generator to Evaluator",
            description="Test pattern generator to evaluator integration",
            interfaces=[IPatternGenerator, IPatternEvaluator],
            test_function=self._test_generator_to_evaluator
        )
        
        # Test: Evaluator -> Optimizer
        self.test_cases["evaluator_to_optimizer"] = IntegrationTestCase(
            test_id="evaluator_to_optimizer",
            name="Evaluator to Optimizer",
            description="Test evaluator to optimizer integration",
            interfaces=[IPatternEvaluator, IPatternOptimizer],
            test_function=self._test_evaluator_to_optimizer
        )
        
        # Test: Multiple Patterns -> Deduplicator
        self.test_cases["patterns_deduplication"] = IntegrationTestCase(
            test_id="patterns_deduplication",
            name="Pattern Deduplication",
            description="Test deduplication of multiple patterns",
            interfaces=[IPatternGenerator, IPatternDeduplicator],
            test_function=self._test_patterns_deduplication
        )
        
        # Test: Patterns -> Validator
        self.test_cases["pattern_validation"] = IntegrationTestCase(
            test_id="pattern_validation",
            name="Pattern Validation",
            description="Test pattern validation integration",
            interfaces=[IPatternGenerator, IPatternValidator],
            test_function=self._test_pattern_validation
        )
        
    def _test_generator_to_evaluator(self) -> TestResult:
        """Test generator to evaluator integration."""
        try:
            generator = self.interface_map.get(IPatternGenerator)
            evaluator = self.interface_map.get(IPatternEvaluator)
            
            if not generator or not evaluator:
                return TestResult("generator_to_evaluator", TestStatus.SKIPPED, 0.0, "Missing implementations")
                
            start_time = time.time()
            
            # Generate patterns
            request = GenerationRequest(
                label="urgent",
                positive_examples=["urgent message", "critical alert"],
                negative_examples=["normal text", "regular message"],
                n_patterns=2
            )
            patterns = generator.generate_patterns(request)
            
            # Evaluate patterns
            test_texts = ["urgent message", "normal text", "critical alert", "regular message"]
            test_labels = ["urgent", "normal", "urgent", "normal"]
            
            for pattern in patterns:
                metrics = evaluator.evaluate_pattern(pattern, test_texts, test_labels)
                assert isinstance(metrics, EvaluationMetrics)
                assert 0 <= metrics.precision <= 1
                assert 0 <= metrics.recall <= 1
                
            execution_time = time.time() - start_time
            
            return TestResult(
                "generator_to_evaluator",
                TestStatus.PASSED,
                execution_time,
                details={"patterns_evaluated": len(patterns)}
            )
            
        except Exception as e:
            return TestResult("generator_to_evaluator", TestStatus.ERROR, 0.0, str(e))
            
    def _test_evaluator_to_optimizer(self) -> TestResult:
        """Test evaluator to optimizer integration."""
        try:
            evaluator = self.interface_map.get(IPatternEvaluator)
            optimizer = self.interface_map.get(IPatternOptimizer)
            
            if not evaluator or not optimizer:
                return TestResult("evaluator_to_optimizer", TestStatus.SKIPPED, 0.0, "Missing implementations")
                
            start_time = time.time()
            
            # Create test patterns
            patterns = [
                Pattern("urgent", 0.8, "urgent"),
                Pattern("critical", 0.7, "urgent")
            ]
            
            test_texts = ["urgent message", "critical alert", "normal text"]
            test_labels = ["urgent", "urgent", "normal"]
            
            # Evaluate patterns
            metrics = evaluator.evaluate_patterns(patterns, test_texts, test_labels)
            
            # Optimize patterns
            config = OptimizationConfig(max_iterations=3, convergence_threshold=0.05)
            optimized = optimizer.optimize_patterns(patterns, test_texts, test_labels, config)
            
            assert isinstance(optimized, list)
            assert len(optimized) <= len(patterns)
            assert all(isinstance(p, Pattern) for p in optimized)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "evaluator_to_optimizer",
                TestStatus.PASSED,
                execution_time,
                details={"patterns_optimized": len(optimized)}
            )
            
        except Exception as e:
            return TestResult("evaluator_to_optimizer", TestStatus.ERROR, 0.0, str(e))
            
    def _test_patterns_deduplication(self) -> TestResult:
        """Test pattern deduplication."""
        try:
            deduplicator = self.interface_map.get(IPatternDeduplicator)
            if not deduplicator:
                return TestResult("patterns_deduplication", TestStatus.SKIPPED, 0.0, "No deduplicator")
                
            start_time = time.time()
            
            # Create patterns with duplicates
            patterns = [
                Pattern("urgent", 0.8, "urgent"),
                Pattern("urgent", 0.8, "urgent"),  # Duplicate
                Pattern("critical", 0.7, "urgent"),
                Pattern("important", 0.6, "urgent")
            ]
            
            test_texts = ["urgent message", "critical alert", "important note"]
            test_labels = ["urgent", "urgent", "urgent"]
            
            # Deduplicate
            unique_patterns = deduplicator.deduplicate_patterns(patterns, test_texts, test_labels)
            
            assert isinstance(unique_patterns, list)
            assert len(unique_patterns) <= len(patterns)
            assert all(isinstance(p, Pattern) for p in unique_patterns)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "patterns_deduplication",
                TestStatus.PASSED,
                execution_time,
                details={
                    "original_count": len(patterns),
                    "deduplicated_count": len(unique_patterns),
                    "reduction": len(patterns) - len(unique_patterns)
                }
            )
            
        except Exception as e:
            return TestResult("patterns_deduplication", TestStatus.ERROR, 0.0, str(e))
            
    def _test_pattern_validation(self) -> TestResult:
        """Test pattern validation."""
        try:
            validator = self.interface_map.get(IPatternValidator)
            if not validator:
                return TestResult("pattern_validation", TestStatus.SKIPPED, 0.0, "No validator")
                
            start_time = time.time()
            
            # Create patterns with some invalid ones
            patterns = [
                Pattern("urgent", 0.8, "urgent"),
                Pattern("[invalid", 0.7, "urgent"),  # Invalid regex
                Pattern("critical", 0.9, "urgent"),
                Pattern("*", 0.5, "urgent")  # Invalid regex
            ]
            
            # Validate patterns
            validation_results = validator.validate_patterns(patterns)
            
            assert isinstance(validation_results, list)
            assert len(validation_results) == len(patterns)
            assert all(isinstance(result, bool) for result in validation_results)
            
            # Check individual validation
            for i, pattern in enumerate(patterns):
                individual_result = validator.validate_pattern(pattern)
                assert individual_result == validation_results[i]
                
            execution_time = time.time() - start_time
            
            return TestResult(
                "pattern_validation",
                TestStatus.PASSED,
                execution_time,
                details={
                    "patterns_tested": len(patterns),
                    "valid_patterns": sum(validation_results),
                    "invalid_patterns": len(patterns) - sum(validation_results)
                }
            )
            
        except Exception as e:
            return TestResult("pattern_validation", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Analysis Integration Tests
    # ========================================================================
    
    def _create_analysis_tests(self) -> None:
        """Create analysis integration tests."""
        
        # Test: Performance Analyzer Integration
        self.test_cases["performance_analysis"] = IntegrationTestCase(
            test_id="performance_analysis",
            name="Performance Analysis Integration",
            description="Test performance analyzer with other components",
            interfaces=[IPerformanceAnalyzer, IPatternGenerator],
            test_function=self._test_performance_analysis
        )
        
        # Test: Quality Analyzer Integration
        self.test_cases["quality_analysis"] = IntegrationTestCase(
            test_id="quality_analysis",
            name="Quality Analysis Integration",
            description="Test quality analyzer with patterns and datasets",
            interfaces=[IQualityAnalyzer, IPatternEvaluator],
            test_function=self._test_quality_analysis
        )
        
    def _test_performance_analysis(self) -> TestResult:
        """Test performance analyzer integration."""
        try:
            perf_analyzer = self.interface_map.get(IPerformanceAnalyzer)
            if not perf_analyzer:
                return TestResult("performance_analysis", TestStatus.SKIPPED, 0.0, "No performance analyzer")
                
            start_time = time.time()
            
            # Simulate operations
            operation_start = time.time()
            time.sleep(0.1)  # Simulate work
            operation_end = time.time()
            
            # Analyze latency
            latency_analysis = perf_analyzer.analyze_latency("pattern_generation", operation_start, operation_end)
            assert isinstance(latency_analysis, dict)
            
            # Analyze throughput
            throughput = perf_analyzer.analyze_throughput(10, 1.0)
            assert isinstance(throughput, (int, float))
            assert throughput > 0
            
            # Generate report
            report = perf_analyzer.generate_performance_report()
            assert isinstance(report, dict)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "performance_analysis",
                TestStatus.PASSED,
                execution_time,
                details={"throughput": throughput}
            )
            
        except Exception as e:
            return TestResult("performance_analysis", TestStatus.ERROR, 0.0, str(e))
            
    def _test_quality_analysis(self) -> TestResult:
        """Test quality analyzer integration."""
        try:
            quality_analyzer = self.interface_map.get(IQualityAnalyzer)
            if not quality_analyzer:
                return TestResult("quality_analysis", TestStatus.SKIPPED, 0.0, "No quality analyzer")
                
            start_time = time.time()
            
            # Create test data
            pattern = Pattern("urgent", 0.8, "urgent")
            metrics = EvaluationMetrics(0.85, 0.75, 0.8, 0.9, 0.8, 2, 3, 8, 17)
            texts = ["urgent message", "normal text", "critical alert"]
            labels = ["urgent", "normal", "urgent"]
            
            # Analyze pattern quality
            pattern_quality = quality_analyzer.analyze_pattern_quality(pattern, metrics)
            assert isinstance(pattern_quality, dict)
            
            # Analyze dataset quality
            dataset_quality = quality_analyzer.analyze_dataset_quality(texts, labels)
            assert isinstance(dataset_quality, dict)
            
            # Generate report
            quality_report = quality_analyzer.generate_quality_report()
            assert isinstance(quality_report, dict)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "quality_analysis",
                TestStatus.PASSED,
                execution_time,
                details={"pattern_quality_keys": list(pattern_quality.keys())}
            )
            
        except Exception as e:
            return TestResult("quality_analysis", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Pipeline Integration Tests
    # ========================================================================
    
    def _create_pipeline_tests(self) -> None:
        """Create pipeline integration tests."""
        
        # Test: Full Pipeline Integration
        self.test_cases["full_pipeline"] = IntegrationTestCase(
            test_id="full_pipeline",
            name="Full Pipeline Integration",
            description="Test complete pipeline from input to output",
            interfaces=[IPipeline, IPatternGenerator, IPatternEvaluator, IExporter],
            test_function=self._test_full_pipeline
        )
        
        # Test: Pipeline Orchestrator
        self.test_cases["pipeline_orchestrator"] = IntegrationTestCase(
            test_id="pipeline_orchestrator",
            name="Pipeline Orchestrator Integration",
            description="Test pipeline orchestrator with components",
            interfaces=[IPipelineOrchestrator, IPatternGenerator, IPatternEvaluator],
            test_function=self._test_pipeline_orchestrator
        )
        
    def _test_full_pipeline(self) -> TestResult:
        """Test full pipeline integration."""
        try:
            pipeline = self.interface_map.get(IPipeline)
            if not pipeline:
                return TestResult("full_pipeline", TestStatus.SKIPPED, 0.0, "No pipeline implementation")
                
            start_time = time.time()
            
            # Test data
            texts = ["urgent message", "normal text", "critical alert", "regular message"]
            labels = ["urgent", "normal", "urgent", "normal"]
            label_descriptions = {
                "urgent": "Messages requiring immediate attention",
                "normal": "Regular messages"
            }
            
            # Validate inputs
            input_valid = pipeline.validate_inputs(texts, labels)
            assert input_valid
            
            # Run pipeline
            results = pipeline.run(texts, labels, label_descriptions)
            
            assert isinstance(results, dict)
            assert "urgent" in results
            assert "normal" in results
            assert all(isinstance(patterns, list) for patterns in results.values())
            
            # Generate report
            report = pipeline.generate_report()
            assert isinstance(report, dict)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "full_pipeline",
                TestStatus.PASSED,
                execution_time,
                details={
                    "labels_processed": len(results),
                    "total_patterns": sum(len(patterns) for patterns in results.values())
                }
            )
            
        except Exception as e:
            return TestResult("full_pipeline", TestStatus.ERROR, 0.0, str(e))
            
    def _test_pipeline_orchestrator(self) -> TestResult:
        """Test pipeline orchestrator integration."""
        try:
            orchestrator = self.interface_map.get(IPipelineOrchestrator)
            if not orchestrator:
                return TestResult("pipeline_orchestrator", TestStatus.SKIPPED, 0.0, "No orchestrator")
                
            start_time = time.time()
            
            # Register mock components
            orchestrator.register_component("generator", self.interface_map.get(IPatternGenerator))
            orchestrator.register_component("evaluator", self.interface_map.get(IPatternEvaluator))
            
            # Test component retrieval
            generator = orchestrator.get_component("generator")
            evaluator = orchestrator.get_component("evaluator")
            
            assert generator is not None
            assert evaluator is not None
            
            # Test stage execution
            stage_input = {"test": "data"}
            stage_output = orchestrator.execute_stage("test_stage", stage_input)
            
            assert isinstance(stage_output, dict)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "pipeline_orchestrator",
                TestStatus.PASSED,
                execution_time,
                details={"components_registered": 2}
            )
            
        except Exception as e:
            return TestResult("pipeline_orchestrator", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Utility Integration Tests
    # ========================================================================
    
    def _create_utility_tests(self) -> None:
        """Create utility integration tests."""
        
        # Test: Cache Integration
        self.test_cases["cache_integration"] = IntegrationTestCase(
            test_id="cache_integration",
            name="Cache Integration",
            description="Test cache integration with other components",
            interfaces=[ICache, IPatternGenerator],
            test_function=self._test_cache_integration
        )
        
        # Test: Configuration Integration
        self.test_cases["configuration_integration"] = IntegrationTestCase(
            test_id="configuration_integration",
            name="Configuration Integration",
            description="Test configuration integration across components",
            interfaces=[IConfiguration, IPatternGenerator, ILLMProvider],
            test_function=self._test_configuration_integration
        )
        
        # Test: Logger Integration
        self.test_cases["logger_integration"] = IntegrationTestCase(
            test_id="logger_integration",
            name="Logger Integration",
            description="Test logger integration across components",
            interfaces=[ILogger, IPatternGenerator, IPatternEvaluator],
            test_function=self._test_logger_integration
        )
        
    def _test_cache_integration(self) -> TestResult:
        """Test cache integration."""
        try:
            cache = self.interface_map.get(ICache)
            if not cache:
                return TestResult("cache_integration", TestStatus.SKIPPED, 0.0, "No cache implementation")
                
            start_time = time.time()
            
            # Test basic cache operations
            cache.set("test_key", "test_value")
            assert cache.exists("test_key")
            
            value = cache.get("test_key")
            assert value == "test_value"
            
            # Test with TTL
            cache.set("ttl_key", "ttl_value", ttl=1)
            assert cache.exists("ttl_key")
            
            # Test deletion
            cache.delete("test_key")
            assert not cache.exists("test_key")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "cache_integration",
                TestStatus.PASSED,
                execution_time,
                details={"operations_tested": 5}
            )
            
        except Exception as e:
            return TestResult("cache_integration", TestStatus.ERROR, 0.0, str(e))
            
    def _test_configuration_integration(self) -> TestResult:
        """Test configuration integration."""
        try:
            config = self.interface_map.get(IConfiguration)
            if not config:
                return TestResult("configuration_integration", TestStatus.SKIPPED, 0.0, "No config implementation")
                
            start_time = time.time()
            
            # Test configuration operations
            config.set("test.setting", "test_value")
            value = config.get("test.setting")
            assert value == "test_value"
            
            # Test default values
            default_value = config.get("non.existent.key", "default")
            assert default_value == "default"
            
            # Test validation
            is_valid = config.validate()
            assert isinstance(is_valid, bool)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "configuration_integration",
                TestStatus.PASSED,
                execution_time,
                details={"is_valid": is_valid}
            )
            
        except Exception as e:
            return TestResult("configuration_integration", TestStatus.ERROR, 0.0, str(e))
            
    def _test_logger_integration(self) -> TestResult:
        """Test logger integration."""
        try:
            logger = self.interface_map.get(ILogger)
            if not logger:
                return TestResult("logger_integration", TestStatus.SKIPPED, 0.0, "No logger implementation")
                
            start_time = time.time()
            
            # Test all log levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "logger_integration",
                TestStatus.PASSED,
                execution_time,
                details={"log_levels_tested": 5}
            )
            
        except Exception as e:
            return TestResult("logger_integration", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Cross-Module Integration Tests
    # ========================================================================
    
    def _create_cross_module_tests(self) -> None:
        """Create cross-module integration tests."""
        
        # Test: Complete Generation -> Evaluation -> Optimization Flow
        self.test_cases["complete_flow"] = IntegrationTestCase(
            test_id="complete_flow",
            name="Complete Processing Flow",
            description="Test complete flow from generation through optimization",
            interfaces=[IPatternGenerator, IPatternEvaluator, IPatternOptimizer, IPatternDeduplicator],
            test_function=self._test_complete_flow
        )
        
    def _test_complete_flow(self) -> TestResult:
        """Test complete processing flow."""
        try:
            generator = self.interface_map.get(IPatternGenerator)
            evaluator = self.interface_map.get(IPatternEvaluator)
            optimizer = self.interface_map.get(IPatternOptimizer)
            deduplicator = self.interface_map.get(IPatternDeduplicator)
            
            if not all([generator, evaluator, optimizer, deduplicator]):
                return TestResult("complete_flow", TestStatus.SKIPPED, 0.0, "Missing implementations")
                
            start_time = time.time()
            
            # Step 1: Generate patterns
            request = GenerationRequest(
                label="urgent",
                positive_examples=["urgent message", "critical alert"],
                negative_examples=["normal text", "regular message"],
                n_patterns=5
            )
            patterns = generator.generate_patterns(request)
            
            # Step 2: Deduplicate
            test_texts = ["urgent message", "normal text", "critical alert", "regular message"]
            test_labels = ["urgent", "normal", "urgent", "normal"]
            
            unique_patterns = deduplicator.deduplicate_patterns(patterns, test_texts, test_labels)
            
            # Step 3: Evaluate
            metrics = evaluator.evaluate_patterns(unique_patterns, test_texts, test_labels)
            
            # Step 4: Optimize
            config = OptimizationConfig(max_iterations=2, convergence_threshold=0.1)
            optimized_patterns = optimizer.optimize_patterns(unique_patterns, test_texts, test_labels, config)
            
            # Verify flow
            assert len(optimized_patterns) <= len(unique_patterns)
            assert len(unique_patterns) <= len(patterns)
            assert all(isinstance(p, Pattern) for p in optimized_patterns)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "complete_flow",
                TestStatus.PASSED,
                execution_time,
                details={
                    "original_patterns": len(patterns),
                    "after_deduplication": len(unique_patterns),
                    "after_optimization": len(optimized_patterns),
                    "final_reduction": len(patterns) - len(optimized_patterns)
                }
            )
            
        except Exception as e:
            return TestResult("complete_flow", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Data Flow Tests
    # ========================================================================
    
    def _create_data_flow_tests(self) -> None:
        """Create data flow tests."""
        
        # Test: Data Consistency Across Pipeline
        self.test_cases["data_consistency"] = IntegrationTestCase(
            test_id="data_consistency",
            name="Data Consistency",
            description="Test data consistency across pipeline stages",
            interfaces=[IPatternGenerator, IPatternEvaluator, IExporter],
            test_function=self._test_data_consistency
        )
        
    def _test_data_consistency(self) -> TestResult:
        """Test data consistency across pipeline."""
        try:
            generator = self.interface_map.get(IPatternGenerator)
            evaluator = self.interface_map.get(IPatternEvaluator)
            exporter = self.interface_map.get(IExporter)
            
            if not all([generator, evaluator, exporter]):
                return TestResult("data_consistency", TestStatus.SKIPPED, 0.0, "Missing implementations")
                
            start_time = time.time()
            
            # Generate patterns
            request = GenerationRequest(
                label="test",
                positive_examples=["test1", "test2"],
                negative_examples=["neg1", "neg2"],
                n_patterns=3
            )
            patterns = generator.generate_patterns(request)
            
            # Evaluate patterns
            test_texts = ["test1", "neg1", "test2", "neg2"]
            test_labels = ["test", "other", "test", "other"]
            
            for pattern in patterns:
                metrics = evaluator.evaluate_pattern(pattern, test_texts, test_labels)
                
                # Verify data consistency
                assert pattern.label == request.label
                assert isinstance(metrics.precision, float)
                assert 0 <= metrics.precision <= 1
                
            # Export patterns
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                exporter.export_patterns(patterns, tmp.name, format="json")
                
            execution_time = time.time() - start_time
            
            return TestResult(
                "data_consistency",
                TestStatus.PASSED,
                execution_time,
                details={"patterns_processed": len(patterns)}
            )
            
        except Exception as e:
            return TestResult("data_consistency", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Error Propagation Tests
    # ========================================================================
    
    def _create_error_propagation_tests(self) -> None:
        """Create error propagation tests."""
        
        # Test: Error Handling Across Components
        self.test_cases["error_propagation"] = IntegrationTestCase(
            test_id="error_propagation",
            name="Error Propagation",
            description="Test error handling and propagation across components",
            interfaces=[IPatternGenerator, IPatternEvaluator],
            test_function=self._test_error_propagation
        )
        
    def _test_error_propagation(self) -> TestResult:
        """Test error propagation across components."""
        try:
            generator = self.interface_map.get(IPatternGenerator)
            evaluator = self.interface_map.get(IPatternEvaluator)
            
            if not generator or not evaluator:
                return TestResult("error_propagation", TestStatus.SKIPPED, 0.0, "Missing implementations")
                
            start_time = time.time()
            
            # Test invalid inputs
            invalid_request = GenerationRequest(
                label="",  # Invalid empty label
                positive_examples=[],  # Empty examples
                negative_examples=[],
                n_patterns=0  # Invalid count
            )
            
            error_caught = False
            try:
                patterns = generator.generate_patterns(invalid_request)
            except Exception:
                error_caught = True
                
            # Test invalid pattern evaluation
            invalid_pattern = Pattern("", 0.0, "")  # Invalid pattern
            test_texts = ["test"]
            test_labels = ["test"]
            
            eval_error_caught = False
            try:
                metrics = evaluator.evaluate_pattern(invalid_pattern, test_texts, test_labels)
            except Exception:
                eval_error_caught = True
                
            execution_time = time.time() - start_time
            
            return TestResult(
                "error_propagation",
                TestStatus.PASSED,
                execution_time,
                details={
                    "generator_error_caught": error_caught,
                    "evaluator_error_caught": eval_error_caught
                }
            )
            
        except Exception as e:
            return TestResult("error_propagation", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Performance Integration Tests
    # ========================================================================
    
    def _create_performance_integration_tests(self) -> None:
        """Create performance integration tests."""
        
        # Test: Concurrent Operations
        self.test_cases["concurrent_operations"] = IntegrationTestCase(
            test_id="concurrent_operations",
            name="Concurrent Operations",
            description="Test concurrent operations across components",
            interfaces=[IPatternGenerator, IPatternEvaluator],
            test_function=self._test_concurrent_operations
        )
        
    def _test_concurrent_operations(self) -> TestResult:
        """Test concurrent operations."""
        try:
            generator = self.interface_map.get(IPatternGenerator)
            if not generator:
                return TestResult("concurrent_operations", TestStatus.SKIPPED, 0.0, "No generator")
                
            start_time = time.time()
            
            # Create multiple requests
            requests = [
                GenerationRequest(f"label_{i}", [f"example_{i}"], [f"neg_{i}"], 2)
                for i in range(3)
            ]
            
            # Test batch generation
            batch_results = generator.generate_patterns_batch(requests)
            
            assert isinstance(batch_results, list)
            assert len(batch_results) == len(requests)
            assert all(isinstance(result, list) for result in batch_results)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                "concurrent_operations",
                TestStatus.PASSED,
                execution_time,
                details={
                    "batch_requests": len(requests),
                    "total_patterns": sum(len(result) for result in batch_results)
                }
            )
            
        except Exception as e:
            return TestResult("concurrent_operations", TestStatus.ERROR, 0.0, str(e))
            
    # ========================================================================
    # Test Execution
    # ========================================================================
    
    def register_implementation(self, interface: Type, implementation: Any) -> None:
        """Register an implementation for an interface."""
        self.interface_map[interface] = implementation
        
    def run_single_test(self, test_id: str) -> TestResult:
        """Run a single test case."""
        if test_id not in self.test_cases:
            raise ValueError(f"Test {test_id} not found")
            
        test_case = self.test_cases[test_id]
        
        if not test_case.test_function:
            return TestResult(test_id, TestStatus.SKIPPED, 0.0, "No test function")
            
        try:
            test_case.status = TestStatus.RUNNING
            result = test_case.test_function()
            test_case.status = result.status
            test_case.execution_time = result.execution_time
            test_case.error_message = result.error_message
            
            self.test_results[test_id] = result
            return result
            
        except Exception as e:
            result = TestResult(test_id, TestStatus.ERROR, 0.0, str(e))
            test_case.status = TestStatus.ERROR
            test_case.error_message = str(e)
            self.test_results[test_id] = result
            return result
            
    def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all integration tests."""
        results = {}
        
        for test_id in self.test_cases.keys():
            result = self.run_single_test(test_id)
            results[test_id] = result
            
        return results
        
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        total_tests = len(self.test_cases)
        passed_tests = sum(1 for r in self.test_results.values() if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in self.test_results.values() if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in self.test_results.values() if r.status == TestStatus.ERROR)
        skipped_tests = sum(1 for r in self.test_results.values() if r.status == TestStatus.SKIPPED)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": f"{passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%",
                "total_execution_time": sum(r.execution_time for r in self.test_results.values())
            },
            "test_results": {
                test_id: {
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                    "details": result.details
                }
                for test_id, result in self.test_results.items()
            },
            "interface_coverage": {
                interface.__name__: sum(
                    1 for test in self.test_cases.values() 
                    if interface in test.interfaces
                )
                for interface in {
                    interface for test in self.test_cases.values() 
                    for interface in test.interfaces
                }
            }
        }
        
        return report
        
    def export_report(self, output_path: Path) -> None:
        """Export test report to file."""
        report = self.generate_test_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"Integration test report exported to {output_path}")