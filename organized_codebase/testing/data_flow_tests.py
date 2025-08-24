"""
Data Flow Tests for comprehensive pipeline validation.
Tests complete data pipelines and concurrent operations.
"""

import asyncio
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from src.interfaces import (
    Pattern, GenerationRequest, EvaluationMetrics, OptimizationConfig,
    IPatternGenerator, IPatternEvaluator, IPatternOptimizer,
    IPatternDeduplicator, IPipeline, IExporter
)


@dataclass
class DataFlowResult:
    """Result of a data flow test."""
    flow_id: str
    status: str
    execution_time: float
    data_integrity: bool
    throughput: float
    error_count: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class DataFlowTester:
    """
    Tests complete data flows through the pipeline system.
    Validates data integrity, performance, and concurrent processing.
    """
    
    def __init__(self):
        self.test_results: Dict[str, DataFlowResult] = {}
        self.interface_map: Dict[type, Any] = {}
        self.test_data_cache: Dict[str, Any] = {}
        
    def register_implementation(self, interface: type, implementation: Any) -> None:
        """Register an implementation for testing."""
        self.interface_map[interface] = implementation
        
    # ========================================================================
    # Complete Pipeline Data Flow Tests
    # ========================================================================
    
    def test_complete_generation_flow(self) -> DataFlowResult:
        """Test complete generation flow from input to final patterns."""
        flow_id = "complete_generation_flow"
        start_time = time.time()
        
        try:
            # Get implementations
            generator = self.interface_map.get(IPatternGenerator)
            evaluator = self.interface_map.get(IPatternEvaluator)
            deduplicator = self.interface_map.get(IPatternDeduplicator)
            optimizer = self.interface_map.get(IPatternOptimizer)
            
            if not all([generator, evaluator, deduplicator, optimizer]):
                return DataFlowResult(
                    flow_id, "skipped", 0.0, False, 0.0, 0,
                    details={"reason": "Missing implementations"}
                )
            
            # Input data
            test_data = self._create_test_dataset()
            error_count = 0
            processed_items = 0
            
            # Process each label
            final_results = {}
            
            for label, data in test_data.items():
                try:
                    # Step 1: Generate patterns
                    request = GenerationRequest(
                        label=label,
                        positive_examples=data["positive"],
                        negative_examples=data["negative"],
                        n_patterns=10
                    )
                    
                    generated_patterns = generator.generate_patterns(request)
                    assert isinstance(generated_patterns, list)
                    assert all(isinstance(p, Pattern) for p in generated_patterns)
                    assert all(p.label == label for p in generated_patterns)
                    
                    # Step 2: Deduplicate patterns
                    all_texts = data["positive"] + data["negative"]
                    all_labels = [label] * len(data["positive"]) + ["other"] * len(data["negative"])
                    
                    unique_patterns = deduplicator.deduplicate_patterns(
                        generated_patterns, all_texts, all_labels
                    )
                    assert isinstance(unique_patterns, list)
                    assert len(unique_patterns) <= len(generated_patterns)
                    assert all(isinstance(p, Pattern) for p in unique_patterns)
                    
                    # Step 3: Evaluate patterns
                    evaluation_results = {}
                    for i, pattern in enumerate(unique_patterns):
                        metrics = evaluator.evaluate_pattern(pattern, all_texts, all_labels)
                        evaluation_results[f"pattern_{i}"] = metrics
                        
                        # Validate metrics
                        assert isinstance(metrics, EvaluationMetrics)
                        assert 0 <= metrics.precision <= 1
                        assert 0 <= metrics.recall <= 1
                        assert 0 <= metrics.f1_score <= 1
                    
                    # Step 4: Optimize patterns
                    config = OptimizationConfig(
                        max_iterations=3,
                        convergence_threshold=0.05,
                        enable_fusion=True,
                        enable_repair=True
                    )
                    
                    optimized_patterns = optimizer.optimize_patterns(
                        unique_patterns, all_texts, all_labels, config
                    )
                    assert isinstance(optimized_patterns, list)
                    assert all(isinstance(p, Pattern) for p in optimized_patterns)
                    
                    # Step 5: Final evaluation
                    final_metrics = {}
                    for i, pattern in enumerate(optimized_patterns):
                        final_metrics[f"optimized_{i}"] = evaluator.evaluate_pattern(
                            pattern, all_texts, all_labels
                        )
                    
                    # Store results
                    final_results[label] = {
                        "original_count": len(generated_patterns),
                        "after_deduplication": len(unique_patterns),
                        "after_optimization": len(optimized_patterns),
                        "final_patterns": optimized_patterns,
                        "metrics": final_metrics
                    }
                    
                    processed_items += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing label {label}: {e}")
            
            execution_time = time.time() - start_time
            throughput = processed_items / execution_time if execution_time > 0 else 0
            data_integrity = error_count == 0 and len(final_results) == len(test_data)
            
            return DataFlowResult(
                flow_id=flow_id,
                status="passed" if error_count == 0 else "failed",
                execution_time=execution_time,
                data_integrity=data_integrity,
                throughput=throughput,
                error_count=error_count,
                details={
                    "labels_processed": processed_items,
                    "total_labels": len(test_data),
                    "results_summary": {
                        label: {
                            "reduction": result["original_count"] - result["after_optimization"],
                            "final_count": result["after_optimization"]
                        }
                        for label, result in final_results.items()
                    }
                }
            )
            
        except Exception as e:
            return DataFlowResult(
                flow_id, "error", time.time() - start_time, False, 0.0, 1,
                details={"error": str(e)}
            )
            
    def test_concurrent_generation_flow(self) -> DataFlowResult:
        """Test concurrent processing of multiple generation requests."""
        flow_id = "concurrent_generation_flow"
        start_time = time.time()
        
        try:
            generator = self.interface_map.get(IPatternGenerator)
            if not generator:
                return DataFlowResult(
                    flow_id, "skipped", 0.0, False, 0.0, 0,
                    details={"reason": "No generator implementation"}
                )
            
            # Create concurrent requests
            test_data = self._create_test_dataset()
            requests = []
            
            for label, data in test_data.items():
                for i in range(3):  # 3 variants per label
                    requests.append(GenerationRequest(
                        label=f"{label}_{i}",
                        positive_examples=data["positive"][:3],  # Subset for speed
                        negative_examples=data["negative"][:3],
                        n_patterns=5
                    ))
            
            # Test batch processing
            batch_results = generator.generate_patterns_batch(requests)
            
            # Validate results
            assert isinstance(batch_results, list)
            assert len(batch_results) == len(requests)
            
            total_patterns = 0
            error_count = 0
            
            for i, (request, result) in enumerate(zip(requests, batch_results)):
                try:
                    assert isinstance(result, list)
                    assert all(isinstance(p, Pattern) for p in result)
                    assert len(result) <= request.n_patterns
                    total_patterns += len(result)
                except AssertionError:
                    error_count += 1
                    
            execution_time = time.time() - start_time
            throughput = len(requests) / execution_time if execution_time > 0 else 0
            data_integrity = error_count == 0
            
            return DataFlowResult(
                flow_id=flow_id,
                status="passed" if error_count == 0 else "failed",
                execution_time=execution_time,
                data_integrity=data_integrity,
                throughput=throughput,
                error_count=error_count,
                details={
                    "concurrent_requests": len(requests),
                    "total_patterns_generated": total_patterns,
                    "average_patterns_per_request": total_patterns / len(requests) if requests else 0
                }
            )
            
        except Exception as e:
            return DataFlowResult(
                flow_id, "error", time.time() - start_time, False, 0.0, 1,
                details={"error": str(e)}
            )
            
    def test_pipeline_data_flow(self) -> DataFlowResult:
        """Test complete pipeline data flow."""
        flow_id = "pipeline_data_flow"
        start_time = time.time()
        
        try:
            pipeline = self.interface_map.get(IPipeline)
            if not pipeline:
                return DataFlowResult(
                    flow_id, "skipped", 0.0, False, 0.0, 0,
                    details={"reason": "No pipeline implementation"}
                )
            
            # Prepare test data
            texts = []
            labels = []
            label_descriptions = {}
            
            test_data = self._create_test_dataset()
            for label, data in test_data.items():
                texts.extend(data["positive"])
                labels.extend([label] * len(data["positive"]))
                texts.extend(data["negative"])
                labels.extend(["other"] * len(data["negative"]))
                label_descriptions[label] = f"Examples of {label} category"
                
            # Validate inputs
            input_valid = pipeline.validate_inputs(texts, labels)
            assert input_valid, "Pipeline input validation failed"
            
            # Run pipeline
            results = pipeline.run(texts, labels, label_descriptions)
            
            # Validate pipeline results
            assert isinstance(results, dict)
            error_count = 0
            total_patterns = 0
            
            for label in test_data.keys():
                if label not in results:
                    error_count += 1
                    continue
                    
                patterns = results[label]
                try:
                    assert isinstance(patterns, list)
                    assert all(isinstance(p, Pattern) for p in patterns)
                    assert all(p.label == label for p in patterns)
                    total_patterns += len(patterns)
                except AssertionError:
                    error_count += 1
            
            # Generate pipeline report
            report = pipeline.generate_report()
            assert isinstance(report, dict)
            
            execution_time = time.time() - start_time
            throughput = len(texts) / execution_time if execution_time > 0 else 0
            data_integrity = error_count == 0
            
            return DataFlowResult(
                flow_id=flow_id,
                status="passed" if error_count == 0 else "failed",
                execution_time=execution_time,
                data_integrity=data_integrity,
                throughput=throughput,
                error_count=error_count,
                details={
                    "input_texts": len(texts),
                    "unique_labels": len(set(labels)),
                    "output_labels": len(results),
                    "total_patterns": total_patterns,
                    "pipeline_report": report
                }
            )
            
        except Exception as e:
            return DataFlowResult(
                flow_id, "error", time.time() - start_time, False, 0.0, 1,
                details={"error": str(e)}
            )
            
    # ========================================================================
    # Resource Management Tests
    # ========================================================================
    
    def test_resource_cleanup_flow(self) -> DataFlowResult:
        """Test proper resource cleanup in data flows."""
        flow_id = "resource_cleanup_flow"
        start_time = time.time()
        
        try:
            # Test resource creation and cleanup
            import resource
            import gc
            
            initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            
            # Perform multiple operations that should clean up resources
            generator = self.interface_map.get(IPatternGenerator)
            if not generator:
                return DataFlowResult(
                    flow_id, "skipped", 0.0, False, 0.0, 0,
                    details={"reason": "No generator implementation"}
                )
            
            # Generate many patterns and ensure cleanup
            for i in range(10):
                request = GenerationRequest(
                    label=f"test_{i}",
                    positive_examples=[f"example_{i}_1", f"example_{i}_2"],
                    negative_examples=[f"neg_{i}_1", f"neg_{i}_2"],
                    n_patterns=5
                )
                
                patterns = generator.generate_patterns(request)
                
                # Force cleanup
                del patterns
                
                # Periodic garbage collection
                if i % 3 == 0:
                    gc.collect()
            
            # Final cleanup
            gc.collect()
            final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            memory_increase = final_memory - initial_memory
            
            execution_time = time.time() - start_time
            
            # Memory increase should be reasonable (less than 50MB)
            memory_ok = memory_increase < 50 * 1024 * 1024
            
            return DataFlowResult(
                flow_id=flow_id,
                status="passed" if memory_ok else "failed",
                execution_time=execution_time,
                data_integrity=memory_ok,
                throughput=10 / execution_time if execution_time > 0 else 0,
                error_count=0 if memory_ok else 1,
                details={
                    "initial_memory": initial_memory,
                    "final_memory": final_memory,
                    "memory_increase_mb": memory_increase / (1024 * 1024),
                    "operations_performed": 10
                }
            )
            
        except Exception as e:
            return DataFlowResult(
                flow_id, "error", time.time() - start_time, False, 0.0, 1,
                details={"error": str(e)}
            )
            
    # ========================================================================
    # Error Recovery Tests
    # ========================================================================
    
    def test_error_recovery_flow(self) -> DataFlowResult:
        """Test error recovery and graceful degradation."""
        flow_id = "error_recovery_flow"
        start_time = time.time()
        
        try:
            generator = self.interface_map.get(IPatternGenerator)
            evaluator = self.interface_map.get(IPatternEvaluator)
            
            if not generator or not evaluator:
                return DataFlowResult(
                    flow_id, "skipped", 0.0, False, 0.0, 0,
                    details={"reason": "Missing implementations"}
                )
            
            # Test with various error conditions
            error_scenarios = [
                # Empty label
                GenerationRequest("", ["example"], ["neg"], 3),
                # Empty examples
                GenerationRequest("test", [], [], 3),
                # Invalid n_patterns
                GenerationRequest("test", ["example"], ["neg"], -1),
                # Very large n_patterns
                GenerationRequest("test", ["example"], ["neg"], 10000),
            ]
            
            recovery_count = 0
            total_scenarios = len(error_scenarios)
            
            for scenario in error_scenarios:
                try:
                    # This should either work or raise a proper exception
                    patterns = generator.generate_patterns(scenario)
                    
                    if patterns is not None:
                        # If it worked, validate the results
                        if isinstance(patterns, list) and all(isinstance(p, Pattern) for p in patterns):
                            recovery_count += 1
                        
                except (ValueError, TypeError, Exception) as e:
                    # Proper error handling counts as recovery
                    if isinstance(e, (ValueError, TypeError)):
                        recovery_count += 1
                    
            # Test evaluation error recovery
            invalid_patterns = [
                Pattern("", 0.5, "test"),  # Empty pattern
                Pattern("[invalid", 0.5, "test"),  # Invalid regex
                Pattern("valid", -1.0, "test"),  # Invalid confidence
            ]
            
            for pattern in invalid_patterns:
                try:
                    metrics = evaluator.evaluate_pattern(pattern, ["test"], ["test"])
                    if isinstance(metrics, EvaluationMetrics):
                        recovery_count += 1
                except Exception as e:
                    if isinstance(e, (ValueError, TypeError)):
                        recovery_count += 1
            
            total_tests = total_scenarios + len(invalid_patterns)
            recovery_rate = recovery_count / total_tests if total_tests > 0 else 0
            
            execution_time = time.time() - start_time
            
            return DataFlowResult(
                flow_id=flow_id,
                status="passed" if recovery_rate >= 0.8 else "failed",  # 80% recovery rate
                execution_time=execution_time,
                data_integrity=recovery_rate >= 0.8,
                throughput=total_tests / execution_time if execution_time > 0 else 0,
                error_count=total_tests - recovery_count,
                details={
                    "total_error_scenarios": total_tests,
                    "recovered_scenarios": recovery_count,
                    "recovery_rate": f"{recovery_rate:.1%}",
                    "failed_scenarios": total_tests - recovery_count
                }
            )
            
        except Exception as e:
            return DataFlowResult(
                flow_id, "error", time.time() - start_time, False, 0.0, 1,
                details={"error": str(e)}
            )
            
    # ========================================================================
    # Performance Data Flow Tests
    # ========================================================================
    
    def test_high_throughput_flow(self) -> DataFlowResult:
        """Test high throughput data processing."""
        flow_id = "high_throughput_flow"
        start_time = time.time()
        
        try:
            generator = self.interface_map.get(IPatternGenerator)
            if not generator:
                return DataFlowResult(
                    flow_id, "skipped", 0.0, False, 0.0, 0,
                    details={"reason": "No generator implementation"}
                )
            
            # Create many small requests for throughput testing
            requests = []
            for i in range(50):  # 50 small requests
                requests.append(GenerationRequest(
                    label=f"high_throughput_{i}",
                    positive_examples=[f"example_{i}"],
                    negative_examples=[f"neg_{i}"],
                    n_patterns=2  # Small number for speed
                ))
            
            # Process requests and measure throughput
            processed = 0
            errors = 0
            
            batch_results = generator.generate_patterns_batch(requests)
            
            for result in batch_results:
                if isinstance(result, list) and len(result) > 0:
                    processed += 1
                else:
                    errors += 1
            
            execution_time = time.time() - start_time
            throughput = processed / execution_time if execution_time > 0 else 0
            
            # Target: at least 10 requests per second
            target_throughput = 10
            meets_target = throughput >= target_throughput
            
            return DataFlowResult(
                flow_id=flow_id,
                status="passed" if meets_target and errors == 0 else "failed",
                execution_time=execution_time,
                data_integrity=errors == 0,
                throughput=throughput,
                error_count=errors,
                details={
                    "total_requests": len(requests),
                    "processed_successfully": processed,
                    "target_throughput": target_throughput,
                    "actual_throughput": throughput,
                    "meets_target": meets_target
                }
            )
            
        except Exception as e:
            return DataFlowResult(
                flow_id, "error", time.time() - start_time, False, 0.0, 1,
                details={"error": str(e)}
            )
            
    # ========================================================================
    # Data Export Flow Tests
    # ========================================================================
    
    def test_export_data_flow(self) -> DataFlowResult:
        """Test complete data export flow."""
        flow_id = "export_data_flow"
        start_time = time.time()
        
        try:
            generator = self.interface_map.get(IPatternGenerator)
            exporter = self.interface_map.get(IExporter)
            
            if not generator or not exporter:
                return DataFlowResult(
                    flow_id, "skipped", 0.0, False, 0.0, 0,
                    details={"reason": "Missing implementations"}
                )
            
            # Generate patterns for export
            request = GenerationRequest(
                label="export_test",
                positive_examples=["export example 1", "export example 2"],
                negative_examples=["not export 1", "not export 2"],
                n_patterns=5
            )
            
            patterns = generator.generate_patterns(request)
            
            # Test different export formats
            import tempfile
            export_results = {}
            
            formats = ["json", "excel", "csv"]
            for format_type in formats:
                try:
                    with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as tmp:
                        if format_type == "json":
                            exporter.export_patterns(patterns, tmp.name, format="json")
                        elif format_type == "excel":
                            data = {"patterns": [{"pattern": p.pattern, "confidence": p.confidence, "label": p.label} for p in patterns]}
                            exporter.export_to_excel(data, tmp.name)
                        elif format_type == "csv":
                            import pandas as pd
                            df = pd.DataFrame([{"pattern": p.pattern, "confidence": p.confidence, "label": p.label} for p in patterns])
                            exporter.export_to_csv(df, tmp.name)
                        
                        # Verify file was created
                        export_path = Path(tmp.name)
                        if export_path.exists() and export_path.stat().st_size > 0:
                            export_results[format_type] = "success"
                        else:
                            export_results[format_type] = "failed"
                            
                except Exception as e:
                    export_results[format_type] = f"error: {str(e)}"
            
            execution_time = time.time() - start_time
            successful_exports = sum(1 for status in export_results.values() if status == "success")
            data_integrity = successful_exports == len(formats)
            
            return DataFlowResult(
                flow_id=flow_id,
                status="passed" if data_integrity else "failed",
                execution_time=execution_time,
                data_integrity=data_integrity,
                throughput=len(formats) / execution_time if execution_time > 0 else 0,
                error_count=len(formats) - successful_exports,
                details={
                    "patterns_exported": len(patterns),
                    "export_formats": formats,
                    "export_results": export_results,
                    "successful_exports": successful_exports
                }
            )
            
        except Exception as e:
            return DataFlowResult(
                flow_id, "error", time.time() - start_time, False, 0.0, 1,
                details={"error": str(e)}
            )
            
    # ========================================================================
    # Test Utilities
    # ========================================================================
    
    def _create_test_dataset(self) -> Dict[str, Dict[str, List[str]]]:
        """Create a comprehensive test dataset."""
        if "test_dataset" in self.test_data_cache:
            return self.test_data_cache["test_dataset"]
            
        dataset = {
            "urgent": {
                "positive": [
                    "urgent message requiring immediate attention",
                    "critical system failure detected",
                    "emergency response needed now",
                    "urgent: please respond immediately",
                    "critical alert: system down"
                ],
                "negative": [
                    "normal operational message",
                    "regular status update",
                    "standard notification",
                    "routine maintenance notice",
                    "general information"
                ]
            },
            "error": {
                "positive": [
                    "error: connection failed",
                    "system error encountered",
                    "failure in processing request",
                    "error code 500 returned",
                    "exception thrown during execution"
                ],
                "negative": [
                    "success: operation completed",
                    "processing finished successfully",
                    "connection established",
                    "request handled properly",
                    "normal execution flow"
                ]
            },
            "warning": {
                "positive": [
                    "warning: disk space running low",
                    "caution: high memory usage detected",
                    "alert: unusual activity noticed",
                    "warning: performance degradation",
                    "notice: potential issue identified"
                ],
                "negative": [
                    "information: system status normal",
                    "update: maintenance completed",
                    "notice: scheduled downtime",
                    "info: configuration updated",
                    "status: all systems operational"
                ]
            }
        }
        
        self.test_data_cache["test_dataset"] = dataset
        return dataset
        
    # ========================================================================
    # Test Execution
    # ========================================================================
    
    def run_all_data_flow_tests(self) -> Dict[str, DataFlowResult]:
        """Run all data flow tests."""
        test_methods = [
            self.test_complete_generation_flow,
            self.test_concurrent_generation_flow,
            self.test_pipeline_data_flow,
            self.test_resource_cleanup_flow,
            self.test_error_recovery_flow,
            self.test_high_throughput_flow,
            self.test_export_data_flow
        ]
        
        results = {}
        
        for test_method in test_methods:
            try:
                result = test_method()
                results[result.flow_id] = result
                self.test_results[result.flow_id] = result
                
                print(f"[{result.status.upper()}] {result.flow_id} - {result.execution_time:.2f}s")
                
            except Exception as e:
                error_result = DataFlowResult(
                    f"{test_method.__name__}", "error", 0.0, False, 0.0, 1,
                    details={"error": str(e)}
                )
                results[test_method.__name__] = error_result
                
        return results
        
    def generate_data_flow_report(self) -> Dict[str, Any]:
        """Generate comprehensive data flow test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.status == "passed")
        failed_tests = sum(1 for r in self.test_results.values() if r.status == "failed")
        error_tests = sum(1 for r in self.test_results.values() if r.status == "error")
        skipped_tests = sum(1 for r in self.test_results.values() if r.status == "skipped")
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": f"{passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%",
                "total_execution_time": sum(r.execution_time for r in self.test_results.values()),
                "average_throughput": sum(r.throughput for r in self.test_results.values()) / total_tests if total_tests > 0 else 0,
                "data_integrity_rate": f"{sum(1 for r in self.test_results.values() if r.data_integrity)/total_tests*100:.1f}%" if total_tests > 0 else "0%"
            },
            "detailed_results": {
                flow_id: {
                    "status": result.status,
                    "execution_time": result.execution_time,
                    "data_integrity": result.data_integrity,
                    "throughput": result.throughput,
                    "error_count": result.error_count,
                    "details": result.details
                }
                for flow_id, result in self.test_results.items()
            },
            "performance_metrics": {
                "fastest_test": min(self.test_results.items(), key=lambda x: x[1].execution_time)[0] if self.test_results else None,
                "slowest_test": max(self.test_results.items(), key=lambda x: x[1].execution_time)[0] if self.test_results else None,
                "highest_throughput": max(self.test_results.items(), key=lambda x: x[1].throughput)[0] if self.test_results else None,
                "total_errors": sum(r.error_count for r in self.test_results.values())
            }
        }
        
        return report
        
    def export_data_flow_report(self, output_path: Path) -> None:
        """Export data flow test report."""
        report = self.generate_data_flow_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"Data flow test report exported to {output_path}")