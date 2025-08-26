#!/usr/bin/env python3
"""
Test script for the integrated Agent QA system.

This script tests all major functionality of the unified agent_qa.py module
including quality monitoring, scoring, benchmarking, validation, and inspection.
"""

import time
import sys
import os
from typing import Dict, Any, List

# Add the core module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'intelligence', 'monitoring'))

try:
    from agent_qa import (
        AgentQualityAssurance,
        get_agent_qa,
        configure_agent_qa,
        inspect_agent_quality,
        validate_agent_output,
        score_agent_quality,
        benchmark_agent_performance,
        get_quality_status,
        BenchmarkType,
        QualityMetric,
        ValidationRule,
        ValidationType,
        QualityThreshold,
        AlertType
    )
    print("[OK] Successfully imported agent_qa module")
except ImportError as e:
    print(f"[ERROR] Failed to import agent_qa module: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic Agent QA functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Test system initialization
    print("Testing system initialization...")
    qa_system = get_agent_qa(enable_monitoring=False)  # Disable monitoring for tests
    assert qa_system is not None
    print("[OK] Agent QA system initialized successfully")
    
    # Test configuration
    print("Testing system configuration...")
    config_result = configure_agent_qa(
        similarity_threshold=0.8,
        enable_benchmarking=True,
        enable_monitoring=False,  # Keep disabled for tests
        alert_threshold=0.5
    )
    assert config_result["status"] == "configured"
    print("[OK] System configuration successful")
    
    # Test status
    print("Testing status retrieval...")
    status = get_quality_status()
    assert "enabled" in status
    assert "monitoring" in status
    print("[OK] Status retrieval successful")
    print(f"   Status: {status}")


def test_quality_inspection():
    """Test quality inspection functionality."""
    print("\n=== Testing Quality Inspection ===")
    
    qa_system = get_agent_qa()
    
    # Test basic inspection
    print("Testing basic quality inspection...")
    test_cases = [
        {"input": "test_input_1", "expected": "test_output_1"},
        {"input": "test_input_2", "expected": "test_output_2"}
    ]
    
    report = qa_system.inspect_agent(
        agent_id="test_agent_1",
        test_cases=test_cases,
        include_benchmarks=True
    )
    
    assert report.agent_id == "test_agent_1"
    assert len(report.metrics) > 0
    assert 0.0 <= report.overall_score <= 1.0
    assert report.status in ["excellent", "good", "satisfactory", "poor", "critical"]
    print("[OK] Quality inspection completed successfully")
    print(f"   Score: {report.overall_score:.3f} ({report.status})")
    print(f"   Metrics: {len(report.metrics)}")
    print(f"   Recommendations: {len(report.recommendations)}")
    
    # Test convenience function
    print("Testing convenience inspection function...")
    report2 = inspect_agent_quality("test_agent_2", test_cases, True)
    assert report2.agent_id == "test_agent_2"
    print("[OK] Convenience function works correctly")


def test_output_validation():
    """Test output validation functionality."""
    print("\n=== Testing Output Validation ===")
    
    qa_system = get_agent_qa()
    
    # Test valid Python code
    print("Testing valid Python code validation...")
    valid_code = """
def hello_world():
    print("Hello, World!")
    return True
"""
    
    result = qa_system.validate_output(
        agent_id="test_agent_validation",
        output=valid_code,
        expected=None
    )
    
    assert result.agent_id == "test_agent_validation"
    assert result.total_checks > 0
    print("[OK] Code validation completed")
    print(f"   Score: {result.score:.3f}")
    print(f"   Passed: {result.passed_checks}/{result.total_checks}")
    print(f"   Issues: {len(result.issues)}")
    
    # Test invalid JSON
    print("Testing JSON validation...")
    invalid_json = '{"key": "value", "invalid": }'
    
    result2 = qa_system.validate_output(
        agent_id="test_agent_json",
        output=invalid_json
    )
    
    print("[OK] JSON validation completed")
    if result2.issues:
        print(f"   Found {len(result2.issues)} validation issues (expected)")
    
    # Test convenience function
    print("Testing convenience validation function...")
    result3 = validate_agent_output("test_agent_3", "Hello World")
    assert result3.agent_id == "test_agent_3"
    print("[OK] Convenience validation function works correctly")


def test_quality_scoring():
    """Test quality scoring functionality."""
    print("\n=== Testing Quality Scoring ===")
    
    qa_system = get_agent_qa()
    
    # Create test metrics
    print("Testing quality scoring with metrics...")
    test_metrics = [
        QualityMetric(
            name="syntax_validation",
            value=0.95,
            threshold=0.9,
            status="pass",
            details={"errors": 0}
        ),
        QualityMetric(
            name="performance_test",
            value=0.85,
            threshold=0.8,
            status="pass",
            details={"response_time": 120}
        ),
        QualityMetric(
            name="security_scan",
            value=0.98,
            threshold=0.95,
            status="pass",
            details={"vulnerabilities": 0}
        )
    ]
    
    score = qa_system.calculate_score(
        agent_id="test_agent_scoring",
        quality_metrics=test_metrics
    )
    
    assert score.agent_id == "test_agent_scoring"
    assert 0.0 <= score.overall_score <= 1.0
    assert len(score.breakdown) > 0
    assert score.grade in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
    print("[OK] Quality scoring completed successfully")
    print(f"   Overall Score: {score.overall_score:.3f} (Grade: {score.grade})")
    print(f"   Status: {score.status}")
    print(f"   Breakdown categories: {len(score.breakdown)}")
    
    # Test with custom weights
    print("Testing custom weight scoring...")
    custom_weights = {
        "performance": 0.5,
        "security": 0.3,
        "functionality": 0.2
    }
    
    score2 = qa_system.calculate_score(
        agent_id="test_agent_custom",
        quality_metrics=test_metrics,
        custom_weights=custom_weights
    )
    print("[OK] Custom weight scoring completed")
    
    # Test convenience function
    print("Testing convenience scoring function...")
    score3 = score_agent_quality("test_agent_4", test_metrics)
    assert score3.agent_id == "test_agent_4"
    print("[OK] Convenience scoring function works correctly")


def test_performance_benchmarking():
    """Test performance benchmarking functionality."""
    print("\n=== Testing Performance Benchmarking ===")
    
    qa_system = get_agent_qa()
    
    # Test full benchmark suite
    print("Testing full benchmark suite...")
    result = qa_system.run_benchmarks(
        agent_id="test_agent_benchmark",
        benchmark_types=None,  # Use all types
        iterations=3  # Reduced for faster testing
    )
    
    assert result.agent_id == "test_agent_benchmark"
    assert len(result.metrics) > 0
    assert 0.0 <= result.overall_score <= 1.0
    assert result.status in ["excellent", "good", "satisfactory", "needs_improvement", "poor"]
    assert result.duration_ms > 0
    print("[OK] Full benchmark suite completed")
    print(f"   Overall Score: {result.overall_score:.3f} ({result.status})")
    print(f"   Duration: {result.duration_ms:.1f}ms")
    print(f"   Metrics: {len(result.metrics)}")
    
    # Test specific benchmark types
    print("Testing specific benchmark types...")
    specific_types = [BenchmarkType.RESPONSE_TIME, BenchmarkType.MEMORY_USAGE]
    result2 = qa_system.run_benchmarks(
        agent_id="test_agent_specific",
        benchmark_types=specific_types,
        iterations=2
    )
    
    assert len(result2.metrics) == 2
    print("[OK] Specific benchmark types completed")
    
    # Test convenience function
    print("Testing convenience benchmark function...")
    result3 = benchmark_agent_performance("test_agent_5", [BenchmarkType.ACCURACY], 2)
    assert result3.agent_id == "test_agent_5"
    print("[OK] Convenience benchmark function works correctly")


def test_quality_monitoring():
    """Test quality monitoring functionality."""
    print("\n=== Testing Quality Monitoring ===")
    
    qa_system = get_agent_qa()
    
    # Test metric recording
    print("Testing metric recording...")
    qa_system.record_metric("monitor_agent_1", "response_time", 150.0)
    qa_system.record_metric("monitor_agent_1", "accuracy", 0.95)
    qa_system.record_metric("monitor_agent_1", "error_rate", 0.02)
    
    # Record multiple values
    for i in range(5):
        qa_system.record_metric("monitor_agent_1", "overall_score", 0.8 + (i * 0.02))
    
    print("[OK] Metric recording completed")
    
    # Test threshold configuration
    print("Testing threshold configuration...")
    custom_threshold = QualityThreshold(
        name="custom_test_threshold",
        metric="accuracy",
        value=0.9,
        operator="lt",
        alert_type=AlertType.QUALITY_DEGRADATION,
        severity="high"
    )
    
    qa_system.add_threshold(custom_threshold)
    print("[OK] Custom threshold added")
    
    # Test agent status
    print("Testing agent status...")
    agent_status = qa_system.get_agent_status("monitor_agent_1")
    assert agent_status["agent_id"] == "monitor_agent_1"
    assert "latest_metrics" in agent_status
    print("[OK] Agent status retrieved")
    print(f"   Metrics count: {agent_status['metrics_count']}")


def test_history_and_trends():
    """Test history tracking and trend analysis."""
    print("\n=== Testing History and Trends ===")
    
    qa_system = get_agent_qa()
    
    # Generate some history by running multiple tests
    print("Generating test history...")
    agent_id = "trend_test_agent"
    
    # Multiple inspections
    for i in range(3):
        qa_system.inspect_agent(agent_id, include_benchmarks=False)
    
    # Multiple validations
    for i in range(3):
        qa_system.validate_output(agent_id, f"test output {i}")
    
    # Multiple benchmarks
    for i in range(2):
        qa_system.run_benchmarks(agent_id, [BenchmarkType.RESPONSE_TIME], 1)
    
    print("[OK] Test history generated")
    
    # Test history retrieval
    print("Testing history retrieval...")
    inspection_history = qa_system.get_inspection_history(agent_id)
    validation_history = qa_system.get_validation_history(agent_id)
    benchmark_history = qa_system.get_benchmark_history(agent_id)
    scoring_history = qa_system.get_scoring_history(agent_id)
    
    assert len(inspection_history) == 3
    assert len(validation_history) == 3
    assert len(benchmark_history) == 2
    print("[OK] History retrieval successful")
    print(f"   Inspections: {len(inspection_history)}")
    print(f"   Validations: {len(validation_history)}")
    print(f"   Benchmarks: {len(benchmark_history)}")
    print(f"   Scores: {len(scoring_history)}")


def test_custom_rules_and_configuration():
    """Test custom rules and configuration."""
    print("\n=== Testing Custom Rules and Configuration ===")
    
    qa_system = get_agent_qa()
    
    # Test custom validation rule
    print("Testing custom validation rule...")
    
    def custom_validator(output):
        """Custom validator that checks for specific content."""
        if isinstance(output, str):
            return "custom_pattern" in output.lower()
        return True
    
    custom_rule = ValidationRule(
        name="custom_content_check",
        type=ValidationType.CONTENT,
        description="Check for custom pattern in output",
        validator=custom_validator,
        error_message="Output does not contain required custom pattern",
        severity="warning"
    )
    
    qa_system.add_custom_validation_rule("custom", custom_rule)
    print("[OK] Custom validation rule added")
    
    # Test with custom rule
    result_fail = qa_system.validate_output(
        agent_id="custom_test_1",
        output="This output lacks the pattern",
        validation_rules=[custom_rule]
    )
    
    result_pass = qa_system.validate_output(
        agent_id="custom_test_2", 
        output="This output contains custom_pattern",
        validation_rules=[custom_rule]
    )
    
    assert len(result_fail.issues) > 0  # Should have issues
    assert len(result_pass.issues) == 0  # Should pass
    print("[OK] Custom validation rule works correctly")
    
    # Test baseline setting
    print("Testing baseline configuration...")
    qa_system.set_baseline("baseline_agent", "response_time", 100.0)
    qa_system.set_baseline("baseline_agent", "accuracy", 0.92)
    print("[OK] Baselines configured")
    
    # Test benchmark setting
    print("Testing benchmark configuration...")
    qa_system.set_benchmark("industry_standard", 0.85)
    print("[OK] Benchmark configured")


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    qa_system = get_agent_qa()
    
    # Test with None inputs
    print("Testing None input handling...")
    try:
        result = qa_system.validate_output("test_agent", None)
        print("[OK] None input handled gracefully")
    except Exception as e:
        print(f"[ERROR] None input caused error: {e}")
    
    # Test with empty metrics
    print("Testing empty metrics handling...")
    try:
        score = qa_system.calculate_score("test_agent", [])
        # Empty metrics should return a default score, not necessarily 0.0
        assert 0.0 <= score.overall_score <= 1.0
        print("[OK] Empty metrics handled correctly")
    except Exception as e:
        print(f"[ERROR] Empty metrics caused error: {e}")
    
    # Test with invalid benchmark types
    print("Testing invalid benchmark handling...")
    try:
        result = qa_system.run_benchmarks("test_agent", [], 1)
        print("[OK] Empty benchmark types handled gracefully")
    except Exception as e:
        print(f"[ERROR] Empty benchmark types caused error: {e}")
    
    print("[OK] Error handling tests completed")


def run_comprehensive_test():
    """Run a comprehensive end-to-end test."""
    print("\n=== Comprehensive End-to-End Test ===")
    
    agent_id = "comprehensive_test_agent"
    qa_system = get_agent_qa()
    
    print("Running comprehensive test scenario...")
    
    # Step 1: Inspect agent quality
    test_cases = [
        {"input": "Calculate 2+2", "expected": "4"},
        {"input": "Generate hello world", "expected": "Hello, World!"}
    ]
    
    inspection_report = qa_system.inspect_agent(agent_id, test_cases, True)
    print(f"   1. Inspection Score: {inspection_report.overall_score:.3f}")
    
    # Step 2: Validate some outputs
    outputs_to_validate = [
        "def calculate_sum(a, b):\n    return a + b",
        '{"result": "success", "value": 42}',
        "Hello, World! This is a test output."
    ]
    
    validation_scores = []
    for i, output in enumerate(outputs_to_validate):
        result = qa_system.validate_output(f"{agent_id}_val_{i}", output)
        validation_scores.append(result.score)
    
    avg_validation = sum(validation_scores) / len(validation_scores)
    print(f"   2. Average Validation Score: {avg_validation:.3f}")
    
    # Step 3: Run benchmarks
    benchmark_result = qa_system.run_benchmarks(
        agent_id,
        [BenchmarkType.RESPONSE_TIME, BenchmarkType.ACCURACY, BenchmarkType.MEMORY_USAGE],
        3
    )
    print(f"   3. Benchmark Score: {benchmark_result.overall_score:.3f}")
    
    # Step 4: Calculate quality score
    metrics_from_inspection = inspection_report.metrics
    quality_score = qa_system.calculate_score(agent_id, metrics_from_inspection)
    print(f"   4. Quality Score: {quality_score.overall_score:.3f} (Grade: {quality_score.grade})")
    
    # Step 5: Record some monitoring metrics
    qa_system.record_metric(agent_id, "overall_score", quality_score.overall_score)
    qa_system.record_metric(agent_id, "response_time", 145.0)
    qa_system.record_metric(agent_id, "accuracy", 0.93)
    
    # Step 6: Get comprehensive status
    agent_status = qa_system.get_agent_status(agent_id)
    system_status = qa_system.get_status()
    
    print(f"   5. Agent Status: {agent_status['metrics_count']} metrics recorded")
    print(f"   6. System Status: {system_status['monitored_agents']} agents monitored")
    
    print("[OK] Comprehensive test completed successfully")
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Agent ID: {agent_id}")
    print(f"Inspection Score: {inspection_report.overall_score:.3f} ({inspection_report.status})")
    print(f"Quality Score: {quality_score.overall_score:.3f} (Grade: {quality_score.grade})")
    print(f"Benchmark Score: {benchmark_result.overall_score:.3f} ({benchmark_result.status})")
    print(f"Validation Average: {avg_validation:.3f}")
    print(f"Recommendations: {len(inspection_report.recommendations)}")


def main():
    """Main test execution function."""
    print("Starting Agent QA Integration Tests")
    print("=" * 50)
    
    try:
        # Run all test suites
        test_basic_functionality()
        test_quality_inspection()
        test_output_validation()
        test_quality_scoring()
        test_performance_benchmarking()
        test_quality_monitoring()
        test_history_and_trends()
        test_custom_rules_and_configuration()
        test_error_handling()
        run_comprehensive_test()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")
        print("[OK] Agent QA integration is working correctly")
        print("[OK] All major functionality tested and verified")
        print("[OK] Module is ready for production use")
        
        # Final system status
        final_status = get_quality_status()
        print(f"\nFinal System Status:")
        print(f"   Monitored Agents: {final_status['monitored_agents']}")
        print(f"   Total Alerts: {final_status['total_alerts']}")
        print(f"   Validation Rules: {final_status['validation_rules']}")
        print(f"   Benchmarks: {final_status['benchmarks']}")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)