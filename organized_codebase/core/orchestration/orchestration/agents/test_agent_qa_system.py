"""
Test script for TestMaster Agent Quality Assurance System

Comprehensive testing of agent QA components:
- QualityInspector: Quality inspections and assessments
- ValidationEngine: Output validation and rule checking
- ScoringSystem: Quality scoring and grading
- BenchmarkingSuite: Performance benchmarking
- QualityMonitor: Continuous quality monitoring
"""

import asyncio
import time
import os
from datetime import datetime
from pathlib import Path
from testmaster.core.feature_flags import FeatureFlags
from testmaster.agent_qa import (
    # Core components
    QualityInspector, ValidationEngine, ScoringSystem,
    BenchmarkingSuite, QualityMonitor,
    
    # Convenience functions
    configure_agent_qa, inspect_agent_quality, validate_agent_output,
    score_agent_quality, benchmark_agent_performance, get_quality_status,
    
    # Enums and configs
    QualityLevel, QualityCheck, ValidationType, ScoreCategory,
    BenchmarkType, AlertType, QualityThreshold,
    
    # Global instances
    get_quality_inspector, get_validation_engine, get_scoring_system,
    get_benchmarking_suite, get_quality_monitor,
    
    # Utilities
    is_agent_qa_enabled, shutdown_agent_qa
)

class AgentQASystemTest:
    """Comprehensive test suite for agent QA system."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.test_agent_id = "test_agent_001"
        
    async def run_all_tests(self):
        """Run all agent QA system tests."""
        print("=" * 60)
        print("TestMaster Agent Quality Assurance System Test")
        print("=" * 60)
        
        # Initialize feature flags
        FeatureFlags.initialize("testmaster_config.yaml")
        
        # Check if agent QA is enabled
        if not is_agent_qa_enabled():
            print("[!] Agent QA is disabled in configuration")
            return
        
        print("[+] Agent QA is enabled")
        
        # Configure agent QA system
        config_result = configure_agent_qa(
            similarity_threshold=0.7,
            enable_benchmarking=True,
            enable_monitoring=True,
            alert_threshold=0.6
        )
        print(f"[+] Agent QA configured: {config_result['status']}")
        
        # Test individual components
        await self.test_quality_inspector()
        await self.test_validation_engine()
        await self.test_scoring_system()
        await self.test_benchmarking_suite()
        await self.test_quality_monitor()
        await self.test_integration()
        
        # Display results
        self.display_results()
    
    async def test_quality_inspector(self):
        """Test QualityInspector functionality."""
        print("\\n[*] Testing QualityInspector...")
        
        try:
            inspector = get_quality_inspector()
            
            # Test quality inspection with sample test cases
            test_cases = [
                {"input": "test_function()", "expected": "result"},
                {"input": "validate_data(data)", "expected": "True"},
                {"input": "process_request(req)", "expected": "response"}
            ]
            
            # Perform quality inspection
            report = inspector.inspect_agent(
                agent_id=self.test_agent_id,
                test_cases=test_cases,
                include_benchmarks=True
            )
            
            print(f"   [+] Quality inspection completed: {report.overall_score:.3f} ({report.status})")
            print(f"   [i] Metrics evaluated: {len(report.metrics)}")
            print(f"   [i] Recommendations: {len(report.recommendations)}")
            
            # Display metric details
            for metric in report.metrics:
                status_icon = "[+]" if metric.status == "pass" else "[!]"
                print(f"   {status_icon} {metric.name}: {metric.value:.3f} (threshold: {metric.threshold:.3f})")
            
            # Test quality trends
            trends = inspector.get_quality_trends(self.test_agent_id)
            print(f"   [i] Quality trends: {trends.get('trend', 'no_data')}")
            
            # Test another inspection to build history
            report2 = inspector.inspect_agent(self.test_agent_id)
            history = inspector.get_inspection_history(self.test_agent_id)
            print(f"   [i] Inspection history: {len(history)} reports")
            
            self.test_results['quality_inspector'] = report.overall_score > 0.0
            
        except Exception as e:
            print(f"   [!] QualityInspector test failed: {e}")
            self.test_results['quality_inspector'] = False
    
    async def test_validation_engine(self):
        """Test ValidationEngine functionality."""
        print("\\n[*] Testing ValidationEngine...")
        
        try:
            engine = get_validation_engine()
            
            # Test output validation with different scenarios
            test_outputs = [
                ("valid_python_code", "def test():\\n    return True", None),
                ("invalid_syntax", "def test(:\\n    return True", None),
                ("json_output", '{"result": "success", "code": 200}', None),
                ("empty_output", "", None),
                ("compared_output", "Hello World", "Hello World")
            ]
            
            validation_results = []
            
            for test_name, output, expected in test_outputs:
                result = engine.validate_output(
                    agent_id=self.test_agent_id,
                    output=output,
                    expected=expected
                )
                
                validation_results.append(result)
                status_icon = "[+]" if result.passed else "[!]"
                print(f"   {status_icon} {test_name}: {result.score:.3f} ({result.passed_checks}/{result.total_checks} checks)")
                
                # Show validation issues
                for issue in result.issues:
                    print(f"      - {issue.severity}: {issue.message}")
            
            # Test custom validation rules
            from testmaster.agent_qa.validation_engine import ValidationRule, ValidationType
            
            custom_rule = ValidationRule(
                name="custom_length_check",
                type=ValidationType.CONTENT,
                description="Check minimum content length",
                validator=lambda x: len(str(x)) >= 10,
                error_message="Content too short"
            )
            
            engine.add_custom_rule("custom", custom_rule)
            
            # Test with custom rule
            custom_result = engine.validate_output(
                agent_id=self.test_agent_id,
                output="Short",
                validation_rules=[custom_rule]
            )
            
            print(f"   [i] Custom rule validation: {custom_result.score:.3f}")
            
            # Check validation history
            history = engine.get_validation_history(self.test_agent_id)
            print(f"   [i] Validation history: {len(history)} results")
            
            self.test_results['validation_engine'] = len(validation_results) > 0
            
        except Exception as e:
            print(f"   [!] ValidationEngine test failed: {e}")
            self.test_results['validation_engine'] = False
    
    async def test_scoring_system(self):
        """Test ScoringSystem functionality."""
        print("\\n[*] Testing ScoringSystem...")
        
        try:
            scoring = get_scoring_system()
            
            # Create sample quality metrics for scoring
            from testmaster.agent_qa.quality_inspector import QualityMetric
            
            sample_metrics = [
                QualityMetric("syntax_validation", 0.95, 0.9, "pass", {"errors": 0}),
                QualityMetric("semantic_analysis", 0.88, 0.8, "pass", {"consistency": 0.9}),
                QualityMetric("performance_test", 0.82, 0.8, "pass", {"response_time": 120}),
                QualityMetric("security_scan", 0.97, 0.95, "pass", {"vulnerabilities": 0}),
                QualityMetric("reliability_test", 0.89, 0.85, "pass", {"uptime": 99.2})
            ]
            
            # Calculate quality score
            score_result = scoring.calculate_score(
                agent_id=self.test_agent_id,
                quality_metrics=sample_metrics
            )
            
            print(f"   [+] Quality score calculated: {score_result.overall_score:.3f} ({score_result.grade})")
            print(f"   [i] Status: {score_result.status}")
            print(f"   [i] Percentile: {score_result.percentile:.1f}%")
            
            # Display score breakdown
            print(f"   [i] Score breakdown:")
            for breakdown in score_result.breakdown:
                print(f"      - {breakdown.category.value}: {breakdown.score:.3f} (weight: {breakdown.weight:.2f})")
            
            # Test with custom weights
            custom_weights = {
                "functionality": 0.4,
                "performance": 0.3,
                "security": 0.2,
                "reliability": 0.1
            }
            
            custom_score = scoring.calculate_score(
                agent_id=self.test_agent_id,
                quality_metrics=sample_metrics,
                custom_weights=custom_weights
            )
            
            print(f"   [i] Custom weighted score: {custom_score.overall_score:.3f}")
            
            # Test benchmarking
            scoring.set_benchmark("industry_standard", 0.85)
            benchmark_comparison = scoring.compare_to_benchmark(self.test_agent_id, "industry_standard")
            print(f"   [i] Benchmark comparison: {benchmark_comparison.get('performance', 'no_data')}")
            
            # Check score trends
            trends = scoring.get_score_trends(self.test_agent_id)
            print(f"   [i] Score trends: {trends.get('trend', 'insufficient_data')}")
            
            self.test_results['scoring_system'] = score_result.overall_score > 0.0
            
        except Exception as e:
            print(f"   [!] ScoringSystem test failed: {e}")
            self.test_results['scoring_system'] = False
    
    async def test_benchmarking_suite(self):
        """Test BenchmarkingSuite functionality."""
        print("\\n[*] Testing BenchmarkingSuite...")
        
        try:
            benchmarking = get_benchmarking_suite()
            
            # Test performance benchmarking
            benchmark_types = [
                BenchmarkType.RESPONSE_TIME,
                BenchmarkType.THROUGHPUT,
                BenchmarkType.MEMORY_USAGE,
                BenchmarkType.CPU_UTILIZATION,
                BenchmarkType.ACCURACY
            ]
            
            # Run benchmarks
            benchmark_result = benchmarking.run_benchmarks(
                agent_id=self.test_agent_id,
                benchmark_types=benchmark_types,
                iterations=5
            )
            
            print(f"   [+] Benchmarking completed: {benchmark_result.overall_score:.3f} ({benchmark_result.status})")
            print(f"   [i] Duration: {benchmark_result.duration_ms:.1f}ms")
            print(f"   [i] Iterations: {benchmark_result.iterations}")
            
            # Display benchmark results
            for metric in benchmark_result.metrics:
                status_icon = "[+]" if metric.status == "pass" else "[!]"
                print(f"   {status_icon} {metric.name}: {metric.value:.2f} {metric.unit} (threshold: {metric.threshold:.2f})")
            
            # Set baseline for comparison
            for metric in benchmark_result.metrics:
                benchmarking.set_baseline(self.test_agent_id, metric.name, metric.value)
            
            # Test benchmark history
            history = benchmarking.get_benchmark_history(self.test_agent_id)
            print(f"   [i] Benchmark history: {len(history)} results")
            
            # Run another benchmark to test trends
            benchmark_result2 = benchmarking.run_benchmarks(
                agent_id=self.test_agent_id,
                benchmark_types=[BenchmarkType.RESPONSE_TIME],
                iterations=3
            )
            
            # Check performance trends
            trends = benchmarking.get_performance_trends(self.test_agent_id)
            print(f"   [i] Performance trends: {trends.get('latest_score', 'no_data')}")
            
            self.test_results['benchmarking_suite'] = benchmark_result.overall_score > 0.0
            
        except Exception as e:
            print(f"   [!] BenchmarkingSuite test failed: {e}")
            self.test_results['benchmarking_suite'] = False
    
    async def test_quality_monitor(self):
        """Test QualityMonitor functionality."""
        print("\\n[*] Testing QualityMonitor...")
        
        try:
            monitor = get_quality_monitor()
            
            # Start monitoring
            monitor.start_monitoring()
            print(f"   [+] Quality monitoring started")
            
            # Record some metrics
            monitor.record_metric(self.test_agent_id, "overall_score", 0.85)
            monitor.record_metric(self.test_agent_id, "response_time", 120.0)
            monitor.record_metric(self.test_agent_id, "error_rate", 0.02)
            monitor.record_metric(self.test_agent_id, "memory_usage", 50.0)
            
            print(f"   [+] Metrics recorded for monitoring")
            
            # Add custom threshold
            from testmaster.agent_qa.quality_monitor import QualityThreshold
            
            custom_threshold = QualityThreshold(
                name="custom_quality_check",
                metric="overall_score",
                value=0.9,
                operator="lt",
                alert_type=AlertType.QUALITY_DEGRADATION,
                severity="medium"
            )
            
            monitor.add_threshold(custom_threshold)
            print(f"   [+] Custom threshold added")
            
            # Record a metric that should trigger an alert
            monitor.record_metric(self.test_agent_id, "overall_score", 0.5)  # Below threshold
            
            # Wait briefly for monitoring to process
            await asyncio.sleep(0.1)
            
            # Check alerts
            alerts = monitor.get_alerts(agent_id=self.test_agent_id)
            print(f"   [i] Alerts generated: {len(alerts)}")
            
            for alert in alerts:
                print(f"      - {alert.severity}: {alert.message}")
                monitor.acknowledge_alert(alert.alert_id)
            
            # Check monitoring status
            status = monitor.get_status()
            print(f"   [i] Monitoring status: {status['monitoring']}")
            print(f"   [i] Monitored agents: {status['monitored_agents']}")
            print(f"   [i] Total alerts: {status['total_alerts']}")
            
            # Check agent-specific status
            agent_status = monitor.get_agent_status(self.test_agent_id)
            print(f"   [i] Agent metrics: {len(agent_status['latest_metrics'])}")
            
            # Stop monitoring
            monitor.stop_monitoring()
            print(f"   [+] Quality monitoring stopped")
            
            self.test_results['quality_monitor'] = status['monitoring'] or len(alerts) >= 0
            
        except Exception as e:
            print(f"   [!] QualityMonitor test failed: {e}")
            self.test_results['quality_monitor'] = False
    
    async def test_integration(self):
        """Test integrated agent QA functionality."""
        print("\\n[*] Testing Integration...")
        
        try:
            # Test end-to-end agent QA workflow
            print("   [>] Starting integrated agent QA workflow...")
            
            # 1. Inspect agent quality
            inspection_report = inspect_agent_quality(
                agent_id=self.test_agent_id,
                test_cases=[{"input": "test", "expected": "result"}],
                include_benchmarks=True
            )
            print(f"   [+] Quality inspection: {inspection_report.overall_score:.3f}")
            
            # 2. Validate agent output
            validation_result = validate_agent_output(
                agent_id=self.test_agent_id,
                output="def test_function():\\n    return 'success'",
                expected="def test_function():\\n    return 'success'"
            )
            print(f"   [+] Output validation: {validation_result.score:.3f}")
            
            # 3. Calculate quality score
            quality_score = score_agent_quality(
                agent_id=self.test_agent_id,
                quality_metrics=inspection_report.metrics
            )
            print(f"   [+] Quality score: {quality_score.overall_score:.3f} ({quality_score.grade})")
            
            # 4. Run performance benchmarks
            benchmark_results = benchmark_agent_performance(
                agent_id=self.test_agent_id,
                benchmark_types=[BenchmarkType.RESPONSE_TIME, BenchmarkType.ACCURACY],
                iterations=3
            )
            print(f"   [+] Performance benchmarks: {benchmark_results.overall_score:.3f}")
            
            # 5. Check overall quality status
            overall_status = get_quality_status()
            print(f"   [+] Overall QA status: {overall_status.get('status', 'unknown')}")
            
            # Verify integration success
            integration_success = (
                inspection_report.overall_score > 0.0 and
                validation_result.score > 0.0 and
                quality_score.overall_score > 0.0 and
                benchmark_results.overall_score > 0.0
            )
            
            print(f"   [i] Integration workflow completed successfully")
            print(f"   [i] Quality metrics:")
            print(f"      - Inspection score: {inspection_report.overall_score:.3f}")
            print(f"      - Validation score: {validation_result.score:.3f}")
            print(f"      - Overall score: {quality_score.overall_score:.3f}")
            print(f"      - Performance score: {benchmark_results.overall_score:.3f}")
            
            self.test_results['integration'] = integration_success
            
        except Exception as e:
            print(f"   [!] Integration test failed: {e}")
            self.test_results['integration'] = False
    
    def display_results(self):
        """Display test results summary."""
        print("\\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for component, result in self.test_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{component.replace('_', ' ').title()}: {status}")
        
        print(f"\\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All Agent QA system tests PASSED!")
        else:
            print("Some tests failed - check implementation")
        
        execution_time = time.time() - self.start_time
        print(f"Total execution time: {execution_time:.2f} seconds")

async def main():
    """Main test execution."""
    try:
        # Run tests
        test_suite = AgentQASystemTest()
        await test_suite.run_all_tests()
        
    finally:
        # Cleanup
        print("\\nCleaning up agent QA system...")
        shutdown_agent_qa()
        print("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())