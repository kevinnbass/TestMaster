#!/usr/bin/env python3
"""
🏗️ MODULE: Performance Validation Core - Main Framework Orchestration
==================================================================

📋 PURPOSE:
    Main orchestration layer for the performance validation framework.
    Coordinates all performance testing components and provides unified interface.

🎯 CORE FUNCTIONALITY:
    • Framework initialization and configuration
    • Component orchestration and coordination
    • Comprehensive performance validation workflow
    • Result aggregation and reporting

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 07:45:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract main orchestration from performance_validation_framework.py via STEELCLAD
   └─ Changes: Created dedicated module for framework orchestration
   └─ Impact: Clean separation of orchestration from individual components

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: All performance modules, logging, json
🎯 Integration Points: All performance validation child modules
⚡ Performance Notes: Orchestration layer with minimal processing overhead
🔒 Security Notes: Configuration validation and safe file operations

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via framework validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: All performance validation child modules
📤 Provides: Complete performance validation framework capabilities
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Import all child modules
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_models import PerformanceTest, PerformanceResult, LoadTestScenario
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_benchmarker import PerformanceBenchmarker
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.cc_1.load_test_executor import LoadTestExecutor
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_regression_detector import PerformanceRegressionDetector


class PerformanceValidationFramework:
    """Main performance validation framework orchestrator"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the performance validation framework"""
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('PerformanceValidationFramework')
        
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Initialize components
        self.benchmarker = PerformanceBenchmarker(
            output_dir=self.config.get('output_dir', 'performance_validation')
        )
        self.load_test_executor = LoadTestExecutor()
        self.regression_detector = PerformanceRegressionDetector(self.benchmarker)
        
        # Framework state
        self.validation_history = []
        self.active_validations = {}
        
        self.logger.info("Performance Validation Framework initialized")
    
    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load framework configuration"""
        default_config = {
            'output_dir': 'performance_validation',
            'auto_detect_regressions': True,
            'regression_thresholds': {
                'response_time_increase_percent': 10.0,
                'throughput_decrease_percent': 10.0,
                'error_rate_increase_percent': 5.0,
                'memory_increase_percent': 20.0
            },
            'default_test_duration_seconds': 60,
            'default_concurrent_users': 10
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config file {config_file}: {str(e)}")
        
        return default_config
    
    def register_performance_test(self, test: PerformanceTest):
        """Register a new performance test"""
        self.benchmarker.register_test(test)
        self.logger.info(f"Registered performance test: {test.test_name}")
    
    def run_performance_validation(self, test_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run complete performance validation suite"""
        
        validation_id = f"validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting performance validation: {validation_id}")
        
        # Track active validation
        self.active_validations[validation_id] = {
            'start_time': datetime.now(timezone.utc),
            'status': 'running'
        }
        
        validation_results = {
            'validation_id': validation_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'benchmark_results': {},
            'regression_analysis': {},
            'summary': {}
        }
        
        try:
            # Run benchmark suite
            benchmark_results = self.benchmarker.execute_benchmark_suite(test_ids)
            validation_results['benchmark_results'] = benchmark_results
            
            # Convert results to dict for regression detection
            current_results = {}
            for result in benchmark_results.get('results', []):
                if isinstance(result, dict) and 'test_id' in result:
                    current_results[result['test_id']] = self._dict_to_result(result)
            
            # Perform regression detection if enabled
            if self.config.get('auto_detect_regressions', True):
                regression_report = self.regression_detector.detect_regressions(current_results)
                validation_results['regression_analysis'] = regression_report
                
                # Log regressions
                if regression_report['regressions_detected']:
                    self.logger.warning(
                        f"Performance regressions detected: {len(regression_report['regressions_detected'])}"
                    )
                    for regression in regression_report['regressions_detected']:
                        self.logger.warning(
                            f"  - {regression['test_id']}: {regression['regression_type']} "
                            f"regression ({regression['change_percent']:.1f}% change)"
                        )
            
            # Generate summary
            validation_results['summary'] = self._generate_validation_summary(validation_results)
            
            # Mark validation as complete
            self.active_validations[validation_id]['status'] = 'completed'
            self.active_validations[validation_id]['end_time'] = datetime.now(timezone.utc)
            
            # Store in history
            self.validation_history.append(validation_results)
            
            # Save results to file
            self._save_validation_results(validation_results)
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {str(e)}")
            self.active_validations[validation_id]['status'] = 'failed'
            self.active_validations[validation_id]['error'] = str(e)
            raise
        
        finally:
            # Clean up active validation
            if validation_id in self.active_validations:
                del self.active_validations[validation_id]
        
        self.logger.info(f"Performance validation completed: {validation_id}")
        return validation_results
    
    async def run_load_test(self, scenario: LoadTestScenario) -> Dict[str, Any]:
        """Run an async load test scenario"""
        self.logger.info(f"Starting load test: {scenario.name}")
        
        try:
            results = await self.load_test_executor.execute_http_load_test(scenario)
            
            # Analyze load test results
            if results.get('summary'):
                summary = results['summary']
                self.logger.info(
                    f"Load test completed: {summary.get('total_requests', 0)} requests, "
                    f"{summary.get('avg_response_time_ms', 0):.2f}ms avg response time, "
                    f"{summary.get('error_rate_percent', 0):.2f}% error rate"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Load test failed: {str(e)}")
            raise
    
    def analyze_performance_trends(self, test_id: str, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends for a specific test"""
        return self.regression_detector.get_trend_analysis(test_id, days)
    
    def _dict_to_result(self, result_dict: Dict[str, Any]) -> PerformanceResult:
        """Convert dictionary to PerformanceResult object"""
        return PerformanceResult(
            test_id=result_dict.get('test_id', ''),
            execution_timestamp=datetime.fromisoformat(result_dict.get('execution_timestamp', datetime.now(timezone.utc).isoformat())),
            duration_seconds=result_dict.get('duration_seconds', 0.0),
            total_requests=result_dict.get('total_requests', 0),
            successful_requests=result_dict.get('successful_requests', 0),
            failed_requests=result_dict.get('failed_requests', 0),
            avg_response_time_ms=result_dict.get('avg_response_time_ms', 0.0),
            min_response_time_ms=result_dict.get('min_response_time_ms', 0.0),
            max_response_time_ms=result_dict.get('max_response_time_ms', 0.0),
            p50_response_time_ms=result_dict.get('p50_response_time_ms', 0.0),
            p95_response_time_ms=result_dict.get('p95_response_time_ms', 0.0),
            p99_response_time_ms=result_dict.get('p99_response_time_ms', 0.0),
            throughput_rps=result_dict.get('throughput_rps', 0.0),
            error_rate_percent=result_dict.get('error_rate_percent', 0.0),
            memory_usage_mb=result_dict.get('memory_usage_mb', 0.0),
            cpu_usage_percent=result_dict.get('cpu_usage_percent', 0.0),
            errors=result_dict.get('errors', []),
            success=result_dict.get('success', True)
        )
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            'validation_id': validation_results['validation_id'],
            'timestamp': validation_results['timestamp'],
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'regressions_detected': 0,
            'improvements_detected': 0,
            'overall_status': 'passed'
        }
        
        # Summarize benchmark results
        if 'benchmark_results' in validation_results:
            benchmark = validation_results['benchmark_results']
            if 'results' in benchmark:
                summary['total_tests'] = len(benchmark['results'])
                summary['passed_tests'] = len([r for r in benchmark['results'] if r.get('success', False)])
                summary['failed_tests'] = summary['total_tests'] - summary['passed_tests']
        
        # Summarize regression analysis
        if 'regression_analysis' in validation_results:
            regression = validation_results['regression_analysis']
            summary['regressions_detected'] = regression.get('summary', {}).get('regressions_count', 0)
            summary['improvements_detected'] = regression.get('summary', {}).get('improvements_count', 0)
        
        # Determine overall status
        if summary['failed_tests'] > 0 or summary['regressions_detected'] > 0:
            summary['overall_status'] = 'failed'
        elif summary['improvements_detected'] > 0:
            summary['overall_status'] = 'improved'
        
        return summary
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file"""
        output_dir = Path(self.config.get('output_dir', 'performance_validation'))
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = output_dir / f"{results['validation_id']}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Validation results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save validation results: {str(e)}")
    
    def get_validation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent validation history"""
        return self.validation_history[-limit:]
    
    def get_active_validations(self) -> Dict[str, Any]:
        """Get currently active validations"""
        return self.active_validations.copy()