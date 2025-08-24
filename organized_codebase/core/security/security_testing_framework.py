#!/usr/bin/env python3
"""
Comprehensive Security Testing Framework
Agent D Hour 5 - Advanced Security Validation and Testing

Implements comprehensive testing framework for security system validation
following STEELCLAD Anti-Regression Modularization Protocol.
"""

import asyncio
import datetime
import json
import logging
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

# Import security modules
from .monitoring_modules.security_events import SecurityEvent, ThreatLevel, ResponseAction
from .monitoring_modules.threat_detector import ThreatDetectionEngine, ThreatPattern
from .predictive_security_analytics import PredictiveAnalyticsEngine

class TestCategory(Enum):
    """Categories of security tests"""
    VULNERABILITY_DETECTION = "VULNERABILITY_DETECTION"
    RESPONSE_VALIDATION = "RESPONSE_VALIDATION"
    CORRELATION_ACCURACY = "CORRELATION_ACCURACY"
    PERFORMANCE_IMPACT = "PERFORMANCE_IMPACT"
    PREDICTIVE_ACCURACY = "PREDICTIVE_ACCURACY"
    INTEGRATION_TESTING = "INTEGRATION_TESTING"
    REGRESSION_TESTING = "REGRESSION_TESTING"
    STRESS_TESTING = "STRESS_TESTING"

class TestResult(Enum):
    """Test result outcomes"""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"

@dataclass
class SecurityTestCase:
    """Security test case definition"""
    test_id: str
    test_name: str
    category: TestCategory
    description: str
    test_function: str
    expected_outcome: str
    threat_level: ThreatLevel
    test_data: Dict[str, Any]
    timeout_seconds: int = 300
    requires_isolation: bool = False
    dependencies: List[str] = None

@dataclass
class SecurityTestResult:
    """Security test execution result"""
    test_id: str
    test_name: str
    category: TestCategory
    result: TestResult
    execution_time_ms: int
    details: Dict[str, Any]
    error_message: Optional[str] = None
    warnings: List[str] = None
    metrics: Dict[str, float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().isoformat()
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = {}

@dataclass
class TestSuiteResult:
    """Test suite execution results"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    execution_time_ms: int
    test_results: List[SecurityTestResult]
    coverage_metrics: Dict[str, float]
    security_score: float
    timestamp: str

class SecurityTestFramework:
    """Comprehensive security testing framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize security testing framework"""
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Test components
        self.test_cases = {}
        self.test_suites = {}
        self.test_data_generators = {}
        self.mock_components = {}
        
        # Execution state
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_test_workers', 4))
        self.test_isolation_env = {}
        
        # Results storage
        self.test_results_dir = Path(__file__).parent / "test_results"
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Initialize test registry
        self._initialize_test_registry()
        
        self.logger.info("Security Testing Framework initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default testing configuration"""
        return {
            'max_test_workers': 4,
            'default_timeout': 300,
            'isolation_enabled': True,
            'parallel_execution': True,
            'coverage_threshold': 0.8,
            'security_score_threshold': 0.85,
            'test_data_retention_days': 30,
            'performance_baseline_samples': 10
        }
    
    def _initialize_test_registry(self):
        """Initialize comprehensive test case registry"""
        # Vulnerability detection tests
        self._register_vulnerability_tests()
        
        # Response validation tests
        self._register_response_tests()
        
        # Correlation accuracy tests
        self._register_correlation_tests()
        
        # Performance impact tests
        self._register_performance_tests()
        
        # Predictive accuracy tests
        self._register_predictive_tests()
        
        # Integration tests
        self._register_integration_tests()
        
        # Regression tests
        self._register_regression_tests()
        
        # Stress tests
        self._register_stress_tests()
    
    def _register_vulnerability_tests(self):
        """Register vulnerability detection test cases"""
        # Code injection detection test
        self.register_test_case(SecurityTestCase(
            test_id="vuln_001_code_injection",
            test_name="Code Injection Detection",
            category=TestCategory.VULNERABILITY_DETECTION,
            description="Test detection of code injection vulnerabilities",
            test_function="test_code_injection_detection",
            expected_outcome="CRITICAL threat detected with high confidence",
            threat_level=ThreatLevel.CRITICAL,
            test_data={
                'malicious_code': [
                    'eval(user_input)',
                    'exec(malicious_payload)',
                    'subprocess.call(cmd, shell=True)',
                    '__import__("os").system("rm -rf /")'
                ],
                'safe_code': [
                    'json.loads(user_input)',
                    'ast.literal_eval(safe_data)',
                    'validated_function(sanitized_input)'
                ]
            },
            timeout_seconds=60
        ))
        
        # SQL injection detection test
        self.register_test_case(SecurityTestCase(
            test_id="vuln_002_sql_injection",
            test_name="SQL Injection Detection",
            description="Test detection of SQL injection patterns",
            category=TestCategory.VULNERABILITY_DETECTION,
            test_function="test_sql_injection_detection",
            expected_outcome="CRITICAL threat detected for injection patterns",
            threat_level=ThreatLevel.CRITICAL,
            test_data={
                'injection_patterns': [
                    "SELECT * FROM users WHERE id = %s" % user_input,
                    "UPDATE users SET password = '%s' WHERE id = %d" % (new_pass, user_id),
                    "'; DROP TABLE users; --",
                    "1' OR '1'='1"
                ],
                'safe_patterns': [
                    "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                    "User.objects.filter(id=user_id)"
                ]
            }
        ))
        
        # Hardcoded secrets detection test
        self.register_test_case(SecurityTestCase(
            test_id="vuln_003_hardcoded_secrets",
            test_name="Hardcoded Secrets Detection",
            category=TestCategory.VULNERABILITY_DETECTION,
            description="Test detection of hardcoded credentials and secrets",
            test_function="test_hardcoded_secrets_detection",
            expected_outcome="HIGH threat detected for hardcoded secrets",
            threat_level=ThreatLevel.HIGH,
            test_data={
                'secret_patterns': [
                    'password = "supersecret123"',
                    'api_key = "ak_1234567890abcdef"',
                    'private_key = "-----BEGIN PRIVATE KEY-----"',
                    'token = "ghp_abcd1234567890"'
                ],
                'acceptable_patterns': [
                    'password = os.getenv("PASSWORD")',
                    'api_key = config.get("api_key")',
                    'token = request.headers.get("Authorization")'
                ]
            }
        ))
    
    def _register_response_tests(self):
        """Register automated response validation tests"""
        self.register_test_case(SecurityTestCase(
            test_id="resp_001_quarantine",
            test_name="Quarantine Response Validation",
            category=TestCategory.RESPONSE_VALIDATION,
            description="Test quarantine response execution and file isolation",
            test_function="test_quarantine_response",
            expected_outcome="File successfully quarantined with metadata",
            threat_level=ThreatLevel.CRITICAL,
            test_data={
                'test_file_content': 'eval("malicious_code")',
                'expected_quarantine_location': 'QUARANTINE/',
                'metadata_required': ['event_id', 'original_path', 'quarantine_time']
            },
            requires_isolation=True
        ))
        
        self.register_test_case(SecurityTestCase(
            test_id="resp_002_alert_generation",
            test_name="Alert Generation Response",
            category=TestCategory.RESPONSE_VALIDATION,
            description="Test alert generation and notification systems",
            test_function="test_alert_generation",
            expected_outcome="Alert generated with proper format and content",
            threat_level=ThreatLevel.HIGH,
            test_data={
                'alert_channels': ['file', 'log', 'notification'],
                'required_fields': ['timestamp', 'threat_level', 'description', 'evidence']
            }
        ))
        
        self.register_test_case(SecurityTestCase(
            test_id="resp_003_escalation",
            test_name="Response Escalation Testing",
            category=TestCategory.RESPONSE_VALIDATION,
            description="Test proper escalation of security responses",
            test_function="test_response_escalation",
            expected_outcome="Responses escalate appropriately based on threat level",
            threat_level=ThreatLevel.EMERGENCY,
            test_data={
                'escalation_chain': ['LOG_ONLY', 'ALERT', 'QUARANTINE', 'BLOCK', 'EMERGENCY_SHUTDOWN'],
                'escalation_timeouts': [60, 300, 600, 1800, 3600]
            }
        ))
    
    def _register_correlation_tests(self):
        """Register correlation accuracy tests"""
        self.register_test_case(SecurityTestCase(
            test_id="corr_001_temporal",
            test_name="Temporal Correlation Accuracy",
            category=TestCategory.CORRELATION_ACCURACY,
            description="Test accuracy of temporal event correlation",
            test_function="test_temporal_correlation",
            expected_outcome="Related events correlated within time window",
            threat_level=ThreatLevel.MEDIUM,
            test_data={
                'time_windows': [60, 300, 600, 3600],
                'event_sequences': [
                    ['login_attempt', 'failed_auth', 'account_lockout'],
                    ['file_access', 'permission_denied', 'privilege_escalation']
                ]
            }
        ))
        
        self.register_test_case(SecurityTestCase(
            test_id="corr_002_pattern_matching",
            test_name="Pattern Matching Correlation",
            category=TestCategory.CORRELATION_ACCURACY,
            description="Test pattern-based event correlation accuracy",
            test_function="test_pattern_correlation",
            expected_outcome="Attack patterns correctly identified and correlated",
            threat_level=ThreatLevel.HIGH,
            test_data={
                'attack_patterns': [
                    'reconnaissance_scan',
                    'vulnerability_exploitation',
                    'privilege_escalation',
                    'data_exfiltration'
                ],
                'correlation_confidence_threshold': 0.8
            }
        ))
    
    def _register_performance_tests(self):
        """Register performance impact tests"""
        self.register_test_case(SecurityTestCase(
            test_id="perf_001_scan_performance",
            test_name="Security Scan Performance Impact",
            category=TestCategory.PERFORMANCE_IMPACT,
            description="Measure performance impact of security scanning",
            test_function="test_scan_performance_impact",
            expected_outcome="CPU impact < 15%, Memory impact < 100MB",
            threat_level=ThreatLevel.INFO,
            test_data={
                'cpu_threshold': 15.0,
                'memory_threshold_mb': 100.0,
                'scan_duration_seconds': 60,
                'baseline_samples': 10
            }
        ))
        
        self.register_test_case(SecurityTestCase(
            test_id="perf_002_correlation_performance",
            test_name="Correlation Engine Performance",
            category=TestCategory.PERFORMANCE_IMPACT,
            description="Test correlation engine performance under load",
            test_function="test_correlation_performance",
            expected_outcome="Correlation latency < 200ms, Memory efficient",
            threat_level=ThreatLevel.INFO,
            test_data={
                'latency_threshold_ms': 200.0,
                'event_load_counts': [100, 500, 1000, 5000],
                'memory_growth_threshold_mb': 50.0
            }
        ))
    
    def _register_predictive_tests(self):
        """Register predictive analytics accuracy tests"""
        self.register_test_case(SecurityTestCase(
            test_id="pred_001_threat_prediction",
            test_name="Threat Probability Prediction Accuracy",
            category=TestCategory.PREDICTIVE_ACCURACY,
            description="Test accuracy of threat probability predictions",
            test_function="test_threat_prediction_accuracy",
            expected_outcome="Prediction accuracy > 70%",
            threat_level=ThreatLevel.MEDIUM,
            test_data={
                'accuracy_threshold': 0.7,
                'prediction_horizons': [1, 4, 12, 24],
                'historical_data_size': 1000
            }
        ))
        
        self.register_test_case(SecurityTestCase(
            test_id="pred_002_false_positive_rate",
            test_name="Prediction False Positive Rate",
            category=TestCategory.PREDICTIVE_ACCURACY,
            description="Test false positive rate of security predictions",
            test_function="test_prediction_false_positive_rate",
            expected_outcome="False positive rate < 10%",
            threat_level=ThreatLevel.MEDIUM,
            test_data={
                'false_positive_threshold': 0.1,
                'test_scenarios': 100,
                'confidence_levels': ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
            }
        ))
    
    def _register_integration_tests(self):
        """Register integration tests"""
        self.register_test_case(SecurityTestCase(
            test_id="intg_001_end_to_end",
            test_name="End-to-End Security Pipeline",
            category=TestCategory.INTEGRATION_TESTING,
            description="Test complete security pipeline from detection to response",
            test_function="test_end_to_end_pipeline",
            expected_outcome="Complete pipeline execution within performance limits",
            threat_level=ThreatLevel.CRITICAL,
            test_data={
                'pipeline_stages': ['detection', 'correlation', 'prediction', 'response'],
                'max_pipeline_time_seconds': 30,
                'test_threat_scenarios': 5
            },
            timeout_seconds=120
        ))
        
        self.register_test_case(SecurityTestCase(
            test_id="intg_002_component_interaction",
            test_name="Component Interaction Testing",
            category=TestCategory.INTEGRATION_TESTING,
            description="Test interactions between security components",
            test_function="test_component_interactions",
            expected_outcome="All components interact correctly with proper data flow",
            threat_level=ThreatLevel.MEDIUM,
            test_data={
                'component_pairs': [
                    ('detector', 'correlator'),
                    ('correlator', 'predictor'),
                    ('predictor', 'responder'),
                    ('monitor', 'dashboard')
                ]
            }
        ))
    
    def _register_regression_tests(self):
        """Register regression tests"""
        self.register_test_case(SecurityTestCase(
            test_id="regr_001_detection_regression",
            test_name="Threat Detection Regression",
            category=TestCategory.REGRESSION_TESTING,
            description="Ensure threat detection capabilities are not degraded",
            test_function="test_detection_regression",
            expected_outcome="All historical threats still detected with same accuracy",
            threat_level=ThreatLevel.HIGH,
            test_data={
                'historical_test_cases': 'load_from_regression_database',
                'accuracy_tolerance': 0.05,
                'minimum_detection_rate': 0.95
            }
        ))
    
    def _register_stress_tests(self):
        """Register stress tests"""
        self.register_test_case(SecurityTestCase(
            test_id="stress_001_high_load",
            test_name="High Load Stress Test",
            category=TestCategory.STRESS_TESTING,
            description="Test system behavior under high security event load",
            test_function="test_high_load_stress",
            expected_outcome="System maintains performance under 10x normal load",
            threat_level=ThreatLevel.INFO,
            test_data={
                'load_multipliers': [2, 5, 10, 20],
                'duration_minutes': 10,
                'performance_degradation_threshold': 0.3
            },
            timeout_seconds=1200
        ))
    
    def register_test_case(self, test_case: SecurityTestCase):
        """Register a test case"""
        if test_case.dependencies is None:
            test_case.dependencies = []
        self.test_cases[test_case.test_id] = test_case
        self.logger.debug(f"Registered test case: {test_case.test_id}")
    
    def register_test_suite(self, suite_name: str, test_ids: List[str]):
        """Register a test suite"""
        self.test_suites[suite_name] = test_ids
        self.logger.info(f"Registered test suite '{suite_name}' with {len(test_ids)} tests")
    
    async def run_test_case(self, test_id: str) -> SecurityTestResult:
        """Run a single test case"""
        test_case = self.test_cases.get(test_id)
        if not test_case:
            return SecurityTestResult(
                test_id=test_id,
                test_name="Unknown Test",
                category=TestCategory.VULNERABILITY_DETECTION,
                result=TestResult.ERROR,
                execution_time_ms=0,
                details={},
                error_message=f"Test case {test_id} not found"
            )
        
        start_time = datetime.datetime.now()
        
        try:
            # Setup test isolation if required
            if test_case.requires_isolation:
                isolation_context = await self._setup_test_isolation(test_case)
            else:
                isolation_context = None
            
            # Execute test function
            test_function = getattr(self, test_case.test_function)
            result = await asyncio.wait_for(
                test_function(test_case),
                timeout=test_case.timeout_seconds
            )
            
            execution_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            # Cleanup isolation
            if isolation_context:
                await self._cleanup_test_isolation(isolation_context)
            
            return SecurityTestResult(
                test_id=test_id,
                test_name=test_case.test_name,
                category=test_case.category,
                result=result['result'],
                execution_time_ms=int(execution_time),
                details=result.get('details', {}),
                error_message=result.get('error_message'),
                warnings=result.get('warnings', []),
                metrics=result.get('metrics', {})
            )
            
        except asyncio.TimeoutError:
            execution_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            return SecurityTestResult(
                test_id=test_id,
                test_name=test_case.test_name,
                category=test_case.category,
                result=TestResult.ERROR,
                execution_time_ms=int(execution_time),
                details={},
                error_message=f"Test timed out after {test_case.timeout_seconds} seconds"
            )
            
        except Exception as e:
            execution_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            return SecurityTestResult(
                test_id=test_id,
                test_name=test_case.test_name,
                category=test_case.category,
                result=TestResult.ERROR,
                execution_time_ms=int(execution_time),
                details={},
                error_message=str(e)
            )
    
    async def run_test_suite(self, suite_name: str) -> TestSuiteResult:
        """Run a complete test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        test_ids = self.test_suites[suite_name]
        start_time = datetime.datetime.now()
        
        # Run tests (parallel or sequential based on config)
        if self.config.get('parallel_execution', True):
            tasks = [self.run_test_case(test_id) for test_id in test_ids]
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            test_results = []
            for test_id in test_ids:
                result = await self.run_test_case(test_id)
                test_results.append(result)
        
        # Process results
        valid_results = [r for r in test_results if isinstance(r, SecurityTestResult)]
        
        # Calculate statistics
        total_tests = len(valid_results)
        passed_tests = len([r for r in valid_results if r.result == TestResult.PASS])
        failed_tests = len([r for r in valid_results if r.result == TestResult.FAIL])
        skipped_tests = len([r for r in valid_results if r.result == TestResult.SKIP])
        error_tests = len([r for r in valid_results if r.result == TestResult.ERROR])
        
        execution_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate coverage and security metrics
        coverage_metrics = await self._calculate_coverage_metrics(valid_results)
        security_score = self._calculate_security_score(valid_results)
        
        suite_result = TestSuiteResult(
            suite_name=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            execution_time_ms=int(execution_time),
            test_results=valid_results,
            coverage_metrics=coverage_metrics,
            security_score=security_score,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # Store results
        await self._store_test_results(suite_result)
        
        return suite_result
    
    async def test_code_injection_detection(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Test code injection detection capabilities"""
        try:
            detector = ThreatDetectionEngine()
            test_data = test_case.test_data
            
            results = {
                'malicious_detections': 0,
                'safe_non_detections': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
            
            # Test malicious code detection
            for malicious_code in test_data['malicious_code']:
                # Create temporary file with malicious content
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(f"# Test file\n{malicious_code}\n")
                    temp_file = f.name
                
                try:
                    detections = detector.scan_file(temp_file)
                    if any(d.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH] for d in detections):
                        results['malicious_detections'] += 1
                    else:
                        results['false_negatives'] += 1
                finally:
                    os.unlink(temp_file)
            
            # Test safe code (should not trigger false positives)
            for safe_code in test_data['safe_code']:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(f"# Safe test file\n{safe_code}\n")
                    temp_file = f.name
                
                try:
                    detections = detector.scan_file(temp_file)
                    if any(d.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH] for d in detections):
                        results['false_positives'] += 1
                    else:
                        results['safe_non_detections'] += 1
                finally:
                    os.unlink(temp_file)
            
            # Calculate accuracy
            total_malicious = len(test_data['malicious_code'])
            total_safe = len(test_data['safe_code'])
            
            detection_rate = results['malicious_detections'] / total_malicious if total_malicious > 0 else 0
            false_positive_rate = results['false_positives'] / total_safe if total_safe > 0 else 0
            
            # Determine test result
            if detection_rate >= 0.9 and false_positive_rate <= 0.1:
                test_result = TestResult.PASS
            else:
                test_result = TestResult.FAIL
            
            return {
                'result': test_result,
                'details': results,
                'metrics': {
                    'detection_rate': detection_rate,
                    'false_positive_rate': false_positive_rate,
                    'accuracy': (results['malicious_detections'] + results['safe_non_detections']) / (total_malicious + total_safe)
                }
            }
            
        except Exception as e:
            return {
                'result': TestResult.ERROR,
                'error_message': f"Code injection test error: {str(e)}",
                'details': {}
            }
    
    async def test_sql_injection_detection(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Test SQL injection detection capabilities"""
        try:
            detector = ThreatDetectionEngine()
            test_data = test_case.test_data
            
            results = {'detected': 0, 'missed': 0, 'false_positives': 0}
            
            # Test injection patterns
            for pattern in test_data['injection_patterns']:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(f'query = "{pattern}"\n')
                    temp_file = f.name
                
                try:
                    detections = detector.scan_file(temp_file)
                    if any('sql' in d.pattern_name.lower() or 'injection' in d.pattern_name.lower() 
                          for d in detections):
                        results['detected'] += 1
                    else:
                        results['missed'] += 1
                finally:
                    os.unlink(temp_file)
            
            # Test safe patterns
            for pattern in test_data['safe_patterns']:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(f'{pattern}\n')
                    temp_file = f.name
                
                try:
                    detections = detector.scan_file(temp_file)
                    if any('sql' in d.pattern_name.lower() or 'injection' in d.pattern_name.lower() 
                          for d in detections):
                        results['false_positives'] += 1
                finally:
                    os.unlink(temp_file)
            
            detection_rate = results['detected'] / len(test_data['injection_patterns'])
            
            return {
                'result': TestResult.PASS if detection_rate >= 0.8 else TestResult.FAIL,
                'details': results,
                'metrics': {'detection_rate': detection_rate}
            }
            
        except Exception as e:
            return {
                'result': TestResult.ERROR,
                'error_message': str(e),
                'details': {}
            }
    
    async def test_hardcoded_secrets_detection(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Test hardcoded secrets detection"""
        try:
            detector = ThreatDetectionEngine()
            test_data = test_case.test_data
            
            secret_detections = 0
            total_secrets = len(test_data['secret_patterns'])
            
            for secret in test_data['secret_patterns']:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(f'{secret}\n')
                    temp_file = f.name
                
                try:
                    detections = detector.scan_file(temp_file)
                    if any('secret' in d.pattern_name.lower() or 'credential' in d.pattern_name.lower()
                          for d in detections):
                        secret_detections += 1
                finally:
                    os.unlink(temp_file)
            
            detection_rate = secret_detections / total_secrets
            
            return {
                'result': TestResult.PASS if detection_rate >= 0.7 else TestResult.FAIL,
                'details': {'secrets_detected': secret_detections, 'total_secrets': total_secrets},
                'metrics': {'detection_rate': detection_rate}
            }
            
        except Exception as e:
            return {
                'result': TestResult.ERROR,
                'error_message': str(e),
                'details': {}
            }
    
    async def _setup_test_isolation(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Setup isolated test environment"""
        isolation_id = f"test_{test_case.test_id}_{datetime.datetime.now().strftime('%H%M%S')}"
        temp_dir = tempfile.mkdtemp(prefix=f"security_test_{isolation_id}_")
        
        return {
            'isolation_id': isolation_id,
            'temp_directory': temp_dir,
            'created_files': [],
            'modified_files': []
        }
    
    async def _cleanup_test_isolation(self, isolation_context: Dict[str, Any]):
        """Clean up isolated test environment"""
        import shutil
        temp_dir = isolation_context.get('temp_directory')
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    async def _calculate_coverage_metrics(self, test_results: List[SecurityTestResult]) -> Dict[str, float]:
        """Calculate test coverage metrics"""
        # Simple coverage calculation based on categories and components tested
        categories_tested = set(result.category for result in test_results)
        total_categories = len(TestCategory)
        
        category_coverage = len(categories_tested) / total_categories
        
        # Component coverage (based on test functions executed)
        components_tested = set()
        for result in test_results:
            if result.result in [TestResult.PASS, TestResult.FAIL]:
                components_tested.add(result.test_name.split()[0].lower())
        
        component_coverage = min(len(components_tested) / 10.0, 1.0)  # Assume 10 key components
        
        return {
            'category_coverage': category_coverage,
            'component_coverage': component_coverage,
            'overall_coverage': (category_coverage + component_coverage) / 2
        }
    
    def _calculate_security_score(self, test_results: List[SecurityTestResult]) -> float:
        """Calculate overall security score based on test results"""
        if not test_results:
            return 0.0
        
        # Weight by test category importance
        category_weights = {
            TestCategory.VULNERABILITY_DETECTION: 0.25,
            TestCategory.RESPONSE_VALIDATION: 0.20,
            TestCategory.CORRELATION_ACCURACY: 0.15,
            TestCategory.PREDICTIVE_ACCURACY: 0.15,
            TestCategory.INTEGRATION_TESTING: 0.10,
            TestCategory.PERFORMANCE_IMPACT: 0.10,
            TestCategory.REGRESSION_TESTING: 0.03,
            TestCategory.STRESS_TESTING: 0.02
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in test_results:
            weight = category_weights.get(result.category, 0.1)
            if result.result == TestResult.PASS:
                score = 1.0
            elif result.result == TestResult.FAIL:
                score = 0.0
            elif result.result == TestResult.SKIP:
                score = 0.5  # Partial credit
            else:  # ERROR
                score = 0.0
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _store_test_results(self, suite_result: TestSuiteResult):
        """Store test results to file"""
        try:
            results_file = self.test_results_dir / f"test_results_{suite_result.suite_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to serializable format
            results_dict = asdict(suite_result)
            
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
                
            self.logger.info(f"Test results stored to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error storing test results: {e}")
    
    def get_test_cases(self, category: TestCategory = None) -> List[SecurityTestCase]:
        """Get test cases, optionally filtered by category"""
        test_cases = list(self.test_cases.values())
        if category:
            test_cases = [tc for tc in test_cases if tc.category == category]
        return test_cases
    
    def get_test_suites(self) -> List[str]:
        """Get list of available test suites"""
        return list(self.test_suites.keys())
    
    async def validate_security_system(self) -> Dict[str, Any]:
        """Run comprehensive security system validation"""
        # Create comprehensive test suite
        all_test_ids = list(self.test_cases.keys())
        self.register_test_suite('comprehensive_validation', all_test_ids)
        
        # Run comprehensive suite
        results = await self.run_test_suite('comprehensive_validation')
        
        # Generate validation report
        validation_report = {
            'validation_timestamp': datetime.datetime.now().isoformat(),
            'overall_security_score': results.security_score,
            'test_summary': {
                'total_tests': results.total_tests,
                'passed_tests': results.passed_tests,
                'failed_tests': results.failed_tests,
                'success_rate': results.passed_tests / results.total_tests if results.total_tests > 0 else 0
            },
            'coverage_metrics': results.coverage_metrics,
            'recommendations': self._generate_recommendations(results),
            'detailed_results': [asdict(tr) for tr in results.test_results]
        }
        
        return validation_report
    
    def _generate_recommendations(self, suite_result: TestSuiteResult) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if suite_result.security_score < 0.8:
            recommendations.append("Security score below threshold - immediate attention required")
        
        if suite_result.failed_tests > 0:
            recommendations.append(f"{suite_result.failed_tests} critical tests failed - review and fix")
        
        if suite_result.coverage_metrics.get('overall_coverage', 0) < 0.8:
            recommendations.append("Test coverage below 80% - add more comprehensive tests")
        
        # Category-specific recommendations
        category_failures = {}
        for result in suite_result.test_results:
            if result.result == TestResult.FAIL:
                category = result.category
                if category not in category_failures:
                    category_failures[category] = 0
                category_failures[category] += 1
        
        for category, failures in category_failures.items():
            recommendations.append(f"Address {failures} failures in {category.value} testing")
        
        if not recommendations:
            recommendations.append("All security tests passed - system validation successful")
        
        return recommendations

# Factory function
def create_security_testing_framework(config: Dict[str, Any] = None) -> SecurityTestFramework:
    """Factory function to create security testing framework"""
    return SecurityTestFramework(config)