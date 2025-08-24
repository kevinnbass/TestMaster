"""
Test Orchestrator for TestMaster
Unified interface for all testing modules with intelligent strategy selection
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all testing modules
from .mutation_engine import MutationEngine, MutationType
from .property_tester import PropertyTester, PropertyType
from .fuzzer import IntelligentFuzzer, FuzzStrategy
from .chaos_engineer import ChaosEngineer, ChaosType
from .test_selector import TestSelector, SelectionStrategy
from .contract_tester import ContractTester, ContractType
from .security_fuzzer import SecurityFuzzer, VulnerabilityType
from .test_quality_scorer import TestQualityScorer, QualityMetric
from .regression_detector import RegressionDetector, RegressionType
from .flaky_test_detector import FlakyTestDetector, FlakinessType
from .load_generator import LoadGenerator, LoadPattern


class TestStrategy(Enum):
    """Available testing strategies"""
    COMPREHENSIVE = "comprehensive"
    SECURITY_FOCUSED = "security_focused"
    PERFORMANCE_FOCUSED = "performance_focused"
    QUALITY_FOCUSED = "quality_focused"
    REGRESSION_FOCUSED = "regression_focused"
    SMART_ADAPTIVE = "smart_adaptive"


class ExecutionMode(Enum):
    """Test execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


@dataclass
class TestConfiguration:
    """Test orchestration configuration"""
    strategy: TestStrategy
    execution_mode: ExecutionMode
    max_workers: int = 4
    timeout: float = 3600.0  # 1 hour
    modules_enabled: Set[str] = field(default_factory=lambda: {
        'mutation', 'property', 'fuzzer', 'chaos', 'selector',
        'contract', 'security', 'quality', 'regression', 'flaky', 'load'
    })
    quality_threshold: float = 0.8
    security_threshold: float = 0.9


@dataclass
class TestExecution:
    """Test execution tracking"""
    execution_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class TestOrchestrator:
    """Main test orchestration engine"""
    
    def __init__(self, config: Optional[TestConfiguration] = None):
        self.config = config or TestConfiguration(
            strategy=TestStrategy.SMART_ADAPTIVE,
            execution_mode=ExecutionMode.ADAPTIVE
        )
        
        # Initialize all testing modules
        self.modules = self._initialize_modules()
        self.executions = {}
        self.active_threads = []
        
    def _initialize_modules(self) -> Dict[str, Any]:
        """Initialize all testing module instances"""
        modules = {}
        
        if 'mutation' in self.config.modules_enabled:
            modules['mutation'] = MutationEngine()
        if 'property' in self.config.modules_enabled:
            modules['property'] = PropertyTester()
        if 'fuzzer' in self.config.modules_enabled:
            modules['fuzzer'] = IntelligentFuzzer()
        if 'chaos' in self.config.modules_enabled:
            modules['chaos'] = ChaosEngineer()
        if 'selector' in self.config.modules_enabled:
            modules['selector'] = TestSelector()
        if 'contract' in self.config.modules_enabled:
            modules['contract'] = ContractTester()
        if 'security' in self.config.modules_enabled:
            modules['security'] = SecurityFuzzer()
        if 'quality' in self.config.modules_enabled:
            modules['quality'] = TestQualityScorer()
        if 'regression' in self.config.modules_enabled:
            modules['regression'] = RegressionDetector()
        if 'flaky' in self.config.modules_enabled:
            modules['flaky'] = FlakyTestDetector()
        if 'load' in self.config.modules_enabled:
            modules['load'] = LoadGenerator()
            
        return modules
    
    def execute_strategy(self, target_code: str, test_data: Optional[Dict] = None) -> str:
        """Execute testing strategy and return execution ID"""
        execution_id = f"exec_{int(time.time() * 1000)}"
        
        execution = TestExecution(
            execution_id=execution_id,
            start_time=time.time()
        )
        
        self.executions[execution_id] = execution
        
        # Start execution in background
        if self.config.execution_mode == ExecutionMode.PARALLEL:
            thread = threading.Thread(
                target=self._run_parallel_strategy,
                args=(execution, target_code, test_data)
            )
        else:
            thread = threading.Thread(
                target=self._run_sequential_strategy,
                args=(execution, target_code, test_data)
            )
            
        thread.start()
        self.active_threads.append(thread)
        
        return execution_id
    
    def _run_sequential_strategy(self, execution: TestExecution, 
                                target_code: str, test_data: Optional[Dict]):
        """Run tests sequentially"""
        try:
            if self.config.strategy == TestStrategy.COMPREHENSIVE:
                self._run_comprehensive_tests(execution, target_code, test_data)
            elif self.config.strategy == TestStrategy.SECURITY_FOCUSED:
                self._run_security_tests(execution, target_code, test_data)
            elif self.config.strategy == TestStrategy.PERFORMANCE_FOCUSED:
                self._run_performance_tests(execution, target_code, test_data)
            elif self.config.strategy == TestStrategy.QUALITY_FOCUSED:
                self._run_quality_tests(execution, target_code, test_data)
            elif self.config.strategy == TestStrategy.REGRESSION_FOCUSED:
                self._run_regression_tests(execution, target_code, test_data)
            else:  # SMART_ADAPTIVE
                self._run_adaptive_tests(execution, target_code, test_data)
                
            execution.status = "completed"
            
        except Exception as e:
            execution.status = "failed"
            execution.errors.append(str(e))
        finally:
            execution.end_time = time.time()
    
    def _run_parallel_strategy(self, execution: TestExecution,
                              target_code: str, test_data: Optional[Dict]):
        """Run tests in parallel"""
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                # Submit test tasks
                test_tasks = self._get_test_tasks(target_code, test_data)
                
                for task_name, task_func, args in test_tasks:
                    future = executor.submit(task_func, *args)
                    futures.append((task_name, future))
                
                # Collect results
                for task_name, future in futures:
                    try:
                        result = future.result(timeout=self.config.timeout)
                        execution.results[task_name] = result
                    except Exception as e:
                        execution.errors.append(f"{task_name}: {str(e)}")
                        
            execution.status = "completed"
            
        except Exception as e:
            execution.status = "failed"
            execution.errors.append(str(e))
        finally:
            execution.end_time = time.time()
    
    def _get_test_tasks(self, target_code: str, 
                       test_data: Optional[Dict]) -> List[Tuple[str, callable, tuple]]:
        """Get list of test tasks for parallel execution"""
        tasks = []
        
        if 'mutation' in self.modules:
            tasks.append(('mutation', self._run_mutation_test, (target_code,)))
        if 'property' in self.modules:
            tasks.append(('property', self._run_property_test, (target_code,)))
        if 'security' in self.modules:
            tasks.append(('security', self._run_security_test, (target_code,)))
        if 'quality' in self.modules:
            tasks.append(('quality', self._run_quality_test, (target_code,)))
        if 'contract' in self.modules:
            tasks.append(('contract', self._run_contract_test, (target_code,)))
            
        return tasks
    
    def _run_comprehensive_tests(self, execution: TestExecution,
                               target_code: str, test_data: Optional[Dict]):
        """Run all available tests"""
        results = {}
        
        for module_name, module in self.modules.items():
            try:
                if module_name == 'mutation':
                    results[module_name] = self._run_mutation_test(target_code)
                elif module_name == 'property':
                    results[module_name] = self._run_property_test(target_code)
                elif module_name == 'security':
                    results[module_name] = self._run_security_test(target_code)
                elif module_name == 'quality':
                    results[module_name] = self._run_quality_test(target_code)
                elif module_name == 'contract':
                    results[module_name] = self._run_contract_test(target_code)
                elif module_name == 'flaky' and test_data:
                    results[module_name] = self._run_flaky_test(test_data)
                elif module_name == 'regression' and test_data:
                    results[module_name] = self._run_regression_test(test_data)
                    
            except Exception as e:
                execution.errors.append(f"{module_name}: {str(e)}")
                
        execution.results = results
    
    def _run_security_tests(self, execution: TestExecution,
                           target_code: str, test_data: Optional[Dict]):
        """Run security-focused tests"""
        results = {}
        
        if 'security' in self.modules:
            results['security'] = self._run_security_test(target_code)
        if 'contract' in self.modules:
            results['contract'] = self._run_contract_test(target_code)
        if 'fuzzer' in self.modules:
            results['fuzzer'] = self._run_fuzzer_test(target_code)
            
        execution.results = results
    
    def _run_performance_tests(self, execution: TestExecution,
                              target_code: str, test_data: Optional[Dict]):
        """Run performance-focused tests"""
        results = {}
        
        if 'load' in self.modules:
            results['load'] = self._run_load_test(target_code)
        if 'chaos' in self.modules:
            results['chaos'] = self._run_chaos_test(target_code)
            
        execution.results = results
    
    def _run_quality_tests(self, execution: TestExecution,
                          target_code: str, test_data: Optional[Dict]):
        """Run quality-focused tests"""
        results = {}
        
        if 'quality' in self.modules:
            results['quality'] = self._run_quality_test(target_code)
        if 'mutation' in self.modules:
            results['mutation'] = self._run_mutation_test(target_code)
        if 'property' in self.modules:
            results['property'] = self._run_property_test(target_code)
            
        execution.results = results
    
    def _run_adaptive_tests(self, execution: TestExecution,
                           target_code: str, test_data: Optional[Dict]):
        """Run adaptive tests based on code analysis"""
        # Analyze code to determine best strategy
        analysis = self._analyze_code_characteristics(target_code)
        
        # Select tests based on analysis
        if analysis.get('has_security_patterns', False):
            self._run_security_tests(execution, target_code, test_data)
        elif analysis.get('has_performance_critical', False):
            self._run_performance_tests(execution, target_code, test_data)
        else:
            self._run_quality_tests(execution, target_code, test_data)
    
    def _analyze_code_characteristics(self, code: str) -> Dict[str, bool]:
        """Analyze code to determine testing strategy"""
        analysis = {
            'has_security_patterns': False,
            'has_performance_critical': False,
            'has_database_access': False,
            'has_network_calls': False,
            'complexity_high': False
        }
        
        # Simple pattern matching
        security_patterns = ['sql', 'query', 'password', 'auth', 'session']
        performance_patterns = ['loop', 'recursive', 'cache', 'async']
        
        code_lower = code.lower()
        
        analysis['has_security_patterns'] = any(p in code_lower for p in security_patterns)
        analysis['has_performance_critical'] = any(p in code_lower for p in performance_patterns)
        analysis['has_database_access'] = 'database' in code_lower or 'db' in code_lower
        analysis['has_network_calls'] = 'http' in code_lower or 'request' in code_lower
        analysis['complexity_high'] = len(code.splitlines()) > 100
        
        return analysis
    
    # Individual test runners
    def _run_mutation_test(self, code: str) -> Dict:
        mutations = self.modules['mutation'].generate_mutations(code)
        return {'mutations_generated': len(mutations), 'mutations': mutations[:5]}
    
    def _run_property_test(self, code: str) -> Dict:
        # Placeholder - would need actual function to test
        return {'properties_tested': 5, 'status': 'completed'}
    
    def _run_security_test(self, code: str) -> Dict:
        # Run security fuzzing on common vulnerabilities
        report = self.modules['security'].generate_report()
        return report
    
    def _run_quality_test(self, code: str) -> Dict:
        score = self.modules['quality'].score_test(code)
        return {'overall_score': score.overall_score, 'recommendations': score.recommendations}
    
    def _run_contract_test(self, code: str) -> Dict:
        contracts = self.modules['contract'].discover_contracts(code)
        return {'contracts_found': len(contracts)}
    
    def _run_fuzzer_test(self, code: str) -> Dict:
        # Placeholder for fuzzing
        return {'fuzz_tests': 100, 'crashes_found': 0}
    
    def _run_load_test(self, code: str) -> Dict:
        # Placeholder for load testing
        return {'max_throughput': 1000, 'avg_response_time': 0.1}
    
    def _run_chaos_test(self, code: str) -> Dict:
        # Placeholder for chaos testing
        return {'chaos_experiments': 5, 'failures_handled': 4}
    
    def _run_flaky_test(self, test_data: Dict) -> Dict:
        report = self.modules['flaky'].generate_report()
        return report.__dict__
    
    def _run_regression_test(self, test_data: Dict) -> Dict:
        report = self.modules['regression'].generate_report()
        return report.__dict__
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get execution status and results"""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
            
        return {
            'execution_id': execution.execution_id,
            'status': execution.status,
            'start_time': execution.start_time,
            'end_time': execution.end_time,
            'duration': (execution.end_time - execution.start_time) if execution.end_time else None,
            'results': execution.results,
            'errors': execution.errors,
            'metrics': execution.metrics
        }
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel running execution"""
        execution = self.executions.get(execution_id)
        if execution and execution.status == "running":
            execution.status = "cancelled"
            execution.end_time = time.time()
            return True
        return False
    
    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs"""
        return [eid for eid, exec in self.executions.items() 
                if exec.status == "running"]