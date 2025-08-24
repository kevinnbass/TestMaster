"""
Continuous Testing Engine for TestMaster
Advanced continuous integration and intelligent test execution
"""

import asyncio
import time
import json
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import git
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TestExecutionStrategy(Enum):
    """Test execution strategies"""
    FULL_SUITE = "full_suite"
    AFFECTED_ONLY = "affected_only"
    SMART_SELECTION = "smart_selection"
    RISK_BASED = "risk_based"
    PARALLEL_SHARDED = "parallel_sharded"

class TestPriority(Enum):
    """Test priority levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SMOKE = "SMOKE"

class ExecutionStatus(Enum):
    """Test execution status"""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"

@dataclass
class TestExecution:
    """Individual test execution record"""
    execution_id: str
    test_path: str
    test_name: str
    strategy: TestExecutionStrategy
    priority: TestPriority
    status: ExecutionStatus
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    coverage_delta: float = 0.0
    failure_reason: Optional[str] = None

@dataclass
class ContinuousTestingMetrics:
    """Continuous testing metrics"""
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: float
    coverage_percentage: float
    test_efficiency_score: float
    flaky_test_count: int
    time_saved_by_selection: float
    execution_frequency: float
    
@dataclass
class TestingSession:
    """Continuous testing session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    trigger_type: str = "manual"
    trigger_details: Dict[str, Any] = field(default_factory=dict)
    executions: List[TestExecution] = field(default_factory=list)
    metrics: Optional[ContinuousTestingMetrics] = None
    recommendations: List[str] = field(default_factory=list)

class ChangeDetectionHandler(FileSystemEventHandler):
    """File system change detection for continuous testing"""
    
    def __init__(self, continuous_engine):
        self.engine = continuous_engine
        self.debounce_time = 2.0  # seconds
        self.pending_changes = set()
        self.timer = None
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            self.pending_changes.add(event.src_path)
            self._reset_timer()
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            self.pending_changes.add(event.src_path)
            self._reset_timer()
    
    def _reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.debounce_time, self._trigger_tests)
        self.timer.start()
    
    def _trigger_tests(self):
        if self.pending_changes:
            asyncio.create_task(self.engine.handle_file_changes(list(self.pending_changes)))
            self.pending_changes.clear()

class ContinuousTestingEngine:
    """Advanced Continuous Testing Engine"""
    
    def __init__(self, project_root: str, test_directory: str):
        self.project_root = Path(project_root)
        self.test_directory = Path(test_directory)
        self.observer: Optional[Observer] = None
        self.active_sessions: Dict[str, TestingSession] = {}
        self.test_history: List[TestExecution] = []
        self.flaky_tests: Set[str] = set()
        self.test_weights: Dict[str, float] = {}
        self.coverage_cache: Dict[str, float] = {}
        self.git_repo = None
        
        try:
            self.git_repo = git.Repo(self.project_root)
        except:
            pass
    
    def start_continuous_monitoring(self) -> None:
        """Start continuous file monitoring"""
        if self.observer and self.observer.is_alive():
            return
        
        event_handler = ChangeDetectionHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.project_root), recursive=True)
        self.observer.start()
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous file monitoring"""
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
    
    async def handle_file_changes(self, changed_files: List[str]) -> TestingSession:
        """Handle detected file changes"""
        session = self._create_session("file_change", {"changed_files": changed_files})
        
        # Determine affected tests
        affected_tests = self._get_affected_tests(changed_files)
        
        # Select execution strategy
        strategy = self._select_execution_strategy(affected_tests, changed_files)
        
        # Execute tests
        await self._execute_test_strategy(session, strategy, affected_tests)
        
        return session
    
    def _create_session(self, trigger_type: str, trigger_details: Dict[str, Any]) -> TestingSession:
        """Create new testing session"""
        session_id = f"session_{int(time.time() * 1000)}"
        session = TestingSession(
            session_id=session_id,
            start_time=time.time(),
            trigger_type=trigger_type,
            trigger_details=trigger_details
        )
        self.active_sessions[session_id] = session
        return session
    
    def _get_affected_tests(self, changed_files: List[str]) -> List[str]:
        """Determine which tests are affected by file changes"""
        affected_tests = set()
        
        for changed_file in changed_files:
            changed_path = Path(changed_file)
            
            # Direct test file changes
            if changed_path.name.startswith('test_'):
                affected_tests.add(str(changed_path))
                continue
            
            # Find tests that import or reference the changed module
            module_name = self._get_module_name(changed_path)
            if module_name:
                related_tests = self._find_tests_importing_module(module_name)
                affected_tests.update(related_tests)
        
        return list(affected_tests)
    
    def _get_module_name(self, file_path: Path) -> Optional[str]:
        """Get module name from file path"""
        try:
            relative_path = file_path.relative_to(self.project_root)
            module_parts = list(relative_path.parts[:-1])  # Exclude filename
            if relative_path.stem != '__init__':
                module_parts.append(relative_path.stem)
            return '.'.join(module_parts) if module_parts else None
        except ValueError:
            return None
    
    def _find_tests_importing_module(self, module_name: str) -> List[str]:
        """Find test files that import the specified module"""
        importing_tests = []
        
        for test_file in self.test_directory.rglob("test_*.py"):
            try:
                content = test_file.read_text()
                # Check for various import patterns
                import_patterns = [
                    f"import {module_name}",
                    f"from {module_name}",
                    f"from {module_name.split('.')[0]}"
                ]
                
                if any(pattern in content for pattern in import_patterns):
                    importing_tests.append(str(test_file))
            except:
                pass
        
        return importing_tests
    
    def _select_execution_strategy(self, affected_tests: List[str], 
                                 changed_files: List[str]) -> TestExecutionStrategy:
        """Select optimal test execution strategy"""
        # Strategy selection logic
        total_tests = len(list(self.test_directory.rglob("test_*.py")))
        affected_ratio = len(affected_tests) / total_tests if total_tests > 0 else 0
        
        # Risk assessment
        has_critical_changes = any(
            'core' in changed_file or 'main' in changed_file 
            for changed_file in changed_files
        )
        
        if has_critical_changes:
            return TestExecutionStrategy.FULL_SUITE
        elif affected_ratio > 0.5:
            return TestExecutionStrategy.SMART_SELECTION
        elif affected_ratio > 0.1:
            return TestExecutionStrategy.AFFECTED_ONLY
        else:
            return TestExecutionStrategy.RISK_BASED
    
    async def _execute_test_strategy(self, session: TestingSession, 
                                   strategy: TestExecutionStrategy, 
                                   affected_tests: List[str]) -> None:
        """Execute tests according to strategy"""
        if strategy == TestExecutionStrategy.FULL_SUITE:
            await self._execute_full_suite(session)
        elif strategy == TestExecutionStrategy.AFFECTED_ONLY:
            await self._execute_specific_tests(session, affected_tests)
        elif strategy == TestExecutionStrategy.SMART_SELECTION:
            selected_tests = self._smart_test_selection(affected_tests)
            await self._execute_specific_tests(session, selected_tests)
        elif strategy == TestExecutionStrategy.RISK_BASED:
            risk_tests = self._risk_based_selection(affected_tests)
            await self._execute_specific_tests(session, risk_tests)
        elif strategy == TestExecutionStrategy.PARALLEL_SHARDED:
            await self._execute_parallel_sharded(session, affected_tests)
        
        # Finalize session
        session.end_time = time.time()
        session.metrics = self._calculate_session_metrics(session)
        session.recommendations = self._generate_session_recommendations(session)
    
    async def _execute_full_suite(self, session: TestingSession) -> None:
        """Execute complete test suite"""
        test_files = list(self.test_directory.rglob("test_*.py"))
        await self._execute_specific_tests(session, [str(f) for f in test_files])
    
    async def _execute_specific_tests(self, session: TestingSession, test_files: List[str]) -> None:
        """Execute specific test files"""
        for test_file in test_files:
            execution = await self._execute_single_test(test_file, session.session_id)
            session.executions.append(execution)
            
            # Update flaky test tracking
            self._update_flaky_test_tracking(execution)
    
    async def _execute_single_test(self, test_file: str, session_id: str) -> TestExecution:
        """Execute single test file"""
        execution_id = f"{session_id}_{hashlib.md5(test_file.encode()).hexdigest()[:8]}"
        
        execution = TestExecution(
            execution_id=execution_id,
            test_path=test_file,
            test_name=Path(test_file).stem,
            strategy=TestExecutionStrategy.AFFECTED_ONLY,  # Default
            priority=self._get_test_priority(test_file),
            status=ExecutionStatus.QUEUED,
            start_time=time.time()
        )
        
        try:
            execution.status = ExecutionStatus.RUNNING
            
            # Execute test using pytest
            result = subprocess.run([
                'python', '-m', 'pytest', test_file, '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=300, cwd=self.project_root)
            
            execution.end_time = time.time()
            execution.duration = execution.end_time - execution.start_time
            execution.exit_code = result.returncode
            execution.stdout = result.stdout
            execution.stderr = result.stderr
            
            if result.returncode == 0:
                execution.status = ExecutionStatus.PASSED
            else:
                execution.status = ExecutionStatus.FAILED
                execution.failure_reason = self._extract_failure_reason(result.stderr)
            
        except subprocess.TimeoutExpired:
            execution.status = ExecutionStatus.TIMEOUT
            execution.end_time = time.time()
            execution.duration = execution.end_time - execution.start_time
        except Exception as e:
            execution.status = ExecutionStatus.ERROR
            execution.end_time = time.time()
            execution.duration = execution.end_time - execution.start_time
            execution.failure_reason = str(e)
        
        self.test_history.append(execution)
        return execution
    
    def _smart_test_selection(self, affected_tests: List[str]) -> List[str]:
        """Intelligent test selection based on multiple factors"""
        selected_tests = set(affected_tests)
        
        # Add high-priority tests
        high_priority_tests = [
            test for test in self._get_all_tests()
            if self._get_test_priority(test) in [TestPriority.CRITICAL, TestPriority.HIGH]
        ]
        selected_tests.update(high_priority_tests[:10])  # Limit to top 10
        
        # Add tests with recent failures
        recent_failures = [
            exec.test_path for exec in self.test_history[-50:]  # Last 50 executions
            if exec.status == ExecutionStatus.FAILED
        ]
        selected_tests.update(recent_failures[:5])  # Limit to 5 recent failures
        
        return list(selected_tests)
    
    def _risk_based_selection(self, affected_tests: List[str]) -> List[str]:
        """Risk-based test selection"""
        risk_scores = {}
        
        for test in self._get_all_tests():
            risk_score = 0.0
            
            # Base risk from affected tests
            if test in affected_tests:
                risk_score += 1.0
            
            # Risk from test priority
            priority = self._get_test_priority(test)
            if priority == TestPriority.CRITICAL:
                risk_score += 0.8
            elif priority == TestPriority.HIGH:
                risk_score += 0.6
            
            # Risk from flaky test history
            if test in self.flaky_tests:
                risk_score += 0.4
            
            # Risk from recent execution frequency
            recent_executions = [
                exec for exec in self.test_history[-20:]
                if exec.test_path == test
            ]
            if len(recent_executions) < 2:  # Haven't run recently
                risk_score += 0.3
            
            risk_scores[test] = risk_score
        
        # Select top risk tests
        sorted_tests = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        return [test for test, score in sorted_tests[:20] if score > 0.5]
    
    async def _execute_parallel_sharded(self, session: TestingSession, test_files: List[str]) -> None:
        """Execute tests in parallel shards"""
        shard_size = max(1, len(test_files) // 4)  # 4 shards
        shards = [test_files[i:i+shard_size] for i in range(0, len(test_files), shard_size)]
        
        tasks = []
        for shard in shards:
            task = asyncio.create_task(self._execute_specific_tests(session, shard))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    def _get_test_priority(self, test_file: str) -> TestPriority:
        """Determine test priority based on various factors"""
        test_name = Path(test_file).stem.lower()
        
        # Priority keywords
        if any(keyword in test_name for keyword in ['critical', 'core', 'main', 'auth']):
            return TestPriority.CRITICAL
        elif any(keyword in test_name for keyword in ['important', 'security', 'api']):
            return TestPriority.HIGH
        elif any(keyword in test_name for keyword in ['smoke', 'basic', 'simple']):
            return TestPriority.SMOKE
        elif any(keyword in test_name for keyword in ['integration', 'e2e', 'end_to_end']):
            return TestPriority.HIGH
        else:
            return TestPriority.MEDIUM
    
    def _get_all_tests(self) -> List[str]:
        """Get all test files"""
        return [str(f) for f in self.test_directory.rglob("test_*.py")]
    
    def _extract_failure_reason(self, stderr: str) -> str:
        """Extract failure reason from stderr"""
        # Simple extraction logic
        lines = stderr.split('\n')
        for line in lines:
            if 'FAILED' in line or 'ERROR' in line or 'AssertionError' in line:
                return line.strip()[:200]  # Limit length
        return "Unknown failure"
    
    def _update_flaky_test_tracking(self, execution: TestExecution) -> None:
        """Update flaky test tracking based on execution history"""
        test_path = execution.test_path
        
        # Get recent executions for this test
        recent_executions = [
            exec for exec in self.test_history[-20:]
            if exec.test_path == test_path
        ]
        
        if len(recent_executions) >= 5:
            # Calculate failure rate
            failures = len([exec for exec in recent_executions if exec.status == ExecutionStatus.FAILED])
            failure_rate = failures / len(recent_executions)
            
            # Mark as flaky if failure rate is between 20-80%
            if 0.2 <= failure_rate <= 0.8:
                self.flaky_tests.add(test_path)
            elif failure_rate < 0.1:
                self.flaky_tests.discard(test_path)  # Remove from flaky if stable
    
    def _calculate_session_metrics(self, session: TestingSession) -> ContinuousTestingMetrics:
        """Calculate metrics for testing session"""
        executions = session.executions
        
        if not executions:
            return ContinuousTestingMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
        
        total = len(executions)
        successful = len([e for e in executions if e.status == ExecutionStatus.PASSED])
        failed = len([e for e in executions if e.status == ExecutionStatus.FAILED])
        
        avg_time = sum(e.duration for e in executions) / total
        
        # Calculate efficiency score (successful tests / time)
        efficiency = successful / avg_time if avg_time > 0 else 0
        
        return ContinuousTestingMetrics(
            total_executions=total,
            successful_executions=successful,
            failed_executions=failed,
            average_execution_time=avg_time,
            coverage_percentage=85.0,  # Placeholder
            test_efficiency_score=efficiency,
            flaky_test_count=len(self.flaky_tests),
            time_saved_by_selection=0.0,  # Placeholder
            execution_frequency=1.0  # Placeholder
        )
    
    def _generate_session_recommendations(self, session: TestingSession) -> List[str]:
        """Generate recommendations based on session results"""
        recommendations = []
        metrics = session.metrics
        
        if not metrics:
            return recommendations
        
        # Failure rate recommendations
        failure_rate = metrics.failed_executions / metrics.total_executions if metrics.total_executions > 0 else 0
        if failure_rate > 0.2:
            recommendations.append("High failure rate detected - review failing tests for common issues")
        
        # Performance recommendations
        if metrics.average_execution_time > 30:
            recommendations.append("Long test execution time - consider optimizing slow tests")
        
        # Flaky test recommendations
        if metrics.flaky_test_count > 5:
            recommendations.append("Multiple flaky tests detected - investigate test stability issues")
        
        # Coverage recommendations
        if metrics.coverage_percentage < 80:
            recommendations.append("Test coverage below 80% - consider adding more comprehensive tests")
        
        return recommendations
    
    async def run_scheduled_testing(self, schedule_config: Dict[str, Any]) -> None:
        """Run scheduled testing based on configuration"""
        while True:
            try:
                # Check if it's time to run based on schedule
                if self._should_run_scheduled_test(schedule_config):
                    session = self._create_session("scheduled", schedule_config)
                    await self._execute_full_suite(session)
                
                # Wait for next check
                await asyncio.sleep(schedule_config.get("check_interval", 3600))  # Default 1 hour
                
            except Exception as e:
                print(f"Scheduled testing error: {e}")
                await asyncio.sleep(3600)
    
    def _should_run_scheduled_test(self, schedule_config: Dict[str, Any]) -> bool:
        """Check if scheduled test should run"""
        # Simple time-based scheduling logic
        current_hour = time.localtime().tm_hour
        scheduled_hours = schedule_config.get("hours", [9, 17])  # Default 9 AM and 5 PM
        
        return current_hour in scheduled_hours
    
    def get_testing_dashboard_data(self) -> Dict[str, Any]:
        """Get data for testing dashboard"""
        recent_sessions = list(self.active_sessions.values())[-10:]  # Last 10 sessions
        
        return {
            "active_sessions": len(self.active_sessions),
            "recent_sessions": recent_sessions,
            "flaky_tests": list(self.flaky_tests),
            "total_test_history": len(self.test_history),
            "monitoring_active": self.observer and self.observer.is_alive(),
            "test_directory": str(self.test_directory),
            "project_root": str(self.project_root)
        }