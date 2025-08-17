"""
Automated Investigation System

Inspired by LangGraph's supervisor delegation patterns for
automated investigation and analysis tasks.

Features:
- Investigate 2-hour idle modules
- Analyze test coverage gaps  
- Generate improvement recommendations
- Root cause analysis for failures
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import subprocess

from ..core.layer_manager import requires_layer


class InvestigationType(Enum):
    """Types of investigations."""
    IDLE_MODULE = "idle_module"
    COVERAGE_GAP = "coverage_gap"
    TEST_FAILURE = "test_failure"
    PERFORMANCE_ISSUE = "performance_issue"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    CODE_QUALITY = "code_quality"
    SECURITY_SCAN = "security_scan"


class InvestigationPriority(Enum):
    """Investigation priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceType(Enum):
    """Types of evidence collected."""
    FILE_ANALYSIS = "file_analysis"
    GIT_HISTORY = "git_history"
    DEPENDENCY_MAP = "dependency_map"
    TEST_RESULTS = "test_results"
    PERFORMANCE_DATA = "performance_data"
    CODE_METRICS = "code_metrics"
    EXTERNAL_TOOLS = "external_tools"


@dataclass
class Evidence:
    """A piece of evidence from investigation."""
    evidence_type: EvidenceType
    source: str
    data: Dict[str, Any]
    confidence: float  # 0-100
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    """A finding from investigation analysis."""
    finding_id: str
    title: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    category: str
    
    # Supporting evidence
    evidence: List[Evidence] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    impact_score: float = 0.0
    effort_estimate: int = 0  # minutes


@dataclass
class Investigation:
    """An investigation task."""
    investigation_id: str
    investigation_type: InvestigationType
    priority: InvestigationPriority
    target: str  # File, module, or system to investigate
    
    # Investigation context
    context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Status tracking
    status: str = "pending"  # pending, running, completed, failed
    progress_percent: float = 0.0
    
    # Results
    findings: List[Finding] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    
    # Summary
    summary: Optional[str] = None
    overall_assessment: Optional[str] = None


@dataclass
class InvestigationResult:
    """Result of an investigation."""
    investigation: Investigation
    success: bool
    execution_time_seconds: float
    
    # Analysis results
    total_findings: int
    critical_findings: int
    high_findings: int
    evidence_count: int
    
    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    long_term_recommendations: List[str] = field(default_factory=list)
    
    # Next steps
    follow_up_investigations: List[str] = field(default_factory=list)


class AutoInvestigator:
    """
    Automated investigation system using LangGraph delegation patterns.
    
    Performs systematic analysis of code issues, idle modules,
    and system health using supervisor-delegated tasks.
    """
    
    @requires_layer("layer3_orchestration", "automated_investigation")
    def __init__(self, watch_paths: Union[str, List[str]]):
        """
        Initialize auto investigator.
        
        Args:
            watch_paths: Directories to investigate
        """
        self.watch_paths = [Path(p) for p in (watch_paths if isinstance(watch_paths, list) else [watch_paths])]
        
        # Investigation storage
        self._investigations: Dict[str, Investigation] = {}
        self._investigation_results: Dict[str, InvestigationResult] = {}
        
        # Investigation strategies
        self._investigators: Dict[InvestigationType, Callable] = {}
        self._evidence_collectors: Dict[EvidenceType, Callable] = {}
        
        # Statistics
        self._stats = {
            'total_investigations': 0,
            'completed_investigations': 0,
            'findings_generated': 0,
            'evidence_collected': 0,
            'avg_investigation_time': 0.0
        }
        
        # Setup investigation strategies
        self._setup_investigators()
        self._setup_evidence_collectors()
        
        print("🔍 Auto investigator initialized")
        print(f"   📁 Investigating: {', '.join(str(p) for p in self.watch_paths)}")
    
    def start_investigation(self, investigation_type: InvestigationType,
                          target: str, priority: InvestigationPriority = InvestigationPriority.NORMAL,
                          context: Dict[str, Any] = None,
                          parameters: Dict[str, Any] = None) -> str:
        """
        Start a new investigation.
        
        Args:
            investigation_type: Type of investigation
            target: Target to investigate (file, module, etc.)
            priority: Investigation priority
            context: Investigation context
            parameters: Investigation parameters
            
        Returns:
            Investigation ID
        """
        investigation_id = self._generate_investigation_id()
        
        investigation = Investigation(
            investigation_id=investigation_id,
            investigation_type=investigation_type,
            priority=priority,
            target=target,
            context=context or {},
            parameters=parameters or {}
        )
        
        self._investigations[investigation_id] = investigation
        self._stats['total_investigations'] += 1
        
        print(f"🔍 Started investigation: {investigation_type.value} for {target}")
        
        # Execute investigation
        try:
            result = self._execute_investigation(investigation)
            self._investigation_results[investigation_id] = result
            
            if result.success:
                self._stats['completed_investigations'] += 1
                print(f"✅ Investigation completed: {investigation_id}")
            else:
                print(f"❌ Investigation failed: {investigation_id}")
                
        except Exception as e:
            print(f"⚠️ Error in investigation {investigation_id}: {e}")
            investigation.status = "failed"
        
        return investigation_id
    
    def investigate_idle_module(self, module_path: str, idle_hours: float) -> str:
        """
        Investigate an idle module.
        
        Args:
            module_path: Path to idle module
            idle_hours: Hours the module has been idle
            
        Returns:
            Investigation ID
        """
        return self.start_investigation(
            investigation_type=InvestigationType.IDLE_MODULE,
            target=module_path,
            priority=InvestigationPriority.HIGH if idle_hours > 168 else InvestigationPriority.NORMAL,  # 7 days
            context={"idle_hours": idle_hours}
        )
    
    def investigate_coverage_gap(self, source_file: str, coverage_percentage: float) -> str:
        """
        Investigate a coverage gap.
        
        Args:
            source_file: Source file with low coverage
            coverage_percentage: Current coverage percentage
            
        Returns:
            Investigation ID
        """
        priority = InvestigationPriority.HIGH if coverage_percentage < 50 else InvestigationPriority.NORMAL
        
        return self.start_investigation(
            investigation_type=InvestigationType.COVERAGE_GAP,
            target=source_file,
            priority=priority,
            context={"coverage_percentage": coverage_percentage}
        )
    
    def investigate_test_failure(self, test_file: str, error_message: str,
                               source_file: str = None) -> str:
        """
        Investigate a test failure.
        
        Args:
            test_file: Test file that failed
            error_message: Error message from failure
            source_file: Related source file
            
        Returns:
            Investigation ID
        """
        return self.start_investigation(
            investigation_type=InvestigationType.TEST_FAILURE,
            target=test_file,
            priority=InvestigationPriority.HIGH,
            context={
                "error_message": error_message,
                "source_file": source_file
            }
        )
    
    def _execute_investigation(self, investigation: Investigation) -> InvestigationResult:
        """Execute an investigation using delegation patterns."""
        start_time = datetime.now()
        investigation.started_at = start_time
        investigation.status = "running"
        
        try:
            # Get investigator for this type
            investigator = self._investigators.get(investigation.investigation_type)
            
            if not investigator:
                raise ValueError(f"No investigator for type: {investigation.investigation_type}")
            
            # Execute investigation
            investigator(investigation)
            
            # Complete investigation
            investigation.completed_at = datetime.now()
            investigation.status = "completed"
            investigation.progress_percent = 100.0
            
            # Generate summary
            self._generate_investigation_summary(investigation)
            
            # Calculate metrics
            execution_time = (investigation.completed_at - start_time).total_seconds()
            
            # Count findings by severity
            critical_findings = len([f for f in investigation.findings if f.severity == "critical"])
            high_findings = len([f for f in investigation.findings if f.severity == "high"])
            
            # Extract recommendations
            immediate_actions = []
            long_term_recommendations = []
            
            for finding in investigation.findings:
                if finding.severity in ["critical", "high"]:
                    immediate_actions.extend(finding.action_items)
                else:
                    long_term_recommendations.extend(finding.recommendations)
            
            result = InvestigationResult(
                investigation=investigation,
                success=True,
                execution_time_seconds=execution_time,
                total_findings=len(investigation.findings),
                critical_findings=critical_findings,
                high_findings=high_findings,
                evidence_count=len(investigation.evidence),
                immediate_actions=list(set(immediate_actions)),  # Remove duplicates
                long_term_recommendations=list(set(long_term_recommendations))
            )
            
            self._stats['findings_generated'] += result.total_findings
            self._stats['evidence_collected'] += result.evidence_count
            
            return result
            
        except Exception as e:
            investigation.status = "failed"
            investigation.completed_at = datetime.now()
            
            return InvestigationResult(
                investigation=investigation,
                success=False,
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                total_findings=0,
                critical_findings=0,
                high_findings=0,
                evidence_count=0
            )
    
    def _setup_investigators(self):
        """Setup investigation strategies."""
        
        def investigate_idle_module_impl(investigation: Investigation):
            """Investigate an idle module."""
            module_path = Path(investigation.target)
            idle_hours = investigation.context.get('idle_hours', 0)
            
            investigation.progress_percent = 10.0
            
            # Collect evidence
            file_analysis = self._collect_evidence(EvidenceType.FILE_ANALYSIS, str(module_path))
            if file_analysis:
                investigation.evidence.append(file_analysis)
            
            investigation.progress_percent = 30.0
            
            git_history = self._collect_evidence(EvidenceType.GIT_HISTORY, str(module_path))
            if git_history:
                investigation.evidence.append(git_history)
            
            investigation.progress_percent = 50.0
            
            dependency_map = self._collect_evidence(EvidenceType.DEPENDENCY_MAP, str(module_path))
            if dependency_map:
                investigation.evidence.append(dependency_map)
            
            investigation.progress_percent = 70.0
            
            # Generate findings
            findings = []
            
            # Idle duration finding
            if idle_hours > 168:  # 7 days
                severity = "high"
                title = "Module has been idle for over a week"
            elif idle_hours > 72:  # 3 days
                severity = "medium"
                title = "Module has been idle for several days"
            else:
                severity = "low"
                title = "Module is idle"
            
            idle_finding = Finding(
                finding_id=f"idle_{investigation.investigation_id}_1",
                title=title,
                description=f"Module {module_path.name} has been idle for {idle_hours:.1f} hours",
                severity=severity,
                category="maintenance",
                confidence=95.0,
                impact_score=min(idle_hours / 24, 10),  # Max 10 days impact
                effort_estimate=30
            )
            
            # Add recommendations based on analysis
            if file_analysis and file_analysis.data.get('line_count', 0) < 50:
                idle_finding.recommendations.append("Consider if this small module is still needed")
                idle_finding.action_items.append("Review module necessity")
            else:
                idle_finding.recommendations.append("Review module for potential updates or improvements")
                idle_finding.action_items.append("Schedule module review")
            
            if dependency_map and dependency_map.data.get('dependents_count', 0) == 0:
                idle_finding.recommendations.append("Module has no dependents - consider deprecation")
                idle_finding.action_items.append("Evaluate module for deprecation")
            
            findings.append(idle_finding)
            
            # Git history finding
            if git_history:
                last_commit_days = git_history.data.get('days_since_last_commit', 0)
                if last_commit_days > 30:
                    git_finding = Finding(
                        finding_id=f"idle_{investigation.investigation_id}_2",
                        title="No recent commits",
                        description=f"Last commit was {last_commit_days} days ago",
                        severity="medium",
                        category="maintenance",
                        confidence=90.0,
                        recommendations=["Check if module needs updates", "Review for technical debt"]
                    )
                    findings.append(git_finding)
            
            investigation.findings.extend(findings)
            investigation.progress_percent = 100.0
        
        def investigate_coverage_gap_impl(investigation: Investigation):
            """Investigate a coverage gap."""
            source_file = Path(investigation.target)
            coverage_pct = investigation.context.get('coverage_percentage', 0)
            
            investigation.progress_percent = 20.0
            
            # Collect evidence
            file_analysis = self._collect_evidence(EvidenceType.FILE_ANALYSIS, str(source_file))
            if file_analysis:
                investigation.evidence.append(file_analysis)
            
            investigation.progress_percent = 40.0
            
            test_results = self._collect_evidence(EvidenceType.TEST_RESULTS, str(source_file))
            if test_results:
                investigation.evidence.append(test_results)
            
            investigation.progress_percent = 60.0
            
            code_metrics = self._collect_evidence(EvidenceType.CODE_METRICS, str(source_file))
            if code_metrics:
                investigation.evidence.append(code_metrics)
            
            investigation.progress_percent = 80.0
            
            # Generate findings
            findings = []
            
            # Coverage gap finding
            if coverage_pct < 30:
                severity = "critical"
                title = "Very low test coverage"
            elif coverage_pct < 50:
                severity = "high"
                title = "Low test coverage"
            elif coverage_pct < 70:
                severity = "medium"
                title = "Below target test coverage"
            else:
                severity = "low"
                title = "Moderate coverage gap"
            
            coverage_finding = Finding(
                finding_id=f"coverage_{investigation.investigation_id}_1",
                title=title,
                description=f"Module {source_file.name} has {coverage_pct:.1f}% test coverage",
                severity=severity,
                category="quality",
                confidence=95.0,
                impact_score=(100 - coverage_pct) / 10,  # Higher impact for lower coverage
                effort_estimate=max(30, int((100 - coverage_pct) * 2))  # 2 min per % point
            )
            
            # Add specific recommendations
            if file_analysis:
                function_count = file_analysis.data.get('function_count', 0)
                class_count = file_analysis.data.get('class_count', 0)
                
                if function_count > class_count * 3:  # Function-heavy
                    coverage_finding.recommendations.append("Focus on unit testing individual functions")
                elif class_count > 0:  # Class-based
                    coverage_finding.recommendations.append("Create comprehensive class-level tests")
                
                if file_analysis.data.get('complexity_score', 0) > 80:
                    coverage_finding.recommendations.append("Break down complex functions before testing")
                    coverage_finding.action_items.append("Refactor complex code")
            
            coverage_finding.recommendations.extend([
                f"Increase coverage from {coverage_pct:.1f}% to at least 80%",
                "Focus on edge cases and error conditions",
                "Add integration tests if module has external dependencies"
            ])
            
            coverage_finding.action_items.extend([
                "Generate missing unit tests",
                "Review uncovered code paths",
                "Add test assertions for edge cases"
            ])
            
            findings.append(coverage_finding)
            
            # Code quality finding
            if code_metrics:
                complexity = code_metrics.data.get('complexity_score', 0)
                if complexity > 80:
                    quality_finding = Finding(
                        finding_id=f"coverage_{investigation.investigation_id}_2",
                        title="High code complexity affects testability",
                        description=f"Module complexity score is {complexity:.1f}/100",
                        severity="medium",
                        category="quality",
                        confidence=85.0,
                        recommendations=[
                            "Refactor complex functions to improve testability",
                            "Extract helper functions",
                            "Simplify conditional logic"
                        ],
                        action_items=[
                            "Identify most complex functions",
                            "Plan refactoring approach"
                        ]
                    )
                    findings.append(quality_finding)
            
            investigation.findings.extend(findings)
            investigation.progress_percent = 100.0
        
        def investigate_test_failure_impl(investigation: Investigation):
            """Investigate a test failure."""
            test_file = Path(investigation.target)
            error_message = investigation.context.get('error_message', '')
            source_file = investigation.context.get('source_file')
            
            investigation.progress_percent = 25.0
            
            # Collect evidence
            file_analysis = self._collect_evidence(EvidenceType.FILE_ANALYSIS, str(test_file))
            if file_analysis:
                investigation.evidence.append(file_analysis)
            
            investigation.progress_percent = 50.0
            
            if source_file:
                source_analysis = self._collect_evidence(EvidenceType.FILE_ANALYSIS, source_file)
                if source_analysis:
                    investigation.evidence.append(source_analysis)
            
            investigation.progress_percent = 75.0
            
            git_history = self._collect_evidence(EvidenceType.GIT_HISTORY, str(test_file))
            if git_history:
                investigation.evidence.append(git_history)
            
            # Generate findings
            findings = []
            
            # Categorize error
            error_category = self._categorize_test_error(error_message)
            
            if error_category == "import":
                severity = "medium"
                title = "Import error in test"
                recommendations = [
                    "Check import paths and module availability",
                    "Verify all dependencies are installed",
                    "Check for circular import issues"
                ]
                action_items = [
                    "Fix import statements",
                    "Update dependency declarations"
                ]
            elif error_category == "assertion":
                severity = "high"
                title = "Test assertion failure"
                recommendations = [
                    "Review test expectations vs actual behavior",
                    "Check if source code changes broke test assumptions",
                    "Update test data or mocks if needed"
                ]
                action_items = [
                    "Analyze assertion failure details",
                    "Update test or fix source code"
                ]
            elif error_category == "syntax":
                severity = "high"
                title = "Syntax error in test"
                recommendations = [
                    "Fix syntax errors in test code",
                    "Check for Python version compatibility",
                    "Validate code formatting"
                ]
                action_items = [
                    "Fix syntax errors immediately"
                ]
            else:
                severity = "medium"
                title = "Test execution error"
                recommendations = [
                    "Debug test execution environment",
                    "Check for test dependencies and setup",
                    "Review error message for specific issues"
                ]
                action_items = [
                    "Debug test failure cause"
                ]
            
            failure_finding = Finding(
                finding_id=f"failure_{investigation.investigation_id}_1",
                title=title,
                description=f"Test {test_file.name} failed: {error_message[:200]}",
                severity=severity,
                category="testing",
                confidence=90.0,
                impact_score=8.0,  # Test failures are high impact
                effort_estimate=30,
                recommendations=recommendations,
                action_items=action_items
            )
            
            findings.append(failure_finding)
            
            # Git history analysis
            if git_history:
                recent_changes = git_history.data.get('recent_commits', [])
                if recent_changes:
                    git_finding = Finding(
                        finding_id=f"failure_{investigation.investigation_id}_2",
                        title="Recent changes may have caused failure",
                        description=f"Found {len(recent_changes)} recent commits",
                        severity="medium",
                        category="maintenance",
                        confidence=70.0,
                        recommendations=[
                            "Review recent changes for breaking modifications",
                            "Consider if test needs updating due to code changes"
                        ],
                        action_items=[
                            "Review recent commit history",
                            "Identify potential breaking changes"
                        ]
                    )
                    findings.append(git_finding)
            
            investigation.findings.extend(findings)
            investigation.progress_percent = 100.0
        
        # Register investigators
        self._investigators = {
            InvestigationType.IDLE_MODULE: investigate_idle_module_impl,
            InvestigationType.COVERAGE_GAP: investigate_coverage_gap_impl,
            InvestigationType.TEST_FAILURE: investigate_test_failure_impl
        }
    
    def _setup_evidence_collectors(self):
        """Setup evidence collection strategies."""
        
        def collect_file_analysis(target: str) -> Optional[Evidence]:
            """Collect file analysis evidence."""
            try:
                file_path = Path(target)
                if not file_path.exists():
                    return None
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST for analysis
                tree = ast.parse(content)
                
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
                imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                
                lines = [line.strip() for line in content.split('\\n')]
                code_lines = [line for line in lines if line and not line.startswith('#')]
                
                data = {
                    'file_size': file_path.stat().st_size,
                    'line_count': len(code_lines),
                    'class_count': len(classes),
                    'function_count': len(functions),
                    'import_count': len(imports),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'complexity_score': min(len(code_lines) / 10 + len(classes) * 5 + len(functions) * 2, 100)
                }
                
                return Evidence(
                    evidence_type=EvidenceType.FILE_ANALYSIS,
                    source=target,
                    data=data,
                    confidence=95.0
                )
                
            except Exception as e:
                print(f"⚠️ Error collecting file analysis for {target}: {e}")
                return None
        
        def collect_git_history(target: str) -> Optional[Evidence]:
            """Collect git history evidence."""
            try:
                # Get git log for file
                result = subprocess.run([
                    'git', 'log', '-10', '--format=%H|%an|%ad|%s', '--', target
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode != 0:
                    return None
                
                commits = []
                lines = result.stdout.strip().split('\\n')
                
                for line in lines:
                    if '|' in line:
                        parts = line.split('|', 3)
                        if len(parts) >= 4:
                            commits.append({
                                'hash': parts[0],
                                'author': parts[1],
                                'date': parts[2],
                                'message': parts[3]
                            })
                
                # Calculate days since last commit
                days_since_last = 0
                if commits:
                    try:
                        from dateutil import parser
                        last_commit_date = parser.parse(commits[0]['date'])
                        days_since_last = (datetime.now() - last_commit_date.replace(tzinfo=None)).days
                    except:
                        days_since_last = 0
                
                data = {
                    'commit_count': len(commits),
                    'recent_commits': commits[:5],  # Last 5 commits
                    'days_since_last_commit': days_since_last,
                    'authors': list(set(commit['author'] for commit in commits))
                }
                
                return Evidence(
                    evidence_type=EvidenceType.GIT_HISTORY,
                    source=target,
                    data=data,
                    confidence=85.0
                )
                
            except Exception as e:
                print(f"⚠️ Error collecting git history for {target}: {e}")
                return None
        
        def collect_dependency_map(target: str) -> Optional[Evidence]:
            """Collect dependency mapping evidence."""
            try:
                file_path = Path(target)
                if not file_path.exists():
                    return None
                
                # Find files that import this module
                dependents = []
                module_name = file_path.stem
                
                for watch_path in self.watch_paths:
                    for py_file in watch_path.rglob("*.py"):
                        if py_file == file_path:
                            continue
                        
                        try:
                            with open(py_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Simple dependency check
                            if (f"import {module_name}" in content or
                                f"from {module_name}" in content or
                                f"from .{module_name}" in content):
                                dependents.append(str(py_file))
                        except:
                            continue
                
                # Analyze this module's dependencies
                dependencies = []
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                dependencies.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                dependencies.append(node.module)
                except:
                    pass
                
                data = {
                    'dependencies': dependencies,
                    'dependents': dependents,
                    'dependencies_count': len(dependencies),
                    'dependents_count': len(dependents),
                    'coupling_score': len(dependencies) + len(dependents)
                }
                
                return Evidence(
                    evidence_type=EvidenceType.DEPENDENCY_MAP,
                    source=target,
                    data=data,
                    confidence=80.0
                )
                
            except Exception as e:
                print(f"⚠️ Error collecting dependency map for {target}: {e}")
                return None
        
        def collect_test_results(target: str) -> Optional[Evidence]:
            """Collect test results evidence."""
            try:
                # Look for corresponding test file
                source_path = Path(target)
                
                # Find test files
                test_files = []
                test_name = f"test_{source_path.stem}.py"
                
                for watch_path in self.watch_paths:
                    for test_file in watch_path.rglob(test_name):
                        test_files.append(str(test_file))
                    
                    # Also check for tests directory
                    tests_dir = watch_path / "tests"
                    if tests_dir.exists():
                        for test_file in tests_dir.rglob(f"*{source_path.stem}*.py"):
                            test_files.append(str(test_file))
                
                data = {
                    'test_files': test_files,
                    'test_files_count': len(test_files),
                    'has_tests': len(test_files) > 0
                }
                
                return Evidence(
                    evidence_type=EvidenceType.TEST_RESULTS,
                    source=target,
                    data=data,
                    confidence=75.0
                )
                
            except Exception as e:
                print(f"⚠️ Error collecting test results for {target}: {e}")
                return None
        
        def collect_code_metrics(target: str) -> Optional[Evidence]:
            """Collect code quality metrics evidence."""
            try:
                file_path = Path(target)
                if not file_path.exists():
                    return None
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Calculate various metrics
                total_lines = len(content.split('\\n'))
                code_lines = len([line for line in content.split('\\n') if line.strip() and not line.strip().startswith('#')])
                
                functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                # Calculate complexity (simplified)
                complexity_score = 0
                complexity_score += len(functions) * 2
                complexity_score += len(classes) * 5
                complexity_score += code_lines / 10
                complexity_score = min(complexity_score, 100)
                
                # Calculate maintainability
                avg_function_lines = code_lines / max(len(functions), 1)
                maintainability = max(0, 100 - avg_function_lines - complexity_score / 2)
                
                data = {
                    'total_lines': total_lines,
                    'code_lines': code_lines,
                    'comment_ratio': (total_lines - code_lines) / max(total_lines, 1),
                    'function_count': len(functions),
                    'class_count': len(classes),
                    'avg_function_lines': avg_function_lines,
                    'complexity_score': complexity_score,
                    'maintainability_score': maintainability
                }
                
                return Evidence(
                    evidence_type=EvidenceType.CODE_METRICS,
                    source=target,
                    data=data,
                    confidence=85.0
                )
                
            except Exception as e:
                print(f"⚠️ Error collecting code metrics for {target}: {e}")
                return None
        
        # Register evidence collectors
        self._evidence_collectors = {
            EvidenceType.FILE_ANALYSIS: collect_file_analysis,
            EvidenceType.GIT_HISTORY: collect_git_history,
            EvidenceType.DEPENDENCY_MAP: collect_dependency_map,
            EvidenceType.TEST_RESULTS: collect_test_results,
            EvidenceType.CODE_METRICS: collect_code_metrics
        }
    
    def _collect_evidence(self, evidence_type: EvidenceType, target: str) -> Optional[Evidence]:
        """Collect specific type of evidence."""
        collector = self._evidence_collectors.get(evidence_type)
        
        if collector:
            try:
                return collector(target)
            except Exception as e:
                print(f"⚠️ Error collecting {evidence_type.value} evidence: {e}")
        
        return None
    
    def _categorize_test_error(self, error_message: str) -> str:
        """Categorize test error message."""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in ["import", "module", "not found"]):
            return "import"
        elif any(keyword in error_lower for keyword in ["assertion", "assert", "expected"]):
            return "assertion"
        elif any(keyword in error_lower for keyword in ["syntax", "invalid syntax"]):
            return "syntax"
        elif any(keyword in error_lower for keyword in ["timeout", "time"]):
            return "timeout"
        else:
            return "runtime"
    
    def _generate_investigation_summary(self, investigation: Investigation):
        """Generate investigation summary."""
        findings_count = len(investigation.findings)
        critical_count = len([f for f in investigation.findings if f.severity == "critical"])
        high_count = len([f for f in investigation.findings if f.severity == "high"])
        
        if findings_count == 0:
            investigation.summary = "No significant issues found."
            investigation.overall_assessment = "HEALTHY"
        elif critical_count > 0:
            investigation.summary = f"Found {critical_count} critical issue(s) requiring immediate attention."
            investigation.overall_assessment = "CRITICAL"
        elif high_count > 0:
            investigation.summary = f"Found {high_count} high-priority issue(s) that should be addressed soon."
            investigation.overall_assessment = "NEEDS_ATTENTION"
        else:
            investigation.summary = f"Found {findings_count} minor issue(s) for future consideration."
            investigation.overall_assessment = "MINOR_ISSUES"
    
    def _generate_investigation_id(self) -> str:
        """Generate unique investigation ID."""
        import time
        return f"inv_{int(time.time() * 1000)}_{hash(datetime.now()) % 10000}"
    
    def get_investigation_result(self, investigation_id: str) -> Optional[InvestigationResult]:
        """Get result of an investigation."""
        return self._investigation_results.get(investigation_id)
    
    def get_all_investigations(self) -> List[Investigation]:
        """Get all investigations."""
        return list(self._investigations.values())
    
    def get_recent_investigations(self, hours: int = 24) -> List[Investigation]:
        """Get recent investigations."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            inv for inv in self._investigations.values()
            if inv.created_at > cutoff
        ]
    
    def get_investigation_statistics(self) -> Dict[str, Any]:
        """Get investigation statistics."""
        total_investigations = len(self._investigations)
        
        # Count by type
        type_counts = {}
        for inv_type in InvestigationType:
            type_counts[inv_type.value] = len([
                inv for inv in self._investigations.values()
                if inv.investigation_type == inv_type
            ])
        
        # Count by status
        status_counts = {}
        for status in ["pending", "running", "completed", "failed"]:
            status_counts[status] = len([
                inv for inv in self._investigations.values()
                if inv.status == status
            ])
        
        # Calculate average time
        completed_investigations = [
            inv for inv in self._investigations.values()
            if inv.status == "completed" and inv.started_at and inv.completed_at
        ]
        
        avg_time = 0.0
        if completed_investigations:
            total_time = sum(
                (inv.completed_at - inv.started_at).total_seconds()
                for inv in completed_investigations
            )
            avg_time = total_time / len(completed_investigations)
        
        return {
            "total_investigations": total_investigations,
            "type_distribution": type_counts,
            "status_distribution": status_counts,
            "average_investigation_time_seconds": avg_time,
            "total_findings": sum(len(inv.findings) for inv in self._investigations.values()),
            "total_evidence": sum(len(inv.evidence) for inv in self._investigations.values()),
            "statistics": dict(self._stats)
        }
    
    def export_investigation_report(self, output_path: str = "investigation_report.json"):
        """Export comprehensive investigation report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_investigation_statistics(),
            "investigations": [],
            "results": []
        }
        
        # Add investigations
        for investigation in self._investigations.values():
            inv_data = {
                "investigation_id": investigation.investigation_id,
                "investigation_type": investigation.investigation_type.value,
                "priority": investigation.priority.value,
                "target": investigation.target,
                "status": investigation.status,
                "progress_percent": investigation.progress_percent,
                "created_at": investigation.created_at.isoformat(),
                "started_at": investigation.started_at.isoformat() if investigation.started_at else None,
                "completed_at": investigation.completed_at.isoformat() if investigation.completed_at else None,
                "summary": investigation.summary,
                "overall_assessment": investigation.overall_assessment,
                "findings_count": len(investigation.findings),
                "evidence_count": len(investigation.evidence)
            }
            report["investigations"].append(inv_data)
        
        # Add results
        for result in self._investigation_results.values():
            result_data = {
                "investigation_id": result.investigation.investigation_id,
                "success": result.success,
                "execution_time_seconds": result.execution_time_seconds,
                "total_findings": result.total_findings,
                "critical_findings": result.critical_findings,
                "high_findings": result.high_findings,
                "evidence_count": result.evidence_count,
                "immediate_actions": result.immediate_actions,
                "long_term_recommendations": result.long_term_recommendations
            }
            report["results"].append(result_data)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Investigation report exported to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting investigation report: {e}")


# Convenience functions for common investigations
def investigate_idle_modules_in_directory(directory: str, idle_threshold_hours: float = 168) -> List[str]:
    """Investigate all idle modules in a directory."""
    investigator = AutoInvestigator(directory)
    investigation_ids = []
    
    # Find idle modules (simplified)
    for py_file in Path(directory).rglob("*.py"):
        if py_file.exists():
            last_modified = datetime.fromtimestamp(py_file.stat().st_mtime)
            hours_idle = (datetime.now() - last_modified).total_seconds() / 3600
            
            if hours_idle > idle_threshold_hours:
                inv_id = investigator.investigate_idle_module(str(py_file), hours_idle)
                investigation_ids.append(inv_id)
    
    return investigation_ids