"""
Coverage Intelligence System

Intelligent coverage analysis with critical path identification
and gap prioritization for strategic test improvements.

Features:
- Critical path identification through code flow analysis
- Coverage gap prioritization based on business impact
- Risk assessment for uncovered code paths
- Strategic test recommendations
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import networkx as nx

from core.layer_manager import requires_layer


class PathCriticality(Enum):
    """Levels of path criticality."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GapType(Enum):
    """Types of coverage gaps."""
    MISSING_FUNCTION = "missing_function"
    MISSING_BRANCH = "missing_branch"
    MISSING_EXCEPTION = "missing_exception"
    MISSING_EDGE_CASE = "missing_edge_case"
    MISSING_INTEGRATION = "missing_integration"
    MISSING_ERROR_PATH = "missing_error_path"


class RiskLevel(Enum):
    """Risk levels for uncovered code."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class CriticalPath:
    """A critical execution path in the codebase."""
    path_id: str
    module_path: str
    function_name: str
    start_line: int
    end_line: int
    
    # Path characteristics
    criticality: PathCriticality
    risk_level: RiskLevel
    complexity_score: float
    
    # Path flow
    execution_flow: List[str] = field(default_factory=list)
    decision_points: List[int] = field(default_factory=list)
    exception_handlers: List[int] = field(default_factory=list)
    
    # Dependencies
    calls_functions: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    
    # Coverage status
    is_covered: bool = False
    coverage_percentage: float = 0.0
    test_count: int = 0
    
    # Business context
    business_impact: str = "unknown"
    user_facing: bool = False
    data_sensitive: bool = False
    
    # Analysis metadata
    identified_at: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None


@dataclass
class CoverageGap:
    """A gap in test coverage."""
    gap_id: str
    module_path: str
    gap_type: GapType
    
    # Gap location
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    branch_condition: Optional[str] = None
    
    # Gap assessment
    priority_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    impact_assessment: str = ""
    
    # Gap context
    surrounding_context: str = ""
    related_functions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Recommendations
    suggested_tests: List[str] = field(default_factory=list)
    test_approach: str = ""
    estimated_effort: str = "medium"
    
    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class ModuleCoverage:
    """Coverage analysis for a module."""
    module_path: str
    
    # Coverage metrics
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    overall_score: float = 0.0
    
    # Critical paths
    critical_paths: List[CriticalPath] = field(default_factory=list)
    uncovered_critical_paths: List[CriticalPath] = field(default_factory=list)
    
    # Coverage gaps
    coverage_gaps: List[CoverageGap] = field(default_factory=list)
    high_priority_gaps: List[CoverageGap] = field(default_factory=list)
    
    # Risk assessment
    overall_risk: RiskLevel = RiskLevel.LOW
    risk_factors: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    priority_level: str = "normal"
    
    # Analysis metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    test_files: List[str] = field(default_factory=list)


@dataclass
class CoverageIntelligenceReport:
    """Complete coverage intelligence report."""
    modules: Dict[str, ModuleCoverage] = field(default_factory=dict)
    
    # Overall metrics
    overall_coverage: float = 0.0
    critical_path_coverage: float = 0.0
    high_risk_modules: List[str] = field(default_factory=list)
    
    # Priority recommendations
    top_priority_gaps: List[CoverageGap] = field(default_factory=list)
    critical_uncovered_paths: List[CriticalPath] = field(default_factory=list)
    
    # Analysis summary
    total_modules_analyzed: int = 0
    total_critical_paths: int = 0
    total_coverage_gaps: int = 0
    
    # Timestamps
    generated_at: datetime = field(default_factory=datetime.now)
    coverage_data_source: str = "static_analysis"


class CoverageIntelligence:
    """
    Coverage intelligence system with critical path identification.
    
    Analyzes code structure to identify critical execution paths,
    assess coverage gaps, and provide strategic test recommendations.
    """
    
    @requires_layer("layer3_orchestration", "coverage_intelligence")
    def __init__(self, watch_paths: Union[str, List[str]]):
        """
        Initialize coverage intelligence system.
        
        Args:
            watch_paths: Directories to analyze
        """
        self.watch_paths = [Path(p) for p in (watch_paths if isinstance(watch_paths, list) else [watch_paths])]
        
        # Analysis cache
        self._coverage_report: Optional[CoverageIntelligenceReport] = None
        self._analysis_cache: Dict[str, Any] = {}
        
        # Critical path patterns
        self._critical_patterns = {
            'authentication': ['login', 'logout', 'auth', 'verify', 'token'],
            'data_access': ['save', 'delete', 'update', 'create', 'query'],
            'payment': ['payment', 'charge', 'refund', 'billing'],
            'security': ['validate', 'sanitize', 'encrypt', 'decrypt'],
            'api_endpoints': ['endpoint', 'route', 'handler', 'controller'],
            'error_handling': ['error', 'exception', 'fail', 'catch']
        }
        
        # Risk indicators
        self._risk_indicators = {
            'user_facing': ['api', 'endpoint', 'view', 'controller', 'handler'],
            'data_sensitive': ['password', 'token', 'key', 'secret', 'user', 'customer'],
            'financial': ['payment', 'price', 'cost', 'billing', 'charge'],
            'security': ['auth', 'login', 'permission', 'role', 'access'],
            'critical_flow': ['main', 'core', 'primary', 'essential']
        }
        
        # Statistics
        self._stats = {
            'modules_analyzed': 0,
            'critical_paths_identified': 0,
            'coverage_gaps_found': 0,
            'high_risk_gaps': 0,
            'last_analysis': None
        }
        
        print("🎯 Coverage intelligence initialized")
        print(f"   📁 Analyzing: {', '.join(str(p) for p in self.watch_paths)}")
    
    def analyze_coverage_intelligence(self, coverage_data: Dict[str, float] = None,
                                    force_reanalysis: bool = False) -> CoverageIntelligenceReport:
        """
        Perform comprehensive coverage intelligence analysis.
        
        Args:
            coverage_data: External coverage data if available
            force_reanalysis: Force re-analysis of all modules
            
        Returns:
            Complete coverage intelligence report
        """
        print("🔍 Analyzing coverage intelligence...")
        
        if self._coverage_report and not force_reanalysis:
            return self._coverage_report
        
        report = CoverageIntelligenceReport()
        
        # Phase 1: Analyze individual modules
        print("   📊 Phase 1: Analyzing module coverage...")
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
            
            for py_file in watch_path.rglob("*.py"):
                if self._should_analyze_file(py_file):
                    module_coverage = self._analyze_module_coverage(py_file, coverage_data)
                    if module_coverage:
                        report.modules[str(py_file)] = module_coverage
                        self._stats['modules_analyzed'] += 1
        
        # Phase 2: Identify critical paths
        print("   🛤️ Phase 2: Identifying critical paths...")
        self._identify_critical_paths(report)
        
        # Phase 3: Detect coverage gaps
        print("   🔍 Phase 3: Detecting coverage gaps...")
        self._detect_coverage_gaps(report)
        
        # Phase 4: Assess risks and priorities
        print("   ⚠️ Phase 4: Assessing risks and priorities...")
        self._assess_risks_and_priorities(report)
        
        # Phase 5: Generate recommendations
        print("   💡 Phase 5: Generating recommendations...")
        self._generate_recommendations(report)
        
        # Calculate overall metrics
        self._calculate_overall_metrics(report)
        
        report.total_modules_analyzed = len(report.modules)
        report.coverage_data_source = "hybrid" if coverage_data else "static_analysis"
        
        self._coverage_report = report
        self._stats['last_analysis'] = datetime.now()
        
        print(f"✅ Coverage analysis complete: {len(report.modules)} modules, "
              f"{report.total_critical_paths} critical paths, {report.total_coverage_gaps} gaps")
        
        return report
    
    def _analyze_module_coverage(self, file_path: Path, coverage_data: Dict[str, float] = None) -> Optional[ModuleCoverage]:
        """Analyze coverage for a single module."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Create module coverage
            module_coverage = ModuleCoverage(
                module_path=str(file_path)
            )
            
            # Get coverage metrics from external data if available
            if coverage_data:
                file_key = str(file_path)
                module_coverage.line_coverage = coverage_data.get(file_key, 0.0)
                module_coverage.overall_score = module_coverage.line_coverage
            
            # Analyze critical paths
            module_coverage.critical_paths = self._extract_critical_paths(tree, str(file_path))
            self._stats['critical_paths_identified'] += len(module_coverage.critical_paths)
            
            # Find test files
            module_coverage.test_files = self._find_test_files(file_path)
            
            # Calculate function coverage
            module_coverage.function_coverage = self._calculate_function_coverage(tree, module_coverage.test_files)
            
            # Assess overall risk
            module_coverage.overall_risk = self._assess_module_risk(module_coverage)
            
            return module_coverage
            
        except Exception as e:
            print(f"⚠️ Error analyzing module coverage {file_path}: {e}")
            return None
    
    def _extract_critical_paths(self, tree: ast.AST, module_path: str) -> List[CriticalPath]:
        """Extract critical execution paths from AST."""
        critical_paths = []
        path_id = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Analyze function for critical patterns
                criticality = self._assess_path_criticality(node, module_path)
                
                if criticality != PathCriticality.LOW:
                    path_id += 1
                    
                    critical_path = CriticalPath(
                        path_id=f"{Path(module_path).stem}_path_{path_id}",
                        module_path=module_path,
                        function_name=node.name,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        criticality=criticality,
                        risk_level=self._assess_path_risk(node, module_path),
                        complexity_score=self._calculate_path_complexity(node)
                    )
                    
                    # Extract execution flow
                    critical_path.execution_flow = self._extract_execution_flow(node)
                    critical_path.decision_points = self._find_decision_points(node)
                    critical_path.exception_handlers = self._find_exception_handlers(node)
                    critical_path.calls_functions = self._extract_function_calls(node)
                    
                    # Assess business context
                    critical_path.user_facing = self._is_user_facing(node, module_path)
                    critical_path.data_sensitive = self._is_data_sensitive(node, module_path)
                    critical_path.business_impact = self._assess_business_impact(node, module_path)
                    
                    critical_paths.append(critical_path)
        
        return critical_paths
    
    def _assess_path_criticality(self, node: ast.FunctionDef, module_path: str) -> PathCriticality:
        """Assess the criticality of a code path."""
        function_name = node.name.lower()
        module_name = Path(module_path).name.lower()
        
        # Check for critical patterns
        critical_score = 0
        
        for category, patterns in self._critical_patterns.items():
            for pattern in patterns:
                if pattern in function_name or pattern in module_name:
                    if category in ['authentication', 'payment', 'security']:
                        critical_score += 3
                    elif category in ['data_access', 'api_endpoints']:
                        critical_score += 2
                    else:
                        critical_score += 1
        
        # Check function complexity
        complexity = self._calculate_path_complexity(node)
        if complexity > 15:
            critical_score += 2
        elif complexity > 10:
            critical_score += 1
        
        # Check for error handling
        has_exception_handling = any(isinstance(child, (ast.Try, ast.Raise, ast.ExceptHandler))
                                   for child in ast.walk(node))
        if not has_exception_handling and critical_score > 0:
            critical_score += 1  # Critical paths without error handling are more critical
        
        # Map score to criticality
        if critical_score >= 6:
            return PathCriticality.CRITICAL
        elif critical_score >= 4:
            return PathCriticality.HIGH
        elif critical_score >= 2:
            return PathCriticality.MEDIUM
        else:
            return PathCriticality.LOW
    
    def _assess_path_risk(self, node: ast.FunctionDef, module_path: str) -> RiskLevel:
        """Assess the risk level of a code path."""
        function_name = node.name.lower()
        module_name = Path(module_path).name.lower()
        
        risk_score = 0
        
        # Check risk indicators
        for category, indicators in self._risk_indicators.items():
            for indicator in indicators:
                if indicator in function_name or indicator in module_name:
                    if category in ['financial', 'security']:
                        risk_score += 3
                    elif category in ['user_facing', 'data_sensitive']:
                        risk_score += 2
                    else:
                        risk_score += 1
        
        # Check for external dependencies
        external_calls = [call for call in self._extract_function_calls(node)
                         if not call.startswith('self.') and '.' in call]
        risk_score += min(len(external_calls), 3)
        
        # Check for file/database operations
        has_io_operations = any(
            call in self._extract_function_calls(node)
            for call in ['open', 'read', 'write', 'save', 'delete', 'execute', 'query']
        )
        if has_io_operations:
            risk_score += 2
        
        # Map score to risk level
        if risk_score >= 8:
            return RiskLevel.SEVERE
        elif risk_score >= 6:
            return RiskLevel.HIGH
        elif risk_score >= 4:
            return RiskLevel.MODERATE
        elif risk_score >= 2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _calculate_path_complexity(self, node: ast.AST) -> float:
        """Calculate cyclomatic complexity of a path."""
        complexity = 1.0  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity
    
    def _extract_execution_flow(self, node: ast.FunctionDef) -> List[str]:
        """Extract execution flow steps from function."""
        flow_steps = []
        
        for stmt in node.body:
            if isinstance(stmt, ast.If):
                flow_steps.append(f"condition: line {stmt.lineno}")
            elif isinstance(stmt, (ast.For, ast.While)):
                flow_steps.append(f"loop: line {stmt.lineno}")
            elif isinstance(stmt, ast.Try):
                flow_steps.append(f"try_block: line {stmt.lineno}")
            elif isinstance(stmt, ast.Return):
                flow_steps.append(f"return: line {stmt.lineno}")
            elif isinstance(stmt, ast.Raise):
                flow_steps.append(f"raise: line {stmt.lineno}")
        
        return flow_steps
    
    def _find_decision_points(self, node: ast.FunctionDef) -> List[int]:
        """Find decision points (if statements, loops) in function."""
        decision_points = []
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                decision_points.append(child.lineno)
        
        return decision_points
    
    def _find_exception_handlers(self, node: ast.FunctionDef) -> List[int]:
        """Find exception handling points in function."""
        exception_points = []
        
        for child in ast.walk(node):
            if isinstance(child, (ast.Try, ast.ExceptHandler, ast.Raise)):
                exception_points.append(child.lineno)
        
        return exception_points
    
    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls from AST node."""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    try:
                        calls.append(ast.unparse(child.func))
                    except:
                        pass
        
        return calls
    
    def _is_user_facing(self, node: ast.FunctionDef, module_path: str) -> bool:
        """Check if function is user-facing."""
        indicators = self._risk_indicators['user_facing']
        function_name = node.name.lower()
        module_name = Path(module_path).name.lower()
        
        return any(indicator in function_name or indicator in module_name for indicator in indicators)
    
    def _is_data_sensitive(self, node: ast.FunctionDef, module_path: str) -> bool:
        """Check if function handles sensitive data."""
        indicators = self._risk_indicators['data_sensitive']
        function_name = node.name.lower()
        module_name = Path(module_path).name.lower()
        
        return any(indicator in function_name or indicator in module_name for indicator in indicators)
    
    def _assess_business_impact(self, node: ast.FunctionDef, module_path: str) -> str:
        """Assess business impact of function."""
        function_name = node.name.lower()
        module_name = Path(module_path).name.lower()
        
        if any(term in function_name or term in module_name 
               for term in ['payment', 'billing', 'charge', 'refund']):
            return "revenue_critical"
        elif any(term in function_name or term in module_name 
                 for term in ['auth', 'login', 'security', 'permission']):
            return "security_critical"
        elif any(term in function_name or term in module_name 
                 for term in ['user', 'customer', 'profile']):
            return "user_experience"
        elif any(term in function_name or term in module_name 
                 for term in ['data', 'save', 'delete', 'update']):
            return "data_integrity"
        else:
            return "operational"
    
    def _find_test_files(self, source_file: Path) -> List[str]:
        """Find test files for a source module."""
        test_files = []
        
        # Common test patterns
        source_stem = source_file.stem
        test_patterns = [
            f"test_{source_stem}.py",
            f"{source_stem}_test.py",
            f"test_{source_stem}_*.py",
            f"{source_stem}_tests.py"
        ]
        
        # Search in common test directories
        test_dirs = ['tests', 'test', 'testing']
        
        for test_dir in test_dirs:
            test_path = source_file.parent / test_dir
            if test_path.exists():
                for pattern in test_patterns:
                    test_files.extend(str(f) for f in test_path.glob(pattern))
        
        # Search in same directory
        for pattern in test_patterns:
            test_files.extend(str(f) for f in source_file.parent.glob(pattern))
        
        return test_files
    
    def _calculate_function_coverage(self, tree: ast.AST, test_files: List[str]) -> float:
        """Calculate estimated function coverage based on test files."""
        if not test_files:
            return 0.0
        
        # Count functions in source
        source_functions = [node.name for node in ast.walk(tree) 
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        
        if not source_functions:
            return 100.0  # No functions to test
        
        # Count tested functions (simplified heuristic)
        tested_functions = set()
        
        for test_file in test_files:
            try:
                if Path(test_file).exists():
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_content = f.read()
                    
                    for func_name in source_functions:
                        if func_name in test_content:
                            tested_functions.add(func_name)
            except:
                continue
        
        return (len(tested_functions) / len(source_functions)) * 100.0
    
    def _assess_module_risk(self, module_coverage: ModuleCoverage) -> RiskLevel:
        """Assess overall risk level for a module."""
        risk_factors = []
        risk_score = 0
        
        # Low coverage risk
        if module_coverage.overall_score < 50:
            risk_score += 3
            risk_factors.append("Low overall coverage")
        elif module_coverage.overall_score < 80:
            risk_score += 1
            risk_factors.append("Moderate coverage")
        
        # Critical path risk
        uncovered_critical = [p for p in module_coverage.critical_paths if not p.is_covered]
        if uncovered_critical:
            critical_count = len([p for p in uncovered_critical if p.criticality == PathCriticality.CRITICAL])
            high_count = len([p for p in uncovered_critical if p.criticality == PathCriticality.HIGH])
            
            risk_score += critical_count * 3 + high_count * 2
            risk_factors.append(f"{len(uncovered_critical)} uncovered critical paths")
        
        # Function coverage risk
        if module_coverage.function_coverage < 30:
            risk_score += 2
            risk_factors.append("Very low function coverage")
        
        # Test file availability
        if not module_coverage.test_files:
            risk_score += 2
            risk_factors.append("No test files found")
        
        module_coverage.risk_factors = risk_factors
        
        # Map score to risk level
        if risk_score >= 8:
            return RiskLevel.SEVERE
        elif risk_score >= 6:
            return RiskLevel.HIGH
        elif risk_score >= 4:
            return RiskLevel.MODERATE
        elif risk_score >= 2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _identify_critical_paths(self, report: CoverageIntelligenceReport):
        """Identify critical paths across all modules."""
        all_critical_paths = []
        
        for module_coverage in report.modules.values():
            all_critical_paths.extend(module_coverage.critical_paths)
            
            # Mark uncovered critical paths
            uncovered = [p for p in module_coverage.critical_paths if not p.is_covered]
            module_coverage.uncovered_critical_paths = uncovered
        
        report.total_critical_paths = len(all_critical_paths)
        
        # Identify most critical uncovered paths
        uncovered_critical = [p for module_cov in report.modules.values() 
                             for p in module_cov.uncovered_critical_paths
                             if p.criticality in [PathCriticality.CRITICAL, PathCriticality.HIGH]]
        
        # Sort by criticality and risk
        uncovered_critical.sort(key=lambda p: (
            -p.criticality.value.count('critical'),
            -p.risk_level.value.count('severe'),
            -p.complexity_score
        ))
        
        report.critical_uncovered_paths = uncovered_critical[:10]  # Top 10
    
    def _detect_coverage_gaps(self, report: CoverageIntelligenceReport):
        """Detect and categorize coverage gaps."""
        total_gaps = 0
        
        for module_path, module_coverage in report.modules.items():
            gaps = []
            
            # Missing function coverage
            if module_coverage.function_coverage < 80:
                gap = CoverageGap(
                    gap_id=f"{Path(module_path).stem}_missing_functions",
                    module_path=module_path,
                    gap_type=GapType.MISSING_FUNCTION,
                    priority_score=self._calculate_gap_priority(module_coverage.function_coverage, 'function'),
                    risk_level=module_coverage.overall_risk,
                    impact_assessment=f"Only {module_coverage.function_coverage:.1f}% of functions tested",
                    suggested_tests=["Add unit tests for uncovered functions"],
                    test_approach="Unit testing with edge cases"
                )
                gaps.append(gap)
            
            # Missing exception handling tests
            for critical_path in module_coverage.critical_paths:
                if critical_path.exception_handlers and not critical_path.is_covered:
                    gap = CoverageGap(
                        gap_id=f"{critical_path.path_id}_exception_handling",
                        module_path=module_path,
                        gap_type=GapType.MISSING_EXCEPTION,
                        function_name=critical_path.function_name,
                        priority_score=self._calculate_gap_priority_from_path(critical_path),
                        risk_level=critical_path.risk_level,
                        impact_assessment=f"Exception handling in {critical_path.function_name} not tested",
                        suggested_tests=[f"Test exception scenarios in {critical_path.function_name}"],
                        test_approach="Error path testing"
                    )
                    gaps.append(gap)
            
            # Missing edge case coverage
            complex_paths = [p for p in module_coverage.critical_paths 
                           if p.complexity_score > 10 and not p.is_covered]
            for path in complex_paths:
                gap = CoverageGap(
                    gap_id=f"{path.path_id}_edge_cases",
                    module_path=module_path,
                    gap_type=GapType.MISSING_EDGE_CASE,
                    function_name=path.function_name,
                    priority_score=self._calculate_gap_priority_from_path(path),
                    risk_level=path.risk_level,
                    impact_assessment=f"Complex function {path.function_name} lacks edge case testing",
                    suggested_tests=[f"Add edge case tests for {path.function_name}"],
                    test_approach="Boundary value analysis"
                )
                gaps.append(gap)
            
            module_coverage.coverage_gaps = gaps
            total_gaps += len(gaps)
            
            # Identify high priority gaps
            module_coverage.high_priority_gaps = [
                gap for gap in gaps if gap.priority_score > 70
            ]
            
            self._stats['coverage_gaps_found'] += len(gaps)
            self._stats['high_risk_gaps'] += len(module_coverage.high_priority_gaps)
        
        report.total_coverage_gaps = total_gaps
    
    def _calculate_gap_priority(self, coverage_value: float, gap_category: str) -> float:
        """Calculate priority score for a coverage gap."""
        base_priority = max(0, 100 - coverage_value)  # Lower coverage = higher priority
        
        # Adjust by category
        category_multipliers = {
            'function': 1.0,
            'branch': 1.2,
            'exception': 1.5,
            'integration': 0.8
        }
        
        multiplier = category_multipliers.get(gap_category, 1.0)
        return min(100.0, base_priority * multiplier)
    
    def _calculate_gap_priority_from_path(self, critical_path: CriticalPath) -> float:
        """Calculate priority score from critical path characteristics."""
        priority = 50.0  # Base priority
        
        # Criticality bonus
        if critical_path.criticality == PathCriticality.CRITICAL:
            priority += 30
        elif critical_path.criticality == PathCriticality.HIGH:
            priority += 20
        elif critical_path.criticality == PathCriticality.MEDIUM:
            priority += 10
        
        # Risk bonus
        if critical_path.risk_level == RiskLevel.SEVERE:
            priority += 25
        elif critical_path.risk_level == RiskLevel.HIGH:
            priority += 15
        elif critical_path.risk_level == RiskLevel.MODERATE:
            priority += 5
        
        # Complexity bonus
        if critical_path.complexity_score > 15:
            priority += 10
        elif critical_path.complexity_score > 10:
            priority += 5
        
        # Business impact bonus
        if critical_path.business_impact in ['revenue_critical', 'security_critical']:
            priority += 15
        elif critical_path.business_impact in ['user_experience', 'data_integrity']:
            priority += 10
        
        return min(100.0, priority)
    
    def _assess_risks_and_priorities(self, report: CoverageIntelligenceReport):
        """Assess risks and set priorities across modules."""
        # Identify high-risk modules
        high_risk_modules = [
            module_path for module_path, module_cov in report.modules.items()
            if module_cov.overall_risk in [RiskLevel.HIGH, RiskLevel.SEVERE]
        ]
        
        report.high_risk_modules = high_risk_modules
        
        # Collect top priority gaps across all modules
        all_gaps = []
        for module_cov in report.modules.values():
            all_gaps.extend(module_cov.coverage_gaps)
        
        # Sort by priority score
        all_gaps.sort(key=lambda g: g.priority_score, reverse=True)
        report.top_priority_gaps = all_gaps[:20]  # Top 20 gaps
    
    def _generate_recommendations(self, report: CoverageIntelligenceReport):
        """Generate recommendations for each module."""
        for module_path, module_coverage in report.modules.items():
            recommendations = []
            
            # Coverage-based recommendations
            if module_coverage.overall_score < 50:
                recommendations.append("URGENT: Increase overall test coverage - currently critically low")
                module_coverage.priority_level = "critical"
            elif module_coverage.overall_score < 80:
                recommendations.append("Improve test coverage to reach 80% minimum")
                module_coverage.priority_level = "high"
            
            # Critical path recommendations
            if module_coverage.uncovered_critical_paths:
                critical_count = len([p for p in module_coverage.uncovered_critical_paths 
                                    if p.criticality == PathCriticality.CRITICAL])
                if critical_count > 0:
                    recommendations.append(f"CRITICAL: {critical_count} critical paths lack test coverage")
                    module_coverage.priority_level = "critical"
            
            # Gap-specific recommendations
            high_priority_gaps = module_coverage.high_priority_gaps
            if high_priority_gaps:
                recommendations.append(f"Address {len(high_priority_gaps)} high-priority coverage gaps")
                
                # Add specific gap recommendations
                for gap in high_priority_gaps[:3]:  # Top 3 gaps
                    recommendations.extend(gap.suggested_tests)
            
            # Risk-based recommendations
            if module_coverage.overall_risk in [RiskLevel.HIGH, RiskLevel.SEVERE]:
                recommendations.append("High-risk module requires immediate attention")
                recommendations.append("Consider comprehensive integration testing")
            
            # Test file recommendations
            if not module_coverage.test_files:
                recommendations.append("Create test file - no existing tests found")
                module_coverage.priority_level = "high"
            
            module_coverage.recommended_actions = recommendations
    
    def _calculate_overall_metrics(self, report: CoverageIntelligenceReport):
        """Calculate overall metrics for the report."""
        if not report.modules:
            return
        
        # Overall coverage
        total_coverage = sum(m.overall_score for m in report.modules.values())
        report.overall_coverage = total_coverage / len(report.modules)
        
        # Critical path coverage
        total_critical_paths = sum(len(m.critical_paths) for m in report.modules.values())
        covered_critical_paths = sum(len([p for p in m.critical_paths if p.is_covered]) 
                                   for m in report.modules.values())
        
        if total_critical_paths > 0:
            report.critical_path_coverage = (covered_critical_paths / total_critical_paths) * 100
        else:
            report.critical_path_coverage = 100.0
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # Skip common ignore patterns
        ignore_patterns = {
            '__pycache__', '.git', '.vscode', '.idea', 'venv', '.env',
            'node_modules', '.pytest_cache', '.coverage', '.tox'
        }
        
        if any(pattern in str(file_path) for pattern in ignore_patterns):
            return False
        
        return True
    
    def get_module_critical_paths(self, module_path: str) -> List[CriticalPath]:
        """Get critical paths for a specific module."""
        if not self._coverage_report:
            return []
        
        module_coverage = self._coverage_report.modules.get(module_path)
        return module_coverage.critical_paths if module_coverage else []
    
    def get_high_priority_gaps(self, max_count: int = 10) -> List[CoverageGap]:
        """Get high priority coverage gaps across all modules."""
        if not self._coverage_report:
            return []
        
        return self._coverage_report.top_priority_gaps[:max_count]
    
    def get_modules_by_risk_level(self, risk_level: RiskLevel) -> List[str]:
        """Get modules with specific risk level."""
        if not self._coverage_report:
            return []
        
        return [
            module_path for module_path, module_cov in self._coverage_report.modules.items()
            if module_cov.overall_risk == risk_level
        ]
    
    def get_coverage_statistics(self) -> Dict[str, Any]:
        """Get coverage intelligence statistics."""
        if not self._coverage_report:
            return {"error": "No analysis performed yet"}
        
        # Risk distribution
        risk_distribution = {}
        for risk_level in RiskLevel:
            risk_distribution[risk_level.value] = len(self.get_modules_by_risk_level(risk_level))
        
        # Gap type distribution
        gap_distribution = {}
        for gap_type in GapType:
            gap_distribution[gap_type.value] = sum(
                len([g for g in m.coverage_gaps if g.gap_type == gap_type])
                for m in self._coverage_report.modules.values()
            )
        
        # Criticality distribution
        criticality_distribution = {}
        for criticality in PathCriticality:
            criticality_distribution[criticality.value] = sum(
                len([p for p in m.critical_paths if p.criticality == criticality])
                for m in self._coverage_report.modules.values()
            )
        
        return {
            "total_modules": len(self._coverage_report.modules),
            "overall_coverage": self._coverage_report.overall_coverage,
            "critical_path_coverage": self._coverage_report.critical_path_coverage,
            "total_critical_paths": self._coverage_report.total_critical_paths,
            "total_coverage_gaps": self._coverage_report.total_coverage_gaps,
            "high_risk_modules": len(self._coverage_report.high_risk_modules),
            "top_priority_gaps": len(self._coverage_report.top_priority_gaps),
            "critical_uncovered_paths": len(self._coverage_report.critical_uncovered_paths),
            "risk_distribution": risk_distribution,
            "gap_type_distribution": gap_distribution,
            "criticality_distribution": criticality_distribution,
            "statistics": dict(self._stats)
        }
    
    def export_coverage_report(self, output_path: str = "coverage_intelligence_report.json"):
        """Export comprehensive coverage intelligence report."""
        if not self._coverage_report:
            print("⚠️ No analysis performed yet. Run analyze_coverage_intelligence() first.")
            return
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "analysis_timestamp": self._coverage_report.generated_at.isoformat(),
            "statistics": self.get_coverage_statistics(),
            "overall_metrics": {
                "overall_coverage": self._coverage_report.overall_coverage,
                "critical_path_coverage": self._coverage_report.critical_path_coverage,
                "total_modules": len(self._coverage_report.modules),
                "high_risk_modules": len(self._coverage_report.high_risk_modules)
            },
            "modules": {},
            "top_priority_gaps": [],
            "critical_uncovered_paths": []
        }
        
        # Add module details
        for module_path, module_cov in self._coverage_report.modules.items():
            module_data = {
                "overall_score": module_cov.overall_score,
                "function_coverage": module_cov.function_coverage,
                "overall_risk": module_cov.overall_risk.value,
                "priority_level": module_cov.priority_level,
                "critical_paths_count": len(module_cov.critical_paths),
                "uncovered_critical_paths": len(module_cov.uncovered_critical_paths),
                "coverage_gaps_count": len(module_cov.coverage_gaps),
                "high_priority_gaps": len(module_cov.high_priority_gaps),
                "recommended_actions": module_cov.recommended_actions,
                "risk_factors": module_cov.risk_factors,
                "test_files": module_cov.test_files,
                "analyzed_at": module_cov.analyzed_at.isoformat()
            }
            report["modules"][module_path] = module_data
        
        # Add top priority gaps
        for gap in self._coverage_report.top_priority_gaps:
            gap_data = {
                "gap_id": gap.gap_id,
                "module_path": gap.module_path,
                "gap_type": gap.gap_type.value,
                "priority_score": gap.priority_score,
                "risk_level": gap.risk_level.value,
                "impact_assessment": gap.impact_assessment,
                "suggested_tests": gap.suggested_tests,
                "test_approach": gap.test_approach,
                "estimated_effort": gap.estimated_effort
            }
            report["top_priority_gaps"].append(gap_data)
        
        # Add critical uncovered paths
        for path in self._coverage_report.critical_uncovered_paths:
            path_data = {
                "path_id": path.path_id,
                "module_path": path.module_path,
                "function_name": path.function_name,
                "criticality": path.criticality.value,
                "risk_level": path.risk_level.value,
                "complexity_score": path.complexity_score,
                "business_impact": path.business_impact,
                "user_facing": path.user_facing,
                "data_sensitive": path.data_sensitive,
                "line_range": f"{path.start_line}-{path.end_line}"
            }
            report["critical_uncovered_paths"].append(path_data)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Coverage intelligence report exported to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting coverage report: {e}")


# Convenience functions for coverage analysis
def analyze_directory_coverage(directory: str, coverage_data: Dict[str, float] = None) -> CoverageIntelligenceReport:
    """Quick coverage analysis of a directory."""
    intelligence = CoverageIntelligence(directory)
    return intelligence.analyze_coverage_intelligence(coverage_data)


def find_critical_coverage_gaps(directory: str, max_gaps: int = 10) -> List[CoverageGap]:
    """Find critical coverage gaps in a directory."""
    intelligence = CoverageIntelligence(directory)
    intelligence.analyze_coverage_intelligence()
    return intelligence.get_high_priority_gaps(max_gaps)


def assess_module_risk(module_path: str) -> RiskLevel:
    """Assess risk level of a specific module."""
    intelligence = CoverageIntelligence(str(Path(module_path).parent))
    report = intelligence.analyze_coverage_intelligence()
    
    module_coverage = report.modules.get(module_path)
    return module_coverage.overall_risk if module_coverage else RiskLevel.MINIMAL