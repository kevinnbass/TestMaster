from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
#!/usr/bin/env python3
"""
Enhanced Security Intelligence Agent
===================================
Advanced security intelligence agent that integrates classical analysis,
real-time monitoring, quality metrics, and performance data for comprehensive
security-aware test generation and vulnerability assessment.
"""

import ast
import os
import re
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from enum import Enum
from pathlib import Path
from datetime import datetime
import logging

# Import existing components
from testmaster.intelligence.security.security_intelligence_agent import (
    SecurityIntelligenceAgent, SecurityTestStrategy, SecurityTestPlan, SecurityFinding,
    SecurityPlanGenerator
)

# Import enhanced systems we built
from enhanced_realtime_security_monitor import (
    EnhancedRealtimeSecurityMonitor, SecurityEvent, RiskProfile, SecurityTrend
)
from live_code_quality_monitor import LiveCodeQualityMonitor, QualitySnapshot, QualityMetric
from enhanced_incremental_ast_engine import EnhancedIncrementalASTEngine, IncrementalResult
from realtime_metrics_collector import RealtimeMetricsCollector
from performance_profiler import PerformanceProfiler, ProfileData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClassicalAnalysisData:
    """Classical analysis data for security intelligence."""
    file_path: str
    complexity_metrics: Dict[str, float]
    dependency_graph: Dict[str, List[str]]
    code_patterns: List[Dict[str, Any]]
    architectural_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]
    performance_characteristics: Dict[str, float]
    change_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EnhancedSecurityFinding:
    """Enhanced security finding with classical analysis context."""
    finding_id: str
    vulnerability_type: str
    severity: str
    location: str
    description: str
    impact_assessment: Dict[str, Any]
    remediation_plan: Dict[str, Any]
    
    # Classical analysis integration
    complexity_correlation: Optional[float] = None
    quality_correlation: Optional[float] = None
    performance_impact: Optional[float] = None
    change_risk_factor: Optional[float] = None
    dependency_risk: Optional[float] = None
    
    # Context data
    classical_context: Optional[ClassicalAnalysisData] = None
    quality_context: Optional[QualitySnapshot] = None
    performance_context: Optional[ProfileData] = None
    
    # Risk scoring
    integrated_risk_score: float = 0.0
    confidence: float = 1.0
    test_priority: float = 0.5

@dataclass
class IntelligentTestSuite:
    """Intelligent security test suite with prioritization."""
    suite_id: str
    target_file: str
    test_cases: List[Dict[str, Any]]
    execution_priority: float
    expected_coverage: float
    risk_based_ordering: List[str]  # Test case IDs in priority order
    classical_analysis_insights: Dict[str, Any]
    estimated_execution_time: float
    confidence_score: float

class ClassicalAnalysisIntegrator:
    """Integrates classical analysis data with security intelligence."""
    
    def __init__(self):
        self.ast_engine = EnhancedIncrementalASTEngine()
        self.quality_monitor = LiveCodeQualityMonitor()
        self.performance_profiler = PerformanceProfiler()
        self.analysis_cache = {}
        self.dependency_analyzer = DependencyAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        
    def analyze_file_classical(self, file_path: str) -> ClassicalAnalysisData:
        """Perform comprehensive classical analysis of a file."""
        try:
            # Check cache first
            cache_key = self._get_cache_key(file_path)
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if time.time() - cached_result['timestamp'] < 300:  # 5 minutes
                    return cached_result['data']
            
            # AST analysis
            ast_result = self.ast_engine.analyze(file_path)
            
            # Quality analysis
            quality_snapshot = self.quality_monitor.analyze_file_quality(file_path)
            
            # Performance profiling
            session_id = self.performance_profiler.start_profiling(f"classical_{int(time.time())}")
            # Simulate analysis work
            time.sleep(0.1)
            profile_data = self.performance_profiler.stop_profiling(session_id)
            
            # Dependency analysis
            dependency_graph = self.dependency_analyzer.analyze_dependencies(file_path)
            
            # Complexity analysis
            complexity_metrics = self.complexity_analyzer.analyze_complexity(file_path)
            
            # Architectural analysis
            architectural_analysis = self._analyze_architecture(file_path, ast_result)
            
            # Code pattern analysis
            code_patterns = self._analyze_code_patterns(file_path)
            
            classical_data = ClassicalAnalysisData(
                file_path=file_path,
                complexity_metrics=complexity_metrics,
                dependency_graph=dependency_graph,
                code_patterns=code_patterns,
                architectural_analysis=architectural_analysis,
                quality_metrics=self._extract_quality_metrics(quality_snapshot),
                performance_characteristics=self._extract_performance_metrics(profile_data),
                change_history=self._get_change_history(file_path)
            )
            
            # Cache result
            self.analysis_cache[cache_key] = {
                'data': classical_data,
                'timestamp': time.time()
            }
            
            return classical_data
            
        except Exception as e:
            logger.error(f"Error in classical analysis of {file_path}: {e}")
            return self._create_minimal_analysis(file_path)
    
    def correlate_with_security(self, classical_data: ClassicalAnalysisData, 
                              security_findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Correlate classical analysis with security findings."""
        correlations = {
            'complexity_security_correlation': self._correlate_complexity_security(
                classical_data.complexity_metrics, security_findings
            ),
            'quality_security_correlation': self._correlate_quality_security(
                classical_data.quality_metrics, security_findings
            ),
            'dependency_security_correlation': self._correlate_dependency_security(
                classical_data.dependency_graph, security_findings
            ),
            'performance_security_correlation': self._correlate_performance_security(
                classical_data.performance_characteristics, security_findings
            ),
            'architectural_security_correlation': self._correlate_architectural_security(
                classical_data.architectural_analysis, security_findings
            )
        }
        
        return correlations
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for file analysis."""
        try:
            stat = os.stat(file_path)
            content_hash = hashlib.md5(f"{file_path}_{stat.st_mtime}".encode()).hexdigest()
            return content_hash
        except:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def _analyze_architecture(self, file_path: str, ast_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze architectural patterns in the code."""
        return {
            'design_patterns': self._detect_design_patterns(ast_result),
            'architectural_style': self._detect_architectural_style(ast_result),
            'coupling_metrics': self._calculate_coupling_metrics(ast_result),
            'cohesion_metrics': self._calculate_cohesion_metrics(ast_result)
        }
    
    def _analyze_code_patterns(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze code patterns that may affect security."""
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Security-relevant patterns
            security_patterns = [
                {'pattern': r'eval\s*\(', 'type': 'dangerous_function', 'risk': 'high'},
                {'pattern': r'exec\s*\(', 'type': 'dangerous_function', 'risk': 'high'},
                {'pattern': r'subprocess\..*shell\s*=\s*True', 'type': 'shell_injection', 'risk': 'high'},
                {'pattern': r'pickle\.loads?\s*\(', 'type': 'unsafe_deserialization', 'risk': 'medium'},
                {'pattern': r'random\.(random|choice)', 'type': 'weak_randomness', 'risk': 'medium'},
                {'pattern': r'password\s*=\s*["\'][^"\']+["\']', 'type': 'hardcoded_credential', 'risk': 'high'}
            ]
            
            for pattern_info in security_patterns:
                matches = re.finditer(pattern_info['pattern'], content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'pattern_type': pattern_info['type'],
                        'risk_level': pattern_info['risk'],
                        'location': f"line {line_num}",
                        'matched_text': match.group(),
                        'context': self._get_context_around_match(content, match)
                    })
        
        except Exception as e:
            logger.error(f"Error analyzing code patterns in {file_path}: {e}")
        
        return patterns
    
    def _extract_quality_metrics(self, quality_snapshot: Optional[QualitySnapshot]) -> Dict[str, float]:
        """Extract quality metrics for correlation analysis."""
        if not quality_snapshot:
            return {}
        
        return {
            'overall_score': quality_snapshot.overall_score,
            'complexity_score': sum(m.value for m in quality_snapshot.metrics if 'complexity' in m.name),
            'maintainability_score': sum(m.value for m in quality_snapshot.metrics if m.category == 'maintainability'),
            'security_score': sum(m.value for m in quality_snapshot.metrics if m.category == 'security'),
            'reliability_score': sum(m.value for m in quality_snapshot.metrics if m.category == 'reliability')
        }
    
    def _extract_performance_metrics(self, profile_data: Optional[ProfileData]) -> Dict[str, float]:
        """Extract performance metrics for correlation analysis."""
        if not profile_data:
            return {}
        
        return {
            'execution_time': profile_data.duration,
            'memory_usage': profile_data.memory_usage.get('delta_mb', 0),
            'function_calls': profile_data.function_calls,
            'hotspot_count': len(profile_data.hotspots),
            'cpu_usage': profile_data.cpu_usage
        }
    
    def _get_change_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Get change history for the file (simplified implementation)."""
        # In a real implementation, this would integrate with version control
        return [
            {
                'timestamp': time.time() - 86400,  # 1 day ago
                'change_type': 'modification',
                'lines_changed': 10,
                'risk_score': 0.3
            }
        ]
    
    def _create_minimal_analysis(self, file_path: str) -> ClassicalAnalysisData:
        """Create minimal analysis data when full analysis fails."""
        return ClassicalAnalysisData(
            file_path=file_path,
            complexity_metrics={'cyclomatic_complexity': 5.0},
            dependency_graph={},
            code_patterns=[],
            architectural_analysis={},
            quality_metrics={'overall_score': 50.0},
            performance_characteristics={'execution_time': 1.0}
        )
    
    def _correlate_complexity_security(self, complexity_metrics: Dict[str, float], 
                                     security_findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Correlate complexity metrics with security findings."""
        complexity_score = complexity_metrics.get('cyclomatic_complexity', 0)
        finding_count = len(security_findings)
        
        # High complexity often correlates with more security issues
        correlation_strength = min(complexity_score / 20 * finding_count / 5, 1.0)
        
        return {
            'correlation_strength': correlation_strength,
            'complexity_score': complexity_score,
            'finding_count': finding_count,
            'insight': 'High complexity correlates with increased security risk' if correlation_strength > 0.6 else 'Complexity-security correlation is moderate'
        }
    
    def _correlate_quality_security(self, quality_metrics: Dict[str, float], 
                                  security_findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Correlate quality metrics with security findings."""
        quality_score = quality_metrics.get('overall_score', 50.0)
        security_score = quality_metrics.get('security_score', 0)
        finding_count = len(security_findings)
        
        # Lower quality often correlates with more security issues
        correlation_strength = max(0, (100 - quality_score) / 100 * finding_count / 5)
        
        return {
            'correlation_strength': correlation_strength,
            'quality_score': quality_score,
            'security_score': security_score,
            'finding_count': finding_count,
            'insight': 'Poor code quality correlates with security vulnerabilities' if correlation_strength > 0.6 else 'Quality-security correlation is moderate'
        }
    
    def _correlate_dependency_security(self, dependency_graph: Dict[str, List[str]], 
                                     security_findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Correlate dependency structure with security findings."""
        dependency_count = sum(len(deps) for deps in dependency_graph.values())
        finding_count = len(security_findings)
        
        # High dependency coupling can increase security risk
        correlation_strength = min(dependency_count / 20 * finding_count / 5, 1.0)
        
        return {
            'correlation_strength': correlation_strength,
            'dependency_count': dependency_count,
            'finding_count': finding_count,
            'insight': 'High dependency coupling increases security attack surface' if correlation_strength > 0.5 else 'Dependency-security correlation is low'
        }
    
    def _correlate_performance_security(self, performance_metrics: Dict[str, float], 
                                      security_findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Correlate performance characteristics with security findings."""
        execution_time = performance_metrics.get('execution_time', 0)
        memory_usage = performance_metrics.get('memory_usage', 0)
        finding_count = len(security_findings)
        
        # Performance issues might indicate security problems
        performance_risk = (execution_time > 2.0) or (memory_usage > 100)
        correlation_strength = 0.7 if performance_risk and finding_count > 0 else 0.3
        
        return {
            'correlation_strength': correlation_strength,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'finding_count': finding_count,
            'insight': 'Performance issues may indicate security vulnerabilities' if performance_risk else 'Performance-security correlation is low'
        }
    
    def _correlate_architectural_security(self, architectural_analysis: Dict[str, Any], 
                                        security_findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Correlate architectural patterns with security findings."""
        coupling_metrics = architectural_analysis.get('coupling_metrics', {})
        cohesion_metrics = architectural_analysis.get('cohesion_metrics', {})
        finding_count = len(security_findings)
        
        # Poor architecture often correlates with security issues
        coupling_score = coupling_metrics.get('total_coupling', 0)
        cohesion_score = cohesion_metrics.get('average_cohesion', 1.0)
        
        architectural_risk = (coupling_score > 10) or (cohesion_score < 0.5)
        correlation_strength = 0.6 if architectural_risk and finding_count > 0 else 0.2
        
        return {
            'correlation_strength': correlation_strength,
            'coupling_score': coupling_score,
            'cohesion_score': cohesion_score,
            'finding_count': finding_count,
            'insight': 'Poor architectural design correlates with security vulnerabilities' if architectural_risk else 'Architecture-security correlation is low'
        }

class DependencyAnalyzer:
    """Analyzes code dependencies for security impact."""
    
    def analyze_dependencies(self, file_path: str) -> Dict[str, List[str]]:
        """Analyze file dependencies."""
        dependencies = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in dependencies:
                            dependencies[module_name] = []
                        dependencies[module_name].append('import')
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in dependencies:
                            dependencies[module_name] = []
                        for alias in node.names:
                            dependencies[module_name].append(alias.name)
        
        except Exception as e:
            logger.error(f"Error analyzing dependencies in {file_path}: {e}")
        
        return dependencies

class ComplexityAnalyzer:
    """Analyzes code complexity for security correlation."""
    
    def analyze_complexity(self, file_path: str) -> Dict[str, float]:
        """Analyze complexity metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            return {
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
                'cognitive_complexity': self._calculate_cognitive_complexity(tree),
                'nesting_depth': self._calculate_nesting_depth(tree),
                'function_count': len([n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                'class_count': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            }
        
        except Exception as e:
            logger.error(f"Error analyzing complexity in {file_path}: {e}")
            return {'cyclomatic_complexity': 1.0}
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Break, ast.Continue)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return float(complexity)
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> float:
        """Calculate cognitive complexity (simplified)."""
        # Simplified cognitive complexity calculation
        cognitive_score = 0
        nesting_level = 0
        
        def calculate_cognitive(node, nesting=0):
            nonlocal cognitive_score
            
            if isinstance(node, (ast.If, ast.While, ast.For)):
                cognitive_score += 1 + nesting
                nesting += 1
            elif isinstance(node, ast.BoolOp):
                cognitive_score += len(node.values) - 1
            
            for child in ast.iter_child_nodes(node):
                calculate_cognitive(child, nesting)
        
        calculate_cognitive(tree)
        return float(cognitive_score)
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> float:
        """Calculate maximum nesting depth."""
        max_depth = 0
        
        def calculate_depth(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    calculate_depth(child, depth + 1)
                else:
                    calculate_depth(child, depth)
        
        calculate_depth(tree)
        return float(max_depth)

class EnhancedSecurityIntelligenceAgent:
    """Enhanced security intelligence agent with classical analysis integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.base_agent = SecurityIntelligenceAgent(config)
        self.classical_integrator = ClassicalAnalysisIntegrator()
        self.security_monitor = EnhancedRealtimeSecurityMonitor(config)
        
        # Enhanced analysis state
        self.enhanced_findings = {}
        self.intelligent_test_suites = {}
        self.correlation_cache = {}
        
        # Configuration
        self.enable_classical_integration = self.config.get('enable_classical_integration', True)
        self.enable_correlation_analysis = self.config.get('enable_correlation_analysis', True)
        self.risk_threshold = self.config.get('risk_threshold', 70.0)
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_security_with_classical_context(self, file_path: str, 
                                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform security analysis with classical analysis context."""
        try:
            # Perform classical analysis
            classical_data = None
            if self.enable_classical_integration:
                classical_data = self.classical_integrator.analyze_file_classical(file_path)
            
            # Perform base security analysis
            base_context = context or {}
            if classical_data:
                # Enrich context with classical analysis
                base_context.update({
                    'complexity_metrics': classical_data.complexity_metrics,
                    'quality_metrics': classical_data.quality_metrics,
                    'architectural_patterns': classical_data.architectural_analysis,
                    'code_patterns': classical_data.code_patterns
                })
            
            base_analysis = self.base_agent.analyze_security(file_path, base_context)
            
            # Enhance findings with classical context
            enhanced_findings = []
            if classical_data and self.enable_correlation_analysis:
                enhanced_findings = self._enhance_findings_with_classical_context(
                    base_analysis.get('findings', []), classical_data
                )
            
            # Generate intelligent test suite
            intelligent_suite = self._generate_intelligent_test_suite(
                file_path, enhanced_findings, classical_data
            )
            
            # Calculate integrated risk assessment
            risk_assessment = self._calculate_integrated_risk_assessment(
                enhanced_findings, classical_data
            )
            
            # Generate correlation insights
            correlation_insights = {}
            if classical_data and enhanced_findings:
                correlation_insights = self.classical_integrator.correlate_with_security(
                    classical_data, base_analysis.get('findings', [])
                )
            
            return {
                **base_analysis,
                'enhanced_analysis': {
                    'classical_data': classical_data.__dict__ if classical_data else None,
                    'enhanced_findings': [f.__dict__ for f in enhanced_findings],
                    'intelligent_test_suite': intelligent_suite.__dict__ if intelligent_suite else None,
                    'integrated_risk_assessment': risk_assessment,
                    'correlation_insights': correlation_insights,
                    'analysis_timestamp': time.time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced security analysis of {file_path}: {e}")
            return {'error': str(e)}
    
    def _enhance_findings_with_classical_context(self, base_findings: List[SecurityFinding],
                                                classical_data: ClassicalAnalysisData) -> List[EnhancedSecurityFinding]:
        """Enhance security findings with classical analysis context."""
        enhanced_findings = []
        
        for i, finding in enumerate(base_findings):
            # Calculate correlations
            complexity_correlation = self._calculate_complexity_correlation(finding, classical_data)
            quality_correlation = self._calculate_quality_correlation(finding, classical_data)
            performance_impact = self._calculate_performance_impact(finding, classical_data)
            change_risk_factor = self._calculate_change_risk_factor(finding, classical_data)
            dependency_risk = self._calculate_dependency_risk(finding, classical_data)
            
            # Calculate integrated risk score
            integrated_risk = self._calculate_integrated_risk_score(
                finding, complexity_correlation, quality_correlation, 
                performance_impact, change_risk_factor, dependency_risk
            )
            
            # Create enhanced finding
            enhanced_finding = EnhancedSecurityFinding(
                finding_id=f"enhanced_{i}_{int(time.time())}",
                vulnerability_type=finding.vulnerability_type.value if hasattr(finding.vulnerability_type, 'value') else str(finding.vulnerability_type),
                severity=finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity),
                location=finding.location,
                description=finding.description,
                impact_assessment=self._create_impact_assessment(finding, classical_data),
                remediation_plan=self._create_remediation_plan(finding, classical_data),
                complexity_correlation=complexity_correlation,
                quality_correlation=quality_correlation,
                performance_impact=performance_impact,
                change_risk_factor=change_risk_factor,
                dependency_risk=dependency_risk,
                classical_context=classical_data,
                integrated_risk_score=integrated_risk,
                confidence=finding.confidence,
                test_priority=self._calculate_test_priority(integrated_risk, finding.confidence)
            )
            
            enhanced_findings.append(enhanced_finding)
        
        return enhanced_findings
    
    def _generate_intelligent_test_suite(self, file_path: str, 
                                       enhanced_findings: List[EnhancedSecurityFinding],
                                       classical_data: Optional[ClassicalAnalysisData]) -> Optional[IntelligentTestSuite]:
        """Generate intelligent test suite with risk-based prioritization."""
        if not enhanced_findings:
            return None
        
        # Generate test cases based on findings
        test_cases = []
        for finding in enhanced_findings:
            test_cases.extend(self._generate_test_cases_for_finding(finding, classical_data))
        
        # Risk-based ordering
        risk_based_ordering = sorted(
            [tc['id'] for tc in test_cases],
            key=lambda tc_id: next(tc['priority'] for tc in test_cases if tc['id'] == tc_id),
            reverse=True
        )
        
        # Calculate execution priority
        execution_priority = sum(f.integrated_risk_score for f in enhanced_findings) / len(enhanced_findings)
        
        # Calculate expected coverage
        expected_coverage = min(0.95, 0.6 + (execution_priority / 100) * 0.35)
        
        # Estimate execution time
        estimated_time = len(test_cases) * 5.0  # 5 minutes per test case average
        
        # Calculate confidence score
        confidence_score = sum(f.confidence for f in enhanced_findings) / len(enhanced_findings)
        
        # Generate classical analysis insights
        classical_insights = {}
        if classical_data:
            classical_insights = {
                'complexity_insights': self._generate_complexity_insights(classical_data),
                'quality_insights': self._generate_quality_insights(classical_data),
                'architectural_insights': self._generate_architectural_insights(classical_data)
            }
        
        suite = IntelligentTestSuite(
            suite_id=f"intelligent_suite_{int(time.time())}",
            target_file=file_path,
            test_cases=test_cases,
            execution_priority=execution_priority,
            expected_coverage=expected_coverage,
            risk_based_ordering=risk_based_ordering,
            classical_analysis_insights=classical_insights,
            estimated_execution_time=estimated_time,
            confidence_score=confidence_score
        )
        
        self.intelligent_test_suites[file_path] = suite
        return suite
    
    def _calculate_complexity_correlation(self, finding: SecurityFinding, 
                                        classical_data: ClassicalAnalysisData) -> float:
        """Calculate correlation between finding and complexity metrics."""
        complexity_score = classical_data.complexity_metrics.get('cyclomatic_complexity', 1.0)
        
        # Higher complexity often correlates with more security issues
        if complexity_score > 15:
            return 0.8
        elif complexity_score > 10:
            return 0.6
        elif complexity_score > 5:
            return 0.4
        else:
            return 0.2
    
    def _calculate_quality_correlation(self, finding: SecurityFinding, 
                                     classical_data: ClassicalAnalysisData) -> float:
        """Calculate correlation between finding and quality metrics."""
        quality_score = classical_data.quality_metrics.get('overall_score', 50.0)
        
        # Lower quality often correlates with more security issues
        return max(0.0, (100 - quality_score) / 100)
    
    def _calculate_performance_impact(self, finding: SecurityFinding, 
                                    classical_data: ClassicalAnalysisData) -> float:
        """Calculate performance impact of the security finding."""
        execution_time = classical_data.performance_characteristics.get('execution_time', 1.0)
        memory_usage = classical_data.performance_characteristics.get('memory_usage', 0)
        
        # Performance issues might indicate or exacerbate security problems
        performance_factor = min(execution_time / 5.0, 1.0) + min(memory_usage / 100.0, 1.0)
        return min(performance_factor, 1.0)
    
    def _calculate_change_risk_factor(self, finding: SecurityFinding, 
                                    classical_data: ClassicalAnalysisData) -> float:
        """Calculate change-based risk factor."""
        change_history = classical_data.change_history
        
        if not change_history:
            return 0.3  # Default low risk
        
        # Recent changes increase risk
        recent_changes = [ch for ch in change_history if time.time() - ch['timestamp'] < 86400 * 7]  # 7 days
        
        if recent_changes:
            avg_risk = sum(ch.get('risk_score', 0.5) for ch in recent_changes) / len(recent_changes)
            return min(avg_risk * 1.5, 1.0)  # Amplify recent change risk
        
        return 0.3
    
    def _calculate_dependency_risk(self, finding: SecurityFinding, 
                                 classical_data: ClassicalAnalysisData) -> float:
        """Calculate dependency-based risk factor."""
        dependency_count = sum(len(deps) for deps in classical_data.dependency_graph.values())
        
        # More dependencies can increase attack surface
        if dependency_count > 20:
            return 0.8
        elif dependency_count > 10:
            return 0.6
        elif dependency_count > 5:
            return 0.4
        else:
            return 0.2
    
    def _calculate_integrated_risk_score(self, finding: SecurityFinding, 
                                       complexity_corr: float, quality_corr: float,
                                       performance_impact: float, change_risk: float,
                                       dependency_risk: float) -> float:
        """Calculate integrated risk score."""
        # Base severity score
        severity_map = {'CRITICAL': 90, 'HIGH': 70, 'MEDIUM': 50, 'LOW': 30}
        base_score = severity_map.get(str(finding.severity).upper(), 50)
        
        # Apply correlation factors
        correlation_factor = (
            complexity_corr * 0.25 +
            quality_corr * 0.25 +
            performance_impact * 0.2 +
            change_risk * 0.15 +
            dependency_risk * 0.15
        )
        
        # Calculate final score
        integrated_score = base_score + (correlation_factor * 30)  # Up to 30 points from correlations
        
        return min(integrated_score, 100.0)
    
    def _calculate_test_priority(self, risk_score: float, confidence: float) -> float:
        """Calculate test execution priority."""
        return (risk_score * 0.7 + confidence * 100 * 0.3) / 100
    
    def _create_impact_assessment(self, finding: SecurityFinding, 
                                classical_data: ClassicalAnalysisData) -> Dict[str, Any]:
        """Create comprehensive impact assessment."""
        return {
            'security_impact': finding.impact,
            'complexity_amplification': classical_data.complexity_metrics.get('cyclomatic_complexity', 1) > 10,
            'quality_degradation_risk': classical_data.quality_metrics.get('overall_score', 50) < 60,
            'performance_implications': classical_data.performance_characteristics.get('execution_time', 1) > 2,
            'architectural_concerns': len(classical_data.architectural_analysis) > 0,
            'dependency_exposure': sum(len(deps) for deps in classical_data.dependency_graph.values()) > 15
        }
    
    def _create_remediation_plan(self, finding: SecurityFinding, 
                               classical_data: ClassicalAnalysisData) -> Dict[str, Any]:
        """Create comprehensive remediation plan."""
        plan = {
            'primary_remediation': finding.remediation,
            'complexity_reduction': [],
            'quality_improvements': [],
            'performance_optimizations': [],
            'architectural_refactoring': []
        }
        
        # Add complexity-based recommendations
        if classical_data.complexity_metrics.get('cyclomatic_complexity', 1) > 15:
            plan['complexity_reduction'].append('Break down complex functions into smaller units')
            plan['complexity_reduction'].append('Reduce nested conditionals and loops')
        
        # Add quality-based recommendations
        if classical_data.quality_metrics.get('overall_score', 50) < 60:
            plan['quality_improvements'].append('Improve code documentation and comments')
            plan['quality_improvements'].append('Add input validation and error handling')
        
        # Add performance-based recommendations
        if classical_data.performance_characteristics.get('execution_time', 1) > 2:
            plan['performance_optimizations'].append('Profile and optimize slow code paths')
            plan['performance_optimizations'].append('Consider caching for repeated operations')
        
        return plan
    
    def _generate_test_cases_for_finding(self, finding: EnhancedSecurityFinding,
                                       classical_data: Optional[ClassicalAnalysisData]) -> List[Dict[str, Any]]:
        """Generate test cases for an enhanced finding."""
        base_test_cases = [
            {
                'id': f"test_{finding.finding_id}_basic",
                'type': 'basic_vulnerability_test',
                'description': f"Basic test for {finding.vulnerability_type}",
                'priority': finding.test_priority,
                'estimated_time': 5.0
            },
            {
                'id': f"test_{finding.finding_id}_edge_case",
                'type': 'edge_case_test',
                'description': f"Edge case test for {finding.vulnerability_type}",
                'priority': finding.test_priority * 0.8,
                'estimated_time': 7.0
            }
        ]
        
        # Add complexity-aware test cases
        if classical_data and classical_data.complexity_metrics.get('cyclomatic_complexity', 1) > 10:
            base_test_cases.append({
                'id': f"test_{finding.finding_id}_complexity",
                'type': 'complexity_aware_test',
                'description': f"Complexity-aware test for {finding.vulnerability_type}",
                'priority': finding.test_priority * 0.9,
                'estimated_time': 10.0
            })
        
        return base_test_cases
    
    def _calculate_integrated_risk_assessment(self, enhanced_findings: List[EnhancedSecurityFinding],
                                            classical_data: Optional[ClassicalAnalysisData]) -> Dict[str, Any]:
        """Calculate integrated risk assessment."""
        if not enhanced_findings:
            return {'overall_risk': 0.0, 'risk_level': 'LOW'}
        
        # Calculate overall risk
        overall_risk = sum(f.integrated_risk_score for f in enhanced_findings) / len(enhanced_findings)
        
        # Determine risk level
        if overall_risk >= 80:
            risk_level = 'CRITICAL'
        elif overall_risk >= 60:
            risk_level = 'HIGH'
        elif overall_risk >= 40:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Calculate risk factors
        risk_factors = {
            'complexity_risk': classical_data.complexity_metrics.get('cyclomatic_complexity', 1) > 15 if classical_data else False,
            'quality_risk': classical_data.quality_metrics.get('overall_score', 50) < 50 if classical_data else False,
            'performance_risk': classical_data.performance_characteristics.get('execution_time', 1) > 3 if classical_data else False,
            'dependency_risk': sum(len(deps) for deps in classical_data.dependency_graph.values()) > 20 if classical_data else False
        }
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'finding_count': len(enhanced_findings),
            'high_priority_findings': len([f for f in enhanced_findings if f.test_priority > 0.7]),
            'recommendation': self._get_risk_recommendation(overall_risk, risk_factors)
        }
    
    def _get_risk_recommendation(self, overall_risk: float, risk_factors: Dict[str, bool]) -> str:
        """Get risk-based recommendation."""
        if overall_risk >= 80:
            return "IMMEDIATE ACTION REQUIRED: Address critical security vulnerabilities"
        elif overall_risk >= 60:
            return "HIGH PRIORITY: Address security issues within 24-48 hours"
        elif overall_risk >= 40:
            return "MEDIUM PRIORITY: Address security issues within 1 week"
        else:
            return "LOW PRIORITY: Monitor and address during regular maintenance"
    
    def _generate_complexity_insights(self, classical_data: ClassicalAnalysisData) -> List[str]:
        """Generate insights from complexity analysis."""
        insights = []
        complexity = classical_data.complexity_metrics.get('cyclomatic_complexity', 1)
        
        if complexity > 20:
            insights.append("CRITICAL: Extremely high complexity increases security risk exponentially")
        elif complexity > 15:
            insights.append("HIGH: High complexity makes security review difficult")
        elif complexity > 10:
            insights.append("MEDIUM: Moderate complexity may hide security issues")
        else:
            insights.append("LOW: Complexity is manageable for security review")
        
        return insights
    
    def _generate_quality_insights(self, classical_data: ClassicalAnalysisData) -> List[str]:
        """Generate insights from quality analysis."""
        insights = []
        quality_score = classical_data.quality_metrics.get('overall_score', 50)
        
        if quality_score < 30:
            insights.append("CRITICAL: Poor code quality strongly correlates with security vulnerabilities")
        elif quality_score < 50:
            insights.append("HIGH: Low code quality increases security risk")
        elif quality_score < 70:
            insights.append("MEDIUM: Moderate quality issues may affect security")
        else:
            insights.append("GOOD: Code quality supports security objectives")
        
        return insights
    
    def _generate_architectural_insights(self, classical_data: ClassicalAnalysisData) -> List[str]:
        """Generate insights from architectural analysis."""
        insights = []
        arch_analysis = classical_data.architectural_analysis
        
        if arch_analysis.get('coupling_metrics', {}).get('total_coupling', 0) > 15:
            insights.append("HIGH: Tight coupling increases attack surface")
        
        if arch_analysis.get('cohesion_metrics', {}).get('average_cohesion', 1.0) < 0.5:
            insights.append("MEDIUM: Low cohesion may indicate design issues")
        
        if not insights:
            insights.append("GOOD: Architectural patterns support security objectives")
        
        return insights


def main():
    """Demo and testing of enhanced security intelligence agent."""
    # Configuration
    config = {
        'enable_classical_integration': True,
        'enable_correlation_analysis': True,
        'risk_threshold': 70.0
    }
    
    # Create enhanced agent
    agent = EnhancedSecurityIntelligenceAgent(config)
    
    # Create test file with security issues
    test_file = 'enhanced_security_test.py'
    test_content = '''
import os
import subprocess
import pickle
import random

def complex_vulnerable_function(user_input, data_source, config_path):
    """A complex function with multiple security vulnerabilities."""
    # Hardcoded credential (security issue)
    api_key = os.getenv('KEY')
    
    # High complexity nested logic
    if user_input:
        if len(user_input) > 10:
            if data_source == "database":
                if config_path:
                    # Command injection vulnerability
                    command = f"cat {config_path} | grep {user_input}"
                    result = os.system(command)
                    
                    # SQL injection risk
                    query = f"SELECT * FROM users WHERE name = '{user_input}'"
                    
                    # Insecure deserialization
                    if result == 0:
                        data = SafePickleHandler.safe_load(user_input.encode())
                        
                        # Weak randomness for security
                        token = str(random.randint(1000, 9999))
                        
                        # Code injection
                        code = f"print('{user_input}')"
                        SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval(code)
                        
                        return data, token
    
    return None, None

class VulnerableComplexClass:
    """A class with architectural and security issues."""
    
    def __init__(self):
        self.secret = os.getenv('SECRET')
        self.connections = []
        self.cache = {}
    
    def process_data(self, input_data, processing_type, validation_level):
        """Overly complex method with multiple responsibilities."""
        if processing_type == "secure":
            if validation_level > 5:
                if len(input_data) > 100:
                    # Poor error handling
                    try:
                        result = self._complex_processing(input_data)
                        if result:
                            if self._validate_result(result):
                                return self._format_output(result)
                    except:
                        pass  # Silent failure
        
        return "ERROR"
    
    def _complex_processing(self, data):
        # Simulated complex processing
        for i in range(10):
            for j in range(10):
                if i * j > 50:
                    data = data.replace(str(i), str(j))
        return data
    
    def _validate_result(self, result):
        # Weak validation
        return len(result) > 0
    
    def _format_output(self, data):
        # Memory-intensive operation
        large_list = [data * 1000 for _ in range(1000)]
        return str(large_list)
'''
    
    try:
        # Write test file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Perform enhanced security analysis
        logger.info("Performing enhanced security analysis with classical context...")
        
        analysis_result = agent.analyze_security_with_classical_context(test_file)
        
        logger.info("Enhanced Security Analysis Results:")
        logger.info(f"  Base findings: {len(analysis_result.get('findings', []))}")
        
        enhanced = analysis_result.get('enhanced_analysis', {})
        if enhanced:
            enhanced_findings = enhanced.get('enhanced_findings', [])
            logger.info(f"  Enhanced findings: {len(enhanced_findings)}")
            
            # Show some enhanced findings
            for i, finding in enumerate(enhanced_findings[:3]):
                logger.info(f"    Finding {i+1}:")
                logger.info(f"      Type: {finding['vulnerability_type']}")
                logger.info(f"      Integrated Risk: {finding['integrated_risk_score']:.1f}")
                logger.info(f"      Test Priority: {finding['test_priority']:.2f}")
                logger.info(f"      Complexity Correlation: {finding['complexity_correlation']:.2f}")
                logger.info(f"      Quality Correlation: {finding['quality_correlation']:.2f}")
            
            # Show intelligent test suite
            test_suite = enhanced.get('intelligent_test_suite')
            if test_suite:
                logger.info(f"  Intelligent Test Suite:")
                logger.info(f"    Test Cases: {len(test_suite['test_cases'])}")
                logger.info(f"    Execution Priority: {test_suite['execution_priority']:.1f}")
                logger.info(f"    Expected Coverage: {test_suite['expected_coverage']:.1%}")
                logger.info(f"    Estimated Time: {test_suite['estimated_execution_time']:.1f} minutes")
            
            # Show integrated risk assessment
            risk_assessment = enhanced.get('integrated_risk_assessment', {})
            logger.info(f"  Integrated Risk Assessment:")
            logger.info(f"    Overall Risk: {risk_assessment.get('overall_risk', 0):.1f}")
            logger.info(f"    Risk Level: {risk_assessment.get('risk_level', 'UNKNOWN')}")
            logger.info(f"    Recommendation: {risk_assessment.get('recommendation', 'N/A')}")
            
            # Show correlation insights
            correlations = enhanced.get('correlation_insights', {})
            if correlations:
                logger.info("  Correlation Insights:")
                for corr_type, corr_data in correlations.items():
                    if isinstance(corr_data, dict) and 'insight' in corr_data:
                        logger.info(f"    {corr_type}: {corr_data['insight']}")
        
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        if os.path.exists(test_file):
            os.remove(test_file)
    
    logger.info("Enhanced security intelligence agent demo completed")


if __name__ == "__main__":
    main()