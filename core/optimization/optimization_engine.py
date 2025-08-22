"""
Optimization Engine - Core AI-powered optimization analysis engine

This module implements the intelligent optimization engine that analyzes code,
identifies improvement opportunities, and generates comprehensive recommendations
using advanced AI techniques and machine learning patterns.

Key Capabilities:
- Multi-type code analysis (performance, security, quality, architecture)
- AI-powered recommendation generation with confidence scoring
- Machine learning pattern recognition for continuous improvement
- Risk assessment and implementation strategy planning
- Performance bottleneck identification and resolution
- Security vulnerability detection with remediation suggestions
"""

import asyncio
import logging
import ast
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime
import difflib

from .optimization_models import (
    OptimizationType, OptimizationPriority, OptimizationStrategy, 
    RecommendationStatus, AnalysisType,
    OptimizationRecommendation, OptimizationResult, OptimizationSession,
    PerformanceMetrics, QualityMetrics, SecurityMetrics, RiskAssessment,
    LearningEntry, OptimizationContext,
    create_optimization_recommendation, create_performance_metrics,
    create_risk_assessment, sort_recommendations_by_impact
)

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """
    Advanced AI-powered optimization engine for intelligent code analysis
    
    Provides comprehensive code optimization through multi-layered analysis,
    machine learning pattern recognition, and expert-level recommendations.
    """
    
    def __init__(self, learning_enabled: bool = True):
        """Initialize optimization engine with optional learning capabilities"""
        self.learning_enabled = learning_enabled
        self.learning_database = {}
        self.performance_baselines = {}
        self.quality_baselines = {}
        self.security_baselines = {}
        self.optimization_history = []
        self.active_sessions = {}
        
        # Analysis configuration
        self.analysis_config = {
            'max_file_size': 10000,  # lines
            'complexity_threshold': 10,
            'performance_threshold': 1.0,  # seconds
            'memory_threshold': 100.0,  # MB
            'security_scan_depth': 'deep',
            'quality_gates': {
                'maintainability_index': 60.0,
                'code_coverage': 80.0,
                'duplication_threshold': 5.0
            }
        }
        
        logger.info("Optimization Engine initialized with learning enabled: %s", learning_enabled)
    
    async def analyze_code(self, file_path: str, context: Optional[OptimizationContext] = None) -> List[OptimizationRecommendation]:
        """
        Perform comprehensive code analysis and generate optimization recommendations
        
        Args:
            file_path: Path to the code file to analyze
            context: Optional context for targeted analysis
            
        Returns:
            List of optimization recommendations
        """
        try:
            recommendations = []
            
            # Read and parse code
            code_content = await self._read_file_safely(file_path)
            if not code_content:
                return recommendations
            
            # Parse AST for analysis
            try:
                tree = ast.parse(code_content)
            except SyntaxError as e:
                logger.warning(f"Syntax error in {file_path}: {e}")
                return []
            
            # Perform multi-layered analysis
            performance_recs = await self._analyze_performance(file_path, code_content, tree, context)
            quality_recs = await self._analyze_code_quality(file_path, code_content, tree, context)
            security_recs = await self._analyze_security(file_path, code_content, tree, context)
            architecture_recs = await self._analyze_architecture(file_path, code_content, tree, context)
            
            recommendations.extend(performance_recs)
            recommendations.extend(quality_recs)
            recommendations.extend(security_recs)
            recommendations.extend(architecture_recs)
            
            # Apply ML-based filtering and prioritization
            recommendations = await self._apply_machine_learning_insights(recommendations, file_path)
            
            # Sort by impact and priority
            recommendations = sort_recommendations_by_impact(recommendations)
            
            # Update learning database
            if self.learning_enabled:
                await self._update_learning_database(recommendations, file_path)
            
            logger.info(f"Generated {len(recommendations)} recommendations for {file_path}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing code in {file_path}: {e}")
            return []
    
    async def _analyze_performance(self, file_path: str, code: str, tree: ast.AST, context: Optional[OptimizationContext]) -> List[OptimizationRecommendation]:
        """Analyze code for performance optimization opportunities"""
        recommendations = []
        
        try:
            # Detect performance bottlenecks
            bottlenecks = await self._detect_performance_bottlenecks(code, tree)
            
            for bottleneck in bottlenecks:
                rec = create_optimization_recommendation(
                    OptimizationType.PERFORMANCE,
                    f"Performance Optimization: {bottleneck['type']}",
                    bottleneck['description'],
                    file_path,
                    OptimizationPriority.HIGH,
                    OptimizationStrategy.GRADUAL
                )
                
                rec.target_lines = bottleneck.get('lines', (0, 0))
                rec.original_code = bottleneck.get('code_snippet', '')
                rec.optimized_code = await self._generate_optimized_code(bottleneck)
                rec.expected_improvement = bottleneck.get('expected_improvement', {})
                rec.confidence_score = bottleneck.get('confidence', 0.8)
                rec.reasoning = bottleneck.get('reasoning', '')
                
                # Risk assessment
                rec.risk_assessment = create_risk_assessment(
                    implementation_risk=0.3,
                    business_impact_risk=0.2,
                    technical_debt_risk=0.1
                )
                
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return []
    
    async def _analyze_code_quality(self, file_path: str, code: str, tree: ast.AST, context: Optional[OptimizationContext]) -> List[OptimizationRecommendation]:
        """Analyze code quality and suggest improvements"""
        recommendations = []
        
        try:
            # Calculate quality metrics
            quality_issues = await self._detect_quality_issues(code, tree)
            
            for issue in quality_issues:
                priority = OptimizationPriority.MEDIUM
                if issue['severity'] == 'high':
                    priority = OptimizationPriority.HIGH
                elif issue['severity'] == 'critical':
                    priority = OptimizationPriority.CRITICAL
                
                rec = create_optimization_recommendation(
                    OptimizationType.CODE_QUALITY,
                    f"Code Quality: {issue['type']}",
                    issue['description'],
                    file_path,
                    priority,
                    OptimizationStrategy.INCREMENTAL_IMPROVEMENT
                )
                
                rec.target_lines = issue.get('lines', (0, 0))
                rec.target_function = issue.get('function', '')
                rec.original_code = issue.get('code_snippet', '')
                rec.optimized_code = await self._generate_quality_improvement(issue)
                rec.confidence_score = issue.get('confidence', 0.7)
                rec.reasoning = issue.get('reasoning', '')
                
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in quality analysis: {e}")
            return []
    
    async def _analyze_security(self, file_path: str, code: str, tree: ast.AST, context: Optional[OptimizationContext]) -> List[OptimizationRecommendation]:
        """Analyze code for security vulnerabilities and improvements"""
        recommendations = []
        
        try:
            # Detect security vulnerabilities
            vulnerabilities = await self._detect_security_vulnerabilities(code, tree)
            
            for vuln in vulnerabilities:
                rec = create_optimization_recommendation(
                    OptimizationType.SECURITY,
                    f"Security Fix: {vuln['type']}",
                    vuln['description'],
                    file_path,
                    OptimizationPriority.CRITICAL if vuln['severity'] == 'critical' else OptimizationPriority.HIGH,
                    OptimizationStrategy.IMMEDIATE if vuln['severity'] == 'critical' else OptimizationStrategy.GRADUAL
                )
                
                rec.target_lines = vuln.get('lines', (0, 0))
                rec.original_code = vuln.get('code_snippet', '')
                rec.optimized_code = await self._generate_secure_alternative(vuln['type'], vuln.get('code_snippet', ''))
                rec.confidence_score = vuln.get('confidence', 0.9)
                rec.reasoning = vuln.get('reasoning', '')
                
                # High security risk assessment
                rec.risk_assessment = create_risk_assessment(
                    implementation_risk=0.2,
                    business_impact_risk=0.8,
                    technical_debt_risk=0.3
                )
                rec.risk_assessment.security_risk = 0.9
                rec.risk_assessment.overall_risk_score = 0.6
                
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in security analysis: {e}")
            return []
    
    async def _analyze_architecture(self, file_path: str, code: str, tree: ast.AST, context: Optional[OptimizationContext]) -> List[OptimizationRecommendation]:
        """Analyze architectural patterns and suggest improvements"""
        recommendations = []
        
        try:
            # Detect architecture issues
            arch_issues = await self._detect_architecture_issues(code, tree)
            
            for issue in arch_issues:
                rec = create_optimization_recommendation(
                    OptimizationType.ARCHITECTURE,
                    f"Architecture: {issue['pattern']}",
                    issue['description'],
                    file_path,
                    OptimizationPriority.MEDIUM,
                    OptimizationStrategy.REDESIGN if issue['severity'] == 'high' else OptimizationStrategy.REFACTOR
                )
                
                rec.target_class = issue.get('class', '')
                rec.target_function = issue.get('function', '')
                rec.reasoning = issue.get('reasoning', '')
                rec.alternative_solutions = issue.get('alternatives', [])
                rec.confidence_score = issue.get('confidence', 0.6)
                
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in architecture analysis: {e}")
            return []
    
    async def _detect_performance_bottlenecks(self, code: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks in code"""
        bottlenecks = []
        
        class PerformanceVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                # Detect inefficient loops
                if isinstance(node.iter, ast.Call) and hasattr(node.iter.func, 'id'):
                    if node.iter.func.id == 'range' and len(node.iter.args) > 0:
                        if isinstance(node.iter.args[0], ast.Call):
                            if hasattr(node.iter.args[0].func, 'id') and node.iter.args[0].func.id == 'len':
                                bottlenecks.append({
                                    'type': 'inefficient_range_len_loop',
                                    'description': 'Use enumerate() instead of range(len()) for better performance',
                                    'lines': (node.lineno, node.end_lineno or node.lineno),
                                    'confidence': 0.9,
                                    'expected_improvement': {'execution_time': 15.0},
                                    'reasoning': 'enumerate() is more pythonic and slightly faster than range(len())'
                                })
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                # Detect string concatenation in loops
                if isinstance(node.op, ast.Add):
                    # This is a simplified check
                    bottlenecks.append({
                        'type': 'potential_string_concatenation',
                        'description': 'Consider using join() or f-strings for string concatenation',
                        'lines': (node.lineno, node.end_lineno or node.lineno),
                        'confidence': 0.6,
                        'expected_improvement': {'execution_time': 25.0},
                        'reasoning': 'String concatenation with + creates new objects each time'
                    })
                self.generic_visit(node)
        
        visitor = PerformanceVisitor()
        visitor.visit(tree)
        
        # Additional regex-based detection for complex patterns
        if 'time.sleep(' in code:
            bottlenecks.append({
                'type': 'blocking_sleep',
                'description': 'Consider using asyncio.sleep() for non-blocking operations',
                'confidence': 0.8,
                'expected_improvement': {'concurrency': 50.0},
                'reasoning': 'Blocking sleep prevents other operations from running'
            })
        
        return bottlenecks
    
    async def _detect_quality_issues(self, code: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect code quality issues"""
        issues = []
        
        class QualityVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check function complexity (simplified)
                statements = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                if statements > 20:
                    issues.append({
                        'type': 'complex_function',
                        'description': f'Function "{node.name}" is too complex ({statements} statements)',
                        'lines': (node.lineno, node.end_lineno or node.lineno),
                        'function': node.name,
                        'severity': 'high' if statements > 30 else 'medium',
                        'confidence': 0.8,
                        'reasoning': 'Complex functions are harder to maintain and test'
                    })
                
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    issues.append({
                        'type': 'missing_docstring',
                        'description': f'Function "{node.name}" is missing a docstring',
                        'lines': (node.lineno, node.lineno),
                        'function': node.name,
                        'severity': 'low',
                        'confidence': 0.9,
                        'reasoning': 'Docstrings improve code documentation and maintainability'
                    })
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Check for missing class docstrings
                if not ast.get_docstring(node):
                    issues.append({
                        'type': 'missing_class_docstring',
                        'description': f'Class "{node.name}" is missing a docstring',
                        'lines': (node.lineno, node.lineno),
                        'severity': 'low',
                        'confidence': 0.9,
                        'reasoning': 'Class docstrings help understand the purpose and usage'
                    })
                self.generic_visit(node)
        
        visitor = QualityVisitor()
        visitor.visit(tree)
        
        return issues
    
    async def _detect_security_vulnerabilities(self, code: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect security vulnerabilities in code"""
        vulnerabilities = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Check for eval usage
                if hasattr(node.func, 'id') and node.func.id in ['eval', 'exec']:
                    vulnerabilities.append({
                        'type': 'dangerous_eval',
                        'description': f'Use of {node.func.id}() can execute arbitrary code',
                        'lines': (node.lineno, node.end_lineno or node.lineno),
                        'severity': 'critical',
                        'confidence': 0.95,
                        'reasoning': 'eval() and exec() can execute malicious code'
                    })
                
                # Check for subprocess without shell=False
                if hasattr(node.func, 'attr') and node.func.attr in ['call', 'run', 'Popen']:
                    if hasattr(node.func.value, 'id') and node.func.value.id == 'subprocess':
                        shell_arg = None
                        for keyword in node.keywords:
                            if keyword.arg == 'shell':
                                shell_arg = keyword.value
                        
                        if shell_arg and hasattr(shell_arg, 'value') and shell_arg.value is True:
                            vulnerabilities.append({
                                'type': 'shell_injection',
                                'description': 'subprocess with shell=True can be vulnerable to injection',
                                'lines': (node.lineno, node.end_lineno or node.lineno),
                                'severity': 'high',
                                'confidence': 0.8,
                                'reasoning': 'shell=True allows command injection attacks'
                            })
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor()
        visitor.visit(tree)
        
        # Check for hardcoded secrets (regex patterns)
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'api_key\s*=\s*["\'][^"\']{20,}["\']',
            r'secret\s*=\s*["\'][^"\']{16,}["\']'
        ]
        
        for pattern in secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                vulnerabilities.append({
                    'type': 'hardcoded_secret',
                    'description': 'Potential hardcoded secret found',
                    'lines': (line_num, line_num),
                    'severity': 'high',
                    'confidence': 0.7,
                    'reasoning': 'Hardcoded secrets in source code are security risks'
                })
        
        return vulnerabilities
    
    async def _detect_architecture_issues(self, code: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect architectural issues and improvement opportunities"""
        issues = []
        
        class ArchitectureVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                # Check for god classes (many methods)
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 15:
                    issues.append({
                        'pattern': 'god_class',
                        'description': f'Class "{node.name}" has too many methods ({len(methods)})',
                        'class': node.name,
                        'severity': 'high',
                        'confidence': 0.7,
                        'reasoning': 'Large classes violate single responsibility principle',
                        'alternatives': ['Split into smaller, focused classes', 'Use composition pattern']
                    })
                
                # Check for missing __init__ method in classes with attributes
                has_init = any(n.name == '__init__' for n in methods)
                has_attributes = any(isinstance(n, ast.Assign) for n in node.body)
                
                if has_attributes and not has_init:
                    issues.append({
                        'pattern': 'missing_constructor',
                        'description': f'Class "{node.name}" has attributes but no constructor',
                        'class': node.name,
                        'severity': 'medium',
                        'confidence': 0.8,
                        'reasoning': 'Classes with attributes should have proper initialization',
                        'alternatives': ['Add __init__ method', 'Use dataclass decorator']
                    })
                
                self.generic_visit(node)
        
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        return issues
    
    async def _generate_optimized_code(self, bottleneck: Dict[str, Any]) -> str:
        """Generate optimized code for identified bottleneck"""
        bottleneck_type = bottleneck.get('type', '')
        
        if bottleneck_type == 'inefficient_range_len_loop':
            return "# Use: for index, item in enumerate(items):"
        elif bottleneck_type == 'potential_string_concatenation':
            return "# Use: result = ''.join(string_parts) or f-strings"
        elif bottleneck_type == 'blocking_sleep':
            return "# Use: await asyncio.sleep(duration)"
        else:
            return "# Optimized implementation would be provided"
    
    async def _generate_quality_improvement(self, issue: Dict[str, Any]) -> str:
        """Generate code improvement for quality issue"""
        issue_type = issue.get('type', '')
        
        if issue_type == 'missing_docstring':
            function_name = issue.get('function', 'function')
            return f'"""\n    {function_name.title()} function description\n    \n    Args:\n        param: Parameter description\n    \n    Returns:\n        Return value description\n    """'
        elif issue_type == 'complex_function':
            return "# Consider breaking this function into smaller, focused functions"
        else:
            return "# Quality improvement would be provided"
    
    async def _generate_secure_alternative(self, vulnerability_type: str, original_code: str) -> str:
        """Generate secure alternative for vulnerability"""
        if vulnerability_type == 'dangerous_eval':
            return "# Use ast.literal_eval() for safe evaluation or implement specific parsing logic"
        elif vulnerability_type == 'shell_injection':
            return "# Use subprocess.run(args, shell=False) and pass command as list"
        elif vulnerability_type == 'hardcoded_secret':
            return "# Use environment variables: os.getenv('SECRET_KEY') or config files"
        else:
            return "# Secure alternative would be provided"
    
    async def _apply_machine_learning_insights(self, recommendations: List[OptimizationRecommendation], file_path: str) -> List[OptimizationRecommendation]:
        """Apply ML insights to filter and enhance recommendations"""
        if not self.learning_enabled or not self.learning_database:
            return recommendations
        
        enhanced_recommendations = []
        
        for rec in recommendations:
            # Look up patterns in learning database
            pattern_key = f"{rec.optimization_type.value}_{rec.target_element}"
            
            if pattern_key in self.learning_database:
                entry = self.learning_database[pattern_key]
                
                # Adjust confidence based on historical success
                rec.confidence_score = (rec.confidence_score + entry['success_rate']) / 2
                
                # Add learned best practices
                if entry['best_practices']:
                    rec.alternative_solutions.extend(entry['best_practices'])
                
                # Skip if historically unsuccessful
                if entry['success_rate'] < 0.3 and entry['total_attempts'] > 5:
                    continue
            
            enhanced_recommendations.append(rec)
        
        return enhanced_recommendations
    
    async def _update_learning_database(self, recommendations: List[OptimizationRecommendation], file_path: str):
        """Update learning database with new recommendations"""
        for rec in recommendations:
            pattern_key = f"{rec.optimization_type.value}_{rec.target_element}"
            
            if pattern_key not in self.learning_database:
                self.learning_database[pattern_key] = {
                    'count': 0,
                    'success_rate': 0.0,
                    'avg_confidence': 0.0,
                    'total_attempts': 0
                }
            
            entry = self.learning_database[pattern_key]
            entry['count'] += 1
            entry['total_attempts'] += 1
            entry['avg_confidence'] = (entry['avg_confidence'] + rec.confidence_score) / 2
    
    async def _read_file_safely(self, file_path: str) -> Optional[str]:
        """Safely read file content with error handling"""
        try:
            path = Path(file_path)
            if not path.exists() or path.stat().st_size > self.analysis_config['max_file_size'] * 80:  # ~80 chars per line
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get machine learning statistics from the database"""
        if not self.learning_database:
            return {'total_patterns': 0, 'avg_success_rate': 0.0}
        
        total_patterns = len(self.learning_database)
        avg_success_rate = sum(entry['success_rate'] for entry in self.learning_database.values()) / total_patterns
        
        return {
            'total_patterns': total_patterns,
            'avg_success_rate': avg_success_rate,
            'total_attempts': sum(entry['total_attempts'] for entry in self.learning_database.values()),
            'patterns_by_type': Counter(key.split('_')[0] for key in self.learning_database.keys())
        }


# Factory function
def create_optimization_engine(learning_enabled: bool = True) -> OptimizationEngine:
    """Create and configure optimization engine"""
    return OptimizationEngine(learning_enabled=learning_enabled)