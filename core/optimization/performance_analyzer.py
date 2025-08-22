"""
Performance Analyzer - Advanced performance optimization and bottleneck detection

This module provides comprehensive performance analysis capabilities for identifying
bottlenecks, memory inefficiencies, and optimization opportunities in Python code.
Uses advanced static analysis, pattern recognition, and performance profiling.

Key Capabilities:
- Algorithmic complexity analysis with Big O notation detection
- Memory usage pattern analysis and optimization suggestions
- I/O operation optimization and async/await recommendations
- Loop optimization and vectorization opportunities
- Database query optimization and N+1 problem detection
- Caching strategy recommendations and implementation guidance
"""

import ast
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path

from .optimization_models import (
    OptimizationType, OptimizationPriority, OptimizationStrategy,
    OptimizationRecommendation, PerformanceMetrics, RiskAssessment,
    create_optimization_recommendation, create_performance_metrics,
    create_risk_assessment
)

logger = logging.getLogger(__name__)


@dataclass
class PerformancePattern:
    """Represents a detected performance pattern or anti-pattern"""
    pattern_type: str
    severity: str
    confidence: float
    description: str
    location: Tuple[int, int]
    code_snippet: str
    optimization_suggestion: str
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    complexity_impact: str = ""
    memory_impact: str = ""


@dataclass
class AlgorithmicComplexity:
    """Represents algorithmic complexity analysis for code segments"""
    time_complexity: str = "O(1)"
    space_complexity: str = "O(1)"
    worst_case: str = ""
    best_case: str = ""
    average_case: str = ""
    complexity_factors: List[str] = field(default_factory=list)
    optimization_potential: float = 0.0


class PerformanceAnalyzer:
    """
    Advanced performance analysis engine for code optimization
    
    Provides comprehensive performance analysis through static code analysis,
    pattern detection, and algorithmic complexity evaluation.
    """
    
    def __init__(self):
        """Initialize performance analyzer with pattern databases"""
        self.performance_patterns = self._load_performance_patterns()
        self.complexity_cache = {}
        self.optimization_history = []
        
        # Analysis configuration
        self.config = {
            'max_loop_depth': 3,
            'complexity_threshold': 'O(n^2)',
            'memory_threshold_mb': 100,
            'io_operation_threshold': 10,
            'cache_hit_ratio_threshold': 0.8,
            'database_query_threshold': 5
        }
        
        logger.info("Performance Analyzer initialized")
    
    async def analyze_performance(self, code: str, tree: ast.AST, file_path: str = "") -> List[OptimizationRecommendation]:
        """
        Comprehensive performance analysis of code
        
        Args:
            code: Source code as string
            tree: AST representation of the code
            file_path: Path to the file being analyzed
            
        Returns:
            List of performance optimization recommendations
        """
        recommendations = []
        
        try:
            # Multi-layer performance analysis
            loop_analysis = await self._analyze_loops(code, tree)
            memory_analysis = await self._analyze_memory_usage(code, tree)
            io_analysis = await self._analyze_io_operations(code, tree)
            algorithm_analysis = await self._analyze_algorithms(code, tree)
            database_analysis = await self._analyze_database_operations(code, tree)
            async_analysis = await self._analyze_async_opportunities(code, tree)
            
            # Convert analysis results to recommendations
            recommendations.extend(self._create_loop_recommendations(loop_analysis, file_path))
            recommendations.extend(self._create_memory_recommendations(memory_analysis, file_path))
            recommendations.extend(self._create_io_recommendations(io_analysis, file_path))
            recommendations.extend(self._create_algorithm_recommendations(algorithm_analysis, file_path))
            recommendations.extend(self._create_database_recommendations(database_analysis, file_path))
            recommendations.extend(self._create_async_recommendations(async_analysis, file_path))
            
            # Apply performance-specific filtering and prioritization
            recommendations = self._prioritize_performance_recommendations(recommendations)
            
            logger.info(f"Generated {len(recommendations)} performance recommendations for {file_path}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return []
    
    async def _analyze_loops(self, code: str, tree: ast.AST) -> List[PerformancePattern]:
        """Analyze loops for performance issues"""
        patterns = []
        
        class LoopAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.loop_depth = 0
                self.nested_loops = []
            
            def visit_For(self, node):
                self.loop_depth += 1
                
                # Detect inefficient range(len()) pattern
                if isinstance(node.iter, ast.Call) and hasattr(node.iter.func, 'id'):
                    if node.iter.func.id == 'range' and len(node.iter.args) > 0:
                        if isinstance(node.iter.args[0], ast.Call) and hasattr(node.iter.args[0].func, 'id'):
                            if node.iter.args[0].func.id == 'len':
                                patterns.append(PerformancePattern(
                                    pattern_type="inefficient_range_len",
                                    severity="medium",
                                    confidence=0.9,
                                    description="Use enumerate() instead of range(len()) for better performance and readability",
                                    location=(node.lineno, node.end_lineno or node.lineno),
                                    code_snippet=f"range(len({ast.unparse(node.iter.args[0].args[0]) if node.iter.args[0].args else 'iterable'}))",
                                    optimization_suggestion="for index, item in enumerate(iterable):",
                                    expected_improvement={"execution_time": 15.0, "readability": 25.0},
                                    complexity_impact="No change in time complexity",
                                    memory_impact="Slightly reduced memory usage"
                                ))
                
                # Detect nested loops with high complexity
                if self.loop_depth > 2:
                    patterns.append(PerformancePattern(
                        pattern_type="deeply_nested_loops",
                        severity="high",
                        confidence=0.8,
                        description=f"Deeply nested loops (depth: {self.loop_depth}) may cause performance issues",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        code_snippet="# Nested loop structure",
                        optimization_suggestion="Consider flattening loops, using vectorized operations, or algorithmic improvements",
                        expected_improvement={"execution_time": 40.0, "scalability": 60.0},
                        complexity_impact="Potential exponential time complexity",
                        memory_impact="May cause stack overflow with deep recursion"
                    ))
                
                # Check for list comprehensions that could be generators
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.ListComp):
                        patterns.append(PerformancePattern(
                            pattern_type="list_comprehension_in_loop",
                            severity="medium",
                            confidence=0.7,
                            description="List comprehension in loop could be memory intensive",
                            location=(node.lineno, node.end_lineno or node.lineno),
                            code_snippet="[...] inside loop",
                            optimization_suggestion="Consider using generator expressions (...) for memory efficiency",
                            expected_improvement={"memory_usage": 30.0, "execution_time": 10.0},
                            complexity_impact="Same time complexity, reduced space complexity",
                            memory_impact="Significantly reduced memory usage"
                        ))
                
                self.generic_visit(node)
                self.loop_depth -= 1
            
            def visit_While(self, node):
                # Similar analysis for while loops
                self.loop_depth += 1
                
                if self.loop_depth > 2:
                    patterns.append(PerformancePattern(
                        pattern_type="deeply_nested_while_loops",
                        severity="high",
                        confidence=0.8,
                        description=f"Deeply nested while loops may cause performance issues",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        code_snippet="# Nested while loop structure",
                        optimization_suggestion="Consider restructuring logic or using break conditions",
                        expected_improvement={"execution_time": 35.0, "maintainability": 40.0}
                    ))
                
                self.generic_visit(node)
                self.loop_depth -= 1
        
        analyzer = LoopAnalyzer()
        analyzer.visit(tree)
        
        return patterns
    
    async def _analyze_memory_usage(self, code: str, tree: ast.AST) -> List[PerformancePattern]:
        """Analyze memory usage patterns"""
        patterns = []
        
        # String concatenation detection
        string_concat_pattern = r'\w+\s*\+=\s*["\'].*["\']'
        matches = list(re.finditer(string_concat_pattern, code))
        
        for match in matches:
            line_num = code[:match.start()].count('\n') + 1
            patterns.append(PerformancePattern(
                pattern_type="string_concatenation",
                severity="medium",
                confidence=0.8,
                description="String concatenation with += creates new objects each time",
                location=(line_num, line_num),
                code_snippet=match.group(),
                optimization_suggestion="Use join() method or f-strings for better performance",
                expected_improvement={"execution_time": 25.0, "memory_usage": 20.0},
                complexity_impact="Reduces from O(n²) to O(n) for multiple concatenations",
                memory_impact="Eliminates intermediate string objects"
            ))
        
        # Large data structure creation
        class MemoryAnalyzer(ast.NodeVisitor):
            def visit_List(self, node):
                if len(node.elts) > 1000:  # Large list literal
                    patterns.append(PerformancePattern(
                        pattern_type="large_list_literal",
                        severity="medium",
                        confidence=0.7,
                        description=f"Large list literal with {len(node.elts)} elements",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        code_snippet=f"[...] # {len(node.elts)} elements",
                        optimization_suggestion="Consider lazy loading, generators, or chunked processing",
                        expected_improvement={"memory_usage": 40.0, "initialization_time": 30.0}
                    ))
                self.generic_visit(node)
            
            def visit_Dict(self, node):
                if len(node.keys) > 1000:  # Large dict literal
                    patterns.append(PerformancePattern(
                        pattern_type="large_dict_literal",
                        severity="medium",
                        confidence=0.7,
                        description=f"Large dictionary literal with {len(node.keys)} keys",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        code_snippet=f"{{...}} # {len(node.keys)} keys",
                        optimization_suggestion="Consider using defaultdict or lazy initialization",
                        expected_improvement={"memory_usage": 35.0, "initialization_time": 25.0}
                    ))
                self.generic_visit(node)
        
        analyzer = MemoryAnalyzer()
        analyzer.visit(tree)
        
        return patterns
    
    async def _analyze_io_operations(self, code: str, tree: ast.AST) -> List[PerformancePattern]:
        """Analyze I/O operations for optimization opportunities"""
        patterns = []
        
        # File I/O analysis
        if 'open(' in code and 'with' not in code:
            patterns.append(PerformancePattern(
                pattern_type="unsafe_file_io",
                severity="high",
                confidence=0.9,
                description="File operations without context manager may cause resource leaks",
                location=(0, 0),  # Would need more sophisticated parsing for exact location
                code_snippet="open(...)",
                optimization_suggestion="Use 'with open(...)' context manager for automatic resource cleanup",
                expected_improvement={"resource_usage": 50.0, "reliability": 60.0}
            ))
        
        # Network I/O patterns
        network_patterns = ['requests.get', 'urllib.request', 'http.client']
        for pattern in network_patterns:
            if pattern in code and 'async' not in code:
                patterns.append(PerformancePattern(
                    pattern_type="synchronous_network_io",
                    severity="medium",
                    confidence=0.7,
                    description="Synchronous network I/O may block application",
                    location=(0, 0),
                    code_snippet=pattern,
                    optimization_suggestion="Consider using async/await with aiohttp for better concurrency",
                    expected_improvement={"concurrency": 80.0, "responsiveness": 70.0}
                ))
        
        return patterns
    
    async def _analyze_algorithms(self, code: str, tree: ast.AST) -> List[PerformancePattern]:
        """Analyze algorithmic efficiency"""
        patterns = []
        
        class AlgorithmAnalyzer(ast.NodeVisitor):
            def visit_For(self, node):
                # Detect potential O(n²) patterns
                inner_loops = 0
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        inner_loops += 1
                
                if inner_loops > 0:
                    complexity = f"O(n^{inner_loops + 1})"
                    if inner_loops >= 2:  # O(n³) or higher
                        patterns.append(PerformancePattern(
                            pattern_type="high_algorithmic_complexity",
                            severity="high",
                            confidence=0.8,
                            description=f"Algorithm has {complexity} time complexity",
                            location=(node.lineno, node.end_lineno or node.lineno),
                            code_snippet="# Nested loop structure",
                            optimization_suggestion="Consider using hash maps, sets, or more efficient algorithms",
                            expected_improvement={"execution_time": 60.0, "scalability": 80.0},
                            complexity_impact=f"Reduce from {complexity} to O(n) or O(n log n)"
                        ))
                
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Detect linear search in loops
                if hasattr(node.func, 'id') and node.func.id == 'index':
                    patterns.append(PerformancePattern(
                        pattern_type="linear_search_in_loop",
                        severity="medium",
                        confidence=0.7,
                        description="Using index() method in loops causes O(n²) behavior",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        code_snippet="list.index(...)",
                        optimization_suggestion="Use dictionary or set for O(1) lookups",
                        expected_improvement={"execution_time": 50.0, "scalability": 70.0}
                    ))
                
                # Detect repeated sorting
                if hasattr(node.func, 'id') and node.func.id == 'sorted':
                    patterns.append(PerformancePattern(
                        pattern_type="repeated_sorting",
                        severity="medium",
                        confidence=0.6,
                        description="Repeated sorting operations can be expensive",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        code_snippet="sorted(...)",
                        optimization_suggestion="Sort once and maintain order, or use heap for partial sorting",
                        expected_improvement={"execution_time": 40.0, "cpu_usage": 30.0}
                    ))
                
                self.generic_visit(node)
        
        analyzer = AlgorithmAnalyzer()
        analyzer.visit(tree)
        
        return patterns
    
    async def _analyze_database_operations(self, code: str, tree: ast.AST) -> List[PerformancePattern]:
        """Analyze database operations for N+1 problems and optimization opportunities"""
        patterns = []
        
        # Detect potential N+1 query problems
        db_patterns = ['cursor.execute', '.query(', '.filter(', '.get(']
        
        for pattern in db_patterns:
            if pattern in code:
                # Simple heuristic: if db operations are in loops
                if re.search(r'for\s+\w+\s+in.*?' + re.escape(pattern), code, re.DOTALL):
                    patterns.append(PerformancePattern(
                        pattern_type="n_plus_one_query",
                        severity="high",
                        confidence=0.8,
                        description="Potential N+1 query problem detected",
                        location=(0, 0),
                        code_snippet=f"for ... {pattern}",
                        optimization_suggestion="Use bulk operations, joins, or eager loading to reduce database queries",
                        expected_improvement={"database_queries": 80.0, "execution_time": 70.0},
                        complexity_impact="Reduces from O(n) queries to O(1) query"
                    ))
        
        # Detect missing query optimization
        if 'SELECT *' in code:
            patterns.append(PerformancePattern(
                pattern_type="select_star_query",
                severity="medium",
                confidence=0.7,
                description="SELECT * queries fetch unnecessary data",
                location=(0, 0),
                code_snippet="SELECT *",
                optimization_suggestion="Specify only required columns in SELECT statements",
                expected_improvement={"network_usage": 40.0, "memory_usage": 30.0}
            ))
        
        return patterns
    
    async def _analyze_async_opportunities(self, code: str, tree: ast.AST) -> List[PerformancePattern]:
        """Analyze opportunities for async/await optimization"""
        patterns = []
        
        # Detect synchronous I/O operations that could benefit from async
        sync_io_patterns = [
            ('time.sleep', 'Use asyncio.sleep() for non-blocking delays'),
            ('requests.get', 'Use aiohttp for async HTTP requests'),
            ('open(', 'Consider aiofiles for async file I/O'),
            ('sqlite3.connect', 'Use aiosqlite for async database operations')
        ]
        
        for pattern, suggestion in sync_io_patterns:
            if pattern in code and 'async def' not in code:
                patterns.append(PerformancePattern(
                    pattern_type="async_opportunity",
                    severity="medium",
                    confidence=0.6,
                    description=f"Synchronous {pattern} could benefit from async implementation",
                    location=(0, 0),
                    code_snippet=pattern,
                    optimization_suggestion=suggestion,
                    expected_improvement={"concurrency": 60.0, "responsiveness": 50.0}
                ))
        
        return patterns
    
    def _create_loop_recommendations(self, patterns: List[PerformancePattern], file_path: str) -> List[OptimizationRecommendation]:
        """Create optimization recommendations from loop analysis"""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type in ["inefficient_range_len", "deeply_nested_loops", "list_comprehension_in_loop"]:
                priority = OptimizationPriority.HIGH if pattern.severity == "high" else OptimizationPriority.MEDIUM
                
                rec = create_optimization_recommendation(
                    OptimizationType.PERFORMANCE,
                    f"Loop Optimization: {pattern.pattern_type.replace('_', ' ').title()}",
                    pattern.description,
                    file_path,
                    priority,
                    OptimizationStrategy.GRADUAL
                )
                
                rec.target_lines = pattern.location
                rec.original_code = pattern.code_snippet
                rec.optimized_code = pattern.optimization_suggestion
                rec.expected_improvement = pattern.expected_improvement
                rec.confidence_score = pattern.confidence
                rec.reasoning = f"{pattern.complexity_impact}. {pattern.memory_impact}"
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_memory_recommendations(self, patterns: List[PerformancePattern], file_path: str) -> List[OptimizationRecommendation]:
        """Create optimization recommendations from memory analysis"""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type in ["string_concatenation", "large_list_literal", "large_dict_literal"]:
                rec = create_optimization_recommendation(
                    OptimizationType.MEMORY,
                    f"Memory Optimization: {pattern.pattern_type.replace('_', ' ').title()}",
                    pattern.description,
                    file_path,
                    OptimizationPriority.MEDIUM,
                    OptimizationStrategy.REFACTOR
                )
                
                rec.target_lines = pattern.location
                rec.original_code = pattern.code_snippet
                rec.optimized_code = pattern.optimization_suggestion
                rec.expected_improvement = pattern.expected_improvement
                rec.confidence_score = pattern.confidence
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_io_recommendations(self, patterns: List[PerformancePattern], file_path: str) -> List[OptimizationRecommendation]:
        """Create optimization recommendations from I/O analysis"""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type in ["unsafe_file_io", "synchronous_network_io"]:
                priority = OptimizationPriority.HIGH if pattern.severity == "high" else OptimizationPriority.MEDIUM
                
                rec = create_optimization_recommendation(
                    OptimizationType.PERFORMANCE,
                    f"I/O Optimization: {pattern.pattern_type.replace('_', ' ').title()}",
                    pattern.description,
                    file_path,
                    priority,
                    OptimizationStrategy.REFACTOR
                )
                
                rec.optimized_code = pattern.optimization_suggestion
                rec.expected_improvement = pattern.expected_improvement
                rec.confidence_score = pattern.confidence
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_algorithm_recommendations(self, patterns: List[PerformancePattern], file_path: str) -> List[OptimizationRecommendation]:
        """Create optimization recommendations from algorithm analysis"""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type in ["high_algorithmic_complexity", "linear_search_in_loop", "repeated_sorting"]:
                priority = OptimizationPriority.HIGH if pattern.severity == "high" else OptimizationPriority.MEDIUM
                
                rec = create_optimization_recommendation(
                    OptimizationType.ALGORITHM,
                    f"Algorithm Optimization: {pattern.pattern_type.replace('_', ' ').title()}",
                    pattern.description,
                    file_path,
                    priority,
                    OptimizationStrategy.ALGORITHMIC_CHANGE
                )
                
                rec.target_lines = pattern.location
                rec.original_code = pattern.code_snippet
                rec.optimized_code = pattern.optimization_suggestion
                rec.expected_improvement = pattern.expected_improvement
                rec.confidence_score = pattern.confidence
                rec.reasoning = pattern.complexity_impact
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_database_recommendations(self, patterns: List[PerformancePattern], file_path: str) -> List[OptimizationRecommendation]:
        """Create optimization recommendations from database analysis"""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type in ["n_plus_one_query", "select_star_query"]:
                priority = OptimizationPriority.HIGH if pattern.pattern_type == "n_plus_one_query" else OptimizationPriority.MEDIUM
                
                rec = create_optimization_recommendation(
                    OptimizationType.PERFORMANCE,
                    f"Database Optimization: {pattern.pattern_type.replace('_', ' ').title()}",
                    pattern.description,
                    file_path,
                    priority,
                    OptimizationStrategy.REFACTOR
                )
                
                rec.optimized_code = pattern.optimization_suggestion
                rec.expected_improvement = pattern.expected_improvement
                rec.confidence_score = pattern.confidence
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_async_recommendations(self, patterns: List[PerformancePattern], file_path: str) -> List[OptimizationRecommendation]:
        """Create optimization recommendations from async analysis"""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type == "async_opportunity":
                rec = create_optimization_recommendation(
                    OptimizationType.PERFORMANCE,
                    f"Async Optimization: {pattern.code_snippet}",
                    pattern.description,
                    file_path,
                    OptimizationPriority.MEDIUM,
                    OptimizationStrategy.REDESIGN
                )
                
                rec.optimized_code = pattern.optimization_suggestion
                rec.expected_improvement = pattern.expected_improvement
                rec.confidence_score = pattern.confidence
                
                recommendations.append(rec)
        
        return recommendations
    
    def _prioritize_performance_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Apply performance-specific prioritization logic"""
        def performance_score(rec: OptimizationRecommendation) -> float:
            base_score = rec.confidence_score
            
            # Boost algorithmic improvements
            if rec.optimization_type == OptimizationType.ALGORITHM:
                base_score *= 1.3
            
            # Boost high-impact improvements
            execution_improvement = rec.expected_improvement.get('execution_time', 0)
            if execution_improvement > 50:
                base_score *= 1.2
            
            # Boost scalability improvements
            scalability_improvement = rec.expected_improvement.get('scalability', 0)
            if scalability_improvement > 70:
                base_score *= 1.1
            
            return base_score
        
        return sorted(recommendations, key=performance_score, reverse=True)
    
    def _load_performance_patterns(self) -> Dict[str, Any]:
        """Load performance pattern database"""
        return {
            'common_bottlenecks': [
                'nested_loops', 'string_concatenation', 'linear_search',
                'repeated_operations', 'memory_leaks', 'blocking_io'
            ],
            'optimization_techniques': [
                'caching', 'vectorization', 'async_operations',
                'algorithm_improvement', 'data_structure_optimization'
            ]
        }
    
    def calculate_complexity(self, node: ast.AST) -> AlgorithmicComplexity:
        """Calculate algorithmic complexity for AST node"""
        # Simplified complexity analysis
        complexity = AlgorithmicComplexity()
        
        loops = len([n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While))])
        if loops == 0:
            complexity.time_complexity = "O(1)"
        elif loops == 1:
            complexity.time_complexity = "O(n)"
        elif loops == 2:
            complexity.time_complexity = "O(n²)"
        else:
            complexity.time_complexity = f"O(n^{loops})"
        
        # Cache result
        node_hash = hash(ast.dump(node))
        self.complexity_cache[node_hash] = complexity
        
        return complexity


# Factory function
def create_performance_analyzer() -> PerformanceAnalyzer:
    """Create and configure performance analyzer"""
    return PerformanceAnalyzer()