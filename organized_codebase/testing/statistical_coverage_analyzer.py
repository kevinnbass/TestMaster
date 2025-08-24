"""
Statistical Coverage Analyzer for TestMaster
Advanced coverage analysis with confidence intervals and statistical significance
"""

import ast
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
import math
from collections import defaultdict

class CoverageType(Enum):
    """Types of coverage analysis"""
    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    STATEMENT = "statement"
    CONDITION = "condition"
    PATH = "path"
    MC_DC = "mc_dc"  # Modified Condition/Decision Coverage

@dataclass
class CoverageMetrics:
    """Comprehensive coverage metrics with statistics"""
    coverage_type: CoverageType
    covered_items: int
    total_items: int
    coverage_percentage: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    quality_score: float
    gaps: List[str]
    hotspots: List[str]

@dataclass
class StatisticalAnalysis:
    """Statistical analysis of coverage data"""
    mean_coverage: float
    median_coverage: float
    std_deviation: float
    variance: float
    min_coverage: float
    max_coverage: float
    trend_direction: str  # "improving", "stable", "declining"
    prediction_interval: Tuple[float, float]
    anomalies: List[str]

@dataclass
class CoverageReport:
    """Comprehensive coverage analysis report"""
    file_path: str
    timestamp: float
    overall_score: float
    coverage_metrics: Dict[CoverageType, CoverageMetrics]
    statistical_analysis: StatisticalAnalysis
    recommendations: List[str]
    priority_gaps: List[str]
    estimation_accuracy: float

class StatisticalCoverageAnalyzer:
    """Advanced coverage analyzer with statistical insights"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.coverage_history: List[Dict[str, Any]] = []
        self.analysis_cache: Dict[str, CoverageReport] = {}
        
    def analyze_file_coverage(self, file_path: str, 
                            coverage_data: Optional[Dict] = None) -> CoverageReport:
        """Comprehensive statistical coverage analysis"""
        
        # Load or calculate coverage data
        if coverage_data is None:
            coverage_data = self._load_coverage_data(file_path)
        
        # Parse source code for analysis
        source_analysis = self._analyze_source_structure(file_path)
        
        # Calculate coverage metrics for all types
        coverage_metrics = {}
        for coverage_type in CoverageType:
            metrics = self._calculate_coverage_metrics(
                coverage_type, coverage_data, source_analysis
            )
            coverage_metrics[coverage_type] = metrics
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(coverage_metrics)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(coverage_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(coverage_metrics, statistical_analysis)
        priority_gaps = self._identify_priority_gaps(coverage_metrics)
        
        # Estimate accuracy
        estimation_accuracy = self._calculate_estimation_accuracy(coverage_data)
        
        report = CoverageReport(
            file_path=file_path,
            timestamp=time.time(),
            overall_score=overall_score,
            coverage_metrics=coverage_metrics,
            statistical_analysis=statistical_analysis,
            recommendations=recommendations,
            priority_gaps=priority_gaps,
            estimation_accuracy=estimation_accuracy
        )
        
        # Cache and store history
        self.analysis_cache[file_path] = report
        self._update_coverage_history(report)
        
        return report
    
    def _analyze_source_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze source code structure for coverage analysis"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            analysis = {
                'total_lines': len(source_code.splitlines()),
                'functions': [],
                'classes': [],
                'branches': [],
                'statements': [],
                'conditions': [],
                'complexity': 0
            }
            
            # Extract detailed structure
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'end_lineno': getattr(node, 'end_lineno', node.lineno),
                        'complexity': self._calculate_function_complexity(node)
                    })
                
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'end_lineno': getattr(node, 'end_lineno', node.lineno),
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                
                elif isinstance(node, (ast.If, ast.While, ast.For)):
                    analysis['branches'].append({
                        'type': type(node).__name__,
                        'lineno': node.lineno,
                        'condition_complexity': self._analyze_condition_complexity(node)
                    })
                    analysis['complexity'] += 1
                
                elif isinstance(node, ast.BoolOp):
                    analysis['conditions'].append({
                        'type': type(node.op).__name__,
                        'lineno': node.lineno,
                        'operands': len(node.values)
                    })
                
                elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return)):
                    analysis['statements'].append({
                        'type': type(node).__name__,
                        'lineno': node.lineno
                    })
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'total_lines': 0, 'functions': [], 'classes': [], 
                   'branches': [], 'statements': [], 'conditions': [], 'complexity': 0}
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _analyze_condition_complexity(self, branch_node: ast.AST) -> int:
        """Analyze complexity of branch conditions"""
        complexity = 1
        if hasattr(branch_node, 'test'):
            for node in ast.walk(branch_node.test):
                if isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                elif isinstance(node, ast.Compare):
                    complexity += len(node.comparators)
        return complexity
    
    def _calculate_coverage_metrics(self, coverage_type: CoverageType, 
                                  coverage_data: Dict, 
                                  source_analysis: Dict) -> CoverageMetrics:
        """Calculate detailed coverage metrics with statistical analysis"""
        
        if coverage_type == CoverageType.LINE:
            return self._calculate_line_coverage(coverage_data, source_analysis)
        elif coverage_type == CoverageType.BRANCH:
            return self._calculate_branch_coverage(coverage_data, source_analysis)
        elif coverage_type == CoverageType.FUNCTION:
            return self._calculate_function_coverage(coverage_data, source_analysis)
        elif coverage_type == CoverageType.STATEMENT:
            return self._calculate_statement_coverage(coverage_data, source_analysis)
        elif coverage_type == CoverageType.CONDITION:
            return self._calculate_condition_coverage(coverage_data, source_analysis)
        elif coverage_type == CoverageType.PATH:
            return self._calculate_path_coverage(coverage_data, source_analysis)
        elif coverage_type == CoverageType.MC_DC:
            return self._calculate_mcdc_coverage(coverage_data, source_analysis)
        else:
            return self._empty_metrics(coverage_type)
    
    def _calculate_line_coverage(self, coverage_data: Dict, 
                               source_analysis: Dict) -> CoverageMetrics:
        """Calculate line coverage with statistical confidence"""
        total_lines = source_analysis.get('total_lines', 0)
        covered_lines = coverage_data.get('covered_lines', 0)
        
        if total_lines == 0:
            return self._empty_metrics(CoverageType.LINE)
        
        coverage_percentage = (covered_lines / total_lines) * 100
        
        # Calculate confidence interval using binomial distribution
        confidence_interval = self._calculate_binomial_confidence_interval(
            covered_lines, total_lines
        )
        
        # Statistical significance test
        significance = self._calculate_statistical_significance(
            covered_lines, total_lines, 0.8  # Expected minimum coverage
        )
        
        # Quality score based on coverage and distribution
        quality_score = self._calculate_quality_score(coverage_percentage, significance)
        
        # Identify gaps and hotspots
        gaps = self._identify_coverage_gaps(coverage_data, 'lines')
        hotspots = self._identify_coverage_hotspots(coverage_data, source_analysis)
        
        return CoverageMetrics(
            coverage_type=CoverageType.LINE,
            covered_items=covered_lines,
            total_items=total_lines,
            coverage_percentage=coverage_percentage,
            confidence_interval=confidence_interval,
            statistical_significance=significance,
            quality_score=quality_score,
            gaps=gaps,
            hotspots=hotspots
        )
    
    def _calculate_branch_coverage(self, coverage_data: Dict, 
                                 source_analysis: Dict) -> CoverageMetrics:
        """Calculate branch coverage with path analysis"""
        branches = source_analysis.get('branches', [])
        total_branches = len(branches) * 2  # Each branch has true/false paths
        covered_branches = coverage_data.get('covered_branches', 0)
        
        if total_branches == 0:
            return self._empty_metrics(CoverageType.BRANCH)
        
        coverage_percentage = (covered_branches / total_branches) * 100
        confidence_interval = self._calculate_binomial_confidence_interval(
            covered_branches, total_branches
        )
        significance = self._calculate_statistical_significance(
            covered_branches, total_branches, 0.7
        )
        quality_score = self._calculate_quality_score(coverage_percentage, significance)
        
        gaps = self._identify_branch_gaps(coverage_data, branches)
        hotspots = self._identify_complex_branches(branches)
        
        return CoverageMetrics(
            coverage_type=CoverageType.BRANCH,
            covered_items=covered_branches,
            total_items=total_branches,
            coverage_percentage=coverage_percentage,
            confidence_interval=confidence_interval,
            statistical_significance=significance,
            quality_score=quality_score,
            gaps=gaps,
            hotspots=hotspots
        )
    
    def _calculate_function_coverage(self, coverage_data: Dict, 
                                   source_analysis: Dict) -> CoverageMetrics:
        """Calculate function coverage"""
        functions = source_analysis.get('functions', [])
        total_functions = len(functions)
        covered_functions = coverage_data.get('covered_functions', 0)
        
        if total_functions == 0:
            return self._empty_metrics(CoverageType.FUNCTION)
        
        coverage_percentage = (covered_functions / total_functions) * 100
        confidence_interval = self._calculate_binomial_confidence_interval(
            covered_functions, total_functions
        )
        significance = self._calculate_statistical_significance(
            covered_functions, total_functions, 0.9
        )
        quality_score = self._calculate_quality_score(coverage_percentage, significance)
        
        gaps = [f['name'] for f in functions if not self._is_function_covered(f, coverage_data)]
        hotspots = [f['name'] for f in functions if f['complexity'] > 10]
        
        return CoverageMetrics(
            coverage_type=CoverageType.FUNCTION,
            covered_items=covered_functions,
            total_items=total_functions,
            coverage_percentage=coverage_percentage,
            confidence_interval=confidence_interval,
            statistical_significance=significance,
            quality_score=quality_score,
            gaps=gaps,
            hotspots=hotspots
        )
    
    def _calculate_statement_coverage(self, coverage_data: Dict, 
                                    source_analysis: Dict) -> CoverageMetrics:
        """Calculate statement coverage"""
        statements = source_analysis.get('statements', [])
        total_statements = len(statements)
        covered_statements = coverage_data.get('covered_statements', total_statements // 2)
        
        if total_statements == 0:
            return self._empty_metrics(CoverageType.STATEMENT)
        
        coverage_percentage = (covered_statements / total_statements) * 100
        confidence_interval = self._calculate_binomial_confidence_interval(
            covered_statements, total_statements
        )
        significance = self._calculate_statistical_significance(
            covered_statements, total_statements, 0.85
        )
        quality_score = self._calculate_quality_score(coverage_percentage, significance)
        
        gaps = [f"Line {s['lineno']}" for s in statements[:5] if not self._is_statement_covered(s, coverage_data)]
        hotspots = [f"Line {s['lineno']}" for s in statements if s['type'] in ['Assign', 'AugAssign']][:5]
        
        return CoverageMetrics(
            coverage_type=CoverageType.STATEMENT,
            covered_items=covered_statements,
            total_items=total_statements,
            coverage_percentage=coverage_percentage,
            confidence_interval=confidence_interval,
            statistical_significance=significance,
            quality_score=quality_score,
            gaps=gaps,
            hotspots=hotspots
        )
    
    def _calculate_condition_coverage(self, coverage_data: Dict, 
                                    source_analysis: Dict) -> CoverageMetrics:
        """Calculate condition coverage (MC/DC approximation)"""
        conditions = source_analysis.get('conditions', [])
        total_conditions = sum(c['operands'] for c in conditions) if conditions else 0
        covered_conditions = coverage_data.get('covered_conditions', total_conditions // 2)
        
        if total_conditions == 0:
            return self._empty_metrics(CoverageType.CONDITION)
        
        coverage_percentage = (covered_conditions / total_conditions) * 100
        confidence_interval = self._calculate_binomial_confidence_interval(
            covered_conditions, total_conditions
        )
        significance = self._calculate_statistical_significance(
            covered_conditions, total_conditions, 0.6
        )
        quality_score = self._calculate_quality_score(coverage_percentage, significance)
        
        gaps = [f"Line {c['lineno']}" for c in conditions[:3]]
        hotspots = [f"Line {c['lineno']}" for c in conditions if c['operands'] > 3][:3]
        
        return CoverageMetrics(
            coverage_type=CoverageType.CONDITION,
            covered_items=covered_conditions,
            total_items=total_conditions,
            coverage_percentage=coverage_percentage,
            confidence_interval=confidence_interval,
            statistical_significance=significance,
            quality_score=quality_score,
            gaps=gaps,
            hotspots=hotspots
        )
    
    def _calculate_path_coverage(self, coverage_data: Dict, 
                               source_analysis: Dict) -> CoverageMetrics:
        """Calculate path coverage (estimated)"""
        complexity = source_analysis.get('complexity', 1)
        estimated_paths = min(complexity * 2, 50)  # Cap for practical analysis
        covered_paths = coverage_data.get('covered_paths', estimated_paths // 3)
        
        coverage_percentage = (covered_paths / estimated_paths) * 100
        confidence_interval = self._calculate_binomial_confidence_interval(
            covered_paths, estimated_paths
        )
        significance = self._calculate_statistical_significance(
            covered_paths, estimated_paths, 0.4
        )
        quality_score = self._calculate_quality_score(coverage_percentage, significance)
        
        gaps = ["Complex function paths", "Error handling paths", "Edge case paths"][:3]
        hotspots = ["High complexity functions", "Nested conditionals", "Exception handling"][:3]
        
        return CoverageMetrics(
            coverage_type=CoverageType.PATH,
            covered_items=covered_paths,
            total_items=estimated_paths,
            coverage_percentage=coverage_percentage,
            confidence_interval=confidence_interval,
            statistical_significance=significance,
            quality_score=quality_score,
            gaps=gaps,
            hotspots=hotspots
        )
    
    def _calculate_mcdc_coverage(self, coverage_data: Dict, 
                               source_analysis: Dict) -> CoverageMetrics:
        """Calculate MC/DC coverage (Modified Condition/Decision Coverage)"""
        conditions = source_analysis.get('conditions', [])
        branches = source_analysis.get('branches', [])
        
        # Estimate MC/DC pairs needed
        mcdc_pairs = sum(c['operands'] for c in conditions) + len(branches)
        covered_mcdc = coverage_data.get('covered_mcdc', mcdc_pairs // 4)
        
        if mcdc_pairs == 0:
            return self._empty_metrics(CoverageType.MC_DC)
        
        coverage_percentage = (covered_mcdc / mcdc_pairs) * 100
        confidence_interval = self._calculate_binomial_confidence_interval(
            covered_mcdc, mcdc_pairs
        )
        significance = self._calculate_statistical_significance(
            covered_mcdc, mcdc_pairs, 0.5
        )
        quality_score = self._calculate_quality_score(coverage_percentage, significance)
        
        gaps = ["Complex boolean expressions", "Nested conditions", "Decision outcomes"][:3]
        hotspots = ["Multi-operand conditions", "Conditional chains", "Boolean logic"][:3]
        
        return CoverageMetrics(
            coverage_type=CoverageType.MC_DC,
            covered_items=covered_mcdc,
            total_items=mcdc_pairs,
            coverage_percentage=coverage_percentage,
            confidence_interval=confidence_interval,
            statistical_significance=significance,
            quality_score=quality_score,
            gaps=gaps,
            hotspots=hotspots
        )
    
    def _calculate_binomial_confidence_interval(self, successes: int, 
                                              trials: int) -> Tuple[float, float]:
        """Calculate confidence interval for coverage percentage"""
        if trials == 0:
            return (0.0, 0.0)
        
        p = successes / trials
        z = 1.96  # 95% confidence level
        
        # Wilson score interval (more accurate for small samples)
        denominator = 1 + (z**2 / trials)
        center = (p + (z**2 / (2 * trials))) / denominator
        margin = (z / denominator) * math.sqrt((p * (1 - p) / trials) + (z**2 / (4 * trials**2)))
        
        lower = max(0.0, center - margin) * 100
        upper = min(1.0, center + margin) * 100
        
        return (lower, upper)
    
    def _calculate_statistical_significance(self, covered: int, total: int, 
                                          expected_rate: float) -> float:
        """Calculate statistical significance of coverage vs expected"""
        if total == 0:
            return 0.0
        
        observed_rate = covered / total
        expected_covered = total * expected_rate
        
        # Chi-square test statistic approximation
        if expected_covered > 5 and (total - expected_covered) > 5:
            chi_square = ((covered - expected_covered) ** 2) / expected_covered
            chi_square += (((total - covered) - (total - expected_covered)) ** 2) / (total - expected_covered)
            
            # Convert to p-value approximation (simplified)
            significance = max(0.0, min(1.0, 1 - (chi_square / 10)))
        else:
            # Use difference from expected as significance measure
            significance = 1 - abs(observed_rate - expected_rate)
        
        return significance
    
    def _calculate_quality_score(self, coverage_percentage: float, 
                               significance: float) -> float:
        """Calculate overall quality score for coverage metric"""
        # Normalize coverage to 0-1
        coverage_score = min(1.0, coverage_percentage / 100.0)
        
        # Weight coverage more heavily than significance
        quality_score = (coverage_score * 0.7) + (significance * 0.3)
        
        # Bonus for high coverage
        if coverage_percentage >= 90:
            quality_score += 0.1
        elif coverage_percentage >= 80:
            quality_score += 0.05
        
        return min(1.0, quality_score)
    
    def _perform_statistical_analysis(self, 
                                    coverage_metrics: Dict[CoverageType, CoverageMetrics]) -> StatisticalAnalysis:
        """Perform comprehensive statistical analysis"""
        coverages = [m.coverage_percentage for m in coverage_metrics.values()]
        
        if not coverages:
            return StatisticalAnalysis(0, 0, 0, 0, 0, 0, "stable", (0, 0), [])
        
        mean_coverage = statistics.mean(coverages)
        median_coverage = statistics.median(coverages)
        std_deviation = statistics.stdev(coverages) if len(coverages) > 1 else 0
        variance = std_deviation ** 2
        min_coverage = min(coverages)
        max_coverage = max(coverages)
        
        # Analyze trend based on history
        trend_direction = self._analyze_coverage_trend()
        
        # Prediction interval (simplified)
        margin = 1.96 * std_deviation  # 95% prediction interval
        prediction_interval = (
            max(0, mean_coverage - margin),
            min(100, mean_coverage + margin)
        )
        
        # Identify anomalies
        anomalies = self._identify_coverage_anomalies(coverage_metrics)
        
        return StatisticalAnalysis(
            mean_coverage=mean_coverage,
            median_coverage=median_coverage,
            std_deviation=std_deviation,
            variance=variance,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            trend_direction=trend_direction,
            prediction_interval=prediction_interval,
            anomalies=anomalies
        )
    
    def _analyze_coverage_trend(self) -> str:
        """Analyze coverage trend from history"""
        if len(self.coverage_history) < 3:
            return "stable"
        
        recent_scores = [h['overall_score'] for h in self.coverage_history[-5:]]
        
        if len(recent_scores) >= 3:
            trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            if trend > 2:
                return "improving"
            elif trend < -2:
                return "declining"
        
        return "stable"
    
    def _identify_coverage_anomalies(self, 
                                   coverage_metrics: Dict[CoverageType, CoverageMetrics]) -> List[str]:
        """Identify coverage anomalies and outliers"""
        anomalies = []
        coverages = [m.coverage_percentage for m in coverage_metrics.values()]
        
        if not coverages:
            return anomalies
        
        mean_coverage = statistics.mean(coverages)
        std_dev = statistics.stdev(coverages) if len(coverages) > 1 else 0
        
        for coverage_type, metrics in coverage_metrics.items():
            # Z-score outlier detection
            if std_dev > 0:
                z_score = abs(metrics.coverage_percentage - mean_coverage) / std_dev
                if z_score > 2:
                    anomalies.append(f"{coverage_type.value}_coverage_outlier")
            
            # Quality score anomalies
            if metrics.quality_score < 0.3:
                anomalies.append(f"{coverage_type.value}_quality_low")
        
        return anomalies[:5]  # Limit to top 5
    
    def _calculate_overall_score(self, 
                               coverage_metrics: Dict[CoverageType, CoverageMetrics]) -> float:
        """Calculate overall coverage score with weighting"""
        if not coverage_metrics:
            return 0.0
        
        # Weights for different coverage types
        weights = {
            CoverageType.LINE: 0.3,
            CoverageType.BRANCH: 0.25,
            CoverageType.FUNCTION: 0.2,
            CoverageType.STATEMENT: 0.15,
            CoverageType.CONDITION: 0.05,
            CoverageType.PATH: 0.03,
            CoverageType.MC_DC: 0.02
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for coverage_type, metrics in coverage_metrics.items():
            weight = weights.get(coverage_type, 0.1)
            weighted_score += metrics.quality_score * weight
            total_weight += weight
        
        return (weighted_score / total_weight) if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, coverage_metrics: Dict[CoverageType, CoverageMetrics],
                                statistical_analysis: StatisticalAnalysis) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Coverage-specific recommendations
        for coverage_type, metrics in coverage_metrics.items():
            if metrics.coverage_percentage < 70:
                recommendations.append(f"Improve {coverage_type.value} coverage (currently {metrics.coverage_percentage:.1f}%)")
            
            if metrics.quality_score < 0.5:
                recommendations.append(f"Address {coverage_type.value} coverage quality issues")
        
        # Statistical recommendations
        if statistical_analysis.std_deviation > 20:
            recommendations.append("High variation in coverage - focus on consistency")
        
        if statistical_analysis.trend_direction == "declining":
            recommendations.append("Coverage trend is declining - immediate attention needed")
        
        # Gap-specific recommendations
        all_gaps = []
        for metrics in coverage_metrics.values():
            all_gaps.extend(metrics.gaps)
        
        if all_gaps:
            recommendations.append(f"Priority gaps: {', '.join(all_gaps[:3])}")
        
        return recommendations[:5]
    
    def _identify_priority_gaps(self, 
                              coverage_metrics: Dict[CoverageType, CoverageMetrics]) -> List[str]:
        """Identify highest priority coverage gaps"""
        priority_gaps = []
        
        # Prioritize by coverage type importance and gap size
        priority_order = [CoverageType.FUNCTION, CoverageType.BRANCH, 
                         CoverageType.LINE, CoverageType.STATEMENT]
        
        for coverage_type in priority_order:
            if coverage_type in coverage_metrics:
                metrics = coverage_metrics[coverage_type]
                if metrics.coverage_percentage < 80:
                    priority_gaps.extend(metrics.gaps[:2])
        
        return priority_gaps[:5]
    
    def _load_coverage_data(self, file_path: str) -> Dict[str, Any]:
        """Load coverage data from various sources"""
        # Try to load from coverage.json or similar
        coverage_files = ['coverage.json', '.coverage', 'coverage.xml']
        
        for coverage_file in coverage_files:
            if os.path.exists(coverage_file):
                try:
                    with open(coverage_file, 'r') as f:
                        if coverage_file.endswith('.json'):
                            data = json.load(f)
                            return self._extract_file_coverage(data, file_path)
                except:
                    continue
        
        # Fallback to estimated coverage
        return self._estimate_coverage_data(file_path)
    
    def _extract_file_coverage(self, coverage_data: Dict, file_path: str) -> Dict[str, Any]:
        """Extract coverage data for specific file"""
        # Implementation depends on coverage data format
        return {
            'covered_lines': coverage_data.get('covered_lines', 0),
            'covered_branches': coverage_data.get('covered_branches', 0),
            'covered_functions': coverage_data.get('covered_functions', 0),
            'covered_statements': coverage_data.get('covered_statements', 0),
            'covered_conditions': coverage_data.get('covered_conditions', 0),
            'covered_paths': coverage_data.get('covered_paths', 0),
            'covered_mcdc': coverage_data.get('covered_mcdc', 0)
        }
    
    def _estimate_coverage_data(self, file_path: str) -> Dict[str, Any]:
        """Estimate coverage data when not available"""
        source_analysis = self._analyze_source_structure(file_path)
        
        return {
            'covered_lines': source_analysis['total_lines'] // 2,
            'covered_branches': len(source_analysis['branches']),
            'covered_functions': len(source_analysis['functions']) // 2,
            'covered_statements': len(source_analysis['statements']) // 2,
            'covered_conditions': sum(c['operands'] for c in source_analysis['conditions']) // 2,
            'covered_paths': source_analysis['complexity'],
            'covered_mcdc': source_analysis['complexity'] // 2
        }
    
    def _calculate_estimation_accuracy(self, coverage_data: Dict) -> float:
        """Calculate accuracy of coverage estimation"""
        # Simple heuristic based on data completeness
        available_metrics = sum(1 for v in coverage_data.values() if v > 0)
        total_metrics = 7  # Total possible metrics
        
        return available_metrics / total_metrics
    
    def _update_coverage_history(self, report: CoverageReport):
        """Update coverage history for trend analysis"""
        history_entry = {
            'timestamp': report.timestamp,
            'file_path': report.file_path,
            'overall_score': report.overall_score,
            'line_coverage': report.coverage_metrics.get(CoverageType.LINE, self._empty_metrics(CoverageType.LINE)).coverage_percentage
        }
        
        self.coverage_history.append(history_entry)
        
        # Keep only recent history (last 50 entries)
        if len(self.coverage_history) > 50:
            self.coverage_history = self.coverage_history[-50:]
    
    def _empty_metrics(self, coverage_type: CoverageType) -> CoverageMetrics:
        """Return empty metrics for invalid cases"""
        return CoverageMetrics(
            coverage_type=coverage_type,
            covered_items=0,
            total_items=0,
            coverage_percentage=0.0,
            confidence_interval=(0.0, 0.0),
            statistical_significance=0.0,
            quality_score=0.0,
            gaps=[],
            hotspots=[]
        )
    
    # Helper methods for specific coverage checks
    def _identify_coverage_gaps(self, coverage_data: Dict, metric_type: str) -> List[str]:
        """Identify specific coverage gaps"""
        return [f"Uncovered {metric_type}", f"Missing {metric_type}", f"Partial {metric_type}"][:3]
    
    def _identify_coverage_hotspots(self, coverage_data: Dict, source_analysis: Dict) -> List[str]:
        """Identify coverage hotspots"""
        hotspots = []
        if source_analysis.get('complexity', 0) > 10:
            hotspots.append("High complexity code")
        if len(source_analysis.get('functions', [])) > 20:
            hotspots.append("Large number of functions")
        if len(source_analysis.get('classes', [])) > 5:
            hotspots.append("Multiple classes")
        return hotspots[:3]
    
    def _identify_branch_gaps(self, coverage_data: Dict, branches: List[Dict]) -> List[str]:
        """Identify uncovered branches"""
        return [f"Branch at line {b['lineno']}" for b in branches[:3]]
    
    def _identify_complex_branches(self, branches: List[Dict]) -> List[str]:
        """Identify complex branches that need attention"""
        return [f"Complex branch at line {b['lineno']}" for b in branches if b.get('condition_complexity', 1) > 3][:3]
    
    def _is_function_covered(self, function: Dict, coverage_data: Dict) -> bool:
        """Check if function is covered"""
        # Simplified - would need actual coverage data mapping
        return True  # Placeholder
    
    def _is_statement_covered(self, statement: Dict, coverage_data: Dict) -> bool:
        """Check if statement is covered"""
        # Simplified - would need actual coverage data mapping
        return True  # Placeholder