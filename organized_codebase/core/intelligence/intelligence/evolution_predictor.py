"""
Code Evolution Predictor
=======================

Revolutionary code evolution prediction and growth analysis system.
Extracted from predictive_code_intelligence.py for enterprise modular architecture.

Agent D Implementation - Hour 15-16: Predictive Intelligence Modularization
"""

import ast
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.data_models import CodeEvolutionAnalysis


class CodeEvolutionPredictor:
    """
    Revolutionary Code Evolution Predictor
    
    Predicts how code will evolve over time using mathematical modeling,
    pattern recognition, and machine learning techniques.
    """
    
    def __init__(self):
        self.evolution_patterns = {
            'class_growth': {'threshold': 15, 'weight': 0.3},
            'method_addition': {'threshold': 10, 'weight': 0.4},
            'complexity_increase': {'threshold': 20, 'weight': 0.5},
            'dependency_growth': {'threshold': 5, 'weight': 0.3}
        }
        
        self.hotspot_indicators = {
            'frequent_changes': 0.4,
            'high_complexity': 0.3,
            'many_dependencies': 0.2,
            'poor_test_coverage': 0.1
        }
        
        self.logger = logging.getLogger(__name__)
    
    def predict_evolution(self, code: str, file_path: str, 
                         historical_data: Optional[Dict[str, Any]] = None) -> CodeEvolutionAnalysis:
        """Predict how code will evolve over time"""
        
        try:
            analysis = CodeEvolutionAnalysis()
            
            # Parse code
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {file_path}: {e}")
                return analysis
            
            # Analyze current state
            analysis.current_state = self._analyze_current_state(tree, code)
            
            # Identify evolution vectors
            analysis.evolution_vectors = self._identify_evolution_vectors(tree, code)
            
            # Predict growth patterns
            analysis.growth_patterns = self._predict_growth_patterns(analysis.current_state, historical_data)
            
            # Analyze complexity trends
            analysis.complexity_trends = self._analyze_complexity_trends(tree, historical_data)
            
            # Predict dependency evolution
            analysis.dependency_evolution = self._predict_dependency_evolution(tree, code)
            
            # Assess feature addition likelihood
            analysis.feature_addition_likelihood = self._assess_feature_addition_likelihood(tree, code)
            
            # Calculate refactoring pressure
            analysis.refactoring_pressure = self._calculate_refactoring_pressure(tree, code)
            
            # Project maintenance burden
            analysis.maintenance_burden_projection = self._project_maintenance_burden(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error predicting code evolution for {file_path}: {e}")
            return CodeEvolutionAnalysis()
    
    def _analyze_current_state(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Analyze current state of code"""
        
        try:
            state = {
                'class_count': 0,
                'method_count': 0,
                'function_count': 0,
                'total_lines': len(code.split('\n')),
                'complexity_metrics': {},
                'dependency_count': 0,
                'import_count': 0,
                'docstring_coverage': 0.0,
                'comment_density': 0.0
            }
            
            # Count AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    state['class_count'] += 1
                    # Count methods in class
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    state['method_count'] += len(methods)
                elif isinstance(node, ast.FunctionDef):
                    state['function_count'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    state['import_count'] += 1
            
            # Calculate complexity metrics
            state['complexity_metrics'] = self._calculate_complexity_metrics(tree)
            
            # Calculate documentation metrics
            state['docstring_coverage'] = self._calculate_docstring_coverage(tree)
            state['comment_density'] = self._calculate_comment_density(code)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error analyzing current state: {e}")
            return {}
    
    def _identify_evolution_vectors(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Identify vectors along which code might evolve"""
        
        try:
            vectors = []
            
            # Class expansion vectors
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    if method_count > 5:  # Classes with many methods likely to grow
                        vectors.append({
                            'type': 'class_expansion',
                            'target': node.name,
                            'likelihood': min(1.0, method_count / 10),
                            'direction': 'method_addition',
                            'driving_factors': ['high_method_count', 'active_development'],
                            'impact_magnitude': method_count * 0.1
                        })
            
            # Feature extension vectors based on code patterns
            if any(word in code.lower() for word in ['config', 'setting', 'option']):
                vectors.append({
                    'type': 'configuration_expansion',
                    'target': 'configuration_system',
                    'likelihood': 0.7,
                    'direction': 'parameter_addition',
                    'driving_factors': ['configuration_presence', 'extensibility_needs'],
                    'impact_magnitude': 0.5
                })
            
            # API expansion vectors
            if any(word in code.lower() for word in ['api', 'endpoint', 'route']):
                vectors.append({
                    'type': 'api_expansion',
                    'target': 'api_surface',
                    'likelihood': 0.6,
                    'direction': 'endpoint_addition',
                    'driving_factors': ['api_presence', 'feature_requests'],
                    'impact_magnitude': 0.4
                })
            
            # Testing expansion vectors
            if 'test' in code.lower() or any(word in code.lower() for word in ['assert', 'mock']):
                vectors.append({
                    'type': 'test_expansion',
                    'target': 'test_coverage',
                    'likelihood': 0.8,
                    'direction': 'test_addition',
                    'driving_factors': ['test_presence', 'quality_requirements'],
                    'impact_magnitude': 0.3
                })
            
            # Documentation expansion vectors
            docstring_coverage = self._calculate_docstring_coverage(tree)
            if docstring_coverage < 0.5:
                vectors.append({
                    'type': 'documentation_expansion',
                    'target': 'documentation_coverage',
                    'likelihood': 0.6,
                    'direction': 'documentation_addition',
                    'driving_factors': ['low_documentation', 'maintainability_needs'],
                    'impact_magnitude': 1.0 - docstring_coverage
                })
            
            return vectors
            
        except Exception as e:
            self.logger.error(f"Error identifying evolution vectors: {e}")
            return []
    
    def _predict_growth_patterns(self, current_state: Dict[str, Any], 
                               historical_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Predict growth patterns based on current state and historical data"""
        
        try:
            patterns = {}
            
            # Predict class growth
            class_count = current_state.get('class_count', 0)
            if class_count > 0:
                base_growth = min(0.5, class_count * 0.1)
                # Adjust based on historical data
                if historical_data and 'class_growth_history' in historical_data:
                    historical_growth = np.mean(historical_data['class_growth_history'])
                    patterns['class_growth_rate'] = (base_growth + historical_growth) / 2
                else:
                    patterns['class_growth_rate'] = base_growth
            
            # Predict method growth
            method_count = current_state.get('method_count', 0)
            function_count = current_state.get('function_count', 0)
            total_functions = method_count + function_count
            
            if total_functions > 0:
                base_growth = min(0.6, total_functions * 0.05)
                patterns['method_growth_rate'] = base_growth
            
            # Predict complexity growth
            complexity = current_state.get('complexity_metrics', {}).get('average_complexity', 0)
            patterns['complexity_growth_rate'] = min(0.4, complexity * 0.02)
            
            # Predict dependency growth
            import_count = current_state.get('import_count', 0)
            patterns['dependency_growth_rate'] = min(0.3, import_count * 0.1)
            
            # Predict line count growth
            total_lines = current_state.get('total_lines', 0)
            if total_lines > 0:
                # Larger files tend to grow faster initially, then slow down
                if total_lines < 200:
                    patterns['line_growth_rate'] = 0.3
                elif total_lines < 500:
                    patterns['line_growth_rate'] = 0.2
                else:
                    patterns['line_growth_rate'] = 0.1
            
            # Predict documentation growth
            doc_coverage = current_state.get('docstring_coverage', 0)
            if doc_coverage < 0.8:
                patterns['documentation_growth_rate'] = 0.5
            else:
                patterns['documentation_growth_rate'] = 0.1
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error predicting growth patterns: {e}")
            return {}
    
    def _analyze_complexity_trends(self, tree: ast.AST, 
                                 historical_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Analyze complexity trends and predict future complexity"""
        
        try:
            trends = {}
            
            # Calculate current complexity distribution
            complexities = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    complexities.append(complexity)
            
            if complexities:
                trends['average_complexity'] = np.mean(complexities)
                trends['max_complexity'] = max(complexities)
                trends['complexity_variance'] = np.var(complexities)
                trends['complexity_std'] = np.std(complexities)
                
                # Predict complexity increase tendency
                high_complexity_ratio = len([c for c in complexities if c > 10]) / len(complexities)
                trends['complexity_increase_tendency'] = min(1.0, high_complexity_ratio * 2)
                
                # Analyze complexity distribution
                simple_functions = len([c for c in complexities if c <= 5])
                moderate_functions = len([c for c in complexities if 5 < c <= 10])
                complex_functions = len([c for c in complexities if c > 10])
                
                total_functions = len(complexities)
                trends['simple_function_ratio'] = simple_functions / total_functions
                trends['moderate_function_ratio'] = moderate_functions / total_functions
                trends['complex_function_ratio'] = complex_functions / total_functions
                
                # Predict refactoring pressure based on complexity
                if trends['complex_function_ratio'] > 0.3:
                    trends['refactoring_urgency'] = 0.8
                elif trends['complex_function_ratio'] > 0.1:
                    trends['refactoring_urgency'] = 0.5
                else:
                    trends['refactoring_urgency'] = 0.2
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing complexity trends: {e}")
            return {}
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function"""
        
        try:
            complexity = 1  # Base complexity
            
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
                elif isinstance(child, (ast.And, ast.Or)):
                    complexity += 1
                elif isinstance(child, ast.ExceptHandler):
                    complexity += 1
            
            return complexity
            
        except Exception as e:
            self.logger.error(f"Error calculating cyclomatic complexity: {e}")
            return 1
    
    def _calculate_complexity_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        
        try:
            complexities = []
            nesting_depths = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    nesting_depth = self._calculate_nesting_depth(node)
                    complexities.append(complexity)
                    nesting_depths.append(nesting_depth)
            
            if not complexities:
                return {'average_complexity': 0, 'average_nesting_depth': 0}
            
            metrics = {
                'average_complexity': np.mean(complexities),
                'max_complexity': max(complexities),
                'min_complexity': min(complexities),
                'complexity_std': np.std(complexities),
                'average_nesting_depth': np.mean(nesting_depths),
                'max_nesting_depth': max(nesting_depths)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity metrics: {e}")
            return {}
    
    def _calculate_nesting_depth(self, node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth in function"""
        
        try:
            max_depth = 0
            
            def calculate_depth(node, current_depth=0):
                nonlocal max_depth
                max_depth = max(max_depth, current_depth)
                
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                        calculate_depth(child, current_depth + 1)
                    else:
                        calculate_depth(child, current_depth)
            
            calculate_depth(node)
            return max_depth
            
        except Exception as e:
            self.logger.error(f"Error calculating nesting depth: {e}")
            return 0
    
    def _predict_dependency_evolution(self, tree: ast.AST, code: str) -> Dict[str, List[str]]:
        """Predict how dependencies might evolve"""
        
        try:
            evolution = {
                'likely_new_dependencies': [],
                'potential_removals': [],
                'upgrade_candidates': [],
                'security_concerns': []
            }
            
            # Analyze current imports
            current_imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        current_imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        current_imports.add(node.module)
            
            # Predict likely new dependencies based on code patterns
            code_lower = code.lower()
            
            if 'async' in code and 'asyncio' not in current_imports:
                evolution['likely_new_dependencies'].append('asyncio')
            
            if any(term in code_lower for term in ['http', 'request', 'api']) and 'requests' not in current_imports:
                evolution['likely_new_dependencies'].append('requests')
            
            if 'json' in code and 'json' not in current_imports:
                evolution['likely_new_dependencies'].append('json')
            
            if any(term in code_lower for term in ['database', 'sql']) and 'sqlalchemy' not in current_imports:
                evolution['likely_new_dependencies'].extend(['sqlalchemy', 'psycopg2'])
            
            if any(term in code_lower for term in ['data', 'analysis', 'csv']) and 'pandas' not in current_imports:
                evolution['likely_new_dependencies'].append('pandas')
            
            if any(term in code_lower for term in ['plot', 'graph', 'chart']) and 'matplotlib' not in current_imports:
                evolution['likely_new_dependencies'].append('matplotlib')
            
            if any(term in code_lower for term in ['ml', 'model', 'predict']) and 'scikit-learn' not in current_imports:
                evolution['likely_new_dependencies'].append('scikit-learn')
            
            # Identify potential removals (unused imports)
            for import_name in current_imports:
                # Simple heuristic: if import appears only once (in import statement)
                if code.count(import_name) <= 1:
                    evolution['potential_removals'].append(import_name)
            
            # Common packages that frequently need updates
            upgrade_candidates_map = {
                'requests': 'frequently_updated',
                'numpy': 'performance_improvements',
                'pandas': 'feature_additions',
                'django': 'security_updates',
                'flask': 'security_updates'
            }
            
            for import_name in current_imports:
                if import_name in upgrade_candidates_map:
                    evolution['upgrade_candidates'].append(import_name)
            
            # Security-sensitive packages
            security_sensitive = ['requests', 'urllib', 'subprocess', 'pickle', 'SafeCodeExecutor.safe_eval', 'exec']
            for import_name in current_imports:
                if import_name in security_sensitive:
                    evolution['security_concerns'].append(import_name)
            
            return evolution
            
        except Exception as e:
            self.logger.error(f"Error predicting dependency evolution: {e}")
            return {}
    
    def _assess_feature_addition_likelihood(self, tree: ast.AST, code: str) -> Dict[str, float]:
        """Assess likelihood of feature additions"""
        
        try:
            likelihood = {}
            code_lower = code.lower()
            
            # Analyze patterns that suggest future feature additions
            if any(word in code_lower for word in ['todo', 'fixme', 'hack', 'temporary']):
                likelihood['planned_features'] = 0.8
            
            # Large classes suggest feature expansion
            class_sizes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    class_sizes.append(method_count)
            
            if class_sizes and max(class_sizes) > 10:
                likelihood['new_methods'] = 0.7
            elif class_sizes and np.mean(class_sizes) > 5:
                likelihood['new_methods'] = 0.5
            
            # Configuration suggests configurability expansion
            if any(word in code_lower for word in ['config', 'setting', 'option', 'parameter']):
                likelihood['configuration_options'] = 0.7
            
            # API patterns suggest endpoint expansion
            if any(word in code_lower for word in ['api', 'endpoint', 'route', 'handler']):
                likelihood['api_endpoints'] = 0.6
            
            # Testing patterns suggest test expansion
            if any(word in code_lower for word in ['test', 'assert', 'mock']):
                likelihood['test_coverage_expansion'] = 0.9
            
            # Error handling suggests robustness improvements
            try_count = len([n for n in ast.walk(tree) if isinstance(n, ast.Try)])
            if try_count > 0:
                likelihood['error_handling_expansion'] = min(0.8, try_count * 0.2)
            
            # Logging suggests monitoring expansion
            if 'log' in code_lower:
                likelihood['monitoring_features'] = 0.6
            
            # Data processing suggests analytics expansion
            if any(word in code_lower for word in ['data', 'process', 'analyze', 'parse']):
                likelihood['analytics_features'] = 0.5
            
            return likelihood
            
        except Exception as e:
            self.logger.error(f"Error assessing feature addition likelihood: {e}")
            return {}
    
    def _calculate_refactoring_pressure(self, tree: ast.AST, code: str) -> Dict[str, float]:
        """Calculate pressure for refactoring"""
        
        try:
            pressure = {}
            
            # Long method pressure
            long_methods = 0
            total_methods = 0
            method_lengths = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_methods += 1
                    length = (node.end_lineno or node.lineno) - node.lineno
                    method_lengths.append(length)
                    if length > 50:
                        long_methods += 1
            
            if total_methods > 0:
                pressure['method_length_pressure'] = long_methods / total_methods
                pressure['average_method_length'] = np.mean(method_lengths)
            
            # Complexity pressure
            high_complexity_functions = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    if complexity > 15:
                        high_complexity_functions += 1
            
            if total_methods > 0:
                pressure['complexity_pressure'] = high_complexity_functions / total_methods
            
            # Code duplication pressure (simplified heuristic)
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            unique_lines = set(lines)
            if len(lines) > 0:
                pressure['duplication_pressure'] = 1 - (len(unique_lines) / len(lines))
            
            # Large class pressure
            large_classes = 0
            total_classes = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    total_classes += 1
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    if method_count > 20:
                        large_classes += 1
            
            if total_classes > 0:
                pressure['class_size_pressure'] = large_classes / total_classes
            
            # Deep nesting pressure
            max_nesting = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    nesting = self._calculate_nesting_depth(node)
                    max_nesting = max(max_nesting, nesting)
            
            pressure['nesting_pressure'] = min(1.0, max_nesting / 10)
            
            return pressure
            
        except Exception as e:
            self.logger.error(f"Error calculating refactoring pressure: {e}")
            return {}
    
    def _project_maintenance_burden(self, analysis: CodeEvolutionAnalysis) -> Dict[str, float]:
        """Project future maintenance burden"""
        
        try:
            burden = {}
            
            # Calculate based on growth patterns
            growth_patterns = analysis.growth_patterns
            
            # Complexity burden
            complexity_growth = growth_patterns.get('complexity_growth_rate', 0)
            burden['complexity_burden'] = min(1.0, complexity_growth * 2)
            
            # Method count burden
            method_growth = growth_patterns.get('method_growth_rate', 0)
            burden['method_count_burden'] = min(1.0, method_growth * 1.5)
            
            # Dependency burden
            dependency_growth = growth_patterns.get('dependency_growth_rate', 0)
            burden['dependency_burden'] = min(1.0, dependency_growth * 3)
            
            # Documentation burden (inverse of documentation growth)
            doc_growth = growth_patterns.get('documentation_growth_rate', 0)
            burden['documentation_burden'] = max(0.0, 1.0 - doc_growth)
            
            # Refactoring burden based on pressure
            refactoring_pressure = analysis.refactoring_pressure
            avg_pressure = np.mean(list(refactoring_pressure.values())) if refactoring_pressure else 0
            burden['refactoring_burden'] = avg_pressure
            
            # Overall maintenance burden (weighted average)
            weights = {
                'complexity_burden': 0.3,
                'method_count_burden': 0.2,
                'dependency_burden': 0.2,
                'documentation_burden': 0.15,
                'refactoring_burden': 0.15
            }
            
            overall_burden = sum(burden.get(key, 0) * weight for key, weight in weights.items())
            burden['overall_maintenance_burden'] = overall_burden
            
            return burden
            
        except Exception as e:
            self.logger.error(f"Error projecting maintenance burden: {e}")
            return {}
    
    def _calculate_docstring_coverage(self, tree: ast.AST) -> float:
        """Calculate docstring coverage percentage"""
        
        try:
            total_functions = 0
            documented_functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    total_functions += 1
                    # Check if first statement is a string (docstring)
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        documented_functions += 1
            
            return documented_functions / max(1, total_functions)
            
        except Exception as e:
            self.logger.error(f"Error calculating docstring coverage: {e}")
            return 0.0
    
    def _calculate_comment_density(self, code: str) -> float:
        """Calculate comment density (comments per line of code)"""
        
        try:
            lines = code.split('\n')
            code_lines = 0
            comment_lines = 0
            
            for line in lines:
                stripped = line.strip()
                if stripped:
                    if stripped.startswith('#'):
                        comment_lines += 1
                    else:
                        code_lines += 1
            
            return comment_lines / max(1, code_lines + comment_lines)
            
        except Exception as e:
            self.logger.error(f"Error calculating comment density: {e}")
            return 0.0


def create_evolution_predictor() -> CodeEvolutionPredictor:
    """Factory function to create CodeEvolutionPredictor instance"""
    
    return CodeEvolutionPredictor()