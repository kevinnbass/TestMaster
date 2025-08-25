"""
Main Comprehensive Codebase Analyzer
====================================

Orchestrates all analysis modules to provide comprehensive codebase insights.
This is the main entry point that coordinates all specialized analyzers.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from .software_metrics import SoftwareMetricsAnalyzer
from .coupling_cohesion import CouplingCohesionAnalyzer
from .inheritance_polymorphism import InheritancePolymorphismAnalyzer
from .graph_analysis import GraphAnalyzer
from .clone_detection import CloneDetectionAnalyzer
from .security_analysis import SecurityAnalyzer
from .taint_analysis import TaintAnalyzer
from .crypto_analysis import CryptographicAnalyzer
from .linguistic_analysis import LinguisticAnalyzer
from .evolution_analysis import EvolutionAnalyzer
from .statistical_analysis import StatisticalAnalyzer
from .structural_analysis import StructuralAnalyzer
from .complexity_analysis import ComplexityAnalyzer
from .quality_analysis import QualityAnalyzer
from .supply_chain_security import SupplyChainSecurityAnalyzer
from .api_analysis import APIAnalyzer
from .testing_analysis import TestingAnalyzer
from .performance_analysis import PerformanceAnalyzer
from .resource_io_analysis import ResourceIOAnalyzer
from .memory_analysis import MemoryAnalyzer
from .database_analysis import DatabaseAnalyzer
from .concurrency_analysis import ConcurrencyAnalyzer
from .error_handling_analysis import ErrorHandlingAnalyzer


class ComprehensiveCodebaseAnalyzer:
    """Main orchestrator for comprehensive codebase analysis."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        
        # Initialize all specialized analyzers
        self.analyzers = {
            'software_metrics': SoftwareMetricsAnalyzer(base_path),
            'coupling_cohesion': CouplingCohesionAnalyzer(base_path),
            'inheritance_polymorphism': InheritancePolymorphismAnalyzer(base_path),
            'graph_analysis': GraphAnalyzer(base_path),
            'clone_detection': CloneDetectionAnalyzer(base_path),
            'security_analysis': SecurityAnalyzer(base_path),
            'taint_analysis': TaintAnalyzer(base_path),
            'cryptographic_analysis': CryptographicAnalyzer(base_path),
            'supply_chain_security': SupplyChainSecurityAnalyzer(base_path),
            'api_analysis': APIAnalyzer(base_path),
            'testing_analysis': TestingAnalyzer(base_path),
            'performance_analysis': PerformanceAnalyzer(base_path),
            'resource_io_analysis': ResourceIOAnalyzer(base_path),
            'memory_analysis': MemoryAnalyzer(base_path),
            'database_analysis': DatabaseAnalyzer(base_path),
            'concurrency_analysis': ConcurrencyAnalyzer(base_path),
            'error_handling_analysis': ErrorHandlingAnalyzer(base_path),
            'linguistic_analysis': LinguisticAnalyzer(base_path),
            'evolution_analysis': EvolutionAnalyzer(base_path),
            'statistical_analysis': StatisticalAnalyzer(base_path),
            'structural_analysis': StructuralAnalyzer(base_path),
            'complexity_analysis': ComplexityAnalyzer(base_path),
            'quality_analysis': QualityAnalyzer(base_path)
        }
        
    def analyze_comprehensive(self, selected_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis across all categories.
        
        Args:
            selected_categories: Optional list of categories to analyze.
                                If None, analyzes all categories.
        
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        print("=" * 80)
        print("COMPREHENSIVE CODEBASE ANALYSIS")
        print("=" * 80)
        print(f"Analyzing codebase: {self.base_path}")
        print(f"Analysis modules: {len(self.analyzers)}")
        print("-" * 80)
        
        # Determine which categories to analyze
        categories_to_analyze = selected_categories if selected_categories else list(self.analyzers.keys())
        
        results = {
            'metadata': {
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'base_path': str(self.base_path),
                'categories_analyzed': categories_to_analyze,
                'total_categories': len(self.analyzers),
                'analysis_version': '1.0.0'
            },
            'analysis_results': {},
            'summary': {},
            'recommendations': []
        }
        
        # Perform analysis for each selected category
        category_times = {}
        errors = {}
        
        for category in categories_to_analyze:
            if category not in self.analyzers:
                print(f"[WARNING] Unknown category: {category}")
                continue
                
            category_start = time.time()
            
            try:
                print(f"[INFO] Starting {category} analysis...")
                analyzer = self.analyzers[category]
                category_results = analyzer.analyze()
                results['analysis_results'][category] = category_results
                
                category_end = time.time()
                category_times[category] = category_end - category_start
                
                print(f"  [OK] Completed {category} analysis in {category_times[category]:.2f}s")
                
            except Exception as e:
                category_end = time.time()
                category_times[category] = category_end - category_start
                errors[category] = str(e)
                
                print(f"  [ERROR] Failed {category} analysis: {str(e)}")
                results['analysis_results'][category] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Generate summary and recommendations
        print("-" * 80)
        print("[INFO] Generating comprehensive summary...")
        
        results['summary'] = self._generate_comprehensive_summary(results['analysis_results'])
        results['recommendations'] = self._generate_comprehensive_recommendations(results['analysis_results'])
        results['cross_analysis'] = self._perform_cross_analysis(results['analysis_results'])
        
        # Add performance metadata
        end_time = time.time()
        total_time = end_time - start_time
        
        results['metadata'].update({
            'total_analysis_time_seconds': total_time,
            'category_analysis_times': category_times,
            'errors': errors,
            'success_rate': len([r for r in results['analysis_results'].values() if 'error' not in r]) / len(categories_to_analyze),
            'categories_completed': len([r for r in results['analysis_results'].values() if 'error' not in r]),
            'categories_failed': len(errors)
        })
        
        print(f"[OK] Comprehensive analysis completed in {total_time:.2f}s")
        print(f"    - Categories analyzed: {len(categories_to_analyze)}")
        print(f"    - Successful: {results['metadata']['categories_completed']}")
        print(f"    - Failed: {results['metadata']['categories_failed']}")
        print(f"    - Success rate: {results['metadata']['success_rate']:.1%}")
        print("=" * 80)
        
        return results
    
    def analyze_category(self, category: str) -> Dict[str, Any]:
        """
        Analyze a specific category.
        
        Args:
            category: Name of the analysis category
            
        Returns:
            Category-specific analysis results
        """
        if category not in self.analyzers:
            raise ValueError(f"Unknown analysis category: {category}")
        
        print(f"[INFO] Performing {category} analysis...")
        start_time = time.time()
        
        try:
            analyzer = self.analyzers[category]
            results = analyzer.analyze()
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            print(f"[OK] {category} analysis completed in {analysis_time:.2f}s")
            
            return {
                'category': category,
                'results': results,
                'metadata': {
                    'analysis_time_seconds': analysis_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'success'
                }
            }
            
        except Exception as e:
            end_time = time.time()
            analysis_time = end_time - start_time
            
            print(f"[ERROR] {category} analysis failed: {str(e)}")
            
            return {
                'category': category,
                'results': {'error': str(e)},
                'metadata': {
                    'analysis_time_seconds': analysis_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'failed',
                    'error': str(e)
                }
            }
    
    def get_available_categories(self) -> List[str]:
        """Get list of available analysis categories."""
        return list(self.analyzers.keys())
    
    def _generate_comprehensive_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary across all analysis results."""
        summary = {
            'overview': {
                'total_python_files': self._count_python_files(),
                'analysis_categories_completed': len([r for r in analysis_results.values() if 'error' not in r]),
                'key_findings': [],
                'overall_health_score': 0.0,
                'critical_issues': []
            },
            'metrics_summary': {},
            'quality_indicators': {},
            'complexity_overview': {},
            'security_overview': {},
            'maintainability_overview': {}
        }
        
        # Extract key metrics from each category
        try:
            # Software metrics summary
            if 'software_metrics' in analysis_results and 'error' not in analysis_results['software_metrics']:
                sm_results = analysis_results['software_metrics']
                if 'halstead_metrics' in sm_results:
                    halstead = sm_results['halstead_metrics']
                    summary['metrics_summary']['halstead_volume'] = halstead.get('volume', 0)
                    summary['metrics_summary']['halstead_difficulty'] = halstead.get('difficulty', 0)
                
                if 'mccabe_complexity' in sm_results:
                    mccabe = sm_results['mccabe_complexity']
                    summary['metrics_summary']['average_complexity'] = mccabe.get('average_complexity', 0)
                    summary['metrics_summary']['high_complexity_functions'] = mccabe.get('high_complexity_functions', 0)
            
            # Quality analysis summary
            if 'quality_analysis' in analysis_results and 'error' not in analysis_results['quality_analysis']:
                qa_results = analysis_results['quality_analysis']
                if 'quality_factors' in qa_results:
                    qf = qa_results['quality_factors']
                    summary['overview']['overall_health_score'] = qf.get('overall_quality_score', 0)
                    summary['quality_indicators'] = {
                        'readability': qf.get('factors', {}).get('readability', {}).get('score', 0),
                        'maintainability': qf.get('factors', {}).get('modularity', {}).get('score', 0),
                        'testability': qf.get('factors', {}).get('testability', {}).get('score', 0)
                    }
            
            # Security analysis summary
            if 'security_analysis' in analysis_results and 'error' not in analysis_results['security_analysis']:
                sa_results = analysis_results['security_analysis']
                if 'security_metrics' in sa_results:
                    sm = sa_results['security_metrics']
                    summary['security_overview'] = {
                        'total_vulnerabilities': sm.get('total_vulnerabilities', 0),
                        'high_severity': sm.get('high_severity', 0),
                        'security_score': sm.get('security_score', 0)
                    }
            
            # Complexity analysis summary
            if 'complexity_analysis' in analysis_results and 'error' not in analysis_results['complexity_analysis']:
                ca_results = analysis_results['complexity_analysis']
                if 'cyclomatic_complexity' in ca_results:
                    cc = ca_results['cyclomatic_complexity']
                    if 'summary' in cc:
                        summary['complexity_overview'] = {
                            'average_complexity': cc['summary'].get('average_complexity', 0),
                            'high_complexity_functions': cc['summary'].get('high_complexity_functions', 0),
                            'very_high_complexity_functions': cc['summary'].get('very_high_complexity_functions', 0)
                        }
            
            # Generate key findings
            summary['overview']['key_findings'] = self._extract_key_findings(analysis_results)
            summary['overview']['critical_issues'] = self._extract_critical_issues(analysis_results)
            
        except Exception as e:
            print(f"[WARNING] Error generating summary: {str(e)}")
        
        return summary
    
    def _generate_comprehensive_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive recommendations based on all analysis results."""
        recommendations = []
        
        try:
            # High complexity recommendations
            if 'complexity_analysis' in analysis_results:
                ca_results = analysis_results['complexity_analysis']
                if 'cyclomatic_complexity' in ca_results:
                    cc = ca_results['cyclomatic_complexity']
                    if cc.get('summary', {}).get('high_complexity_functions', 0) > 5:
                        recommendations.append({
                            'category': 'complexity',
                            'priority': 'high',
                            'title': 'Reduce High Complexity Functions',
                            'description': f"Found {cc['summary']['high_complexity_functions']} functions with high cyclomatic complexity",
                            'action': 'Refactor functions with complexity > 10 into smaller, more focused functions',
                            'impact': 'high',
                            'effort': 'medium'
                        })
            
            # Security recommendations
            if 'security_analysis' in analysis_results:
                sa_results = analysis_results['security_analysis']
                if 'security_metrics' in sa_results:
                    sm = sa_results['security_metrics']
                    if sm.get('high_severity', 0) > 0:
                        recommendations.append({
                            'category': 'security',
                            'priority': 'critical',
                            'title': 'Address High Severity Security Issues',
                            'description': f"Found {sm['high_severity']} high severity security vulnerabilities",
                            'action': 'Review and fix security vulnerabilities, especially SQL injection and command injection risks',
                            'impact': 'critical',
                            'effort': 'high'
                        })
            
            # Clone detection recommendations
            if 'clone_detection' in analysis_results:
                cd_results = analysis_results['clone_detection']
                if 'clone_metrics' in cd_results:
                    cm = cd_results['clone_metrics']
                    if cm.get('clone_ratio', 0) > 0.15:
                        recommendations.append({
                            'category': 'maintainability',
                            'priority': 'medium',
                            'title': 'Reduce Code Duplication',
                            'description': f"Code clone ratio is {cm['clone_ratio']:.1%}, indicating significant duplication",
                            'action': 'Extract common code into reusable functions or modules',
                            'impact': 'medium',
                            'effort': 'medium'
                        })
            
            # Quality recommendations
            if 'quality_analysis' in analysis_results:
                qa_results = analysis_results['quality_analysis']
                if 'quality_factors' in qa_results:
                    qf = qa_results['quality_factors']
                    overall_score = qf.get('overall_quality_score', 0)
                    
                    if overall_score < 70:
                        recommendations.append({
                            'category': 'quality',
                            'priority': 'high',
                            'title': 'Improve Overall Code Quality',
                            'description': f"Overall quality score is {overall_score:.1f}/100",
                            'action': 'Focus on improving readability, testability, and documentation',
                            'impact': 'high',
                            'effort': 'high'
                        })
            
            # Documentation recommendations
            if 'linguistic_analysis' in analysis_results:
                la_results = analysis_results['linguistic_analysis']
                if 'documentation_quality' in la_results:
                    dq = la_results['documentation_quality']
                    if 'summary' in dq and dq['summary'].get('function_docstring_coverage', 0) < 0.5:
                        recommendations.append({
                            'category': 'documentation',
                            'priority': 'medium',
                            'title': 'Improve Documentation Coverage',
                            'description': f"Function docstring coverage is {dq['summary']['function_docstring_coverage']:.1%}",
                            'action': 'Add docstrings to undocumented functions, especially public APIs',
                            'impact': 'medium',
                            'effort': 'low'
                        })
        
        except Exception as e:
            print(f"[WARNING] Error generating recommendations: {str(e)}")
        
        # Sort recommendations by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
        
        return recommendations
    
    def _perform_cross_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-analysis between different categories."""
        cross_analysis = {
            'correlations': [],
            'patterns': [],
            'insights': []
        }
        
        try:
            # Complexity vs Quality correlation
            complexity_score = 0
            quality_score = 0
            
            if 'complexity_analysis' in analysis_results and 'quality_analysis' in analysis_results:
                # Extract scores for correlation analysis
                if 'cyclomatic_complexity' in analysis_results['complexity_analysis']:
                    cc = analysis_results['complexity_analysis']['cyclomatic_complexity']
                    avg_complexity = cc.get('summary', {}).get('average_complexity', 0)
                    complexity_score = max(0, 100 - (avg_complexity * 10))  # Inverse relationship
                
                if 'quality_factors' in analysis_results['quality_analysis']:
                    qf = analysis_results['quality_analysis']['quality_factors']
                    quality_score = qf.get('overall_quality_score', 0)
                
                if complexity_score > 0 and quality_score > 0:
                    correlation = abs(complexity_score - quality_score)
                    cross_analysis['correlations'].append({
                        'factors': ['complexity', 'quality'],
                        'correlation_strength': 'strong' if correlation < 20 else 'moderate' if correlation < 40 else 'weak',
                        'description': 'High complexity correlates with lower quality scores'
                    })
            
            # Security vs Quality correlation
            if 'security_analysis' in analysis_results and 'quality_analysis' in analysis_results:
                sa_results = analysis_results['security_analysis']
                if 'security_metrics' in sa_results:
                    security_score = sa_results['security_metrics'].get('security_score', 0)
                    
                    if quality_score > 0 and security_score > 0:
                        cross_analysis['patterns'].append({
                            'pattern': 'security_quality_relationship',
                            'description': f'Security score ({security_score:.1f}) vs Quality score ({quality_score:.1f})',
                            'insight': 'Higher quality code tends to have better security practices'
                        })
            
            # Clone detection vs Maintainability
            if 'clone_detection' in analysis_results and 'quality_analysis' in analysis_results:
                cd_results = analysis_results['clone_detection']
                if 'clone_metrics' in cd_results:
                    clone_ratio = cd_results['clone_metrics'].get('clone_ratio', 0)
                    
                    if clone_ratio > 0.2:
                        cross_analysis['insights'].append({
                            'type': 'maintainability_risk',
                            'description': f'High code duplication ({clone_ratio:.1%}) may impact maintainability',
                            'recommendation': 'Prioritize refactoring duplicated code sections'
                        })
        
        except Exception as e:
            print(f"[WARNING] Error in cross-analysis: {str(e)}")
        
        return cross_analysis
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis results."""
        findings = []
        
        try:
            # From software metrics
            if 'software_metrics' in analysis_results:
                sm_results = analysis_results['software_metrics']
                if 'mccabe_complexity' in sm_results:
                    mccabe = sm_results['mccabe_complexity']
                    avg_complexity = mccabe.get('average_complexity', 0)
                    if avg_complexity > 8:
                        findings.append(f"High average cyclomatic complexity: {avg_complexity:.1f}")
            
            # From security analysis
            if 'security_analysis' in analysis_results:
                sa_results = analysis_results['security_analysis']
                if 'security_metrics' in sa_results:
                    sm = sa_results['security_metrics']
                    total_vulns = sm.get('total_vulnerabilities', 0)
                    if total_vulns > 10:
                        findings.append(f"Multiple security vulnerabilities detected: {total_vulns}")
            
            # From clone detection
            if 'clone_detection' in analysis_results:
                cd_results = analysis_results['clone_detection']
                if 'clone_metrics' in cd_results:
                    cm = cd_results['clone_metrics']
                    clone_ratio = cm.get('clone_ratio', 0)
                    if clone_ratio > 0.15:
                        findings.append(f"Significant code duplication: {clone_ratio:.1%}")
        
        except Exception as e:
            print(f"[WARNING] Error extracting findings: {str(e)}")
        
        return findings
    
    def _extract_critical_issues(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract critical issues from analysis results."""
        issues = []
        
        try:
            # Critical security issues
            if 'security_analysis' in analysis_results:
                sa_results = analysis_results['security_analysis']
                if 'security_metrics' in sa_results:
                    sm = sa_results['security_metrics']
                    if sm.get('high_severity', 0) > 0:
                        issues.append({
                            'category': 'security',
                            'severity': 'critical',
                            'description': f"{sm['high_severity']} high severity security vulnerabilities",
                            'impact': 'Data breach, system compromise'
                        })
            
            # Critical complexity issues
            if 'complexity_analysis' in analysis_results:
                ca_results = analysis_results['complexity_analysis']
                if 'cyclomatic_complexity' in ca_results:
                    cc = ca_results['cyclomatic_complexity']
                    very_high = cc.get('summary', {}).get('very_high_complexity_functions', 0)
                    if very_high > 0:
                        issues.append({
                            'category': 'complexity',
                            'severity': 'high',
                            'description': f"{very_high} functions with extremely high complexity",
                            'impact': 'Difficult maintenance, increased bug risk'
                        })
        
        except Exception as e:
            print(f"[WARNING] Error extracting critical issues: {str(e)}")
        
        return issues
    
    def _count_python_files(self) -> int:
        """Count total Python files in the codebase."""
        try:
            return len(list(self.base_path.rglob("*.py")))
        except:
            return 0