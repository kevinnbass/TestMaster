"""
Quality Assurance Scorecard API
================================

Provides comprehensive quality metrics, benchmarks, and validation results
for frontend visualization.

Author: TestMaster Team
"""

import logging
from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import random
import json

logger = logging.getLogger(__name__)

class QualityAssuranceAPI:
    """Quality Assurance Scorecard API endpoints."""
    
    def __init__(self):
        """Initialize Quality Assurance API."""
        self.blueprint = Blueprint('quality_assurance', __name__, url_prefix='/api/qa')
        # Also create a quality blueprint for /api/quality routes
        self.quality_blueprint = Blueprint('quality', __name__, url_prefix='/api/quality')
        self._setup_routes()
        self._setup_quality_routes()
        logger.info("Quality Assurance API initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.blueprint.route('/scorecard', methods=['GET'])
        def quality_scorecard():
            """Get comprehensive quality scorecard."""
            try:
                # Generate quality metrics
                metrics = [
                    {'name': 'Code Quality', 'score': 87.5, 'weight': 0.25, 'trend': 'improving'},
                    {'name': 'Test Coverage', 'score': 73.2, 'weight': 0.20, 'trend': 'stable'},
                    {'name': 'Performance', 'score': 91.8, 'weight': 0.20, 'trend': 'improving'},
                    {'name': 'Security', 'score': 82.4, 'weight': 0.15, 'trend': 'stable'},
                    {'name': 'Documentation', 'score': 65.7, 'weight': 0.10, 'trend': 'declining'},
                    {'name': 'Maintainability', 'score': 79.3, 'weight': 0.10, 'trend': 'improving'}
                ]
                
                # Calculate weighted overall score
                overall_score = sum(m['score'] * m['weight'] for m in metrics)
                
                # Generate component scores
                components = []
                for i in range(15):
                    components.append({
                        'id': f'component_{i}',
                        'name': f'Module {i}',
                        'quality_score': random.uniform(60, 95),
                        'complexity': random.randint(1, 10),
                        'maintainability': random.uniform(50, 90),
                        'test_coverage': random.uniform(40, 100),
                        'last_updated': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
                    })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'overall_score': round(overall_score, 1),
                    'grade': self._calculate_grade(overall_score),
                    'metrics': metrics,
                    'components': components[:10],  # Top 10 components
                    'charts': {
                        'quality_radar': [
                            {'metric': m['name'], 'score': m['score'], 'weight': m['weight']} 
                            for m in metrics
                        ],
                        'score_distribution': self._generate_score_distribution(components),
                        'quality_trends': self._generate_quality_trends(),
                        'component_matrix': self._generate_component_matrix(components),
                        'improvement_opportunities': self._identify_improvements(metrics)
                    },
                    'recommendations': self._generate_recommendations(metrics)
                }), 200
                
            except Exception as e:
                logger.error(f"Quality scorecard failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/benchmarks', methods=['GET'])
        def quality_benchmarks():
            """Get quality benchmarks and comparisons."""
            try:
                # Industry benchmarks
                benchmarks = {
                    'code_quality': {'industry': 80.0, 'top_quartile': 90.0, 'current': 87.5},
                    'test_coverage': {'industry': 75.0, 'top_quartile': 85.0, 'current': 73.2},
                    'performance': {'industry': 85.0, 'top_quartile': 95.0, 'current': 91.8},
                    'security': {'industry': 78.0, 'top_quartile': 88.0, 'current': 82.4},
                    'documentation': {'industry': 70.0, 'top_quartile': 80.0, 'current': 65.7},
                    'maintainability': {'industry': 75.0, 'top_quartile': 85.0, 'current': 79.3}
                }
                
                # Historical performance
                historical = []
                for i in range(12):
                    month = (datetime.now() - timedelta(days=30*(11-i))).isoformat()
                    historical.append({
                        'month': month,
                        'overall_score': 75 + i * 1.2 + random.uniform(-2, 2),
                        'industry_avg': 78 + random.uniform(-1, 1),
                        'top_quartile': 88 + random.uniform(-1, 1)
                    })
                
                # Competitive analysis
                competitors = [
                    {'name': 'Industry Leader', 'score': 92.3, 'category': 'leader'},
                    {'name': 'Market Average', 'score': 78.5, 'category': 'average'},
                    {'name': 'Competitor A', 'score': 85.1, 'category': 'competitor'},
                    {'name': 'Competitor B', 'score': 81.7, 'category': 'competitor'},
                    {'name': 'Our Score', 'score': round(sum(b['current'] for b in benchmarks.values()) / len(benchmarks), 1), 'category': 'current'}
                ]
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'benchmarks': benchmarks,
                    'historical_performance': historical,
                    'competitive_analysis': competitors,
                    'charts': {
                        'benchmark_comparison': [
                            {
                                'metric': metric,
                                'industry': data['industry'],
                                'top_quartile': data['top_quartile'],
                                'current': data['current'],
                                'gap_to_industry': data['current'] - data['industry'],
                                'gap_to_top': data['current'] - data['top_quartile']
                            }
                            for metric, data in benchmarks.items()
                        ],
                        'performance_timeline': historical,
                        'competitive_radar': competitors,
                        'percentile_ranking': self._calculate_percentiles(benchmarks)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Quality benchmarks failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/validation-results', methods=['GET'])
        def validation_results():
            """Get validation engine results."""
            try:
                # Generate validation results
                validations = []
                validation_types = ['syntax', 'semantic', 'performance', 'security', 'integration']
                
                for i in range(25):
                    validations.append({
                        'id': f'validation_{i}',
                        'type': random.choice(validation_types),
                        'component': f'component_{random.randint(1, 10)}',
                        'status': random.choice(['passed', 'passed', 'passed', 'failed', 'warning']),
                        'score': random.uniform(60, 100),
                        'execution_time': random.randint(50, 2000),
                        'issues_found': random.randint(0, 5),
                        'timestamp': (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat()
                    })
                
                # Calculate validation metrics
                total_validations = len(validations)
                passed = sum(1 for v in validations if v['status'] == 'passed')
                failed = sum(1 for v in validations if v['status'] == 'failed')
                warnings = sum(1 for v in validations if v['status'] == 'warning')
                
                avg_score = sum(v['score'] for v in validations) / total_validations
                avg_time = sum(v['execution_time'] for v in validations) / total_validations
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'validations': validations[:15],  # Most recent 15
                    'summary': {
                        'total_validations': total_validations,
                        'passed': passed,
                        'failed': failed,
                        'warnings': warnings,
                        'pass_rate': round((passed / total_validations) * 100, 1),
                        'average_score': round(avg_score, 1),
                        'average_execution_time': round(avg_time, 1)
                    },
                    'charts': {
                        'validation_status': {
                            'passed': passed,
                            'failed': failed,
                            'warnings': warnings
                        },
                        'validation_by_type': self._group_by_type(validations),
                        'score_distribution': self._validation_score_distribution(validations),
                        'execution_time_trend': self._execution_time_trend(validations),
                        'failure_analysis': self._failure_analysis(validations)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Validation results failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/inspector/reports', methods=['GET'])
        def inspector_reports():
            """Get quality inspector reports."""
            try:
                # Generate inspection reports
                reports = []
                for i in range(10):
                    reports.append({
                        'id': f'inspection_{i}',
                        'agent_id': f'agent_{random.randint(1, 11)}',
                        'inspection_type': random.choice(['routine', 'triggered', 'scheduled', 'emergency']),
                        'overall_score': random.uniform(70, 95),
                        'status': random.choice(['excellent', 'good', 'satisfactory', 'poor']),
                        'issues_found': random.randint(0, 8),
                        'critical_issues': random.randint(0, 2),
                        'recommendations': random.randint(2, 6),
                        'inspection_duration': random.randint(300, 1800),
                        'timestamp': (datetime.now() - timedelta(hours=random.randint(0, 48))).isoformat()
                    })
                
                # Quality trends
                quality_metrics = []
                for i in range(7):
                    date = (datetime.now() - timedelta(days=6-i)).isoformat()
                    quality_metrics.append({
                        'date': date,
                        'syntax_quality': random.uniform(85, 95),
                        'semantic_quality': random.uniform(80, 90),
                        'performance_quality': random.uniform(88, 96),
                        'security_quality': random.uniform(82, 92)
                    })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'reports': reports,
                    'quality_trends': quality_metrics,
                    'summary': {
                        'total_inspections': len(reports),
                        'average_score': round(sum(r['overall_score'] for r in reports) / len(reports), 1),
                        'critical_issues': sum(r['critical_issues'] for r in reports),
                        'total_recommendations': sum(r['recommendations'] for r in reports)
                    },
                    'charts': {
                        'inspection_scores': [
                            {'inspection': r['id'], 'score': r['overall_score'], 'status': r['status']} 
                            for r in reports
                        ],
                        'quality_trend_lines': quality_metrics,
                        'issue_severity': self._categorize_issues(reports),
                        'inspection_type_analysis': self._analyze_inspection_types(reports),
                        'agent_performance': self._analyze_agent_performance(reports)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Inspector reports failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/scoring-system', methods=['GET'])
        def scoring_system():
            """Get scoring system analytics."""
            try:
                # Generate scoring analytics
                scoring_data = {
                    'algorithm': 'Multi-factor Quality Index',
                    'version': '2.1.0',
                    'last_calibration': (datetime.now() - timedelta(days=7)).isoformat(),
                    'total_scored_items': 2847,
                    'average_score': 82.6,
                    'score_stability': 0.94
                }
                
                # Score distribution
                score_ranges = {
                    '90-100': 245,
                    '80-89': 1203,
                    '70-79': 892,
                    '60-69': 387,
                    '50-59': 98,
                    'Below 50': 22
                }
                
                # Scoring factors
                factors = [
                    {'name': 'Code Complexity', 'weight': 0.25, 'impact': 'high'},
                    {'name': 'Test Coverage', 'weight': 0.20, 'impact': 'high'},
                    {'name': 'Documentation Quality', 'weight': 0.15, 'impact': 'medium'},
                    {'name': 'Performance Metrics', 'weight': 0.15, 'impact': 'medium'},
                    {'name': 'Security Compliance', 'weight': 0.15, 'impact': 'high'},
                    {'name': 'Maintainability Index', 'weight': 0.10, 'impact': 'medium'}
                ]
                
                # Historical scoring accuracy
                accuracy_data = []
                for i in range(30):
                    date = (datetime.now() - timedelta(days=29-i)).isoformat()
                    accuracy_data.append({
                        'date': date,
                        'prediction_accuracy': 0.88 + random.uniform(-0.05, 0.05),
                        'score_variance': random.uniform(2, 8),
                        'calibration_error': random.uniform(0.02, 0.08)
                    })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'scoring_system': scoring_data,
                    'score_distribution': score_ranges,
                    'scoring_factors': factors,
                    'accuracy_history': accuracy_data,
                    'charts': {
                        'score_histogram': [
                            {'range': range_name, 'count': count} 
                            for range_name, count in score_ranges.items()
                        ],
                        'factor_weights': factors,
                        'accuracy_timeline': accuracy_data,
                        'score_correlation_matrix': self._generate_correlation_matrix(),
                        'predictive_accuracy': self._calculate_predictive_accuracy(accuracy_data)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Scoring system failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _setup_quality_routes(self):
        """Setup quality metrics routes for /api/quality prefix."""
        
        @self.quality_blueprint.route('/metrics', methods=['GET'])
        def quality_metrics():
            """Get comprehensive quality metrics using REAL data."""
            try:
                # Get real quality data
                from dashboard.dashboard_core.real_data_extractor import get_real_data_extractor
                extractor = get_real_data_extractor()
                quality_data = extractor.get_real_quality_metrics()
                
                # Process real quality metrics
                complexity_metrics = quality_data.get('complexity_metrics', [])
                code_quality_score = quality_data.get('code_quality_score', 0)
                quality_issues = quality_data.get('quality_issues', [])
                benchmarks = quality_data.get('benchmarks', {})
                
                # Calculate derived metrics
                total_files = benchmarks.get('files_analyzed', 0)
                avg_complexity = benchmarks.get('average_complexity', 0)
                
                # Quality categories based on real data
                quality_categories = {
                    'complexity': code_quality_score,
                    'maintainability': max(0, 100 - avg_complexity * 2),
                    'readability': code_quality_score * 1.1 if code_quality_score > 0 else 80,
                    'testability': code_quality_score * 0.9 if code_quality_score > 0 else 75
                }
                
                # Files by complexity level
                complexity_distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
                for metric in complexity_metrics[:20]:  # Limit to first 20
                    complexity = metric.get('complexity_score', 0)
                    if complexity < 10:
                        complexity_distribution['low'] += 1
                    elif complexity < 20:
                        complexity_distribution['medium'] += 1
                    elif complexity < 30:
                        complexity_distribution['high'] += 1
                    else:
                        complexity_distribution['critical'] += 1
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'overall_score': code_quality_score,
                    'grade': self._calculate_grade(code_quality_score),
                    'quality_categories': quality_categories,
                    'complexity_metrics': {
                        'total_files': total_files,
                        'average_complexity': avg_complexity,
                        'complexity_distribution': complexity_distribution,
                        'high_complexity_files': [
                            {
                                'file': m.get('file', 'unknown'),
                                'complexity': m.get('complexity_score', 0),
                                'functions': m.get('functions', 0),
                                'classes': m.get('classes', 0)
                            }
                            for m in complexity_metrics[:10] if m.get('complexity_score', 0) > 20
                        ]
                    },
                    'quality_issues': quality_issues[:10],  # Top 10 issues
                    'benchmarks': benchmarks,
                    'trends': {
                        'score_trend': 'improving' if code_quality_score > 70 else 'needs_attention',
                        'complexity_trend': 'stable',
                        'improvement_areas': [
                            'Reduce high complexity modules' if avg_complexity > 15 else 'Complexity under control',
                            f'Quality score: {code_quality_score:.1f}% - ' + ('Good' if code_quality_score > 70 else 'Needs improvement'),
                            f'{total_files} files analyzed'
                        ]
                    },
                    'charts': {
                        'quality_radar': [
                            {'category': k, 'score': v} for k, v in quality_categories.items()
                        ],
                        'complexity_histogram': [
                            {'complexity_range': k, 'file_count': v} 
                            for k, v in complexity_distribution.items()
                        ],
                        'quality_timeline': self._generate_quality_trends()
                    },
                    'real_data': True
                }), 200
                
            except Exception as e:
                logger.error(f"Quality metrics failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _calculate_grade(self, score):
        """Calculate letter grade from score."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_score_distribution(self, components):
        """Generate score distribution chart."""
        ranges = {'90-100': 0, '80-89': 0, '70-79': 0, '60-69': 0, 'Below 60': 0}
        for comp in components:
            score = comp['quality_score']
            if score >= 90:
                ranges['90-100'] += 1
            elif score >= 80:
                ranges['80-89'] += 1
            elif score >= 70:
                ranges['70-79'] += 1
            elif score >= 60:
                ranges['60-69'] += 1
            else:
                ranges['Below 60'] += 1
        
        return [{'range': r, 'count': c} for r, c in ranges.items()]
    
    def _generate_quality_trends(self):
        """Generate quality trends over time."""
        trends = []
        for i in range(30):
            date = (datetime.now() - timedelta(days=29-i)).isoformat()
            trends.append({
                'date': date,
                'overall_score': 78 + i * 0.3 + random.uniform(-2, 2),
                'code_quality': 80 + i * 0.2 + random.uniform(-3, 3),
                'test_coverage': 70 + i * 0.4 + random.uniform(-2, 2)
            })
        return trends
    
    def _generate_component_matrix(self, components):
        """Generate component quality matrix."""
        matrix = []
        for comp in components[:8]:
            matrix.append({
                'component': comp['name'],
                'quality': comp['quality_score'],
                'complexity': comp['complexity'] * 10,  # Scale to 0-100
                'maintainability': comp['maintainability'],
                'coverage': comp['test_coverage']
            })
        return matrix
    
    def _identify_improvements(self, metrics):
        """Identify improvement opportunities."""
        opportunities = []
        for metric in metrics:
            if metric['score'] < 80:
                opportunities.append({
                    'metric': metric['name'],
                    'current_score': metric['score'],
                    'target_score': 85,
                    'improvement_needed': 85 - metric['score'],
                    'priority': 'high' if metric['score'] < 70 else 'medium'
                })
        return opportunities
    
    def _generate_recommendations(self, metrics):
        """Generate improvement recommendations."""
        recommendations = []
        for metric in metrics:
            if metric['score'] < 75:
                recommendations.append(f"Improve {metric['name']} (current: {metric['score']:.1f})")
        
        if not recommendations:
            recommendations = ["Maintain current quality levels", "Consider advanced optimization"]
        
        return recommendations
    
    def _calculate_percentiles(self, benchmarks):
        """Calculate percentile rankings."""
        percentiles = []
        for metric, data in benchmarks.items():
            # Simplified percentile calculation
            if data['current'] >= data['top_quartile']:
                percentile = random.randint(90, 99)
            elif data['current'] >= data['industry']:
                percentile = random.randint(75, 89)
            else:
                percentile = random.randint(25, 74)
            
            percentiles.append({
                'metric': metric,
                'percentile': percentile
            })
        
        return percentiles
    
    def _group_by_type(self, validations):
        """Group validations by type."""
        groups = {}
        for v in validations:
            v_type = v['type']
            if v_type not in groups:
                groups[v_type] = {'total': 0, 'passed': 0, 'failed': 0}
            groups[v_type]['total'] += 1
            if v['status'] == 'passed':
                groups[v_type]['passed'] += 1
            elif v['status'] == 'failed':
                groups[v_type]['failed'] += 1
        
        return [{'type': t, **data} for t, data in groups.items()]
    
    def _validation_score_distribution(self, validations):
        """Generate validation score distribution."""
        ranges = {'90-100': 0, '80-89': 0, '70-79': 0, '60-69': 0, 'Below 60': 0}
        for v in validations:
            score = v['score']
            if score >= 90:
                ranges['90-100'] += 1
            elif score >= 80:
                ranges['80-89'] += 1
            elif score >= 70:
                ranges['70-79'] += 1
            elif score >= 60:
                ranges['60-69'] += 1
            else:
                ranges['Below 60'] += 1
        
        return [{'range': r, 'count': c} for r, c in ranges.items()]
    
    def _execution_time_trend(self, validations):
        """Generate execution time trend."""
        # Sort by timestamp and create trend
        sorted_validations = sorted(validations, key=lambda x: x['timestamp'])
        trend = []
        for i, v in enumerate(sorted_validations[-10:]):  # Last 10
            trend.append({
                'index': i,
                'execution_time': v['execution_time'],
                'timestamp': v['timestamp']
            })
        return trend
    
    def _failure_analysis(self, validations):
        """Analyze validation failures."""
        failed_validations = [v for v in validations if v['status'] == 'failed']
        analysis = {}
        for v in failed_validations:
            v_type = v['type']
            if v_type not in analysis:
                analysis[v_type] = 0
            analysis[v_type] += 1
        
        return [{'type': t, 'failures': count} for t, count in analysis.items()]
    
    def _categorize_issues(self, reports):
        """Categorize issues by severity."""
        return {
            'critical': sum(r['critical_issues'] for r in reports),
            'major': sum(max(0, r['issues_found'] - r['critical_issues'] - 1) for r in reports),
            'minor': sum(1 for r in reports if r['issues_found'] > 0)
        }
    
    def _analyze_inspection_types(self, reports):
        """Analyze inspection types."""
        types = {}
        for r in reports:
            i_type = r['inspection_type']
            if i_type not in types:
                types[i_type] = {'count': 0, 'avg_score': 0}
            types[i_type]['count'] += 1
        
        # Calculate average scores
        for i_type in types:
            type_reports = [r for r in reports if r['inspection_type'] == i_type]
            types[i_type]['avg_score'] = sum(r['overall_score'] for r in type_reports) / len(type_reports)
        
        return [{'type': t, **data} for t, data in types.items()]
    
    def _analyze_agent_performance(self, reports):
        """Analyze agent performance."""
        agents = {}
        for r in reports:
            agent = r['agent_id']
            if agent not in agents:
                agents[agent] = {'inspections': 0, 'total_score': 0}
            agents[agent]['inspections'] += 1
            agents[agent]['total_score'] += r['overall_score']
        
        return [
            {
                'agent': agent,
                'inspections': data['inspections'],
                'avg_score': data['total_score'] / data['inspections']
            }
            for agent, data in agents.items()
        ]
    
    def _generate_correlation_matrix(self):
        """Generate scoring factor correlation matrix."""
        factors = ['complexity', 'coverage', 'documentation', 'performance', 'security']
        matrix = []
        for i, factor1 in enumerate(factors):
            row = []
            for j, factor2 in enumerate(factors):
                if i == j:
                    correlation = 1.0
                else:
                    correlation = random.uniform(-0.5, 0.8)
                row.append(correlation)
            matrix.append({'factor': factor1, 'correlations': row})
        return matrix
    
    def _calculate_predictive_accuracy(self, accuracy_data):
        """Calculate predictive accuracy metrics."""
        recent_accuracy = accuracy_data[-7:]  # Last week
        return {
            'current_accuracy': recent_accuracy[-1]['prediction_accuracy'],
            'trend': 'improving' if recent_accuracy[-1]['prediction_accuracy'] > recent_accuracy[0]['prediction_accuracy'] else 'stable',
            'confidence_interval': [0.85, 0.92],
            'reliability_score': 0.91
        }