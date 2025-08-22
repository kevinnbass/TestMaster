"""
Intelligence Integration API
Integrates all new intelligence modules with existing dashboard
"""

from flask import Blueprint, request, jsonify, Response
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add core intelligence path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.intelligence.analysis.technical_debt_analyzer import TechnicalDebtAnalyzer
from core.intelligence.analysis.ml_code_analyzer import MLCodeAnalyzer
from core.intelligence.orchestration.agent_coordinator import AgentCoordinator, IntelligenceAgent, AgentCapability, TaskPriority

# Create blueprint
intelligence_integration_bp = Blueprint('intelligence_integration', __name__, url_prefix='/api/v1/intelligence')

# Global instances
debt_analyzer = None
ml_analyzer = None
coordinator = None

def init_intelligence_integration():
    """Initialize intelligence integration components"""
    global debt_analyzer, ml_analyzer, coordinator
    
    debt_analyzer = TechnicalDebtAnalyzer()
    ml_analyzer = MLCodeAnalyzer()
    coordinator = AgentCoordinator()
    
    # Initialize agents
    _initialize_agents()
    
    # Start coordination
    coordinator.start_coordination()

def _initialize_agents():
    """Initialize intelligence agents"""
    global coordinator
    
    # Technical Debt Agent
    debt_agent = IntelligenceAgent(
        agent_id="debt_agent_v1",
        name="Technical Debt Analyzer V1",
        capabilities=[
            AgentCapability(
                name="debt_analysis",
                description="Comprehensive technical debt analysis",
                task_types=["analyze_debt", "quantify_debt", "debt_remediation"],
                max_concurrent_tasks=2,
                estimated_task_time=45.0,
                specialized_domains=["code_quality", "maintainability", "refactoring"]
            )
        ]
    )
    
    # ML Code Agent
    ml_agent = IntelligenceAgent(
        agent_id="ml_agent_v1",
        name="ML Code Analyzer V1",
        capabilities=[
            AgentCapability(
                name="ml_analysis",
                description="Advanced ML/AI code analysis",
                task_types=["analyze_ml", "tensor_analysis", "model_validation"],
                max_concurrent_tasks=1,
                estimated_task_time=60.0,
                specialized_domains=["machine_learning", "deep_learning", "data_science"]
            )
        ]
    )
    
    # Coverage Integration Agent
    coverage_agent = IntelligenceAgent(
        agent_id="coverage_agent_v1",
        name="Coverage Integration Agent",
        capabilities=[
            AgentCapability(
                name="coverage_integration",
                description="Integrate with existing coverage systems",
                task_types=["coverage_analysis", "gap_analysis", "quality_metrics"],
                max_concurrent_tasks=3,
                estimated_task_time=30.0,
                specialized_domains=["testing", "quality_assurance", "metrics"]
            )
        ]
    )
    
    coordinator.register_agent(debt_agent)
    coordinator.register_agent(ml_agent)
    coordinator.register_agent(coverage_agent)

# API Routes

@intelligence_integration_bp.route('/health', methods=['GET'])
def health_check():
    """Intelligence system health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'debt_analyzer': debt_analyzer is not None,
            'ml_analyzer': ml_analyzer is not None,
            'coordinator': coordinator is not None,
            'active_agents': len(coordinator.agents) if coordinator else 0
        }
    })

@intelligence_integration_bp.route('/debt/analyze', methods=['POST'])
def analyze_technical_debt():
    """Analyze technical debt for existing dashboard integration"""
    try:
        data = request.get_json() or {}
        project_path = data.get('project_path', '.')
        
        if not debt_analyzer:
            return jsonify({'error': 'Debt analyzer not initialized'}), 500
        
        # Run analysis
        result = debt_analyzer.analyze_project(project_path)
        
        # Format for dashboard compatibility
        formatted_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'project_path': project_path,
            'analysis': {
                'total_debt_hours': result.get('metrics', {}).get('total_debt_hours', 0),
                'debt_ratio': result.get('metrics', {}).get('debt_ratio', 0),
                'monthly_interest': result.get('metrics', {}).get('monthly_interest', 0),
                'break_even_point': result.get('metrics', {}).get('break_even_point', 0),
                'total_issues': len(result.get('debt_items', [])),
                'critical_issues': len([
                    item for item in result.get('debt_items', [])
                    if item.get('severity') == 'critical'
                ]),
                'high_priority_issues': len([
                    item for item in result.get('debt_items', [])
                    if item.get('severity') == 'high'
                ]),
                'issues_by_type': result.get('summary', {}).get('debt_by_type', {}),
                'top_recommendations': result.get('recommendations', [])[:5]
            },
            'detailed_results': result
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@intelligence_integration_bp.route('/ml/analyze', methods=['POST'])
def analyze_ml_code():
    """Analyze ML code for dashboard integration"""
    try:
        data = request.get_json() or {}
        project_path = data.get('project_path', '.')
        
        if not ml_analyzer:
            return jsonify({'error': 'ML analyzer not initialized'}), 500
        
        # Run analysis
        result = ml_analyzer.analyze_project(project_path)
        
        # Format for dashboard compatibility
        analysis_result = result.get('analysis_result', {})
        formatted_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'project_path': project_path,
            'analysis': {
                'frameworks_detected': analysis_result.get('frameworks_detected', []),
                'total_issues': len(analysis_result.get('issues', [])),
                'critical_issues': len([
                    issue for issue in analysis_result.get('issues', [])
                    if issue.get('severity') == 'critical'
                ]),
                'security_issues': len([
                    issue for issue in analysis_result.get('issues', [])
                    if issue.get('type') == 'security_issue'
                ]),
                'performance_issues': len([
                    issue for issue in analysis_result.get('issues', [])
                    if 'performance' in issue.get('impact', '')
                ]),
                'issues_by_framework': {},
                'recommendations': analysis_result.get('recommendations', [])
            },
            'detailed_results': result
        }
        
        # Calculate issues by framework
        for issue in analysis_result.get('issues', []):
            framework = issue.get('framework', 'unknown')
            if framework not in formatted_result['analysis']['issues_by_framework']:
                formatted_result['analysis']['issues_by_framework'][framework] = 0
            formatted_result['analysis']['issues_by_framework'][framework] += 1
        
        return jsonify(formatted_result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@intelligence_integration_bp.route('/comprehensive', methods=['POST'])
def comprehensive_analysis():
    """Run comprehensive analysis combining all intelligence"""
    try:
        data = request.get_json() or {}
        project_path = data.get('project_path', '.')
        include_debt = data.get('include_debt', True)
        include_ml = data.get('include_ml', True)
        
        results = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'project_path': project_path,
            'analyses_run': [],
            'combined_metrics': {
                'total_issues': 0,
                'critical_issues': 0,
                'high_priority_issues': 0,
                'estimated_fix_hours': 0,
                'risk_score': 0
            }
        }
        
        # Run debt analysis
        if include_debt and debt_analyzer:
            debt_result = debt_analyzer.analyze_project(project_path)
            results['debt_analysis'] = debt_result
            results['analyses_run'].append('technical_debt')
            
            # Update combined metrics
            debt_items = debt_result.get('debt_items', [])
            results['combined_metrics']['total_issues'] += len(debt_items)
            results['combined_metrics']['critical_issues'] += len([
                item for item in debt_items if item.get('severity') == 'critical'
            ])
            results['combined_metrics']['high_priority_issues'] += len([
                item for item in debt_items if item.get('severity') == 'high'
            ])
            results['combined_metrics']['estimated_fix_hours'] += debt_result.get('metrics', {}).get('total_debt_hours', 0)
        
        # Run ML analysis
        if include_ml and ml_analyzer:
            ml_result = ml_analyzer.analyze_project(project_path)
            results['ml_analysis'] = ml_result
            results['analyses_run'].append('ml_code')
            
            # Update combined metrics
            ml_issues = ml_result.get('analysis_result', {}).get('issues', [])
            results['combined_metrics']['total_issues'] += len(ml_issues)
            results['combined_metrics']['critical_issues'] += len([
                issue for issue in ml_issues if issue.get('severity') == 'critical'
            ])
            results['combined_metrics']['high_priority_issues'] += len([
                issue for issue in ml_issues if issue.get('severity') == 'high'
            ])
        
        # Calculate overall risk score
        total_issues = results['combined_metrics']['total_issues']
        critical_issues = results['combined_metrics']['critical_issues']
        
        if total_issues > 0:
            risk_score = min(100, (critical_issues * 50 + total_issues * 5))
            results['combined_metrics']['risk_score'] = risk_score
        
        # Generate combined recommendations
        all_recommendations = []
        if 'debt_analysis' in results:
            all_recommendations.extend(results['debt_analysis'].get('recommendations', []))
        if 'ml_analysis' in results:
            all_recommendations.extend(results['ml_analysis'].get('analysis_result', {}).get('recommendations', []))
        
        results['combined_recommendations'] = all_recommendations[:10]  # Top 10
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@intelligence_integration_bp.route('/agents/status', methods=['GET'])
def get_agent_status():
    """Get status of all intelligence agents"""
    if not coordinator:
        return jsonify({'error': 'Coordinator not initialized'}), 500
    
    status = coordinator.get_system_status()
    
    # Format for dashboard
    formatted_status = {
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'system': {
            'active_agents': status['active_agents'],
            'total_agents': status['total_agents'],
            'queue_size': status['queue_size'],
            'active_tasks': status['active_tasks']
        },
        'agents': []
    }
    
    # Add detailed agent info
    for agent_id, agent_info in status.get('agent_summary', {}).items():
        agent_details = coordinator.get_agent_status(agent_id)
        if agent_details:
            formatted_status['agents'].append({
                'id': agent_id,
                'name': agent_details['name'],
                'status': agent_details['status'],
                'current_tasks': agent_details['current_tasks'],
                'completed_tasks': agent_details['completed_tasks'],
                'failed_tasks': agent_details['failed_tasks'],
                'workload_score': agent_details['workload_score'],
                'capabilities': agent_details['capabilities']
            })
    
    formatted_status['statistics'] = status.get('statistics', {})
    
    return jsonify(formatted_status)

@intelligence_integration_bp.route('/task/submit', methods=['POST'])
def submit_task():
    """Submit a task to the intelligence system"""
    try:
        data = request.get_json()
        task_type = data.get('task_type')
        description = data.get('description')
        parameters = data.get('parameters', {})
        priority = data.get('priority', 'medium')
        
        if not coordinator:
            return jsonify({'error': 'Coordinator not initialized'}), 500
        
        # Convert priority string to enum
        priority_map = {
            'low': TaskPriority.LOW,
            'medium': TaskPriority.MEDIUM,
            'high': TaskPriority.HIGH,
            'critical': TaskPriority.CRITICAL
        }
        
        task_priority = priority_map.get(priority.lower(), TaskPriority.MEDIUM)
        
        # Submit task
        task_id = coordinator.submit_task(
            task_type=task_type,
            description=description,
            parameters=parameters,
            priority=task_priority
        )
        
        if task_id:
            return jsonify({
                'success': True,
                'task_id': task_id,
                'message': f'Task {task_id} submitted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to submit task - no available agents'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@intelligence_integration_bp.route('/task/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """Get status of a specific task"""
    if not coordinator:
        return jsonify({'error': 'Coordinator not initialized'}), 500
    
    # Check active tasks
    if task_id in coordinator.active_tasks:
        task = coordinator.active_tasks[task_id]
        return jsonify({
            'success': True,
            'task': task.to_dict()
        })
    
    # Check completed tasks
    for task in coordinator.completed_tasks:
        if task.task_id == task_id:
            return jsonify({
                'success': True,
                'task': task.to_dict()
            })
    
    # Check failed tasks
    for task in coordinator.failed_tasks:
        if task.task_id == task_id:
            return jsonify({
                'success': True,
                'task': task.to_dict()
            })
    
    return jsonify({
        'success': False,
        'error': 'Task not found'
    }), 404

@intelligence_integration_bp.route('/metrics/realtime', methods=['GET'])
def get_realtime_metrics():
    """Get real-time intelligence metrics for dashboard"""
    if not coordinator:
        return jsonify({'error': 'Coordinator not initialized'}), 500
    
    system_status = coordinator.get_system_status()
    
    # Format metrics for dashboard charts/graphs
    metrics = {
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'system_health': {
            'status': 'healthy' if system_status['active_agents'] > 0 else 'degraded',
            'active_agents': system_status['active_agents'],
            'total_agents': system_status['total_agents'],
            'queue_size': system_status['queue_size'],
            'active_tasks': system_status['active_tasks']
        },
        'performance': {
            'tasks_completed': system_status['statistics'].get('tasks_completed', 0),
            'tasks_failed': system_status['statistics'].get('tasks_failed', 0),
            'average_completion_time': system_status['statistics'].get('average_completion_time', 0),
            'success_rate': 0
        },
        'agent_workloads': []
    }
    
    # Calculate success rate
    completed = metrics['performance']['tasks_completed']
    failed = metrics['performance']['tasks_failed']
    total = completed + failed
    if total > 0:
        metrics['performance']['success_rate'] = (completed / total) * 100
    
    # Add agent workload data
    for agent_id, agent_info in system_status.get('agent_summary', {}).items():
        agent_details = coordinator.get_agent_status(agent_id)
        if agent_details:
            metrics['agent_workloads'].append({
                'agent_id': agent_id,
                'name': agent_details['name'],
                'workload': agent_details['workload_score'],
                'status': agent_details['status']
            })
    
    return jsonify(metrics)

# Initialize on import
try:
    init_intelligence_integration()
except Exception as e:
    print(f"Warning: Failed to initialize intelligence integration: {e}")

# Export
__all__ = ['intelligence_integration_bp', 'init_intelligence_integration']