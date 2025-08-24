#!/usr/bin/env python3
"""
🏗️ MODULE: REST API Routes - Intelligence API Endpoints
==================================================================

📋 PURPOSE:
    REST API route handlers and endpoint logic extracted
    from unified_intelligence_api.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Health check endpoint
    • Project analysis endpoints
    • Technical debt analysis
    • ML code analysis
    • Coverage analysis
    • Task status monitoring
    • Real-time metrics
    • Documentation generation

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract REST API routes from unified_intelligence_api.py
   └─ Source: Lines 141-392 (251 lines)
   └─ Purpose: Separate API route handling into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: Flask, datetime, logging, task coordination
📤 Provides: REST API endpoint handlers and logic
"""

from flask import request, jsonify
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RestApiRoutes:
    """Handles REST API route setup and request processing."""
    
    def __init__(self, app, coordinator, debt_analyzer, ml_analyzer, 
                 coverage_analyzer, api_manager, analytics_cache, real_time_metrics):
        self.app = app
        self.coordinator = coordinator
        self.debt_analyzer = debt_analyzer
        self.ml_analyzer = ml_analyzer
        self.coverage_analyzer = coverage_analyzer
        self.api_manager = api_manager
        self.analytics_cache = analytics_cache
        self.real_time_metrics = real_time_metrics
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup REST API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """System health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'agents': len(self.coordinator.agents),
                'active_tasks': len(self.coordinator.active_tasks)
            })
        
        @self.app.route('/api/intelligence/analyze', methods=['POST'])
        def analyze_project():
            """Comprehensive project analysis"""
            try:
                data = request.get_json()
                project_path = data.get('project_path', '.')
                analysis_types = data.get('analysis_types', ['all'])
                
                # Submit analysis tasks
                task_ids = []
                
                if 'debt' in analysis_types or 'all' in analysis_types:
                    task_id = self.coordinator.submit_task(
                        'analyze_debt',
                        f'Analyze technical debt for {project_path}',
                        {'project_path': project_path},
                        'HIGH'  # TaskPriority.HIGH equivalent
                    )
                    if task_id:
                        task_ids.append(task_id)
                
                if 'ml' in analysis_types or 'all' in analysis_types:
                    task_id = self.coordinator.submit_task(
                        'analyze_ml',
                        f'Analyze ML code for {project_path}',
                        {'project_path': project_path},
                        'HIGH'
                    )
                    if task_id:
                        task_ids.append(task_id)
                
                if 'coverage' in analysis_types or 'all' in analysis_types:
                    task_id = self.coordinator.submit_task(
                        'analyze_coverage',
                        f'Analyze code coverage for {project_path}',
                        {'project_path': project_path},
                        'MEDIUM'
                    )
                    if task_id:
                        task_ids.append(task_id)
                
                return jsonify({
                    'status': 'analysis_started',
                    'task_ids': task_ids,
                    'message': f'Started {len(task_ids)} analysis tasks'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/intelligence/debt/analyze', methods=['POST'])
        def analyze_technical_debt():
            """Analyze technical debt"""
            try:
                data = request.get_json()
                project_path = data.get('project_path', '.')
                
                result = self.debt_analyzer.analyze_project(project_path)
                
                # Cache result
                analysis_id = f"debt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.analytics_cache[analysis_id] = result
                
                # Update metrics
                self.real_time_metrics['completed_analyses'] += 1
                if 'debt_items' in result:
                    self.real_time_metrics['total_issues_found'] += len(result['debt_items'])
                
                return jsonify({
                    'analysis_id': analysis_id,
                    'result': result
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/intelligence/ml/analyze', methods=['POST'])
        def analyze_ml_code():
            """Analyze ML/AI code"""
            try:
                data = request.get_json()
                project_path = data.get('project_path', '.')
                
                result = self.ml_analyzer.analyze_project(project_path)
                
                # Cache result
                analysis_id = f"ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.analytics_cache[analysis_id] = result
                
                # Update metrics
                self.real_time_metrics['completed_analyses'] += 1
                if 'analysis_result' in result and 'issues' in result['analysis_result']:
                    self.real_time_metrics['total_issues_found'] += len(result['analysis_result']['issues'])
                
                return jsonify({
                    'analysis_id': analysis_id,
                    'result': result
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/intelligence/coverage/analyze', methods=['POST'])
        def analyze_coverage():
            """Analyze code coverage"""
            try:
                data = request.get_json()
                project_path = data.get('project_path', '.')
                
                result = self.coverage_analyzer.analyze_coverage(project_path)
                
                # Cache result  
                analysis_id = f"coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.analytics_cache[analysis_id] = result.__dict__ if hasattr(result, '__dict__') else result
                
                # Update metrics
                self.real_time_metrics['completed_analyses'] += 1
                
                return jsonify({
                    'analysis_id': analysis_id,
                    'result': result.__dict__ if hasattr(result, '__dict__') else result
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/intelligence/comprehensive', methods=['POST'])
        def comprehensive_analysis():
            """Run comprehensive analysis combining all intelligence"""
            try:
                data = request.get_json()
                project_path = data.get('project_path', '.')
                
                # Run all analyses
                debt_result = self.debt_analyzer.analyze_project(project_path)
                ml_result = self.ml_analyzer.analyze_project(project_path)
                coverage_result = self.coverage_analyzer.generate_comprehensive_report(project_path)
                
                # Combine results
                comprehensive_result = {
                    'technical_debt': debt_result,
                    'ml_analysis': ml_result,
                    'coverage_analysis': coverage_result,
                    'summary': {
                        'total_debt_hours': debt_result.get('metrics', {}).get('total_debt_hours', 0),
                        'ml_frameworks_detected': len(ml_result.get('analysis_result', {}).get('frameworks_detected', [])),
                        'coverage_percentage': getattr(coverage_result.get('coverage', {}), 'total_coverage', 0),
                        'total_issues': (
                            len(debt_result.get('debt_items', [])) + 
                            len(ml_result.get('analysis_result', {}).get('issues', []))
                        )
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache result
                analysis_id = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.analytics_cache[analysis_id] = comprehensive_result
                
                return jsonify({
                    'analysis_id': analysis_id,
                    'result': comprehensive_result
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/intelligence/agents/status', methods=['GET'])
        def get_agents_status():
            """Get status of all intelligence agents"""
            return jsonify(self.coordinator.get_system_status())
        
        @self.app.route('/api/intelligence/tasks/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            """Get status of specific task"""
            if task_id in self.coordinator.active_tasks:
                task = self.coordinator.active_tasks[task_id]
                return jsonify(task.to_dict())
            
            # Check completed tasks
            for task in self.coordinator.completed_tasks:
                if task.task_id == task_id:
                    return jsonify(task.to_dict())
            
            # Check failed tasks
            for task in self.coordinator.failed_tasks:
                if task.task_id == task_id:
                    return jsonify(task.to_dict())
            
            return jsonify({'error': 'Task not found'}), 404
        
        @self.app.route('/api/intelligence/analytics/<analysis_id>', methods=['GET'])
        def get_analysis_result(analysis_id):
            """Get cached analysis result"""
            if analysis_id in self.analytics_cache:
                return jsonify(self.analytics_cache[analysis_id])
            return jsonify({'error': 'Analysis not found'}), 404
        
        @self.app.route('/api/intelligence/metrics', methods=['GET'])
        def get_real_time_metrics():
            """Get real-time system metrics"""
            return jsonify({
                'metrics': self.real_time_metrics,
                'system_status': self.coordinator.get_system_status(),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/intelligence/documentation/generate', methods=['POST'])
        def generate_documentation():
            """Generate API documentation"""
            try:
                data = request.get_json()
                api_type = data.get('api_type', 'rest')
                template_name = data.get('template_name', 'comprehensive')
                context = data.get('context', {})
                
                from ...documentation.templates.api.base import ApiType, ApiContext
                
                # Convert string to enum
                api_type_enum = ApiType(api_type)
                
                # Create context object
                api_context = ApiContext(**context)
                
                # Generate documentation
                documentation = self.api_manager.generate_documentation(
                    api_type_enum, template_name, api_context
                )
                
                return jsonify({
                    'documentation': documentation,
                    'api_type': api_type,
                    'template_name': template_name
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500