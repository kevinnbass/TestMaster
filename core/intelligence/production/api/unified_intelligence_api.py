
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
Unified Intelligence API System
Exposes all TestMaster intelligence capabilities to the frontend
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from ..analysis.technical_debt_analyzer import TechnicalDebtAnalyzer
from ..analysis.ml_code_analyzer import MLCodeAnalyzer
from ..orchestration.agent_coordinator import AgentCoordinator, IntelligenceAgent, AgentCapability, TaskPriority
from ...analysis.coverage.interface import UnifiedCoverageAnalyzer
from ..documentation.templates.api.manager import ApiTemplateManager


class IntelligenceAPI:
    """Unified Intelligence API for frontend integration"""
    
    def __init__(self, port: int = 5000):
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.port = port
        
        # Initialize intelligence components
        self.debt_analyzer = TechnicalDebtAnalyzer()
        self.ml_analyzer = MLCodeAnalyzer()
        self.coverage_analyzer = UnifiedCoverageAnalyzer()
        self.api_manager = ApiTemplateManager()
        self.coordinator = AgentCoordinator()
        
        # Analytics storage
        self.analytics_cache = {}
        self.real_time_metrics = {
            'active_analyses': 0,
            'completed_analyses': 0,
            'total_issues_found': 0,
            'average_analysis_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self._setup_routes()
        self._setup_websocket_events()
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize intelligence agents"""
        # Technical Debt Agent
        debt_agent = IntelligenceAgent(
            agent_id="debt_agent",
            name="Technical Debt Analyzer",
            capabilities=[
                AgentCapability(
                    name="debt_analysis",
                    description="Analyze technical debt in codebases",
                    task_types=["analyze_debt", "quantify_debt", "remediation_plan"],
                    max_concurrent_tasks=2,
                    estimated_task_time=30.0,
                    specialized_domains=["code_quality", "maintainability"]
                )
            ]
        )
        
        # ML Analysis Agent
        ml_agent = IntelligenceAgent(
            agent_id="ml_agent", 
            name="ML Code Analyzer",
            capabilities=[
                AgentCapability(
                    name="ml_analysis",
                    description="Analyze ML/AI code for issues",
                    task_types=["analyze_ml", "check_tensor_shapes", "model_architecture"],
                    max_concurrent_tasks=1,
                    estimated_task_time=45.0,
                    specialized_domains=["machine_learning", "deep_learning", "data_science"]
                )
            ]
        )
        
        # Coverage Agent
        coverage_agent = IntelligenceAgent(
            agent_id="coverage_agent",
            name="Coverage Analyzer", 
            capabilities=[
                AgentCapability(
                    name="coverage_analysis",
                    description="Analyze code coverage and quality",
                    task_types=["analyze_coverage", "generate_report", "identify_gaps"],
                    max_concurrent_tasks=3,
                    estimated_task_time=20.0,
                    specialized_domains=["testing", "quality_assurance"]
                )
            ]
        )
        
        # Register agents
        self.coordinator.register_agent(debt_agent)
        self.coordinator.register_agent(ml_agent)
        self.coordinator.register_agent(coverage_agent)
        
        # Start coordination
        self.coordinator.start_coordination()
    
    def _setup_routes(self):
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
                        TaskPriority.HIGH
                    )
                    if task_id:
                        task_ids.append(task_id)
                
                if 'ml' in analysis_types or 'all' in analysis_types:
                    task_id = self.coordinator.submit_task(
                        'analyze_ml',
                        f'Analyze ML code for {project_path}',
                        {'project_path': project_path},
                        TaskPriority.HIGH
                    )
                    if task_id:
                        task_ids.append(task_id)
                
                if 'coverage' in analysis_types or 'all' in analysis_types:
                    task_id = self.coordinator.submit_task(
                        'analyze_coverage',
                        f'Analyze code coverage for {project_path}',
                        {'project_path': project_path},
                        TaskPriority.MEDIUM
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
                
                from ..documentation.templates.api.base import ApiType, ApiContext
                
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
    
    def _setup_websocket_events(self):
        """Setup WebSocket events for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.logger.info("Client connected to intelligence API")
            emit('connected', {
                'status': 'connected',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info("Client disconnected from intelligence API")
        
        @self.socketio.on('subscribe_metrics')
        def handle_subscribe_metrics():
            """Subscribe to real-time metrics updates"""
            def send_metrics():
                while True:
                    self.socketio.emit('metrics_update', {
                        'metrics': self.real_time_metrics,
                        'agents': self.coordinator.get_system_status(),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.socketio.sleep(2)  # Update every 2 seconds
            
            self.socketio.start_background_task(send_metrics)
        
        @self.socketio.on('start_analysis')
        def handle_start_analysis(data):
            """Handle analysis request via WebSocket"""
            project_path = data.get('project_path', '.')
            analysis_type = data.get('analysis_type', 'comprehensive')
            
            # Submit task
            task_id = self.coordinator.submit_task(
                f'analyze_{analysis_type}',
                f'WebSocket analysis: {analysis_type}',
                {'project_path': project_path},
                TaskPriority.HIGH
            )
            
            emit('analysis_started', {
                'task_id': task_id,
                'analysis_type': analysis_type
            })
    
    def run(self, debug: bool = False, host: str = '0.0.0.0'):
        """Run the intelligence API server"""
        self.logger.info(f"Starting Intelligence API server on {host}:{self.port}")
        
        # Register event callbacks for real-time updates
        def on_task_completed(task, agent):
            self.socketio.emit('task_completed', {
                'task_id': task.task_id,
                'agent_id': agent.agent_id,
                'result': task.result
            })
        
        def on_task_failed(task, agent):
            self.socketio.emit('task_failed', {
                'task_id': task.task_id,
                'agent_id': agent.agent_id,
                'error': task.error
            })
        
        self.coordinator.register_event_callback('task_completed', on_task_completed)
        self.coordinator.register_event_callback('task_failed', on_task_failed)
        
        try:
            self.socketio.run(self.app, host=host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            self.logger.info("Shutting down Intelligence API server")
            self.coordinator.stop_coordination()
    
    def stop(self):
        """Stop the intelligence API"""
        self.coordinator.stop_coordination()
        self.logger.info("Intelligence API stopped")


def create_intelligence_api(port: int = 5000) -> IntelligenceAPI:
    """Factory function to create Intelligence API instance"""
    return IntelligenceAPI(port=port)


# Export
__all__ = ['IntelligenceAPI', 'create_intelligence_api']