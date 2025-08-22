
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
Dashboard Server Module
=======================

Main Flask application factory and server startup.
This replaces the monolithic web_monitor.py file.

Author: TestMaster Team
"""

import logging
import time
import sys
import os
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API blueprints
from api.performance import performance_bp, init_performance_api
from api.analytics import analytics_bp, init_analytics_api
from api.workflow import workflow_bp, init_workflow_api
from api.tests import tests_bp
from api.refactor import refactor_bp, init_refactor_api
from api.llm import llm_bp, init_llm_api
from api.health import HealthCheckAPI
from api.monitoring import MonitoringAPI

# Import new visualization API blueprints
from api.intelligence import IntelligenceAPI
from api.test_generation import TestGenerationAPI
from api.security import SecurityAPI
from api.coverage import CoverageAPI
from api.flow_optimization import FlowOptimizationAPI
from api.quality_assurance import QualityAssuranceAPI
from api.telemetry import TelemetryAPI
from api.async_processing import AsyncProcessingAPI
from api.real_codebase_scanner import CodebaseScanner as RealCodebaseScanner

# Import new crew and swarm orchestration blueprints
from api.crew_orchestration import crew_orchestration_bp
from api.swarm_orchestration import swarm_orchestration_bp

# Import observability integration
from api.observability import observability_bp

# Import intelligence integration
from api.intelligence_integration import intelligence_integration_bp, init_intelligence_integration

# Import NEWTON GRAPH DESTROYER - Knowledge Graph API
from api.knowledge_graph import knowledge_graph_bp, init_knowledge_graph_api

# Import production deployment
from api.production_deployment import production_bp

# Import enhanced telemetry
from api.enhanced_telemetry import enhanced_telemetry_bp

# Import robustness enhancements
from api.backend_health_monitor import health_monitor_bp, health_monitor
from api.frontend_data_contracts import data_contract_bp
from api.enhanced_analytics import enhanced_analytics_bp, analytics_engine

# Import Phase 1 orchestration integration
from api.orchestration_flask import orchestration_bp, init_orchestration_api

# Import Phase 2 multi-agent integration
from api.phase2_api import phase2_bp, init_phase2_api

# Import core modules
from dashboard.dashboard_core.monitor import RealTimeMonitor
from dashboard.dashboard_core.cache import MetricsCache

# Import original modules for compatibility
try:
    from testmaster.core.config import get_config
except ImportError:
    get_config = None

logger = logging.getLogger(__name__)


class DashboardServer:
    """
    Main dashboard server class.
    
    Manages the Flask application and all monitoring components.
    """
    
    def __init__(self, host='0.0.0.0', port=5000):
        """
        Initialize the dashboard server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self.start_time = time.time()
        
        # Core components
        self.monitor = RealTimeMonitor(max_history_points=300, collection_interval=0.1)
        self.cache = MetricsCache(default_ttl=300)
        
        # Initialize metrics feed for real data
        try:
            from dashboard.dashboard_core.metrics_feed import MetricsFeed
            from dashboard.dashboard_core.analytics_aggregator import AnalyticsAggregator
            self.aggregator = AnalyticsAggregator(cache_ttl=30)
            self.metrics_feed = MetricsFeed(self.monitor, self.aggregator, update_interval=0.1)
        except ImportError:
            logger.warning("MetricsFeed or AnalyticsAggregator not available")
            self.metrics_feed = None
            self.aggregator = None
        
        # State for API modules
        self.llm_api_enabled = False
        self.refactor_roadmaps = {}
        
        # Create Flask app
        self.app = self._create_app()
        
        # Start background monitoring
        self._start_background_monitoring()
        
        logger.info(f"Dashboard server initialized on {host}:{port}")
    
    def _create_app(self) -> Flask:
        """
        Create and configure the Flask application.
        
        Returns:
            Configured Flask app
        """
        # Create app with explicit static/template folders  
        import os
        static_path = os.path.join(os.path.dirname(__file__), 'static')
        app = Flask(__name__, 
                   static_folder=static_path,
                   template_folder=static_path)
        
        # Enable CORS
        CORS(app)
        
        # Initialize API modules with dependencies
        init_performance_api(self.monitor, self.cache)
        init_analytics_api(self.aggregator)  # Pass analytics aggregator
        init_workflow_api(None, self.refactor_roadmaps)
        init_refactor_api(None, self.refactor_roadmaps)
        init_llm_api(None, self.llm_api_enabled)
        
        # Register blueprints
        try:
            app.register_blueprint(performance_bp, url_prefix='/api/performance')
            logger.info("Performance blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register performance blueprint: {e}")
            
        try:
            app.register_blueprint(analytics_bp, url_prefix='/api/analytics')
            logger.info("Analytics blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register analytics blueprint: {e}")
            
        try:
            app.register_blueprint(workflow_bp, url_prefix='/api/workflow')
            logger.info("Workflow blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register workflow blueprint: {e}")
            
        try:
            app.register_blueprint(tests_bp, url_prefix='/api/tests')
            logger.info("Tests blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register tests blueprint: {e}")
            
        try:
            app.register_blueprint(refactor_bp, url_prefix='/api/refactor')
            logger.info("Refactor blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register refactor blueprint: {e}")
            
        try:
            app.register_blueprint(llm_bp, url_prefix='/api/llm')
            logger.info("LLM blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register LLM blueprint: {e}")
            
        # Register health check API
        try:
            health_api = HealthCheckAPI(self.aggregator)
            app.register_blueprint(health_api.blueprint)
            logger.info("Health check blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register health check blueprint: {e}")
        
        # Register monitoring API
        try:
            monitoring_api = MonitoringAPI(self.aggregator)
            app.register_blueprint(monitoring_api.blueprint)
            logger.info("Monitoring blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register monitoring blueprint: {e}")
        
        # Register new visualization API blueprints
        try:
            intelligence_api = IntelligenceAPI()
            app.register_blueprint(intelligence_api.blueprint)
            logger.info("Intelligence API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register intelligence blueprint: {e}")
        
        try:
            test_generation_api = TestGenerationAPI()
            app.register_blueprint(test_generation_api.blueprint)
            logger.info("Test Generation API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register test generation blueprint: {e}")
        
        try:
            security_api = SecurityAPI()
            app.register_blueprint(security_api.blueprint)
            logger.info("Security API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register security blueprint: {e}")
        
        try:
            coverage_api = CoverageAPI()
            app.register_blueprint(coverage_api.blueprint)
            logger.info("Coverage API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register coverage blueprint: {e}")
        
        try:
            flow_api = FlowOptimizationAPI()
            app.register_blueprint(flow_api.blueprint)
            logger.info("Flow Optimization API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register flow optimization blueprint: {e}")
        
        try:
            qa_api = QualityAssuranceAPI()
            app.register_blueprint(qa_api.blueprint)
            app.register_blueprint(qa_api.quality_blueprint)
            logger.info("Quality Assurance API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register quality assurance blueprint: {e}")
        
        try:
            telemetry_api = TelemetryAPI()
            app.register_blueprint(telemetry_api.blueprint)
            logger.info("Telemetry API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register telemetry blueprint: {e}")
        
        try:
            async_api = AsyncProcessingAPI()
            app.register_blueprint(async_api.blueprint)
            logger.info("Async Processing API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register async processing blueprint: {e}")
        
        try:
            real_scanner = RealCodebaseScanner()
            app.register_blueprint(real_scanner.blueprint)
            logger.info("Real Codebase Scanner API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register real codebase scanner blueprint: {e}")
        
        # Register crew and swarm orchestration blueprints
        try:
            app.register_blueprint(crew_orchestration_bp, url_prefix='/api/crew')
            logger.info("Crew Orchestration API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register crew orchestration blueprint: {e}")
        
        try:
            app.register_blueprint(swarm_orchestration_bp, url_prefix='/api/swarm')
            logger.info("Swarm Orchestration API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register swarm orchestration blueprint: {e}")
        
        # Register observability blueprint
        try:
            app.register_blueprint(observability_bp, url_prefix='/api/observability')
            logger.info("Observability API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register observability blueprint: {e}")
        
        # Register NEWTON GRAPH DESTROYER - Knowledge Graph API
        try:
            init_knowledge_graph_api()  # Initialize the API
            app.register_blueprint(knowledge_graph_bp, url_prefix='/api/knowledge-graph')
            logger.info("Knowledge Graph API registered - COMPETITORS OBLITERATED!")
        except Exception as e:
            logger.error(f"Failed to register knowledge graph blueprint: {e}")
        
        # Register production deployment blueprint
        try:
            app.register_blueprint(production_bp, url_prefix='/api/production')
            logger.info("Production Deployment API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register production deployment blueprint: {e}")
        
        # Register enhanced telemetry blueprint
        try:
            app.register_blueprint(enhanced_telemetry_bp, url_prefix='/api/telemetry')
            logger.info("Enhanced Telemetry API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register enhanced telemetry blueprint: {e}")
        
        # Register robustness enhancement blueprints
        try:
            app.register_blueprint(health_monitor_bp, url_prefix='/api/health-monitor')
            logger.info("Backend Health Monitor API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register backend health monitor blueprint: {e}")
        
        try:
            app.register_blueprint(data_contract_bp, url_prefix='/api/data-contract')
            logger.info("Frontend Data Contracts API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register frontend data contracts blueprint: {e}")
        
        try:
            app.register_blueprint(enhanced_analytics_bp, url_prefix='/api/analytics')
            logger.info("Enhanced Analytics API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register enhanced analytics blueprint: {e}")
        
        # Register Phase 1 orchestration blueprint
        try:
            init_orchestration_api()
            app.register_blueprint(orchestration_bp, url_prefix='/api/orchestration')
            logger.info("Phase 1 Orchestration API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register orchestration blueprint: {e}")
        
        # Register Phase 2 multi-agent blueprint
        try:
            init_phase2_api()
            app.register_blueprint(phase2_bp, url_prefix='/api/phase2')
            logger.info("Phase 2 Multi-Agent API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register Phase 2 blueprint: {e}")
        
        # Register Intelligence Integration blueprint
        try:
            init_intelligence_integration()
            app.register_blueprint(intelligence_integration_bp)
            logger.info("Intelligence Integration API blueprint registered successfully")
        except Exception as e:
            logger.error(f"Failed to register Intelligence Integration blueprint: {e}")
        
        # Main routes
        @app.route('/')
        def dashboard():
            """Serve the main dashboard."""
            import os
            # Force reading the correct modular dashboard
            static_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
            logger.info(f"Attempting to serve modular dashboard from: {static_path}")
            logger.info(f"File exists: {os.path.exists(static_path)}")
            logger.info(f"Flask static folder: {app.static_folder}")
            
            try:
                if os.path.exists(static_path):
                    with open(static_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logger.info(f"Successfully read modular dashboard ({len(content)} chars)")
                        logger.info(f"Title check: {'TestMaster Dashboard v2.0' in content}")
                        return content
                else:
                    logger.error(f"Static index.html not found at: {static_path}")
                    return jsonify({"error": "Dashboard not available", "static_path": static_path}), 503
            except Exception as e:
                logger.error(f"Failed to serve modular dashboard: {e}")
                return jsonify({"error": "Dashboard serving failed", "details": str(e)}), 503

        @app.route('/css/<path:filename>')
        def serve_css(filename):
            """Serve CSS files."""
            import os
            css_path = os.path.join(os.path.dirname(__file__), 'static', 'css')
            return send_from_directory(css_path, filename)
        
        @app.route('/js/<path:filename>')
        def serve_js(filename):
            """Serve JavaScript files."""
            import os
            js_path = os.path.join(os.path.dirname(__file__), 'static', 'js')
            return send_from_directory(js_path, filename)
        
        @app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self.start_time,
                'monitoring_active': self.monitor.running,
                'version': '2.0.0'
            })
        
        @app.route('/api/config')
        def get_config_endpoint():
            """Get configuration information."""
            try:
                if get_config:
                    config = get_config()
                    return jsonify({
                        'status': 'success',
                        'active_profile': getattr(config, '_active_profile', 'unknown'),
                        'configuration_available': True
                    })
                else:
                    return jsonify({
                        'status': 'success',
                        'active_profile': 'development',
                        'configuration_available': False,
                        'message': 'Configuration module not available'
                    })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500

        @app.route('/api/debug/routes')
        def debug_routes():
            """Debug endpoint to list all registered routes."""
            routes = []
            for rule in app.url_map.iter_rules():
                routes.append({
                    'endpoint': rule.endpoint,
                    'methods': list(rule.methods),
                    'rule': rule.rule
                })
            
            return jsonify({
                'status': 'success',
                'total_routes': len(routes),
                'routes': sorted(routes, key=lambda x: x['rule'])
            })
        
        # Error handlers
        @app.errorhandler(404)
        def not_found(error):
            return jsonify({'status': 'error', 'error': 'Endpoint not found'}), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal error: {error}")
            return jsonify({'status': 'error', 'error': 'Internal server error'}), 500
        
        return app
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring components."""
        try:
            # Start real-time monitoring
            self.monitor.start_monitoring()
            logger.info("Background monitoring started")
            
            # Initialize default codebase
            self.monitor.register_codebase('/testmaster')
            
            # Start metrics feed if available
            if self.metrics_feed:
                self.metrics_feed.start()
                logger.info("Metrics feed started")
            
        except Exception as e:
            logger.error(f"Failed to start background monitoring: {e}")
    
    def run(self, debug=False) -> None:
        """
        Run the Flask development server.
        
        Args:
            debug: Enable debug mode
        """
        logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        try:
            self.app.run(host=self.host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Shutdown the server and cleanup resources."""
        logger.info("Shutting down dashboard server")
        
        try:
            if self.metrics_feed:
                self.metrics_feed.stop()
                
            if self.monitor:
                self.monitor.stop_monitoring()
            
            if self.cache:
                self.cache.clear()
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def create_app():
    """
    Application factory function.
    
    Returns:
        Flask application instance
    """
    server = DashboardServer()
    return server.app


def main():
    """Main entry point for command line usage."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run server
    server = DashboardServer()
    
    try:
        server.run(debug=True)
    except KeyboardInterrupt:
        print("\\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())