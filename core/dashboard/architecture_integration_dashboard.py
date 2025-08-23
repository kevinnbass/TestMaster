#!/usr/bin/env python3
"""
Architecture Integration Dashboard - Agent A Hour 6
Dashboard integration at localhost:5000 for architecture monitoring

Integrates architecture health monitoring with the existing dashboard
infrastructure following ADAMANTIUMCLAD protocol requirements.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import Flask, render_template, jsonify, request
from threading import Thread
import time

# Import architecture components
from core.architecture.architecture_integration import get_architecture_framework
from core.services.service_registry import get_service_registry
from web.realtime.websocket_architecture_stream import get_websocket_stream
from core.templates.template_migration_system import get_migration_system
from core.intelligence.production_activator import get_production_activator


class ArchitectureDashboard:
    """
    Architecture Integration Dashboard
    
    Provides web interface for architecture monitoring and management
    at localhost:5000 following ADAMANTIUMCLAD frontend-first protocol.
    """
    
    def __init__(self, port: int = 5000, debug: bool = False):
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.debug = debug
        
        # Architecture components
        self.framework = get_architecture_framework()
        self.registry = get_service_registry()
        self.stream = get_websocket_stream()
        self.migration = get_migration_system()
        self.activator = get_production_activator()
        
        # Flask application
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'architecture-dashboard-key'
        
        # Dashboard state
        self.dashboard_metrics = {}
        self.last_update = datetime.now()
        self.export_formats = ['json', 'png', 'svg']
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info(f"Architecture Dashboard initialized on port {port}")
    
    def _setup_routes(self):
        """Setup Flask routes for dashboard"""
        
        @self.app.route('/')
        def dashboard_home():
            """Main dashboard page"""
            try:
                metrics = self._get_dashboard_metrics()
                return render_template('architecture_dashboard.html', 
                                     metrics=metrics,
                                     timestamp=datetime.now().isoformat())
            except Exception as e:
                self.logger.error(f"Dashboard home error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/health')
        def api_health():
            """Architecture health API endpoint"""
            try:
                health = self.framework.validate_architecture()
                return jsonify({
                    'status': 'operational',
                    'overall_health': health.overall_health,
                    'layer_compliance': health.layer_compliance,
                    'dependency_health': health.dependency_health,
                    'import_success_rate': health.import_success_rate,
                    'recommendations': health.recommendations,
                    'timestamp': health.timestamp.isoformat()
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/services')
        def api_services():
            """Services registry API endpoint"""
            try:
                report = self.registry.get_registration_report()
                return jsonify({
                    'status': 'success',
                    'data': report
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/templates')
        def api_templates():
            """Template migration status API endpoint"""
            try:
                report = self.migration.get_migration_report()
                return jsonify({
                    'status': 'success',
                    'data': report
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/intelligence')
        def api_intelligence():
            """Intelligence modules status API endpoint"""
            try:
                report = self.activator.get_activation_report()
                return jsonify({
                    'status': 'success',
                    'data': report
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/websocket-status')
        def api_websocket_status():
            """WebSocket stream status API endpoint"""
            try:
                metrics = self.stream.get_stream_metrics()
                return jsonify({
                    'status': 'success',
                    'data': metrics
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/comprehensive-status')
        def api_comprehensive_status():
            """Comprehensive system status API endpoint"""
            try:
                metrics = self._get_comprehensive_metrics()
                return jsonify({
                    'status': 'success',
                    'data': metrics,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export/<format>')
        def api_export(format):
            """Export architecture data in specified format"""
            try:
                if format not in self.export_formats:
                    return jsonify({'error': f'Unsupported format: {format}'}), 400
                
                data = self._get_comprehensive_metrics()
                
                if format == 'json':
                    return jsonify(data)
                elif format == 'png':
                    # Generate PNG visualization
                    png_data = self._generate_png_visualization(data)
                    return png_data, 200, {'Content-Type': 'image/png'}
                elif format == 'svg':
                    # Generate SVG visualization
                    svg_data = self._generate_svg_visualization(data)
                    return svg_data, 200, {'Content-Type': 'image/svg+xml'}
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/generate-report')
        def api_generate_report():
            """Generate comprehensive architecture report"""
            try:
                report = self._generate_comprehensive_report()
                return jsonify({
                    'status': 'success',
                    'report': report,
                    'generated_at': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        try:
            # Get architecture health
            health = self.framework.validate_architecture()
            
            # Get service status
            service_report = self.registry.get_registration_report()
            
            # Get template status
            template_report = self.migration.get_migration_report()
            
            # Get intelligence status
            intelligence_report = self.activator.get_activation_report()
            
            # Get WebSocket status
            stream_metrics = self.stream.get_stream_metrics()
            
            metrics = {
                'health': {
                    'overall': health.overall_health * 100,
                    'layer_compliance': health.layer_compliance * 100,
                    'dependency_health': health.dependency_health * 100,
                    'import_success_rate': health.import_success_rate * 100,
                    'status': 'excellent' if health.overall_health > 0.8 else 'good' if health.overall_health > 0.6 else 'needs_attention'
                },
                'services': {
                    'total_registered': service_report['summary']['successful'],
                    'success_rate': service_report['summary']['success_rate'] * 100,
                    'by_layer': service_report['by_layer'],
                    'by_type': service_report['by_type']
                },
                'templates': {
                    'discovered': template_report['templates_discovered'],
                    'migrated': template_report['migrations_successful'],
                    'success_rate': template_report['success_rate'] * 100,
                    'generators_created': template_report['categories_processed']
                },
                'intelligence': {
                    'modules_discovered': intelligence_report['summary']['modules_discovered'],
                    'modules_activated': intelligence_report['summary']['modules_activated'],
                    'success_rate': intelligence_report['summary']['success_rate'] * 100,
                    'services_registered': intelligence_report['summary']['services_registered']
                },
                'websocket': {
                    'available': stream_metrics['websockets_available'],
                    'running': stream_metrics['running'],
                    'port': stream_metrics['port'],
                    'clients_connected': stream_metrics['clients_connected']
                },
                'recommendations': health.recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
            self.dashboard_metrics = metrics
            self.last_update = datetime.now()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics for export"""
        try:
            dashboard_metrics = self._get_dashboard_metrics()
            
            # Add integration status
            integration_report = self.framework.get_integration_report()
            
            comprehensive = {
                'dashboard_metrics': dashboard_metrics,
                'integration_status': integration_report,
                'system_info': {
                    'architecture_framework': 'operational',
                    'total_services': len(self.registry.service_definitions),
                    'total_templates': len(self.migration.discovered_templates),
                    'total_intelligence_modules': len(self.activator.activation_results),
                    'websocket_streaming': dashboard_metrics['websocket']['available']
                },
                'performance_summary': {
                    'architecture_health': dashboard_metrics['health']['overall'],
                    'service_success_rate': dashboard_metrics['services']['success_rate'],
                    'template_migration_rate': dashboard_metrics['templates']['success_rate'],
                    'intelligence_activation_rate': dashboard_metrics['intelligence']['success_rate']
                },
                'export_metadata': {
                    'generated_by': 'Agent A Architecture Dashboard',
                    'version': '1.0.0',
                    'export_time': datetime.now().isoformat(),
                    'data_freshness': (datetime.now() - self.last_update).total_seconds()
                }
            }
            
            return comprehensive
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive metrics: {e}")
            return {'error': str(e)}
    
    def _generate_png_visualization(self, data: Dict[str, Any]) -> bytes:
        """Generate PNG visualization of architecture data"""
        try:
            # Simple PNG generation - in production would use matplotlib or similar
            import base64
            
            # Create a simple visualization data structure
            viz_data = json.dumps(data, indent=2)
            
            # For now, return a base64-encoded placeholder
            # In production, this would generate actual charts/graphs
            placeholder = f"Architecture Visualization - {datetime.now().isoformat()}"
            encoded = base64.b64encode(placeholder.encode()).decode()
            
            return base64.b64decode(encoded + "==")
            
        except Exception as e:
            self.logger.error(f"PNG generation error: {e}")
            return b"PNG generation failed"
    
    def _generate_svg_visualization(self, data: Dict[str, Any]) -> str:
        """Generate SVG visualization of architecture data"""
        try:
            health = data['dashboard_metrics']['health']['overall']
            services = data['dashboard_metrics']['services']['total_registered']
            
            svg = f"""
            <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
                <rect width="800" height="600" fill="#f5f5f5"/>
                <text x="400" y="50" text-anchor="middle" font-size="24" font-weight="bold">Architecture Status</text>
                
                <!-- Health Circle -->
                <circle cx="200" cy="200" r="80" fill="none" stroke="#ddd" stroke-width="10"/>
                <circle cx="200" cy="200" r="80" fill="none" stroke="#4CAF50" stroke-width="10"
                        stroke-dasharray="{health * 502.65} 502.65" transform="rotate(-90 200 200)"/>
                <text x="200" y="210" text-anchor="middle" font-size="20" font-weight="bold">{health:.1f}%</text>
                <text x="200" y="300" text-anchor="middle" font-size="14">Health</text>
                
                <!-- Services Bar -->
                <rect x="350" y="160" width="200" height="40" fill="#ddd"/>
                <rect x="350" y="160" width="{services * 4}" height="40" fill="#2196F3"/>
                <text x="450" y="185" text-anchor="middle" fill="white" font-weight="bold">{services} Services</text>
                
                <!-- Timestamp -->
                <text x="400" y="550" text-anchor="middle" font-size="12" fill="#666">
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </text>
            </svg>
            """
            
            return svg.strip()
            
        except Exception as e:
            self.logger.error(f"SVG generation error: {e}")
            return "<svg><text>SVG generation failed</text></svg>"
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive architecture report"""
        try:
            metrics = self._get_comprehensive_metrics()
            
            report = {
                'executive_summary': {
                    'system_status': 'operational',
                    'architecture_health': f"{metrics['dashboard_metrics']['health']['overall']:.1f}%",
                    'total_services': metrics['system_info']['total_services'],
                    'key_achievements': [
                        f"{metrics['dashboard_metrics']['services']['total_registered']} services registered",
                        f"{metrics['dashboard_metrics']['templates']['migrated']} templates migrated",
                        f"{metrics['dashboard_metrics']['intelligence']['modules_activated']} intelligence modules active"
                    ]
                },
                'detailed_metrics': metrics,
                'recommendations': metrics['dashboard_metrics']['recommendations'],
                'next_steps': [
                    'Continue performance optimization',
                    'Expand intelligence module integration',
                    'Enhance real-time monitoring capabilities'
                ],
                'report_metadata': {
                    'generated_by': 'Agent A Architecture Dashboard',
                    'report_type': 'comprehensive_architecture_status',
                    'generation_time': datetime.now().isoformat(),
                    'data_sources': ['architecture_framework', 'service_registry', 'template_system', 'intelligence_platform']
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return {'error': str(e)}
    
    def create_dashboard_template(self) -> str:
        """Create HTML template for dashboard"""
        template_dir = Path('templates')
        template_dir.mkdir(exist_ok=True)
        
        template_path = template_dir / 'architecture_dashboard.html'
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Architecture Dashboard - Agent A</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { text-align: center; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333; }
        .metric-value { font-size: 36px; font-weight: bold; margin: 10px 0; }
        .metric-value.excellent { color: #4CAF50; }
        .metric-value.good { color: #FFC107; }
        .metric-value.needs_attention { color: #FF9800; }
        .progress-bar { width: 100%; height: 10px; background: #ddd; border-radius: 5px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s ease; }
        .recommendations { background: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 20px; }
        .timestamp { text-align: center; color: #666; font-size: 14px; margin-top: 30px; }
        .export-links { text-align: center; margin: 20px 0; }
        .export-links a { margin: 0 10px; padding: 8px 16px; background: #2196F3; color: white; text-decoration: none; border-radius: 4px; }
        .export-links a:hover { background: #1976D2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Architecture Dashboard</h1>
        <p>Agent A - Real-time Architecture Monitoring</p>
    </div>
    
    <div class="export-links">
        <a href="/api/export/json" target="_blank">Export JSON</a>
        <a href="/api/export/svg" target="_blank">Export SVG</a>
        <a href="/api/generate-report" target="_blank">Generate Report</a>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">Architecture Health</div>
            <div class="metric-value {{ metrics.health.status }}">{{ "%.1f"|format(metrics.health.overall) }}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ metrics.health.overall }}%"></div>
            </div>
            <div>Layer Compliance: {{ "%.1f"|format(metrics.health.layer_compliance) }}%</div>
            <div>Dependency Health: {{ "%.1f"|format(metrics.health.dependency_health) }}%</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Services Registry</div>
            <div class="metric-value excellent">{{ metrics.services.total_registered }}</div>
            <div>Success Rate: {{ "%.1f"|format(metrics.services.success_rate) }}%</div>
            <div>By Layer: {{ metrics.services.by_layer|length }} layers</div>
            <div>By Type: {{ metrics.services.by_type|length }} types</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Template System</div>
            <div class="metric-value excellent">{{ metrics.templates.migrated }}</div>
            <div>Templates Discovered: {{ metrics.templates.discovered }}</div>
            <div>Migration Rate: {{ "%.1f"|format(metrics.templates.success_rate) }}%</div>
            <div>Generators Created: {{ metrics.templates.generators_created }}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Intelligence Platform</div>
            <div class="metric-value excellent">{{ metrics.intelligence.modules_activated }}</div>
            <div>Modules Discovered: {{ metrics.intelligence.modules_discovered }}</div>
            <div>Activation Rate: {{ "%.1f"|format(metrics.intelligence.success_rate) }}%</div>
            <div>Services Registered: {{ metrics.intelligence.services_registered }}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">WebSocket Streaming</div>
            <div class="metric-value {{ 'excellent' if metrics.websocket.available else 'needs_attention' }}">
                {{ 'Available' if metrics.websocket.available else 'Unavailable' }}
            </div>
            <div>Status: {{ 'Running' if metrics.websocket.running else 'Stopped' }}</div>
            <div>Port: {{ metrics.websocket.port }}</div>
            <div>Clients: {{ metrics.websocket.clients_connected }}</div>
        </div>
    </div>
    
    {% if metrics.recommendations %}
    <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
            {% for rec in metrics.recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
    
    <div class="timestamp">
        Last updated: {{ timestamp }}
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            window.location.reload();
        }, 30000);
    </script>
</body>
</html>
        """
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content.strip())
        
        self.logger.info(f"Dashboard template created: {template_path}")
        return str(template_path)
    
    def run(self, host='localhost'):
        """Run the dashboard server"""
        try:
            # Create template if it doesn't exist
            self.create_dashboard_template()
            
            self.logger.info(f"Starting Architecture Dashboard on http://{host}:{self.port}")
            self.app.run(host=host, port=self.port, debug=self.debug, threaded=True)
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            raise
    
    def run_in_background(self):
        """Run dashboard in background thread"""
        def run_server():
            self.run()
        
        thread = Thread(target=run_server, daemon=True)
        thread.start()
        self.logger.info("Dashboard started in background thread")
        return thread


# Global dashboard instance
_dashboard_instance: Optional[ArchitectureDashboard] = None


def get_architecture_dashboard(port: int = 5000) -> ArchitectureDashboard:
    """Get global architecture dashboard instance"""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = ArchitectureDashboard(port=port)
    return _dashboard_instance


def start_architecture_dashboard(port: int = 5000, background: bool = True):
    """Start architecture dashboard"""
    dashboard = get_architecture_dashboard(port)
    
    if background:
        return dashboard.run_in_background()
    else:
        dashboard.run()