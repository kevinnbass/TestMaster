#!/usr/bin/env python3
"""
🏗️ MODULE: Gamma Dashboard Enhancements - Core Dashboard Logic
==================================================================

📋 PURPOSE:
    Core dashboard class with routing, integration, and WebSocket handling
    for the Enhanced Unified Gamma Dashboard. Extracted via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Main dashboard class with initialization
    • Flask route setup and API endpoints
    • WebSocket event configuration
    • Agent E personal analytics integration
    • Backend service coordination

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract core dashboard logic from unified_gamma_dashboard_enhanced.py
   └─ Source: Lines 91-441 (350 lines)
   └─ Purpose: Reduce main file size while preserving 100% functionality

📞 DEPENDENCIES:
==================================================================
🤝 Imports: performance_enhancements (APIUsageTracker, DataIntegrator, PerformanceMonitor)
🤝 Imports: ui_enhancements (ENHANCED_DASHBOARD_HTML)
📤 Provides: EnhancedUnifiedDashboard class for main module
"""

import os
import sys
import json
import time
import threading
import requests
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import psutil
import random

# Import extracted modules
from .performance_enhancements import APIUsageTracker, DataIntegrator, PerformanceMonitor
from .ui_enhancements import ENHANCED_DASHBOARD_HTML

# Agent E Personal Analytics Integration (when available)
try:
    from personal_analytics_service import (
        PersonalAnalyticsService,
        register_personal_analytics_endpoints,
        register_socketio_handlers
    )
    AGENT_E_INTEGRATION_AVAILABLE = True
except ImportError:
    AGENT_E_INTEGRATION_AVAILABLE = False
    print("Agent E Personal Analytics Service not yet available - dashboard ready for integration")

# Import new dashboard modules
try:
    from charts.chart_integration import chart_engine, ChartType
    from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.frontend.data.data_aggregation_pipeline import data_pipeline, AggregationType, FilterCondition, FilterOperator
    from filters.advanced_filter_ui import filter_ui, FilterType, FilterField
    ADVANCED_MODULES_AVAILABLE = True
    print("[SUCCESS] Advanced dashboard modules loaded: Charts, Data Pipeline, Filters")
except ImportError as e:
    ADVANCED_MODULES_AVAILABLE = False
    print(f"[WARNING] Advanced modules not fully available: {e}")


class EnhancedUnifiedDashboard:
    """
    Enhanced unified dashboard with Agent E integration capabilities.
    
    Provides extensible architecture for cross-swarm collaboration while
    maintaining high performance and professional user experience.
    """
    
    def __init__(self, port: int = 5016):
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'enhanced_gamma_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Initialize core subsystems
        self.api_tracker = APIUsageTracker()
        self.data_integrator = DataIntegrator()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize Agent E integration if available
        self.personal_analytics = None
        if AGENT_E_INTEGRATION_AVAILABLE:
            self.personal_analytics = PersonalAnalyticsService()
            print("[SUCCESS] Agent E Personal Analytics Integration Active")
        
        # Enhanced backend services with personal analytics support
        self.backend_services = {
            'core_services': {
                'base_url': 'http://localhost:5000',
                'endpoints': [
                    'health-data', 'analytics-data', 'graph-data', 'linkage-data',
                    'robustness-data', 'analytics-aggregator', 'web-monitoring',
                    'security-orchestration', 'performance-profiler'
                ]
            },
            'visualization_services': {
                'base_url': 'http://localhost:5002',
                'endpoints': ['3d-visualization-data', 'webgl-metrics']
            },
            'budget_services': {
                'base_url': 'http://localhost:5003',
                'endpoints': ['api-usage-tracker', 'unified-data']
            },
            'coordination_services': {
                'base_url': 'http://localhost:5005',
                'endpoints': ['agent-coordination-status', 'multi-agent-metrics']
            },
            'monitoring_services': {
                'base_url': 'http://localhost:5010',
                'endpoints': ['api-usage-stats', 'unified-agent-status', 'cost-estimation']
            },
            # Agent E Personal Analytics Extension Points
            'personal_analytics': {
                'base_url': 'internal',  # Internal service
                'endpoints': [
                    'personal-analytics',
                    'personal-analytics/real-time',
                    'personal-analytics/3d-data',
                    'personal-analytics/productivity',
                    'personal-analytics/quality-metrics'
                ]
            }
        }
        
        self.setup_routes()
        self.setup_socketio_events()
        self.setup_agent_e_integration()
        
    def setup_routes(self):
        """Setup enhanced dashboard routes with Agent E integration points."""
        
        @self.app.route('/')
        def enhanced_dashboard():
            """Main enhanced dashboard with personal analytics panel space."""
            return render_template_string(ENHANCED_DASHBOARD_HTML)
        
        @self.app.route('/charts')
        def charts_dashboard():
            """Advanced charts dashboard page."""
            with open(Path(__file__).parent / 'dashboard_modules' / 'templates' / 'charts_dashboard.html', 'r') as f:
                return f.read()
        
        @self.app.route('/api/unified-status')
        def unified_status():
            """Get unified status from all services."""
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'services': {},
                'personal_analytics_available': AGENT_E_INTEGRATION_AVAILABLE
            }
            
            # Collect status from all backend services
            for service_name, service_config in self.backend_services.items():
                if service_config['base_url'] != 'internal':
                    status_data['services'][service_name] = self._check_service_status(
                        service_config['base_url']
                    )
            
            # Add personal analytics status if available
            if self.personal_analytics:
                status_data['personal_analytics'] = {
                    'status': 'operational',
                    'integration': 'active'
                }
            
            return jsonify(status_data)
        
        # Agent E Personal Analytics Endpoints
        @self.app.route('/api/personal-analytics')
        def personal_analytics_data():
            """Get personal analytics data from Agent E service."""
            if not self.personal_analytics:
                return jsonify({'error': 'Personal analytics not yet integrated'}), 503
            
            try:
                data = self.personal_analytics.get_personal_analytics_data()
                self.api_tracker.track_api_call('personal-analytics', purpose='dashboard_display')
                return jsonify(data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/personal-analytics/real-time')
        def personal_real_time():
            """Get real-time personal metrics."""
            if not self.personal_analytics:
                return jsonify({'error': 'Personal analytics not yet integrated'}), 503
            
            try:
                metrics = self.personal_analytics.get_real_time_metrics()
                return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/personal-analytics/3d-data')
        def personal_3d_data():
            """Get 3D visualization data for personal project structure."""
            if not self.personal_analytics:
                return jsonify({'error': 'Personal analytics not yet integrated'}), 503
            
            try:
                viz_data = self.personal_analytics.get_3d_visualization_data()
                return jsonify(viz_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard-config')
        def dashboard_config():
            """Get dashboard configuration including panel layout."""
            config = {
                'panels': {
                    'system_health': {'row': 0, 'col': 0, 'width': 1, 'height': 1},
                    'api_usage': {'row': 0, 'col': 1, 'width': 1, 'height': 1},
                    'agent_coordination': {'row': 0, 'col': 2, 'width': 1, 'height': 1},
                    # Agent E Personal Analytics Panel Space (2x2)
                    'personal_analytics': {
                        'row': 1, 'col': 0, 
                        'width': 2, 'height': 2,
                        'enabled': AGENT_E_INTEGRATION_AVAILABLE
                    },
                    '3d_visualization': {'row': 1, 'col': 2, 'width': 1, 'height': 2},
                    'security_metrics': {'row': 3, 'col': 0, 'width': 1, 'height': 1},
                    'performance_monitor': {'row': 3, 'col': 1, 'width': 1, 'height': 1},
                    'development_insights': {'row': 3, 'col': 2, 'width': 1, 'height': 1}
                },
                'theme': {
                    'primary': '#00f5ff',
                    'secondary': '#ff00f5',
                    'success': '#00ff7f',
                    'warning': '#ffa500',
                    'danger': '#ff4500',
                    'background': '#0a0e27'
                }
            }
            return jsonify(config)
        
        # Chart API Routes
        if ADVANCED_MODULES_AVAILABLE:
            @self.app.route('/api/charts', methods=['POST'])
            def create_chart():
                """Create a new chart."""
                data = request.json
                chart_config = chart_engine.create_chart(
                    chart_id=data.get('id', str(uuid.uuid4())),
                    chart_type=ChartType(data['type']),
                    data=data['data'],
                    options=data.get('options')
                )
                return jsonify(chart_config)
            
            @self.app.route('/api/charts/<chart_id>')
            def get_chart(chart_id):
                """Get chart configuration."""
                chart = chart_engine.get_chart_config(chart_id)
                if chart:
                    return jsonify(chart)
                return jsonify({'error': 'Chart not found'}), 404
            
            @self.app.route('/api/charts/<chart_id>/data', methods=['PUT'])
            def update_chart_data(chart_id):
                """Update chart data."""
                data = request.json
                success = chart_engine.update_chart_data(chart_id, data['data'])
                return jsonify({'success': success})
            
            @self.app.route('/api/charts/<chart_id>/export/<format>')
            def export_chart(chart_id, format):
                """Export chart in specified format."""
                export_data = chart_engine.export_chart(chart_id, format)
                if export_data:
                    return export_data, 200, {
                        'Content-Type': 'application/octet-stream',
                        'Content-Disposition': f'attachment; filename=chart_{chart_id}.{format}'
                    }
                return jsonify({'error': 'Export failed'}), 500
            
            @self.app.route('/api/data/aggregate', methods=['POST'])
            async def aggregate_data():
                """Perform data aggregation."""
                data = request.json
                result = await data_pipeline.aggregate_data(
                    data=data['data'],
                    group_by=data.get('group_by'),
                    aggregations=data.get('aggregations'),
                    filters=data.get('filters'),
                    time_window=data.get('time_window')
                )
                return jsonify(result.to_dict('records'))
            
            @self.app.route('/api/filters/ui')
            def get_filter_ui():
                """Get filter UI configuration."""
                fields = request.args.getlist('fields')
                ui_config = filter_ui.build_filter_ui(fields if fields else None)
                return jsonify(ui_config)
            
            @self.app.route('/api/filters/apply', methods=['POST'])
            def apply_filters():
                """Apply filter conditions."""
                conditions = request.json.get('conditions', [])
                filter_conditions = filter_ui.apply_filters(conditions)
                return jsonify({
                    'success': True,
                    'filters_applied': len(filter_conditions)
                })
            
            @self.app.route('/api/filters/presets')
            def get_filter_presets():
                """Get available filter presets."""
                return jsonify(filter_ui.get_public_presets())
            
            @self.app.route('/api/metrics/dashboard')
            def get_dashboard_metrics():
                """Get comprehensive dashboard metrics."""
                metrics = {
                    'charts': chart_engine.get_performance_metrics() if ADVANCED_MODULES_AVAILABLE else {},
                    'data_pipeline': data_pipeline.get_performance_metrics() if ADVANCED_MODULES_AVAILABLE else {},
                    'filters': filter_ui.get_metrics() if ADVANCED_MODULES_AVAILABLE else {},
                    'api_usage': self.api_tracker.get_usage_summary()
                }
                return jsonify(metrics)
    
    def setup_socketio_events(self):
        """Setup WebSocket events for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            emit('connection_established', {
                'status': 'connected',
                'timestamp': datetime.now().isoformat(),
                'personal_analytics_available': AGENT_E_INTEGRATION_AVAILABLE
            })
        
        @self.socketio.on('subscribe_updates')
        def handle_subscription(data):
            """Handle subscription to real-time updates."""
            update_type = data.get('type', 'all')
            
            def stream_updates():
                while True:
                    update_data = {
                        'timestamp': datetime.now().isoformat(),
                        'type': update_type
                    }
                    
                    if update_type in ['all', 'system']:
                        update_data['system_health'] = self._get_system_health()
                    
                    if update_type in ['all', 'personal'] and self.personal_analytics:
                        update_data['personal_metrics'] = self.personal_analytics.get_real_time_metrics()
                    
                    self.socketio.emit('dashboard_update', update_data, broadcast=True)
                    time.sleep(5)  # Update every 5 seconds
            
            threading.Thread(target=stream_updates, daemon=True).start()
            emit('subscription_confirmed', {'type': update_type})
        
        # Agent E specific WebSocket handlers
        if AGENT_E_INTEGRATION_AVAILABLE and self.personal_analytics:
            register_socketio_handlers(self.socketio, self.personal_analytics)
    
    def setup_agent_e_integration(self):
        """Setup Agent E integration if service is available."""
        if AGENT_E_INTEGRATION_AVAILABLE and self.personal_analytics:
            # Register personal analytics endpoints
            register_personal_analytics_endpoints(self.app, self.personal_analytics)
            print("[SUCCESS] Agent E API endpoints registered successfully")
    
    def _check_service_status(self, base_url: str) -> Dict[str, Any]:
        """Check if a backend service is operational."""
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            return {
                'status': 'operational' if response.status_code == 200 else 'degraded',
                'response_time': response.elapsed.total_seconds() * 1000
            }
        except:
            return {'status': 'offline', 'response_time': None}
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available // (1024 * 1024),  # MB
            'dashboard_performance': 'optimal' if cpu_percent < 50 else 'degraded'
        }
    
    def run(self):
        """Start the enhanced unified dashboard server."""
        print("[STARTUP] STARTING ENHANCED UNIFIED GAMMA DASHBOARD")
        print("=" * 60)
        print(f"   Enhanced Dashboard: http://localhost:{self.port}")
        print(f"   Agent E Integration: {'ACTIVE [SUCCESS]' if AGENT_E_INTEGRATION_AVAILABLE else 'READY (awaiting service)'}")
        print(f"   Personal Analytics Panel: 2x2 grid space allocated")
        print(f"   3D Visualization API: Ready for project structure rendering")
        print(f"   Real-time Updates: WebSocket streaming active")
        print(f"   Performance Target: <100ms response, 60+ FPS")
        print("=" * 60)
        print()
        
        try:
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)
        except KeyboardInterrupt:
            print("\nEnhanced dashboard stopped by user")
        except Exception as e:
            print(f"Error running enhanced dashboard: {e}")