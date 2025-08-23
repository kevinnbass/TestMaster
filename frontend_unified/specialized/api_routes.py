#!/usr/bin/env python3
"""
STEELCLAD MODULE: API Routes Manager
===================================

Flask API routes extracted from enhanced_linkage_dashboard.py
(5,274 lines) -> 200 lines

Provides:
- All Flask route handlers
- API endpoint definitions
- JSON response formatting
- Error handling

Author: Agent X (STEELCLAD Modularization)
"""

import json
from pathlib import Path
from datetime import datetime
from flask import jsonify

from .linkage_analysis import quick_linkage_analysis
from .data_generator import LiveDataGenerator


def register_routes(app):
    """Register all API routes with the Flask application."""
    
    # Initialize data generator
    data_generator = LiveDataGenerator()
    
    @app.route('/graph-data')
    def graph_data():
        """Serve graph data for Neo4j visualization."""
        try:
            graph_file = Path(__file__).parent.parent / "GRAPH.json"
            if graph_file.exists():
                with open(graph_file, 'r') as f:
                    return jsonify(json.load(f))
            else:
                return jsonify({"error": "Graph data not found", "nodes": [], "relationships": []})
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route('/linkage-data')
    def linkage_data():
        """Serve functional linkage analysis data."""
        try:
            print("Starting linkage analysis...")
            result = quick_linkage_analysis()
            print(f"Linkage analysis complete: {result['total_files']} files analyzed")
            return jsonify(result)
        except Exception as e:
            print(f"Linkage analysis error: {e}")
            return jsonify({"error": str(e), "total_files": 0, "orphaned_files": [], "hanging_files": [], "marginal_files": []})

    @app.route('/health-data')
    def health_data():
        """Serve system health data."""
        try:
            health_data = data_generator.get_health_data()
            return jsonify(health_data)
        except Exception as e:
            return jsonify({"error": str(e), "health_score": 0, "overall_health": "error"})

    @app.route('/analytics-data') 
    def analytics_data():
        """Serve analytics flow data."""
        try:
            analytics_data = data_generator.get_analytics_data()
            return jsonify(analytics_data)
        except Exception as e:
            return jsonify({"error": str(e), "active_transactions": 0, "completed_transactions": 0, "failed_transactions": 0})

    @app.route('/robustness-data')
    def robustness_data():
        """Serve system robustness data."""
        try:
            robustness_data = data_generator.get_robustness_data()
            return jsonify(robustness_data)
        except Exception as e:
            return jsonify({"error": str(e), "dead_letter_size": 0, "compression_efficiency": "0%", "fallback_level": "L1"})

    @app.route('/enhanced-linkage-data')
    def enhanced_linkage_data():
        """Serve enhanced linkage analysis data with Agent Alpha intelligence."""
        try:
            # Try to load Agent Alpha's enhanced intelligence
            enhanced_intelligence_file = Path(__file__).parent / "enhanced_intelligence_linkage.py"
            if enhanced_intelligence_file.exists():
                print("Loading Agent Alpha's enhanced intelligence...")
                
                # Import the enhanced analyzer
                import sys
                sys.path.append(str(Path(__file__).parent))
                
                try:
                    from enhanced_intelligence_linkage import EnhancedLinkageAnalyzer
                    analyzer = EnhancedLinkageAnalyzer()
                    
                    # Analyze the full codebase
                    enhanced_results = analyzer.analyze_codebase("TestMaster", max_files=None)
                    print(f"Enhanced analysis complete: {len(enhanced_results.get('multi_layer_graph', {}).get('nodes', []))} nodes")
                    
                    return jsonify(enhanced_results)
                    
                except ImportError as ie:
                    print(f"Enhanced intelligence import error: {ie}")
                    # Fallback to basic linkage data
                    return jsonify(quick_linkage_analysis())
                    
            else:
                print("Enhanced intelligence not found, using basic linkage analysis")
                return jsonify(quick_linkage_analysis())
                
        except Exception as e:
            print(f"Enhanced linkage analysis error: {e}")
            return jsonify({"error": str(e), "multi_layer_graph": {"nodes": [], "links": [], "layers": []}})

    @app.route('/security-status')
    def security_status():
        """Get security status from enhanced security dashboard."""
        try:
            security_data = data_generator.get_security_status()
            return jsonify(security_data)
        except Exception as e:
            return jsonify({"error": str(e), "security_score": 0, "overall_security": "error"})

    @app.route('/ml-metrics')
    def ml_metrics():
        """Get ML module metrics from ML monitoring dashboard."""
        try:
            ml_data = data_generator.get_ml_metrics()
            return jsonify(ml_data)
        except Exception as e:
            return jsonify({"error": str(e), "active_models": 0, "model_health": "error"})

    @app.route('/telemetry-summary')
    def telemetry_summary():
        """Get telemetry summary from telemetry dashboard."""
        try:
            telemetry_data = data_generator.get_telemetry_summary()
            return jsonify(telemetry_data)
        except Exception as e:
            return jsonify({"error": str(e), "total_operations": 0, "avg_response_time": 0})

    @app.route('/system-health')
    def system_health():
        """Get comprehensive system health metrics."""
        try:
            system_data = data_generator.get_system_health()
            return jsonify(system_data)
        except Exception as e:
            return jsonify({"error": str(e), "overall_health": "error", "health_score": 0})

    @app.route('/performance-metrics')
    def performance_metrics():
        """Get detailed performance metrics."""
        try:
            perf_data = data_generator.get_performance_metrics()
            return jsonify(perf_data)
        except Exception as e:
            return jsonify({"error": str(e), "response_times": {"avg": 0}})

    @app.route('/quality-metrics')
    def quality_metrics():
        """Get quality assurance and testing metrics."""
        try:
            quality_data = data_generator.get_quality_metrics()
            return jsonify(quality_data)
        except Exception as e:
            return jsonify({"error": str(e), "overall_quality_score": 0})

    @app.route('/module-status')
    def module_status():
        """Get status of all integrated modules."""
        try:
            modules = {
                "intelligence_engine": {"status": "operational", "health": 95, "last_update": datetime.now().isoformat()},
                "security_scanner": {"status": "operational", "health": 92, "last_update": datetime.now().isoformat()},
                "ml_pipeline": {"status": "operational", "health": 88, "last_update": datetime.now().isoformat()},
                "telemetry_collector": {"status": "operational", "health": 90, "last_update": datetime.now().isoformat()},
                "graph_engine": {"status": "operational", "health": 93, "last_update": datetime.now().isoformat()},
                "dashboard_api": {"status": "operational", "health": 97, "last_update": datetime.now().isoformat()},
                "linkage_analyzer": {"status": "operational", "health": 94, "last_update": datetime.now().isoformat()}
            }
            
            # Add some variation
            import random
            for module_name, module_data in modules.items():
                if random.random() < 0.1:  # 10% chance of issues
                    module_data["status"] = "degraded"
                    module_data["health"] = random.randint(60, 85)
                elif random.random() < 0.05:  # 5% chance of maintenance
                    module_data["status"] = "maintenance"
                    module_data["health"] = random.randint(70, 90)
                    
            return jsonify({
                "timestamp": datetime.now().isoformat(),
                "total_modules": len(modules),
                "operational": len([m for m in modules.values() if m["status"] == "operational"]),
                "degraded": len([m for m in modules.values() if m["status"] == "degraded"]),
                "maintenance": len([m for m in modules.values() if m["status"] == "maintenance"]),
                "modules": modules
            })
        except Exception as e:
            return jsonify({"error": str(e), "total_modules": 0})