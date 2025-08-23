#!/usr/bin/env python3
"""
Complete Working Dashboard with All Features
"""

import os
import sys
import time
import threading
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add TestMaster to Python path
testmaster_dir = Path(__file__).parent / "TestMaster"
sys.path.insert(0, str(testmaster_dir))

# Flask and SocketIO setup
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit

# Import the template from the main file
from enhanced_linkage_dashboard import ENHANCED_DASHBOARD_HTML, quick_linkage_analysis

# Import the enhanced intelligence analyzer
try:
    from enhanced_intelligence_linkage import EnhancedLinkageAnalyzer
    ENHANCED_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYZER_AVAILABLE = False
    print("Enhanced analyzer not available, using basic linkage analysis only")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'working_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Simple data generator for demo
class WorkingDataGenerator:
    def __init__(self):
        self.health_score = 95
        self.active_transactions = 15
        self.completed_transactions = 1250
        self.failed_transactions = 12
        self.dead_letter_size = 2
        
    def get_health_data(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': self.health_score,
            'status': 'healthy',
            'endpoints': {
                'linkage_api': {'status': 'healthy'},
                'graph_api': {'status': 'healthy'},
                'websocket': {'status': 'connected'}
            }
        }
    
    def get_analytics_data(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'active_transactions': self.active_transactions,
            'completed_transactions': self.completed_transactions,
            'failed_transactions': self.failed_transactions
        }
    
    def get_robustness_data(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'dead_letter_size': self.dead_letter_size,
            'fallback_level': 'L1',
            'compression_efficiency': '94.2%'
        }

data_generator = WorkingDataGenerator()

# Initialize enhanced analyzer
if ENHANCED_ANALYZER_AVAILABLE:
    enhanced_analyzer = EnhancedLinkageAnalyzer()
    print("Enhanced multi-dimensional analyzer loaded")

@app.route('/')
def dashboard():
    """Serve the enhanced live dashboard."""
    return render_template_string(ENHANCED_DASHBOARD_HTML)

@app.route('/linkage-data')
def linkage_data():
    """Serve functional linkage analysis data."""
    try:
        return jsonify(quick_linkage_analysis())
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/enhanced-linkage-data')
def enhanced_linkage_data():
    """Serve enhanced multi-dimensional linkage analysis data."""
    try:
        if ENHANCED_ANALYZER_AVAILABLE:
            # Run enhanced analysis
            results = enhanced_analyzer.analyze_codebase()
            return jsonify(results)
        else:
            # Fallback to basic analysis with enhanced structure
            basic_results = quick_linkage_analysis()
            enhanced_structure = {
                "basic_linkage": basic_results,
                "semantic_dimensions": {"status": "analyzer_not_available"},
                "security_dimensions": {"status": "analyzer_not_available"},
                "quality_dimensions": {"status": "analyzer_not_available"},
                "pattern_dimensions": {"status": "analyzer_not_available"},
                "predictive_dimensions": {"status": "analyzer_not_available"},
                "multi_layer_graph": {
                    "nodes": [],
                    "links": [],
                    "layers": [
                        {"id": "functional", "name": "Functional Linkage", "color": "#3b82f6"},
                        {"id": "semantic", "name": "Semantic Intent", "color": "#10b981"},
                        {"id": "security", "name": "Security Risk", "color": "#ef4444"},
                        {"id": "quality", "name": "Quality Metrics", "color": "#f59e0b"},
                        {"id": "patterns", "name": "Design Patterns", "color": "#8b5cf6"},
                        {"id": "predictive", "name": "Evolution Forecast", "color": "#ec4899"}
                    ]
                },
                "intelligence_summary": {"message": "Enhanced analyzer not loaded"},
                "analysis_timestamp": datetime.now().isoformat()
            }
            return jsonify(enhanced_structure)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/graph-data')
def graph_data():
    """Serve graph data for Neo4j visualization."""
    try:
        graph_file = Path(__file__).parent / "GRAPH.json"
        if graph_file.exists():
            with open(graph_file, 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({"error": "Graph data not found", "nodes": [], "relationships": []})
    except Exception as e:
        return jsonify({"error": str(e)})

# SocketIO handlers
@socketio.on('connect')
def on_connect():
    print('Client connected')
    emit('status', {'msg': 'Connected to working dashboard'})

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

def background_data_broadcast():
    """Broadcast live data to connected clients."""
    while True:
        try:
            if socketio:
                health_data = data_generator.get_health_data()
                analytics_data = data_generator.get_analytics_data()
                robustness_data = data_generator.get_robustness_data()
                
                socketio.emit('health_update', health_data)
                socketio.emit('analytics_update', analytics_data)
                socketio.emit('robustness_update', robustness_data)
                
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Error in broadcast: {e}")
            time.sleep(5)

def main():
    """Launch the working dashboard."""
    print("STARTING Working TestMaster Dashboard")
    print("=" * 50)
    
    print("Template Features:")
    print(f"   - Template Size: {len(ENHANCED_DASHBOARD_HTML):,} characters")
    print(f"   - Tab Navigation: {'OK' if 'tab-navigation' in ENHANCED_DASHBOARD_HTML else 'MISSING'}")
    print(f"   - D3.js Integration: {'OK' if 'd3js.org' in ENHANCED_DASHBOARD_HTML else 'MISSING'}")
    print(f"   - Spatial Visualization: {'OK' if 'loadGraphVisualization' in ENHANCED_DASHBOARD_HTML else 'MISSING'}")
    
    print("\nDashboard Access:")
    print("   - Main Dashboard: http://localhost:5002")
    print("   - Linkage Data API: http://localhost:5002/linkage-data")
    print("   - Graph Data API: http://localhost:5002/graph-data")
    
    print("\nHow to Use:")
    print("   1. Open http://localhost:5002 in your browser")
    print("   2. Click 'Graph View' tab")
    print("   3. Click 'Load Graph' button")
    print("   4. Explore the spatial visualization!")
    
    print("\nStarting background data broadcaster...")
    broadcast_thread = threading.Thread(target=background_data_broadcast, daemon=True)
    broadcast_thread.start()
    
    print("Launching Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5002, debug=False)

if __name__ == '__main__':
    main()