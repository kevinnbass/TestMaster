#!/usr/bin/env python3
"""
Unified TestMaster Dashboard
============================

Combines the aesthetic and features of:
- Port 5002: 3D visualizations, animations, professional UI
- Port 5000: Comprehensive backend endpoints and data

Includes comprehensive API usage tracking before AI analysis.

Author: Agent Alpha (Intelligence Enhancement)
"""

import os
import sys
import time
import json
import logging
import threading
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add TestMaster to path
sys.path.insert(0, str(Path(__file__).parent / "TestMaster"))

class APIUsageTracker:
    """
    Comprehensive API usage tracking system to monitor costs before AI analysis.
    """
    
    def __init__(self):
        self.api_calls = defaultdict(int)
        self.api_costs = {}
        self.model_usage = defaultdict(int) 
        self.api_history = deque(maxlen=10000)
        self.cost_estimates = {
            "gpt-4": 0.03,  # per 1k tokens
            "gpt-4-turbo": 0.01,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
            "gemini-pro": 0.0005,
            "llama-2": 0.0002
        }
        self.daily_budget = 50.0  # $50 daily budget
        self.daily_spending = 0.0
        self.last_reset = datetime.now().date()
    
    def track_api_call(self, endpoint: str, model: str = None, tokens: int = 0, purpose: str = "analysis"):
        """Track an API call with cost estimation."""
        current_time = datetime.now()
        
        # Reset daily spending if new day
        if current_time.date() > self.last_reset:
            self.daily_spending = 0.0
            self.last_reset = current_time.date()
        
        # Estimate cost
        cost = 0.0
        if model and model in self.cost_estimates and tokens > 0:
            cost = (tokens / 1000) * self.cost_estimates[model]
            self.daily_spending += cost
        
        # Record the call
        call_record = {
            "timestamp": current_time.isoformat(),
            "endpoint": endpoint,
            "model": model,
            "tokens": tokens,
            "cost_usd": cost,
            "purpose": purpose,
            "daily_total": self.daily_spending
        }
        
        self.api_calls[endpoint] += 1
        if model:
            self.model_usage[model] += 1
        self.api_history.append(call_record)
        
        # Log warning if approaching budget
        if self.daily_spending > self.daily_budget * 0.8:
            logger.warning(f"API spending at ${self.daily_spending:.2f}, approaching daily budget of ${self.daily_budget}")
        
        return call_record
    
    def check_budget_availability(self, estimated_cost: float) -> tuple[bool, str]:
        """Check if we can afford a planned API operation."""
        if self.daily_spending + estimated_cost > self.daily_budget:
            return False, f"Would exceed daily budget. Current: ${self.daily_spending:.2f}, Estimated: ${estimated_cost:.2f}, Budget: ${self.daily_budget}"
        return True, "Within budget"
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "daily_spending": round(self.daily_spending, 4),
            "daily_budget": self.daily_budget,
            "budget_remaining": round(self.daily_budget - self.daily_spending, 4),
            "total_api_calls": sum(self.api_calls.values()),
            "calls_by_endpoint": dict(self.api_calls),
            "model_usage": dict(self.model_usage),
            "recent_calls": list(self.api_history)[-10:],
            "budget_status": "OK" if self.daily_spending < self.daily_budget * 0.8 else "WARNING"
        }

class UnifiedDashboard:
    """
    Unified dashboard combining 3D aesthetics with comprehensive backend integration.
    """
    
    def __init__(self, port: int = 5003):
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'unified_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Initialize API tracker
        self.api_tracker = APIUsageTracker()
        
        # Backend endpoints to proxy from port 5000
        self.backend_endpoints = [
            'analytics-aggregator', 'web-monitoring', 'test-generation-framework',
            'security-orchestration', 'dashboard-server-apis', 'documentation-orchestrator',
            'unified-coordination-service', 'performance-profiler', 'visualization-dataset',
            'agent-coordination-status', 'health-data', 'analytics-data', 'robustness-data',
            'security-status', 'ml-metrics', 'system-health', 'linkage-data', 'graph-data'
        ]
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all routes for the unified dashboard."""
        
        @self.app.route('/')
        def unified_dashboard():
            """Main unified dashboard with 3D aesthetic and comprehensive features."""
            return render_template_string(UNIFIED_DASHBOARD_HTML)
        
        @self.app.route('/api-usage-tracker')
        def api_usage_tracker():
            """API usage tracking endpoint."""
            return jsonify(self.api_tracker.get_usage_summary())
        
        @self.app.route('/check-ai-budget', methods=['POST'])
        def check_ai_budget():
            """Check if AI operation is within budget."""
            data = request.get_json()
            estimated_cost = data.get('estimated_cost', 0.0)
            model = data.get('model', 'unknown')
            purpose = data.get('purpose', 'analysis')
            
            can_afford, message = self.api_tracker.check_budget_availability(estimated_cost)
            
            return jsonify({
                "can_afford": can_afford,
                "message": message,
                "current_spending": self.api_tracker.daily_spending,
                "budget_remaining": self.api_tracker.daily_budget - self.api_tracker.daily_spending,
                "model": model,
                "purpose": purpose
            })
        
        @self.app.route('/proxy/<endpoint>')
        def proxy_backend(endpoint):
            """Proxy requests to backend on port 5000."""
            try:
                # Track the API call
                self.api_tracker.track_api_call(f"proxy/{endpoint}", purpose="dashboard_data")
                
                # Make request to port 5000
                response = requests.get(f"http://localhost:5000/{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    return jsonify(response.json())
                else:
                    return jsonify({"error": f"Backend returned {response.status_code}"}), 500
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/unified-data')
        def unified_data():
            """Aggregate data from all backend sources."""
            aggregated_data = {
                "timestamp": datetime.now().isoformat(),
                "api_usage": self.api_tracker.get_usage_summary(),
                "backend_status": {},
                "system_overview": {
                    "total_endpoints": len(self.backend_endpoints),
                    "unified_dashboard_port": self.port,
                    "backend_port": 5000,
                    "aesthetic_source": 5002
                }
            }
            
            # Try to fetch data from key endpoints
            key_endpoints = ['health-data', 'analytics-aggregator', 'security-orchestration']
            for endpoint in key_endpoints:
                try:
                    response = requests.get(f"http://localhost:5000/{endpoint}", timeout=5)
                    if response.status_code == 200:
                        aggregated_data["backend_status"][endpoint] = {
                            "status": "operational",
                            "data": response.json()
                        }
                except Exception as e:
                    aggregated_data["backend_status"][endpoint] = {
                        "status": "error", 
                        "error": str(e)
                    }
            
            return jsonify(aggregated_data)
    
    def run(self):
        """Start the unified dashboard server."""
        print("🚀 STARTING UNIFIED TESTMASTER DASHBOARD")
        print("=" * 60)
        print(f"   Unified Dashboard: http://localhost:{self.port}")
        print(f"   Backend Proxy: http://localhost:{self.port}/proxy/<endpoint>")
        print(f"   API Tracking: http://localhost:{self.port}/api-usage-tracker")
        print("   Features: 3D Visualization + Comprehensive Backend Integration")
        print("   API Budget: $50/day with cost tracking")
        print()
        
        try:
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)
        except KeyboardInterrupt:
            print("\\nUnified dashboard stopped by user")
        except Exception as e:
            print(f"Error running unified dashboard: {e}")

# HTML Template combining 3D aesthetic with comprehensive features
UNIFIED_DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TestMaster Unified Dashboard - Alpha/Beta/Gamma Integration</title>
    
    <!-- External Libraries -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #000;
            color: #fff;
            overflow-x: hidden;
        }
        
        /* Animated Background */
        #particle-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1929 100%);
        }
        
        /* Header */
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 80px;
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(20px);
            z-index: 1000;
            display: flex;
            align-items: center;
            padding: 0 2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #00f5ff, #ff00f5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-right: 2rem;
        }
        
        /* API Budget Warning */
        .budget-status {
            margin-left: auto;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .budget-ok { background: rgba(0, 255, 127, 0.2); color: #00ff7f; }
        .budget-warning { background: rgba(255, 165, 0, 0.2); color: #ffa500; }
        .budget-danger { background: rgba(255, 69, 0, 0.2); color: #ff4500; }
        
        /* Main Content */
        .main-content {
            margin-top: 80px;
            padding: 2rem;
            min-height: calc(100vh - 80px);
        }
        
        /* Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .dashboard-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .dashboard-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 255, 255, 0.1);
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            background: linear-gradient(45deg, #00f5ff, #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00ff7f;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }
        
        /* Endpoint Status */
        .endpoint-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .endpoint-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .endpoint-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff7f;
        }
        
        /* API Usage Chart */
        .chart-container {
            height: 200px;
            margin-top: 1rem;
        }
        
        /* Loading Animation */
        .loading {
            text-align: center;
            padding: 2rem;
            color: rgba(255, 255, 255, 0.5);
        }
        
        .spinner {
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-left: 2px solid #00f5ff;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .header {
                padding: 0 1rem;
            }
            
            .main-content {
                padding: 1rem;
            }
        }
    </style>
</head>
<body>
    <!-- Animated Background -->
    <div id="particle-background"></div>
    
    <!-- Header -->
    <div class="header">
        <div class="logo">🚀 TestMaster Unified</div>
        <div id="budget-status" class="budget-status budget-ok">
            API Budget: $0.00 / $50.00
        </div>
    </div>
    
    <!-- Main Content -->
    <div class="main-content">
        <!-- Dashboard Grid -->
        <div class="dashboard-grid">
            <!-- API Usage Card -->
            <div class="dashboard-card">
                <div class="card-title">📊 API Usage Tracking</div>
                <div class="metric-value" id="api-calls">0</div>
                <div class="metric-label">Total API Calls Today</div>
                <div class="chart-container">
                    <canvas id="api-chart"></canvas>
                </div>
            </div>
            
            <!-- System Health Card -->
            <div class="dashboard-card">
                <div class="card-title">⚡ System Health</div>
                <div class="metric-value" id="system-health">--</div>
                <div class="metric-label">Overall Health Score</div>
                <div class="loading" id="health-loading">
                    <div class="spinner"></div>
                    Loading health data...
                </div>
            </div>
            
            <!-- Integration Status Card -->
            <div class="dashboard-card">
                <div class="card-title">🔗 Integration Status</div>
                <div class="metric-value" id="integrations">7</div>
                <div class="metric-label">Active Integrations</div>
                <div class="endpoint-list" id="endpoint-list">
                    <!-- Endpoints will be populated here -->
                </div>
            </div>
        </div>
        
        <!-- Additional Features Grid -->
        <div class="dashboard-grid">
            <!-- Security Status -->
            <div class="dashboard-card">
                <div class="card-title">🛡️ Security Overview</div>
                <div class="metric-value" id="security-score">--</div>
                <div class="metric-label">Security Score</div>
                <div class="loading">
                    <div class="spinner"></div>
                    Loading security data...
                </div>
            </div>
            
            <!-- Analytics Summary -->
            <div class="dashboard-card">
                <div class="card-title">📈 Analytics Engine</div>
                <div class="metric-value" id="analytics-insights">--</div>
                <div class="metric-label">Active Insights</div>
                <div class="loading">
                    <div class="spinner"></div>
                    Loading analytics...
                </div>
            </div>
            
            <!-- Coordination Status -->
            <div class="dashboard-card">
                <div class="card-title">🤖 Agent Coordination</div>
                <div class="metric-value" id="active-agents">3</div>
                <div class="metric-label">Alpha, Beta, Gamma</div>
                <div class="endpoint-item">
                    <span>Multi-Agent System</span>
                    <div class="endpoint-status"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize unified dashboard
        class UnifiedDashboard {
            constructor() {
                this.apiChart = null;
                this.updateInterval = null;
                this.endpoints = [
                    'analytics-aggregator', 'web-monitoring', 'security-orchestration',
                    'dashboard-server-apis', 'documentation-orchestrator', 
                    'unified-coordination-service', 'test-generation-framework'
                ];
                this.init();
            }
            
            async init() {
                this.setupCharts();
                this.startDataUpdates();
                this.setupEndpointList();
            }
            
            setupCharts() {
                // API Usage Chart
                const ctx = document.getElementById('api-chart').getContext('2d');
                this.apiChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'API Calls',
                            data: [],
                            borderColor: '#00f5ff',
                            backgroundColor: 'rgba(0, 245, 255, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: { display: false },
                            x: { display: false }
                        }
                    }
                });
            }
            
            setupEndpointList() {
                const endpointList = document.getElementById('endpoint-list');
                this.endpoints.forEach(endpoint => {
                    const item = document.createElement('div');
                    item.className = 'endpoint-item';
                    item.innerHTML = `
                        <span>${endpoint}</span>
                        <div class="endpoint-status" id="status-${endpoint}"></div>
                    `;
                    endpointList.appendChild(item);
                });
            }
            
            async startDataUpdates() {
                this.updateData();
                this.updateInterval = setInterval(() => this.updateData(), 5000);
            }
            
            async updateData() {
                try {
                    // Update API usage
                    const apiData = await fetch('/api-usage-tracker').then(r => r.json());
                    this.updateAPIUsage(apiData);
                    
                    // Update unified data
                    const unifiedData = await fetch('/unified-data').then(r => r.json());
                    this.updateSystemData(unifiedData);
                    
                } catch (error) {
                    console.error('Failed to update data:', error);
                }
            }
            
            updateAPIUsage(data) {
                document.getElementById('api-calls').textContent = data.total_api_calls || 0;
                
                // Update budget status
                const budgetStatus = document.getElementById('budget-status');
                const spending = data.daily_spending || 0;
                const budget = data.daily_budget || 50;
                
                budgetStatus.textContent = `API Budget: $${spending.toFixed(2)} / $${budget.toFixed(2)}`;
                
                // Update budget status class
                budgetStatus.className = 'budget-status ';
                if (spending > budget * 0.8) {
                    budgetStatus.className += 'budget-danger';
                } else if (spending > budget * 0.5) {
                    budgetStatus.className += 'budget-warning';
                } else {
                    budgetStatus.className += 'budget-ok';
                }
                
                // Update API chart
                const now = new Date().toLocaleTimeString();
                this.apiChart.data.labels.push(now);
                this.apiChart.data.datasets[0].data.push(data.total_api_calls || 0);
                
                // Keep only last 20 data points
                if (this.apiChart.data.labels.length > 20) {
                    this.apiChart.data.labels.shift();
                    this.apiChart.data.datasets[0].data.shift();
                }
                
                this.apiChart.update('none');
            }
            
            updateSystemData(data) {
                if (data.backend_status) {
                    // Update system health
                    if (data.backend_status['health-data']?.data?.system_health) {
                        const health = data.backend_status['health-data'].data.system_health;
                        document.getElementById('system-health').textContent = 
                            (health + '%') || '--';
                        document.getElementById('health-loading').style.display = 'none';
                    }
                    
                    // Update security score
                    if (data.backend_status['security-orchestration']?.data?.real_time_metrics?.security_score) {
                        const score = data.backend_status['security-orchestration'].data.real_time_metrics.security_score;
                        document.getElementById('security-score').textContent = score + '%';
                    }
                    
                    // Update endpoint statuses
                    Object.keys(data.backend_status).forEach(endpoint => {
                        const statusEl = document.getElementById(`status-${endpoint}`);
                        if (statusEl) {
                            const isOperational = data.backend_status[endpoint].status === 'operational';
                            statusEl.style.background = isOperational ? '#00ff7f' : '#ff4500';
                        }
                    });
                }
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new UnifiedDashboard();
        });
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    dashboard = UnifiedDashboard()
    dashboard.run()