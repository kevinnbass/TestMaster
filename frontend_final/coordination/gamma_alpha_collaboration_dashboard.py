#!/usr/bin/env python3
"""
🤝 MODULE: Gamma-Alpha Collaboration Dashboard - API Cost Integration
==================================================================

📋 PURPOSE:
    Enhanced dashboard providing Agent Alpha with comprehensive API cost tracking
    and performance metrics visualization through Agent Gamma's proven dashboard
    infrastructure. Builds on successful Agent E collaboration framework.

🎯 CORE FUNCTIONALITY:
    • Real-time API usage monitoring and cost calculation
    • Performance correlation with cost metrics
    • Interactive cost prediction and optimization recommendations
    • Multi-provider API tracking (OpenAI, Anthropic, etc.)
    • Budget management dashboard with alerts

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-24 01:21:00] | Agent Gamma | 🆕 FEATURE
   └─ Goal: Create API cost dashboard for Agent Alpha collaboration
   └─ Changes: Built comprehensive cost tracking visualization platform
   └─ Impact: Enables Agent Alpha cost management integration with Gamma dashboard

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-24 by Agent Gamma
🔧 Language: Python
📦 Dependencies: Flask, SocketIO, requests, datetime
🎯 Integration Points: Agent Alpha API cost management service
⚡ Performance Notes: <100ms API responses, real-time cost tracking
🔒 Security Notes: Local deployment, API key management

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: [Pending] | Last Run: [Not yet tested]
✅ Integration Tests: [Pending] | Last Run: [Not yet tested]  
✅ Performance Tests: [Target: <100ms] | Last Run: [Not yet tested]
⚠️  Known Issues: Initial deployment - requires Alpha integration

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Agent Alpha cost tracking service
📤 Provides: Dashboard infrastructure for API cost visualization
🚨 Breaking Changes: None - extends existing dashboard capabilities
"""

import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIUsageTracker:
    """
    Enhanced API usage tracking system with real-time cost calculation
    for Agent Alpha integration.
    """
    
    def __init__(self):
        self.api_calls = defaultdict(int)
        self.api_costs = {}
        self.model_usage = defaultdict(int) 
        self.api_history = deque(maxlen=10000)
        self.cost_estimates = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1k tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }
        self.daily_budget = 100.00  # Default daily budget in USD
        self.budget_alerts = []
        
    def track_api_call(self, provider: str, model: str, input_tokens: int, 
                      output_tokens: int, timestamp: Optional[datetime] = None):
        """Track API call with cost calculation."""
        if not timestamp:
            timestamp = datetime.now()
            
        # Calculate cost
        cost = 0.0
        if model in self.cost_estimates:
            rates = self.cost_estimates[model]
            cost = (input_tokens * rates["input"] / 1000) + (output_tokens * rates["output"] / 1000)
        
        call_data = {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": timestamp.isoformat()
        }
        
        self.api_history.append(call_data)
        self.api_calls[f"{provider}:{model}"] += 1
        self.model_usage[model] += input_tokens + output_tokens
        
        # Update daily costs
        today = timestamp.date().isoformat()
        if today not in self.api_costs:
            self.api_costs[today] = 0.0
        self.api_costs[today] += cost
        
        # Check budget alerts
        self._check_budget_alerts(today)
        
        return call_data
    
    def _check_budget_alerts(self, date: str):
        """Check if budget thresholds are exceeded."""
        daily_cost = self.api_costs.get(date, 0.0)
        percentage = (daily_cost / self.daily_budget) * 100
        
        # Clear old alerts for this date
        self.budget_alerts = [alert for alert in self.budget_alerts if alert["date"] != date]
        
        if percentage >= 90:
            self.budget_alerts.append({
                "level": "critical",
                "message": f"Daily budget 90% exceeded: ${daily_cost:.2f}/${self.daily_budget:.2f}",
                "date": date,
                "percentage": percentage
            })
        elif percentage >= 75:
            self.budget_alerts.append({
                "level": "warning", 
                "message": f"Daily budget 75% used: ${daily_cost:.2f}/${self.daily_budget:.2f}",
                "date": date,
                "percentage": percentage
            })
    
    def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive cost summary for dashboard."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        total_cost = 0.0
        daily_costs = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.isoformat()
            cost = self.api_costs.get(date_str, 0.0)
            total_cost += cost
            daily_costs.append({"date": date_str, "cost": cost})
            current_date += timedelta(days=1)
        
        # Calculate model usage breakdown
        model_costs = defaultdict(float)
        for call in list(self.api_history)[-1000:]:  # Last 1000 calls
            model_costs[call["model"]] += call["cost"]
        
        return {
            "total_cost": total_cost,
            "average_daily_cost": total_cost / days,
            "daily_costs": daily_costs,
            "model_breakdown": dict(model_costs),
            "total_calls": sum(self.api_calls.values()),
            "total_tokens": sum(self.model_usage.values()),
            "budget_alerts": self.budget_alerts,
            "budget_remaining": max(0, self.daily_budget - self.api_costs.get(end_date.isoformat(), 0.0))
        }

class GammaAlphaDashboard:
    """
    Enhanced dashboard with Agent Alpha API cost integration.
    """
    
    def __init__(self, port: int = 5002):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'gamma-alpha-collaboration-2025'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.port = port
        
        # Initialize API tracker
        self.api_tracker = APIUsageTracker()
        
        # Generate some sample data
        self._generate_sample_data()
        
        self.setup_routes()
        self.setup_socketio_handlers()
        
        logger.info(f"Gamma-Alpha Dashboard initialized on port {port}")
    
    def _generate_sample_data(self):
        """Generate sample API usage data for demonstration."""
        import random
        
        # Generate last 7 days of sample data
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            
            # Random API calls per day
            num_calls = random.randint(10, 50)
            for _ in range(num_calls):
                provider = random.choice(["openai", "anthropic", "google"])
                model = random.choice(["gpt-4", "claude-3-sonnet", "gpt-4-turbo"])
                input_tokens = random.randint(500, 3000)
                output_tokens = random.randint(100, 1500)
                
                call_time = date - timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                self.api_tracker.track_api_call(provider, model, input_tokens, output_tokens, call_time)
    
    def setup_routes(self):
        """Setup Flask routes for API cost dashboard."""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self.get_dashboard_template())
        
        @self.app.route('/api/cost-summary')
        def cost_summary():
            """Get comprehensive cost summary for Agent Alpha."""
            days = request.args.get('days', 7, type=int)
            return jsonify({
                "status": "success",
                "data": self.api_tracker.get_cost_summary(days),
                "timestamp": datetime.now().isoformat(),
                "gamma_alpha_collaboration": True
            })
        
        @self.app.route('/api/budget-status')
        def budget_status():
            """Get current budget status and alerts."""
            today = datetime.now().date().isoformat()
            daily_spent = self.api_tracker.api_costs.get(today, 0.0)
            
            return jsonify({
                "status": "success",
                "data": {
                    "daily_budget": self.api_tracker.daily_budget,
                    "daily_spent": daily_spent,
                    "daily_remaining": max(0, self.api_tracker.daily_budget - daily_spent),
                    "percentage_used": (daily_spent / self.api_tracker.daily_budget) * 100,
                    "alerts": self.api_tracker.budget_alerts,
                    "recommendations": self._get_cost_recommendations()
                },
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/cost-prediction', methods=['POST'])
        def cost_prediction():
            """Predict future costs based on usage patterns."""
            data = request.get_json() or {}
            
            # Simple prediction based on recent trends
            recent_costs = list(self.api_tracker.api_costs.values())[-7:]
            if recent_costs:
                average_daily = sum(recent_costs) / len(recent_costs)
                
                predictions = {
                    "daily_average": average_daily,
                    "weekly_projection": average_daily * 7,
                    "monthly_projection": average_daily * 30,
                    "confidence": "medium"
                }
            else:
                predictions = {
                    "daily_average": 0,
                    "weekly_projection": 0,
                    "monthly_projection": 0,
                    "confidence": "low"
                }
            
            return jsonify({
                "status": "success",
                "predictions": predictions,
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/track-call', methods=['POST'])
        def track_api_call():
            """Track new API call for Agent Alpha."""
            data = request.get_json()
            
            try:
                call_data = self.api_tracker.track_api_call(
                    provider=data.get('provider', 'unknown'),
                    model=data.get('model', 'unknown'),
                    input_tokens=data.get('input_tokens', 0),
                    output_tokens=data.get('output_tokens', 0)
                )
                
                # Emit real-time update
                self.socketio.emit('cost_update', {
                    "call_data": call_data,
                    "summary": self.api_tracker.get_cost_summary(1)
                })
                
                return jsonify({
                    "status": "success",
                    "call_data": call_data,
                    "message": "API call tracked successfully"
                })
                
            except Exception as e:
                logger.error(f"Error tracking API call: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
    
    def _get_cost_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        # Analyze usage patterns
        model_costs = defaultdict(float)
        for call in list(self.api_tracker.api_history)[-100:]:
            model_costs[call["model"]] += call["cost"]
        
        if model_costs:
            most_expensive = max(model_costs.items(), key=lambda x: x[1])
            if most_expensive[1] > 5.0:  # If spending more than $5 on one model
                recommendations.append(f"Consider using a cheaper alternative to {most_expensive[0]}")
        
        # Budget recommendations
        today_cost = self.api_tracker.api_costs.get(datetime.now().date().isoformat(), 0.0)
        if today_cost > self.api_tracker.daily_budget * 0.8:
            recommendations.append("Consider implementing request batching to reduce API calls")
            recommendations.append("Review model selection - use smaller models for simple tasks")
        
        return recommendations or ["Usage patterns look optimal"]
    
    def setup_socketio_handlers(self):
        """Setup WebSocket handlers for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Client connected to Gamma-Alpha dashboard')
            emit('connected', {
                "message": "Connected to Gamma-Alpha API Cost Dashboard",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Client disconnected from dashboard')
        
        @self.socketio.on('request_cost_update')
        def handle_cost_update_request():
            """Send real-time cost updates."""
            summary = self.api_tracker.get_cost_summary(7)
            emit('cost_update', summary)
    
    def get_dashboard_template(self) -> str:
        """Return HTML template for Gamma-Alpha collaboration dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤝 Gamma-Alpha Collaboration Dashboard - API Cost Management</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 30px 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            margin-bottom: 30px;
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { font-size: 1.1em; opacity: 0.9; }
        .collaboration-badge {
            display: inline-block;
            background: linear-gradient(45deg, #4ade80, #22c55e);
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); }
        .card h2 {
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { font-weight: 600; opacity: 0.9; }
        .metric-value { font-weight: bold; color: #ffd700; }
        .cost-high { color: #f87171 !important; }
        .cost-medium { color: #facc15 !important; }
        .cost-low { color: #4ade80 !important; }
        .alert {
            background: rgba(248, 113, 113, 0.2);
            border-left: 4px solid #f87171;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }
        .alert.warning {
            background: rgba(250, 204, 21, 0.2);
            border-left-color: #facc15;
        }
        .recommendations {
            background: rgba(74, 222, 128, 0.1);
            border-left: 4px solid #4ade80;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
        }
        .recommendations ul { margin-left: 20px; margin-top: 10px; }
        .budget-progress {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .budget-bar {
            height: 100%;
            background: linear-gradient(45deg, #4ade80, #22c55e);
            transition: width 0.3s ease;
        }
        .budget-bar.warning { background: linear-gradient(45deg, #facc15, #eab308); }
        .budget-bar.critical { background: linear-gradient(45deg, #f87171, #ef4444); }
        #connectionStatus {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            font-size: 0.9em;
        }
        .connected { color: #4ade80; }
        .disconnected { color: #f87171; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .loading { animation: pulse 2s infinite; }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div id="connectionStatus" class="disconnected">Connecting...</div>
    
    <div class="container">
        <header>
            <h1>🤝 Gamma-Alpha Collaboration Dashboard</h1>
            <div class="subtitle">API Cost Management & Performance Integration</div>
            <div class="collaboration-badge">Cross-Swarm Partnership: Gamma (Visualization) + Alpha (Cost Management)</div>
        </header>
        
        <div class="dashboard-grid">
            <!-- Budget Overview Card -->
            <div class="card">
                <h2>💰 Daily Budget Status</h2>
                <div id="budgetStatus" class="loading">
                    <div class="metric">
                        <span class="metric-label">Daily Budget</span>
                        <span class="metric-value" id="dailyBudget">$--</span>
                    </div>
                    <div class="budget-progress">
                        <div id="budgetBar" class="budget-bar" style="width: 0%"></div>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Spent Today</span>
                        <span class="metric-value" id="dailySpent">$--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Remaining</span>
                        <span class="metric-value" id="dailyRemaining">$--</span>
                    </div>
                </div>
                <div id="budgetAlerts"></div>
            </div>
            
            <!-- Cost Summary Card -->
            <div class="card">
                <h2>📊 Cost Summary (7 Days)</h2>
                <div id="costSummary" class="loading">
                    <div class="metric">
                        <span class="metric-label">Total Cost</span>
                        <span class="metric-value" id="totalCost">$--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Average Daily</span>
                        <span class="metric-value" id="averageCost">$--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total API Calls</span>
                        <span class="metric-value" id="totalCalls">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Tokens</span>
                        <span class="metric-value" id="totalTokens">--</span>
                    </div>
                </div>
            </div>
            
            <!-- Model Breakdown Card -->
            <div class="card">
                <h2>🤖 Model Usage Breakdown</h2>
                <div id="modelBreakdown" class="loading">
                    <div class="metric">
                        <span class="metric-label">Top Model</span>
                        <span class="metric-value" id="topModel">--</span>
                    </div>
                </div>
            </div>
            
            <!-- Cost Predictions Card -->
            <div class="card">
                <h2>🔮 Cost Predictions</h2>
                <div id="costPredictions" class="loading">
                    <div class="metric">
                        <span class="metric-label">Weekly Projection</span>
                        <span class="metric-value" id="weeklyProjection">$--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Monthly Projection</span>
                        <span class="metric-value" id="monthlyProjection">$--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value" id="predictionConfidence">--</span>
                    </div>
                </div>
            </div>
            
            <!-- Optimization Recommendations Card -->
            <div class="card">
                <h2>💡 Cost Optimization</h2>
                <div id="optimizationCard" class="loading">
                    <div class="recommendations">
                        <div id="recommendations">Loading recommendations...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        class GammaAlphaDashboard {
            constructor() {
                this.socket = io();
                this.setupEventHandlers();
                this.initializeDashboard();
            }
            
            setupEventHandlers() {
                this.socket.on('connect', () => {
                    console.log('Connected to Gamma-Alpha Dashboard');
                    document.getElementById('connectionStatus').textContent = 'Connected';
                    document.getElementById('connectionStatus').className = 'connected';
                });
                
                this.socket.on('cost_update', (data) => {
                    console.log('Received cost update:', data);
                    this.updateCostMetrics();
                });
            }
            
            async initializeDashboard() {
                await this.updateBudgetStatus();
                await this.updateCostSummary();
                await this.updateCostPredictions();
                
                // Set up periodic updates
                setInterval(() => this.updateBudgetStatus(), 30000);
                setInterval(() => this.updateCostSummary(), 60000);
            }
            
            async updateBudgetStatus() {
                try {
                    const response = await fetch('/api/budget-status');
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        const data = result.data;
                        
                        document.getElementById('dailyBudget').textContent = `$${data.daily_budget.toFixed(2)}`;
                        document.getElementById('dailySpent').textContent = `$${data.daily_spent.toFixed(2)}`;
                        document.getElementById('dailyRemaining').textContent = `$${data.daily_remaining.toFixed(2)}`;
                        
                        // Update progress bar
                        const percentage = Math.min(data.percentage_used, 100);
                        const bar = document.getElementById('budgetBar');
                        bar.style.width = `${percentage}%`;
                        
                        if (percentage >= 90) {
                            bar.className = 'budget-bar critical';
                        } else if (percentage >= 75) {
                            bar.className = 'budget-bar warning';
                        } else {
                            bar.className = 'budget-bar';
                        }
                        
                        // Show alerts
                        const alertsContainer = document.getElementById('budgetAlerts');
                        alertsContainer.innerHTML = '';
                        
                        data.alerts.forEach(alert => {
                            const alertEl = document.createElement('div');
                            alertEl.className = `alert ${alert.level}`;
                            alertEl.textContent = alert.message;
                            alertsContainer.appendChild(alertEl);
                        });
                        
                        // Show recommendations
                        const recommendationsEl = document.getElementById('recommendations');
                        recommendationsEl.innerHTML = '<ul>' + 
                            data.recommendations.map(r => `<li>${r}</li>`).join('') + 
                            '</ul>';
                    }
                    
                    document.getElementById('budgetStatus').classList.remove('loading');
                    document.getElementById('optimizationCard').classList.remove('loading');
                    
                } catch (error) {
                    console.error('Error updating budget status:', error);
                }
            }
            
            async updateCostSummary() {
                try {
                    const response = await fetch('/api/cost-summary');
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        const data = result.data;
                        
                        document.getElementById('totalCost').textContent = `$${data.total_cost.toFixed(2)}`;
                        document.getElementById('averageCost').textContent = `$${data.average_daily_cost.toFixed(2)}`;
                        document.getElementById('totalCalls').textContent = data.total_calls.toLocaleString();
                        document.getElementById('totalTokens').textContent = data.total_tokens.toLocaleString();
                        
                        // Update model breakdown
                        const modelBreakdown = Object.entries(data.model_breakdown)
                            .sort(([,a], [,b]) => b - a);
                        
                        if (modelBreakdown.length > 0) {
                            const [topModel, topCost] = modelBreakdown[0];
                            document.getElementById('topModel').textContent = `${topModel} ($${topCost.toFixed(2)})`;
                        }
                    }
                    
                    document.getElementById('costSummary').classList.remove('loading');
                    document.getElementById('modelBreakdown').classList.remove('loading');
                    
                } catch (error) {
                    console.error('Error updating cost summary:', error);
                }
            }
            
            async updateCostPredictions() {
                try {
                    const response = await fetch('/api/cost-prediction', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    });
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        const pred = result.predictions;
                        
                        document.getElementById('weeklyProjection').textContent = `$${pred.weekly_projection.toFixed(2)}`;
                        document.getElementById('monthlyProjection').textContent = `$${pred.monthly_projection.toFixed(2)}`;
                        document.getElementById('predictionConfidence').textContent = pred.confidence;
                    }
                    
                    document.getElementById('costPredictions').classList.remove('loading');
                    
                } catch (error) {
                    console.error('Error updating predictions:', error);
                }
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', () => {
            window.dashboard = new GammaAlphaDashboard();
        });
    </script>
</body>
</html>
        """
    
    def run(self, debug: bool = False):
        """Run the Gamma-Alpha collaboration dashboard."""
        logger.info(f"Starting Gamma-Alpha Dashboard on port {self.port}")
        logger.info("🤝 Cross-swarm collaboration: Agent Gamma + Agent Alpha")
        logger.info("📊 Providing: API cost visualization and budget management")
        
        try:
            self.socketio.run(
                self.app, 
                host='localhost', 
                port=self.port, 
                debug=debug,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")

def main():
    """Main entry point for Gamma-Alpha collaboration dashboard."""
    dashboard = GammaAlphaDashboard(port=5002)
    dashboard.run(debug=False)

if __name__ == "__main__":
    main()