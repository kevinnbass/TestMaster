#!/usr/bin/env python3
"""
API Dashboard Integration Module
=================================

Integrates API usage tracking with Alpha, Beta, and Gamma dashboards.
Provides real-time cost monitoring and usage visualization.

Agent A - Critical Dashboard Integration for Cost Control
"""

import json
import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
from collections import deque

# Import the API tracker
from .api_usage_tracker import (
    get_api_tracker, APICallType, CostWarningLevel,
    get_usage_stats, API_COSTS
)


class APIDashboardIntegration:
    """
    Integrates API usage tracking with multiple dashboards
    Provides real-time updates and visualizations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tracker = get_api_tracker()
        
        # Real-time data buffers
        self.recent_calls = deque(maxlen=100)
        self.cost_timeline = deque(maxlen=1440)  # 24 hours of minute data
        self.alert_queue = deque(maxlen=50)
        
        # Dashboard connections
        self.dashboard_sockets = {}
        self.update_callbacks = []
        
        # Register with tracker
        self.tracker.register_dashboard_callback(self._on_api_call)
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("API Dashboard Integration initialized")
    
    def _on_api_call(self, api_call):
        """Handle new API call from tracker"""
        # Add to recent calls
        self.recent_calls.append({
            'timestamp': api_call.timestamp.isoformat(),
            'model': api_call.model,
            'type': api_call.call_type.value,
            'cost': api_call.estimated_cost,
            'component': api_call.component,
            'tokens': api_call.input_tokens + api_call.output_tokens
        })
        
        # Check for alerts
        if api_call.estimated_cost > 0.1:  # Alert for expensive calls
            self._add_alert('expensive_call', f"High cost call: ${api_call.estimated_cost:.2f} for {api_call.purpose}")
        
        # Notify dashboards
        self._broadcast_update('api_call', asdict(api_call))
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Update cost timeline
                current_stats = get_usage_stats()
                self.cost_timeline.append({
                    'timestamp': datetime.now().isoformat(),
                    'total_cost': current_stats['total_cost'],
                    'call_count': current_stats['total_calls']
                })
                
                # Check budget status
                budget_status = current_stats['budget_status']
                self._check_budget_alerts(budget_status)
                
                # Broadcast periodic update
                self._broadcast_update('stats_update', current_stats)
                
                # Sleep for 1 minute
                threading.Event().wait(60)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                threading.Event().wait(5)
    
    def _check_budget_alerts(self, budget_status):
        """Check budget status and generate alerts"""
        # Daily budget check
        daily_percentage = budget_status['daily']['percentage']
        if daily_percentage > 90:
            self._add_alert('budget_critical', f"Daily budget at {daily_percentage:.1f}%!")
        elif daily_percentage > 75:
            self._add_alert('budget_warning', f"Daily budget at {daily_percentage:.1f}%")
        
        # Hourly budget check
        hourly_percentage = budget_status['hourly']['percentage']
        if hourly_percentage > 90:
            self._add_alert('budget_critical', f"Hourly budget at {hourly_percentage:.1f}%!")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add alert to queue"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message
        }
        self.alert_queue.append(alert)
        self._broadcast_update('alert', alert)
    
    def _broadcast_update(self, update_type: str, data: Any):
        """Broadcast update to all connected dashboards"""
        message = {
            'type': update_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Send to all registered callbacks
        for callback in self.update_callbacks:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Dashboard update failed: {e}")
    
    def register_dashboard(self, dashboard_id: str, callback):
        """Register a dashboard for updates"""
        self.update_callbacks.append(callback)
        self.logger.info(f"Dashboard {dashboard_id} registered for API updates")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        stats = get_usage_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_stats': stats,
            'recent_calls': list(self.recent_calls),
            'cost_timeline': list(self.cost_timeline),
            'alerts': list(self.alert_queue),
            'model_costs': API_COSTS,
            'recommendations': self._generate_dashboard_recommendations(stats)
        }
    
    def _generate_dashboard_recommendations(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dashboard-specific recommendations"""
        recommendations = []
        
        # Cost optimization recommendations
        if stats['total_cost'] > 5.0:
            recommendations.append({
                'type': 'cost',
                'priority': 'high',
                'message': 'Consider implementing caching to reduce API calls',
                'action': 'enable_caching'
            })
        
        # Model optimization
        if 'gpt-4' in stats.get('calls_by_model', {}):
            gpt4_calls = stats['calls_by_model']['gpt-4']
            if gpt4_calls > 10:
                recommendations.append({
                    'type': 'model',
                    'priority': 'medium',
                    'message': f'GPT-4 used {gpt4_calls} times - consider GPT-3.5 for simple tasks',
                    'action': 'optimize_model_selection'
                })
        
        # Component optimization
        if stats.get('calls_by_component'):
            top_component = max(stats['calls_by_component'].items(), key=lambda x: x[1])
            if top_component[1] > 20:
                recommendations.append({
                    'type': 'component',
                    'priority': 'medium',
                    'message': f"Component '{top_component[0]}' has {top_component[1]} calls - consider optimization",
                    'action': 'optimize_component'
                })
        
        return recommendations
    
    def get_cost_projection(self, hours: int = 24) -> Dict[str, float]:
        """Project costs for the next N hours based on current usage"""
        stats = get_usage_stats()
        
        # Calculate average hourly cost from recent data
        recent_hours = list(stats.get('hourly_cost', {}).items())[-24:]
        if recent_hours:
            avg_hourly_cost = sum(cost for _, cost in recent_hours) / len(recent_hours)
        else:
            avg_hourly_cost = 0.0
        
        return {
            'current_total': stats['total_cost'],
            'projected_hourly': avg_hourly_cost,
            'projected_daily': avg_hourly_cost * 24,
            'projected_custom': avg_hourly_cost * hours,
            'projection_hours': hours
        }
    
    def export_dashboard_report(self) -> Dict[str, Any]:
        """Export comprehensive dashboard report"""
        return {
            'generated_at': datetime.now().isoformat(),
            'dashboard_data': self.get_dashboard_data(),
            'cost_projection': self.get_cost_projection(),
            'api_health': self._get_api_health(),
            'optimization_opportunities': self._find_optimization_opportunities()
        }
    
    def _get_api_health(self) -> Dict[str, Any]:
        """Assess overall API usage health"""
        stats = get_usage_stats()
        
        # Calculate health scores
        budget_health = 100 - stats['budget_status']['daily']['percentage']
        success_rate = (stats['successful_calls'] / max(1, stats['total_calls'])) * 100
        
        return {
            'overall_score': (budget_health + success_rate) / 2,
            'budget_health': budget_health,
            'success_rate': success_rate,
            'status': 'healthy' if budget_health > 50 and success_rate > 90 else 'warning'
        }
    
    def _find_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Find specific optimization opportunities"""
        opportunities = []
        stats = get_usage_stats()
        
        # Check for repeated similar calls
        recent_purposes = {}
        for call in self.recent_calls:
            purpose_key = f"{call['component']}_{call['type']}"
            recent_purposes[purpose_key] = recent_purposes.get(purpose_key, 0) + 1
        
        for key, count in recent_purposes.items():
            if count > 5:
                opportunities.append({
                    'type': 'caching',
                    'component': key.split('_')[0],
                    'potential_savings': count * 0.001,  # Rough estimate
                    'recommendation': f"Cache results for {key} to avoid {count} repeated calls"
                })
        
        # Check for expensive models on simple tasks
        if 'gpt-4' in stats.get('cost_by_type', {}):
            for call_type in ['embedding', 'classification']:
                if call_type in stats.get('cost_by_type', {}):
                    opportunities.append({
                        'type': 'model_downgrade',
                        'task': call_type,
                        'potential_savings': stats['cost_by_type'][call_type] * 0.7,
                        'recommendation': f"Use lighter model for {call_type} tasks"
                    })
        
        return opportunities


# Dashboard HTML Template
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>API Usage Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2196F3; color: white; padding: 20px; border-radius: 5px; }}
        .metric-card {{ background: white; padding: 15px; margin: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .alert {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .alert-critical {{ background: #ff5252; color: white; }}
        .alert-warning {{ background: #ffc107; color: #333; }}
        .chart {{ width: 100%; height: 300px; background: white; margin: 20px 0; padding: 20px; border-radius: 5px; }}
        .table {{ width: 100%; background: white; border-radius: 5px; overflow: hidden; }}
        .table th {{ background: #f5f5f5; padding: 10px; text-align: left; }}
        .table td {{ padding: 10px; border-top: 1px solid #eee; }}
        .progress-bar {{ width: 100%; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #4CAF50, #FFC107, #FF5252); transition: width 0.3s; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚨 API Usage & Cost Dashboard</h1>
        <p>Real-time monitoring of AI/LLM API calls and costs</p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
        <div class="metric-card">
            <div class="metric-value">${total_cost:.2f}</div>
            <div class="metric-label">Total Cost</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_calls}</div>
            <div class="metric-label">Total API Calls</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${daily_spent:.2f} / ${daily_limit:.2f}</div>
            <div class="metric-label">Daily Budget</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {daily_percentage}%"></div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
    </div>
    
    <div id="alerts">
        {alerts_html}
    </div>
    
    <div class="chart" id="cost-timeline">
        <h3>Cost Timeline (24h)</h3>
        <!-- Chart would be rendered here with JavaScript -->
    </div>
    
    <div class="table">
        <h3>Recent API Calls</h3>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Model</th>
                    <th>Type</th>
                    <th>Component</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                </tr>
            </thead>
            <tbody>
                {recent_calls_html}
            </tbody>
        </table>
    </div>
    
    <div class="table">
        <h3>Cost by Model</h3>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Calls</th>
                    <th>Total Cost</th>
                    <th>Avg Cost</th>
                </tr>
            </thead>
            <tbody>
                {model_costs_html}
            </tbody>
        </table>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""


def generate_dashboard_html() -> str:
    """Generate HTML dashboard for API usage"""
    integration = APIDashboardIntegration()
    data = integration.get_dashboard_data()
    stats = data['current_stats']
    
    # Generate alerts HTML
    alerts_html = ""
    for alert in data['alerts'][-5:]:  # Show last 5 alerts
        alert_class = 'alert-critical' if 'critical' in alert['type'] else 'alert-warning'
        alerts_html += f'<div class="alert {alert_class}">{alert["message"]}</div>'
    
    # Generate recent calls HTML
    recent_calls_html = ""
    for call in list(data['recent_calls'])[-10:]:  # Show last 10 calls
        recent_calls_html += f"""
        <tr>
            <td>{call['timestamp']}</td>
            <td>{call['model']}</td>
            <td>{call['type']}</td>
            <td>{call['component']}</td>
            <td>{call['tokens']}</td>
            <td>${call['cost']:.4f}</td>
        </tr>
        """
    
    # Generate model costs HTML
    model_costs_html = ""
    for model, calls in stats.get('calls_by_model', {}).items():
        cost = stats.get('cost_by_model', {}).get(model, 0)
        avg_cost = cost / calls if calls > 0 else 0
        model_costs_html += f"""
        <tr>
            <td>{model}</td>
            <td>{calls}</td>
            <td>${cost:.4f}</td>
            <td>${avg_cost:.4f}</td>
        </tr>
        """
    
    # Format the template
    budget = stats['budget_status']['daily']
    
    return DASHBOARD_HTML_TEMPLATE.format(
        total_cost=stats['total_cost'],
        total_calls=stats['total_calls'],
        daily_spent=budget['spent'],
        daily_limit=budget['limit'],
        daily_percentage=budget['percentage'],
        success_rate=(stats['successful_calls'] / max(1, stats['total_calls'])) * 100,
        alerts_html=alerts_html,
        recent_calls_html=recent_calls_html,
        model_costs_html=model_costs_html
    )


def save_dashboard_html(output_path: Path = None):
    """Save dashboard HTML to file"""
    if output_path is None:
        output_path = Path("api_usage_dashboard.html")
    
    html = generate_dashboard_html()
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path


if __name__ == "__main__":
    # Test dashboard integration
    print("API Dashboard Integration Test")
    print("=" * 60)
    
    integration = APIDashboardIntegration()
    
    # Get dashboard data
    data = integration.get_dashboard_data()
    print("Dashboard Data:")
    print(json.dumps(data, indent=2, default=str))
    
    # Generate and save HTML dashboard
    dashboard_path = save_dashboard_html()
    print(f"\nDashboard saved to: {dashboard_path}")
    
    # Get cost projection
    projection = integration.get_cost_projection(24)
    print("\nCost Projection (24h):")
    print(json.dumps(projection, indent=2))
    
    print("\n" + "=" * 60)
    print("CRITICAL: Dashboard integration active - monitoring all API calls")
    print("=" * 60)