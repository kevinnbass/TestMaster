from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
TestMaster Web Monitoring Utilities

Web monitoring and dashboard utilities extracted from the web_monitor/utils.py file.
Provides dashboard generation and monitoring functionality.
"""

from typing import Dict, Any, List, Optional
import json


class WebMonitoringUtils:
    """Utilities for web monitoring operations"""
    
    @staticmethod
    def format_timestamp(timestamp: Any) -> str:
        """Format timestamp for display"""
        if hasattr(timestamp, 'strftime'):
            return timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return str(timestamp)
    
    @staticmethod
    def get_status_indicator(status: str) -> str:
        """Get HTML status indicator"""
        indicators = {
            'active': '<span class="status-indicator status-active"></span>',
            'inactive': '<span class="status-indicator status-inactive"></span>',
            'warning': '<span class="status-indicator status-warning"></span>'
        }
        return indicators.get(status, indicators['warning'])
    
    @staticmethod
    def calculate_success_rate(total: int, success: int) -> float:
        """Calculate success rate percentage"""
        if total == 0:
            return 0.0
        return (success / total) * 100.0
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """Format bytes into human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize text for HTML display"""
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
        }
        return "".join(html_escape_table.get(c, c) for c in text)


class DashboardGenerator:
    """Generates HTML dashboard components"""
    
    def __init__(self):
        self.css_styles = self._load_default_styles()
        self.js_functions = self._load_default_scripts()
    
    def create_metrics_card(self, metrics: Dict[str, Any]) -> str:
        """Create metrics display card"""
        if not metrics or 'system' not in metrics:
            return self._create_error_card("Metrics not available")
        
        system_metrics = metrics['system']
        components = metrics.get('components', {})
        workflow = metrics.get('workflow', {})
        
        return f"""
        <div class="card">
            <h3>📊 System Metrics</h3>
            <div class="metric">
                <span>CPU Usage:</span>
                <span class="metric-value">{system_metrics.get('cpu_usage', 0):.1f}%</span>
            </div>
            <div class="metric">
                <span>Memory Usage:</span>
                <span class="metric-value">{system_metrics.get('memory_usage', 0):.1f}%</span>
            </div>
            <div class="metric">
                <span>Active Agents:</span>
                <span class="metric-value">{components.get('active_agents', 0)}/16</span>
            </div>
            <div class="metric">
                <span>Active Bridges:</span>
                <span class="metric-value">{components.get('active_bridges', 0)}/5</span>
            </div>
            <div class="metric">
                <span>Queue Size:</span>
                <span class="metric-value">{workflow.get('queue_size', 0)}</span>
            </div>
            <div class="metric">
                <span>Events/Second:</span>
                <span class="metric-value">{workflow.get('events_per_second', 0):.1f}</span>
            </div>
        </div>
        """
    
    def create_component_status_card(self, components: Dict[str, str]) -> str:
        """Create component status display card"""
        component_items = ""
        for name, status in components.items():
            indicator = WebMonitoringUtils.get_status_indicator(status)
            display_name = name.replace('_', ' ').title()
            component_items += f"""
            <div class="component-item">
                {indicator}
                <span>{display_name}</span>
            </div>
            """
        
        return f"""
        <div class="card">
            <h3>🔧 Component Status</h3>
            <div class="component-grid">
                {component_items}
            </div>
        </div>
        """
    
    def create_alerts_card(self, alerts: Dict[str, Any]) -> str:
        """Create alerts display card"""
        if not alerts.get('recent_alerts'):
            return """
            <div class="card">
                <h3>🚨 Recent Alerts</h3>
                <p style="text-align: center; opacity: 0.7; padding: 20px;">No recent alerts</p>
            </div>
            """
        
        alert_items = ""
        for alert in alerts['recent_alerts'][-5:]:  # Show last 5
            alert_items += f"""
            <div class="alert alert-{alert.get('severity', 'info')}">
                <strong>{alert.get('severity', 'INFO').upper()}</strong> - {alert.get('component', 'System')}<br>
                {WebMonitoringUtils.sanitize_html(alert.get('message', 'No message'))}<br>
                <small>{WebMonitoringUtils.format_timestamp(alert.get('timestamp', ''))}</small>
            </div>
            """
        
        total_alerts = alerts.get('total_alerts', 0)
        active_alerts = alerts.get('active_alerts', 0)
        
        return f"""
        <div class="card">
            <h3>🚨 Recent Alerts ({total_alerts} total, {active_alerts} active)</h3>
            {alert_items}
        </div>
        """
    
    def create_health_card(self, metrics: Dict[str, Any]) -> str:
        """Create system health display card"""
        workflow = metrics.get('workflow', {})
        security = metrics.get('security', {})
        
        return f"""
        <div class="card">
            <h3>💚 System Health</h3>
            <div class="metric">
                <span>Total Agents:</span>
                <span class="metric-value">16</span>
            </div>
            <div class="metric">
                <span>Total Bridges:</span>
                <span class="metric-value">5</span>
            </div>
            <div class="metric">
                <span>Consensus Decisions:</span>
                <span class="metric-value">{workflow.get('consensus_decisions', 0)}</span>
            </div>
            <div class="metric">
                <span>Security Alerts:</span>
                <span class="metric-value">{security.get('security_alerts', 0)}</span>
            </div>
            <div class="metric">
                <span>Last Update:</span>
                <span class="metric-value">{WebMonitoringUtils.format_timestamp(metrics.get('timestamp', ''))}</span>
            </div>
        </div>
        """
    
    def create_llm_metrics_card(self, llm_metrics: Optional[Dict[str, Any]]) -> str:
        """Create LLM intelligence metrics card"""
        if not llm_metrics or llm_metrics.get('error'):
            return """
            <div class="card">
                <h3>🤖 LLM Intelligence</h3>
                <p style="text-align: center; opacity: 0.7; padding: 20px;">
                    LLM monitoring not available
                </p>
            </div>
            """
        
        api_calls = llm_metrics.get('api_calls', {})
        token_usage = llm_metrics.get('token_usage', {})
        cost_tracking = llm_metrics.get('cost_tracking', {})
        analysis_status = llm_metrics.get('analysis_status', {})
        
        return f"""
        <div class="card">
            <h3>🤖 LLM Intelligence</h3>
            <div class="metric">
                <span>API Calls:</span>
                <span class="metric-value">{api_calls.get('total_calls', 0)}</span>
            </div>
            <div class="metric">
                <span>Success Rate:</span>
                <span class="metric-value">{api_calls.get('success_rate', 0):.1f}%</span>
            </div>
            <div class="metric">
                <span>Tokens Used:</span>
                <span class="metric-value">{token_usage.get('total_tokens', 0):,}</span>
            </div>
            <div class="metric">
                <span>Cost Estimate:</span>
                <span class="metric-value">${cost_tracking.get('total_cost_estimate', 0):.3f}</span>
            </div>
            <div class="metric">
                <span>Calls/Minute:</span>
                <span class="metric-value">{api_calls.get('calls_per_minute', 0):.1f}</span>
            </div>
            <div class="metric">
                <span>Active Analyses:</span>
                <span class="metric-value">{analysis_status.get('active_analyses', 0)}</span>
            </div>
        </div>
        """
    
    def _create_error_card(self, message: str) -> str:
        """Create error display card"""
        return f"""
        <div class="card">
            <h3>❌ Error</h3>
            <p>{WebMonitoringUtils.sanitize_html(message)}</p>
        </div>
        """
    
    def _load_default_styles(self) -> str:
        """Load default CSS styles"""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { 
            text-align: center; 
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        .dashboard { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
        }
        .card { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric-value { font-weight: bold; font-size: 1.1em; }
        .status-indicator { 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            display: inline-block; 
            margin-right: 8px;
        }
        .status-active { background: #4CAF50; }
        .status-inactive { background: #f44336; }
        .status-warning { background: #ff9800; }
        .component-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .component-item {
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        """
    
    def _load_default_scripts(self) -> str:
        """Load default JavaScript functions"""
        return """
        async function fetchData(endpoint) {
            try {
                const response = await fetch(`/api/${endpoint}`);
                return await response.json();
            } catch (error) {
                console.error(`Error fetching ${endpoint}:`, error);
                return null;
            }
        }
        
        function formatTimestamp(timestamp) {
            return new Date(timestamp).toLocaleString();
        }
        """