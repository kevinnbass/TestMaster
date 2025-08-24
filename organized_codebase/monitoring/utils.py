                        history['cpu_usage'].pop(0)
                        history['memory_usage_mb'].pop(0)
                        history['cpu_load'].pop(0)
                        history['network_kb_s'].pop(0)
                        
        except Exception as e:
            logger.error(f"Error collecting performance data: {e}")
    
    def run(self, debug: bool = False):
        """Run the web monitoring server."""
        logger.info(f"Starting TestMaster Web Monitoring Dashboard at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
    
    def stop(self):
        """Stop the monitoring server."""
        logger.info("Stopping web monitoring server...")
        self.monitor.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

# HTML Dashboard Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TestMaster Hybrid Intelligence Platform - Real-Time Monitor</title>
    <style>
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
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
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
        .card h3 { 
            margin-bottom: 15px; 
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child { border-bottom: none; }
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
        .alert { 
            padding: 10px; 
            margin-bottom: 10px; 
            border-radius: 5px; 
            border-left: 4px solid;
        }
        .alert-warning { 
            background: rgba(255, 152, 0, 0.2); 
            border-left-color: #ff9800; 
        }
        .alert-critical { 
            background: rgba(244, 67, 54, 0.2); 
            border-left-color: #f44336; 
        }
        .alert-info { 
            background: rgba(33, 150, 243, 0.2); 
            border-left-color: #2196F3; 
        }
        .refresh-info { 
            text-align: center; 
            margin-top: 20px; 
            opacity: 0.7; 
        }
        .chart-container { 
            height: 200px; 
            margin: 15px 0; 
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            padding: 10px;
        }
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
        .loading { text-align: center; padding: 40px; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 4px solid #fff;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TestMaster Hybrid Intelligence Platform</h1>
            <p>Real-Time Monitoring Dashboard</p>
            <p id="last-update">Loading...</p>
        </div>
        
        <div id="dashboard" class="dashboard">
            <div class="loading">
                <div class="spinner"></div>
                <p>Initializing monitoring systems...</p>
            </div>
        </div>
        
        <div class="refresh-info">
            <p>Dashboard refreshes every 5 seconds | Data collected every 2 seconds</p>
        </div>
    </div>

    <script>
        let metricsHistory = [];
        
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
        
        function getStatusIndicator(status) {
            if (status === 'active') return '<span class="status-indicator status-active"></span>';
            if (status === 'inactive') return '<span class="status-indicator status-inactive"></span>';
            return '<span class="status-indicator status-warning"></span>';
        }
        
        function createMetricsCard(metrics) {
            return `
                <div class="card">
                    <h3>📊 System Metrics</h3>
                    <div class="metric">
                        <span>CPU Usage:</span>
                        <span class="metric-value">${metrics.system.cpu_usage.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Memory Usage:</span>
                        <span class="metric-value">${metrics.system.memory_usage.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Active Agents:</span>
                        <span class="metric-value">${metrics.components.active_agents}/16</span>
                    </div>
                    <div class="metric">
                        <span>Active Bridges:</span>
                        <span class="metric-value">${metrics.components.active_bridges}/5</span>
                    </div>
                    <div class="metric">
                        <span>Queue Size:</span>
                        <span class="metric-value">${metrics.workflow.queue_size}</span>
                    </div>
                    <div class="metric">
                        <span>Events/Second:</span>
                        <span class="metric-value">${metrics.workflow.events_per_second.toFixed(1)}</span>
                    </div>
                </div>
            `;
        }
        
        function createLLMMetricsCard(llmMetrics) {
            if (!llmMetrics || llmMetrics.error) {
                return `
                    <div class="card">
                        <h3>🤖 LLM Intelligence</h3>
                        <p style="text-align: center; opacity: 0.7; padding: 20px;">
                            ${llmMetrics ? llmMetrics.error : 'LLM monitoring not available'}
                        </p>
                    </div>
                `;
            }
            
            return `
                <div class="card">
                    <h3>🤖 LLM Intelligence</h3>
                    <div class="metric">
                        <span>API Calls:</span>
                        <span class="metric-value">${llmMetrics.api_calls.total_calls}</span>
                    </div>
                    <div class="metric">
                        <span>Success Rate:</span>
                        <span class="metric-value">${llmMetrics.api_calls.success_rate.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Tokens Used:</span>
                        <span class="metric-value">${llmMetrics.token_usage.total_tokens.toLocaleString()}</span>
                    </div>
                    <div class="metric">
                        <span>Cost Estimate:</span>
                        <span class="metric-value">$${llmMetrics.cost_tracking.total_cost_estimate.toFixed(3)}</span>
                    </div>
                    <div class="metric">
                        <span>Calls/Minute:</span>
                        <span class="metric-value">${llmMetrics.api_calls.calls_per_minute.toFixed(1)}</span>
                    </div>
                    <div class="metric">
                        <span>Active Analyses:</span>
                        <span class="metric-value">${llmMetrics.analysis_status.active_analyses}</span>
                    </div>
                </div>
            `;
        }
        
        function createComponentsCard(components) {
            const componentItems = Object.entries(components)
                .map(([name, status]) => `
                    <div class="component-item">
                        ${getStatusIndicator(status)}
                        <span>${name.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}</span>
                    </div>
                `).join('');
            
            return `
                <div class="card">
                    <h3>🔧 Component Status</h3>
                    <div class="component-grid">
                        ${componentItems}
                    </div>
                </div>
            `;
        }
        
        function createAlertsCard(alerts) {
            if (!alerts.recent_alerts || alerts.recent_alerts.length === 0) {
                return `
                    <div class="card">
                        <h3>🚨 Recent Alerts</h3>
                        <p style="text-align: center; opacity: 0.7; padding: 20px;">No recent alerts</p>
                    </div>
                `;
            }
            
            const alertItems = alerts.recent_alerts
                .slice(-5)
                .map(alert => `
                    <div class="alert alert-${alert.severity}">
                        <strong>${alert.severity.toUpperCase()}</strong> - ${alert.component}<br>
                        ${alert.message}<br>
                        <small>${formatTimestamp(alert.timestamp)}</small>
                    </div>
                `).join('');
            
            return `
                <div class="card">
                    <h3>🚨 Recent Alerts (${alerts.total_alerts} total, ${alerts.active_alerts} active)</h3>
                    ${alertItems}
                </div>
            `;
        }
        
        function createHealthCard(metrics) {
            return `
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
                        <span class="metric-value">${metrics.workflow.consensus_decisions}</span>
                    </div>
                    <div class="metric">
                        <span>Security Alerts:</span>
                        <span class="metric-value">${metrics.security.security_alerts}</span>
                    </div>
                    <div class="metric">
                        <span>Last Update:</span>
                        <span class="metric-value" id="update-time">${formatTimestamp(metrics.timestamp)}</span>
                    </div>
                </div>
            `;
        }
        
        async function updateDashboard() {
            const [metrics, components, llmMetrics] = await Promise.all([
                fetchData('metrics'),
                fetchData('components'),
                fetchData('llm/metrics')
            ]);
            
            if (!metrics) {
                document.getElementById('dashboard').textContent = '<div class="card"><h3>❌ Error</h3><p>Failed to load metrics</p></div>';
                return;
            }
            
            // Update last update time
            document.getElementById('last-update').textContent = `Last Update: ${formatTimestamp(metrics.timestamp)}`;
            
            // Create dashboard content
            const dashboardHTML = `
                ${createMetricsCard(metrics)}
                ${createLLMMetricsCard(llmMetrics)}
                ${createComponentsCard(metrics.components.component_status)}
                ${createHealthCard(metrics)}
                ${createAlertsCard(metrics.alerts)}
            `;
            
            document.getElementById('dashboard').textContent = dashboardHTML;
        }
        
        // Initialize dashboard
        updateDashboard();
        
        // Auto-refresh every 5 seconds
        setInterval(updateDashboard, 5000);
        
        // Add some basic error handling
        window.addEventListener('error', (e) => {
            console.error('Dashboard error:', e.error);
        });
    </script>
</body>
</html>
"""

def main():
    """Main entry point for web monitoring server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TestMaster Web Monitoring Dashboard")
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and start web server
    server = WebMonitoringServer(port=args.port, host=args.host)
    
    try:
        server.run(debug=args.debug)
    except KeyboardInterrupt:
        server.stop()
        print("\nWeb monitoring server stopped")

if __name__ == "__main__":
    main()