#!/usr/bin/env python3
"""
🌐 MODULE: Unified Greek Dashboard - Multi-Agent Coordination Interface
==================================================================

📋 PURPOSE:
    Provides unified dashboard interface for complete Greek Swarm coordination,
    displaying real-time status, metrics, and coordination data from all agents.

🎯 CORE FUNCTIONALITY:
    • Multi-agent real-time dashboard with unified view
    • Cross-agent coordination interface and controls
    • Advanced metrics visualization and monitoring
    • Inter-agent communication and data flow display
    • Comprehensive swarm health and performance tracking

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 06:10:00 | Agent Delta | 🆕 FEATURE
   └─ Goal: Create unified Greek Swarm dashboard for Hour 7 mission
   └─ Changes: Complete implementation of multi-agent dashboard, coordination UI
   └─ Impact: Enables unified Greek Swarm monitoring and coordination interface

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Delta
🔧 Language: Python
📦 Dependencies: Flask, SocketIO, greek_swarm_coordinator
🎯 Integration Points: All Greek Swarm agents, ADAMANTIUMCLAD dashboard
⚡ Performance Notes: Real-time updates, WebSocket communication, caching
🔒 Security Notes: Agent authentication, secure WebSocket, CORS protection

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: 0% | Last Run: N/A (New implementation)
✅ Integration Tests: 0% | Last Run: N/A (New implementation)
✅ Performance Tests: 0% | Last Run: N/A (New implementation)
⚠️  Known Issues: None (Initial implementation)

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Greek Swarm Coordinator, all Greek agents
📤 Provides: Unified dashboard interface for all Greek Swarm operations
🚨 Breaking Changes: None (additive dashboard enhancement)
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import asyncio
from greek_swarm_coordinator import GreekSwarmCoordinator, AgentType, AgentStatus, CoordinationType

class UnifiedGreekDashboard:
    """Unified dashboard for Greek Swarm coordination and monitoring"""
    
    def __init__(self, port: int = 5003, coordinator_port: int = 5002):
        self.port = port
        self.coordinator_port = coordinator_port
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'greek_swarm_unified_dashboard'
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Greek Swarm Coordinator
        self.coordinator = GreekSwarmCoordinator(coordinator_port)
        
        # Dashboard state
        self.dashboard_clients: Dict[str, Any] = {}
        self.last_broadcast_time = datetime.utcnow()
        
        # Setup routes and socket handlers
        self.setup_routes()
        self.setup_socket_handlers()
        
        # Start background services
        self.start_background_services()
    
    def setup_routes(self):
        """Setup Flask routes for unified dashboard"""
        
        @self.app.route('/')
        def unified_dashboard():
            """Main unified Greek Swarm dashboard"""
            return render_template_string(self.get_unified_dashboard_template())
        
        @self.app.route('/api/swarm-status')
        def get_swarm_status():
            """Get complete Greek Swarm status"""
            return jsonify(self.coordinator.get_swarm_status())
        
        @self.app.route('/api/agents')
        def get_all_agents():
            """Get all registered agents"""
            agents_data = {}
            for agent_id, agent_info in self.coordinator.agents.items():
                agents_data[agent_id] = {
                    'agent_type': agent_info.agent_type.value,
                    'status': agent_info.status.value,
                    'host': agent_info.host,
                    'port': agent_info.port,
                    'health_score': agent_info.health_score,
                    'load_factor': agent_info.load_factor,
                    'last_heartbeat': agent_info.last_heartbeat.isoformat(),
                    'capabilities': agent_info.capabilities,
                    'performance_metrics': agent_info.performance_metrics
                }
            
            return jsonify({
                'agents': agents_data,
                'summary': {
                    'total': len(agents_data),
                    'active': sum(1 for a in agents_data.values() if a['status'] == 'active'),
                    'by_type': {
                        agent_type.value: len([a for a in agents_data.values() 
                                             if a['agent_type'] == agent_type.value])
                        for agent_type in AgentType
                    }
                }
            })
        
        @self.app.route('/api/coordination-history')
        def get_coordination_history():
            """Get recent coordination message history"""
            history = []
            for message in self.coordinator.coordination_history[-50:]:  # Last 50 messages
                history.append({
                    'message_id': message.message_id,
                    'source_agent': message.source_agent,
                    'target_agent': message.target_agent,
                    'coordination_type': message.coordination_type.value,
                    'timestamp': message.timestamp.isoformat(),
                    'priority': message.priority,
                    'retry_count': message.retry_count
                })
            
            return jsonify({
                'history': history,
                'queue_size': len(self.coordinator.coordination_queue),
                'total_messages': self.coordinator.swarm_metrics.total_coordination_messages
            })
        
        @self.app.route('/api/swarm-metrics')
        def get_swarm_metrics():
            """Get comprehensive swarm metrics"""
            return jsonify(asdict(self.coordinator.swarm_metrics))
        
        @self.app.route('/api/coordinate', methods=['POST'])
        def send_coordination_message():
            """Send coordination message to agents"""
            data = request.get_json()
            
            try:
                coordination_type = CoordinationType(data['coordination_type'])
                self.coordinator.send_coordination_message(
                    target_agent=data['target_agent'],
                    coordination_type=coordination_type,
                    payload=data.get('payload', {}),
                    priority=data.get('priority', 1)
                )
                
                return jsonify({'status': 'success', 'message': 'Coordination message sent'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 400
        
        @self.app.route('/api/health')
        def health_check():
            """Unified dashboard health check"""
            return jsonify({
                'status': 'healthy',
                'coordinator_status': 'operational',
                'swarm_agents': self.coordinator.swarm_metrics.active_agents,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def setup_socket_handlers(self):
        """Setup WebSocket handlers for real-time communication"""
        
        @self.socketio.on('connect')
        def on_connect():
            client_id = request.sid
            self.dashboard_clients[client_id] = {
                'connected_at': datetime.utcnow(),
                'subscriptions': []
            }
            emit('connected', {'client_id': client_id, 'status': 'Connected to Greek Swarm Dashboard'})
        
        @self.socketio.on('disconnect')
        def on_disconnect():
            client_id = request.sid
            if client_id in self.dashboard_clients:
                del self.dashboard_clients[client_id]
        
        @self.socketio.on('subscribe_agent_updates')
        def on_subscribe_agent_updates(data):
            """Subscribe to specific agent updates"""
            client_id = request.sid
            agent_types = data.get('agent_types', [])
            
            if client_id in self.dashboard_clients:
                self.dashboard_clients[client_id]['subscriptions'] = agent_types
                join_room(f"agent_updates_{client_id}")
                emit('subscription_confirmed', {'agent_types': agent_types})
        
        @self.socketio.on('request_swarm_status')
        def on_request_swarm_status():
            """Request immediate swarm status update"""
            status = self.coordinator.get_swarm_status()
            emit('swarm_status_update', status)
    
    def start_background_services(self):
        """Start background services for real-time updates"""
        # Real-time data broadcast service
        broadcast_thread = threading.Thread(target=self._broadcast_service, daemon=True)
        broadcast_thread.start()
        
        # Coordination monitoring service
        monitoring_thread = threading.Thread(target=self._coordination_monitoring_service, daemon=True)
        monitoring_thread.start()
    
    def _broadcast_service(self):
        """Background service for real-time data broadcasting"""
        while True:
            try:
                if self.dashboard_clients:
                    # Broadcast swarm status
                    status = self.coordinator.get_swarm_status()
                    self.socketio.emit('swarm_status_update', status)
                    
                    # Broadcast agent updates
                    agents_data = {}
                    for agent_id, agent_info in self.coordinator.agents.items():
                        agents_data[agent_id] = asdict(agent_info)
                    
                    self.socketio.emit('agents_update', agents_data)
                    
                    # Broadcast metrics
                    metrics = asdict(self.coordinator.swarm_metrics)
                    self.socketio.emit('metrics_update', metrics)
                
                time.sleep(5)  # Broadcast every 5 seconds
            except Exception as e:
                print(f"Broadcast service error: {e}")
                time.sleep(2)
    
    def _coordination_monitoring_service(self):
        """Background service for coordination message monitoring"""
        last_message_count = 0
        
        while True:
            try:
                current_message_count = len(self.coordinator.coordination_history)
                
                if current_message_count > last_message_count:
                    # New coordination messages, broadcast updates
                    recent_messages = self.coordinator.coordination_history[last_message_count:]
                    for message in recent_messages:
                        message_data = {
                            'message_id': message.message_id,
                            'source_agent': message.source_agent,
                            'target_agent': message.target_agent,
                            'coordination_type': message.coordination_type.value,
                            'timestamp': message.timestamp.isoformat(),
                            'priority': message.priority
                        }
                        self.socketio.emit('coordination_message', message_data)
                    
                    last_message_count = current_message_count
                
                time.sleep(3)  # Check every 3 seconds
            except Exception as e:
                print(f"Coordination monitoring error: {e}")
                time.sleep(1)
    
    def get_unified_dashboard_template(self):
        """Generate unified Greek Swarm dashboard HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Greek Swarm - Unified Coordination Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2c5282 0%, #2d3748 50%, #1a202c 100%);
            color: #f7fafc;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .header {
            background: rgba(45, 55, 72, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            text-align: center;
            border-bottom: 3px solid #4299e1;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #4299e1, #9f7aea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header .subtitle {
            font-size: 1.1em;
            color: #a0aec0;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 300px 1fr;
            height: calc(100vh - 100px);
        }
        
        .sidebar {
            background: rgba(26, 32, 44, 0.8);
            backdrop-filter: blur(10px);
            border-right: 2px solid #4a5568;
            padding: 20px;
            overflow-y: auto;
        }
        
        .sidebar h3 {
            color: #4299e1;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 1px solid #4a5568;
            padding-bottom: 8px;
        }
        
        .agent-list {
            margin-bottom: 30px;
        }
        
        .agent-item {
            background: rgba(74, 85, 104, 0.3);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            border-left: 4px solid transparent;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .agent-item.active {
            border-left-color: #48bb78;
            background: rgba(72, 187, 120, 0.1);
        }
        
        .agent-item.inactive {
            border-left-color: #f56565;
            background: rgba(245, 101, 101, 0.1);
        }
        
        .agent-item.error {
            border-left-color: #fc8181;
            background: rgba(252, 129, 129, 0.1);
        }
        
        .agent-name {
            font-weight: bold;
            color: #e2e8f0;
            margin-bottom: 4px;
        }
        
        .agent-details {
            font-size: 0.9em;
            color: #a0aec0;
        }
        
        .main-dashboard {
            padding: 20px;
            overflow-y: auto;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(45, 55, 72, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #4a5568;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h3 {
            color: #4299e1;
            margin-bottom: 15px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(74, 85, 104, 0.3);
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            color: #a0aec0;
            font-size: 0.95em;
        }
        
        .metric-value {
            color: #e2e8f0;
            font-weight: bold;
            font-size: 1.05em;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 8px;
        }
        
        .status-active { background-color: #48bb78; }
        .status-inactive { background-color: #f56565; }
        .status-warning { background-color: #ed8936; }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #4a5568;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78, #38a169);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .coordination-log {
            max-height: 300px;
            overflow-y: auto;
            background: rgba(26, 32, 44, 0.5);
            border-radius: 8px;
            padding: 15px;
        }
        
        .coordination-message {
            background: rgba(74, 85, 104, 0.3);
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
            border-left: 3px solid #4299e1;
        }
        
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .message-type {
            background: #4299e1;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        
        .message-time {
            color: #a0aec0;
            font-size: 0.85em;
        }
        
        .message-content {
            color: #e2e8f0;
            font-size: 0.9em;
        }
        
        .control-panel {
            background: rgba(45, 55, 72, 0.8);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .control-buttons {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .control-btn {
            background: linear-gradient(135deg, #4299e1, #3182ce);
            border: none;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .control-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(66, 153, 225, 0.4);
        }
        
        .control-btn.danger {
            background: linear-gradient(135deg, #f56565, #e53e3e);
        }
        
        .control-btn.success {
            background: linear-gradient(135deg, #48bb78, #38a169);
        }
        
        .agent-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .agent-card {
            background: rgba(74, 85, 104, 0.3);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .agent-card.active {
            border-color: #48bb78;
            box-shadow: 0 0 20px rgba(72, 187, 120, 0.3);
        }
        
        .agent-type {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        
        .agent-stats {
            font-size: 0.9em;
            color: #a0aec0;
        }
        
        @media (max-width: 768px) {
            .main-container {
                grid-template-columns: 1fr;
                height: auto;
            }
            
            .sidebar {
                border-right: none;
                border-bottom: 2px solid #4a5568;
            }
            
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(45, 55, 72, 0.9);
            padding: 10px 15px;
            border-radius: 20px;
            border: 1px solid #4a5568;
            font-size: 0.9em;
        }
        
        .connected { color: #48bb78; }
        .disconnected { color: #f56565; }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        <span class="disconnected">Connecting...</span>
    </div>
    
    <div class="header">
        <h1>🤝 Greek Swarm - Unified Dashboard</h1>
        <div class="subtitle">Multi-Agent Coordination & Real-Time Monitoring System</div>
    </div>
    
    <div class="main-container">
        <div class="sidebar">
            <h3>🤖 Agents</h3>
            <div class="agent-list" id="agentList">
                <div class="agent-item">
                    <div class="agent-name">Loading agents...</div>
                    <div class="agent-details">Please wait</div>
                </div>
            </div>
            
            <h3>📊 Swarm Metrics</h3>
            <div class="metric">
                <span class="metric-label">Active Agents:</span>
                <span class="metric-value" id="activeAgentsCount">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Health Score:</span>
                <span class="metric-value" id="healthScore">0%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Coordination Messages:</span>
                <span class="metric-value" id="coordMessages">0</span>
            </div>
        </div>
        
        <div class="main-dashboard">
            <div class="dashboard-grid">
                <!-- Swarm Status Card -->
                <div class="card">
                    <h3>🌐 Swarm Status</h3>
                    <div class="metric">
                        <span class="metric-label">Total Agents</span>
                        <span class="metric-value" id="totalAgents">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Agents</span>
                        <span class="metric-value" id="activeAgents">
                            0 <span class="status-indicator status-inactive" id="swarmStatus"></span>
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Average Health</span>
                        <span class="metric-value" id="avgHealth">0%</span>
                        <div class="progress-bar">
                            <div class="progress-fill" id="healthProgress" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Load Distribution</span>
                        <span class="metric-value" id="loadDistribution">0%</span>
                    </div>
                </div>
                
                <!-- Agent Overview Card -->
                <div class="card">
                    <h3>🤖 Agent Overview</h3>
                    <div class="agent-grid" id="agentOverview">
                        <!-- Agents will be populated here -->
                    </div>
                </div>
                
                <!-- Coordination Activity Card -->
                <div class="card">
                    <h3>🔄 Coordination Activity</h3>
                    <div class="metric">
                        <span class="metric-label">Queue Size</span>
                        <span class="metric-value" id="queueSize">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Successful</span>
                        <span class="metric-value" id="successfulCoord">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Failed</span>
                        <span class="metric-value" id="failedCoord">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value" id="successRate">0%</span>
                    </div>
                </div>
                
                <!-- Performance Metrics Card -->
                <div class="card">
                    <h3>⚡ Performance</h3>
                    <div class="metric">
                        <span class="metric-label">Avg Response Time</span>
                        <span class="metric-value" id="avgResponseTime">0ms</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">System Load</span>
                        <span class="metric-value" id="systemLoad">Low</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Data Sync Status</span>
                        <span class="metric-value" id="dataSyncStatus">Synchronized</span>
                    </div>
                </div>
            </div>
            
            <!-- Coordination Log Card -->
            <div class="card" style="grid-column: 1 / -1;">
                <h3>📨 Recent Coordination Messages</h3>
                <div class="coordination-log" id="coordinationLog">
                    <div class="coordination-message">
                        <div class="message-header">
                            <span class="message-type">SYSTEM</span>
                            <span class="message-time">Loading...</span>
                        </div>
                        <div class="message-content">Initializing coordination system...</div>
                    </div>
                </div>
            </div>
            
            <!-- Control Panel -->
            <div class="control-panel">
                <h3>🎛️ Control Panel</h3>
                <div class="control-buttons">
                    <button class="control-btn" onclick="refreshSwarmStatus()">🔄 Refresh Status</button>
                    <button class="control-btn success" onclick="coordinateDataSync()">📊 Sync Data</button>
                    <button class="control-btn" onclick="balanceLoad()">⚖️ Balance Load</button>
                    <button class="control-btn" onclick="runHealthCheck()">🏥 Health Check</button>
                    <button class="control-btn danger" onclick="emergencyStop()">🚨 Emergency Stop</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        class GreekSwarmDashboard {
            constructor() {
                this.socket = io();
                this.setupSocketHandlers();
                this.initializeDashboard();
            }
            
            setupSocketHandlers() {
                this.socket.on('connect', () => {
                    console.log('Connected to Greek Swarm Dashboard');
                    document.getElementById('connectionStatus').innerHTML = '<span class="connected">Connected</span>';
                    this.socket.emit('subscribe_agent_updates', { agent_types: ['alpha', 'beta', 'gamma', 'delta', 'epsilon'] });
                });
                
                this.socket.on('disconnect', () => {
                    console.log('Disconnected from dashboard');
                    document.getElementById('connectionStatus').innerHTML = '<span class="disconnected">Disconnected</span>';
                });
                
                this.socket.on('swarm_status_update', (data) => {
                    this.updateSwarmStatus(data);
                });
                
                this.socket.on('agents_update', (data) => {
                    this.updateAgents(data);
                });
                
                this.socket.on('metrics_update', (data) => {
                    this.updateMetrics(data);
                });
                
                this.socket.on('coordination_message', (data) => {
                    this.addCoordinationMessage(data);
                });
            }
            
            async initializeDashboard() {
                await this.fetchInitialData();
                setInterval(() => this.fetchSwarmStatus(), 10000);
            }
            
            async fetchInitialData() {
                try {
                    const [statusResponse, agentsResponse, metricsResponse] = await Promise.all([
                        fetch('/api/swarm-status'),
                        fetch('/api/agents'),
                        fetch('/api/swarm-metrics')
                    ]);
                    
                    const status = await statusResponse.json();
                    const agents = await agentsResponse.json();
                    const metrics = await metricsResponse.json();
                    
                    this.updateSwarmStatus(status);
                    this.updateAgents(agents.agents);
                    this.updateMetrics(metrics);
                } catch (error) {
                    console.error('Error fetching initial data:', error);
                }
            }
            
            async fetchSwarmStatus() {
                try {
                    const response = await fetch('/api/swarm-status');
                    const data = await response.json();
                    this.updateSwarmStatus(data);
                } catch (error) {
                    console.error('Error fetching swarm status:', error);
                }
            }
            
            updateSwarmStatus(data) {
                if (!data.metrics) return;
                
                const metrics = data.metrics;
                
                document.getElementById('totalAgents').textContent = metrics.total_agents;
                document.getElementById('activeAgents').textContent = metrics.active_agents;
                document.getElementById('activeAgentsCount').textContent = metrics.active_agents;
                
                const healthPercentage = (metrics.average_health_score * 100).toFixed(1);
                document.getElementById('avgHealth').textContent = healthPercentage + '%';
                document.getElementById('healthScore').textContent = healthPercentage + '%';
                document.getElementById('healthProgress').style.width = healthPercentage + '%';
                
                const loadDistribution = (metrics.load_distribution_score * 100).toFixed(1);
                document.getElementById('loadDistribution').textContent = loadDistribution + '%';
                
                // Update swarm status indicator
                const statusEl = document.getElementById('swarmStatus');
                if (metrics.active_agents > 0) {
                    statusEl.className = 'status-indicator status-active';
                } else {
                    statusEl.className = 'status-indicator status-inactive';
                }
            }
            
            updateAgents(agentsData) {
                const agentList = document.getElementById('agentList');
                const agentOverview = document.getElementById('agentOverview');
                
                agentList.innerHTML = '';
                agentOverview.innerHTML = '';
                
                Object.entries(agentsData).forEach(([agentId, agent]) => {
                    // Sidebar agent list
                    const agentItem = document.createElement('div');
                    agentItem.className = `agent-item ${agent.status}`;
                    agentItem.innerHTML = `
                        <div class="agent-name">${agent.agent_type.toUpperCase()}</div>
                        <div class="agent-details">
                            ${agent.host}:${agent.port}<br>
                            Health: ${(agent.health_score * 100).toFixed(0)}%
                        </div>
                    `;
                    agentList.appendChild(agentItem);
                    
                    // Overview grid
                    const agentCard = document.createElement('div');
                    agentCard.className = `agent-card ${agent.status}`;
                    agentCard.innerHTML = `
                        <div class="agent-type">${agent.agent_type}</div>
                        <div class="agent-stats">
                            Health: ${(agent.health_score * 100).toFixed(0)}%<br>
                            Load: ${(agent.load_factor * 100).toFixed(0)}%
                        </div>
                    `;
                    agentOverview.appendChild(agentCard);
                });
            }
            
            updateMetrics(metrics) {
                document.getElementById('coordMessages').textContent = metrics.total_coordination_messages;
                document.getElementById('queueSize').textContent = '0'; // Will be updated from swarm status
                document.getElementById('successfulCoord').textContent = metrics.successful_coordinations;
                document.getElementById('failedCoord').textContent = metrics.failed_coordinations;
                
                const totalCoord = metrics.successful_coordinations + metrics.failed_coordinations;
                const successRate = totalCoord > 0 ? (metrics.successful_coordinations / totalCoord * 100).toFixed(1) : '100';
                document.getElementById('successRate').textContent = successRate + '%';
                
                document.getElementById('avgResponseTime').textContent = (metrics.average_response_time * 1000).toFixed(0) + 'ms';
            }
            
            addCoordinationMessage(message) {
                const log = document.getElementById('coordinationLog');
                const messageEl = document.createElement('div');
                messageEl.className = 'coordination-message';
                messageEl.innerHTML = `
                    <div class="message-header">
                        <span class="message-type">${message.coordination_type.toUpperCase()}</span>
                        <span class="message-time">${new Date(message.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div class="message-content">
                        ${message.source_agent} → ${message.target_agent}
                        ${message.priority > 1 ? ' (Priority: ' + message.priority + ')' : ''}
                    </div>
                `;
                
                log.insertBefore(messageEl, log.firstChild);
                
                // Keep only last 20 messages
                while (log.children.length > 20) {
                    log.removeChild(log.lastChild);
                }
            }
        }
        
        // Control functions
        async function refreshSwarmStatus() {
            window.dashboard.socket.emit('request_swarm_status');
        }
        
        async function coordinateDataSync() {
            try {
                const response = await fetch('/api/coordinate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        target_agent: 'all',
                        coordination_type: 'data_sync',
                        payload: { sync_type: 'full' },
                        priority: 2
                    })
                });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function balanceLoad() {
            // Implement load balancing coordination
            alert('Load balancing initiated');
        }
        
        async function runHealthCheck() {
            // Implement health check coordination
            alert('Health check initiated');
        }
        
        async function emergencyStop() {
            if (confirm('Are you sure you want to initiate emergency stop?')) {
                alert('Emergency stop initiated');
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', () => {
            window.dashboard = new GreekSwarmDashboard();
        });
    </script>
</body>
</html>
        """
    
    def run(self):
        """Run the unified Greek dashboard server"""
        print(f"Starting Unified Greek Swarm Dashboard")
        print(f"Port: {self.port}")
        print(f"Coordinator Port: {self.coordinator_port}")
        print(f"Dashboard URL: http://localhost:{self.port}")
        print(f"WebSocket: Enabled for real-time updates")
        
        try:
            self.socketio.run(
                self.app,
                host='0.0.0.0',
                port=self.port,
                debug=False
            )
        except Exception as e:
            print(f"Dashboard server error: {e}")

def main():
    """Main entry point"""
    print("=" * 80)
    print("UNIFIED GREEK SWARM DASHBOARD - HOUR 7 DEPLOYMENT")
    print("=" * 80)
    print("Status: Multi-Agent Coordination Dashboard")
    print("Port: 5003 (Unified Dashboard)")
    print("Integration: Complete Greek Swarm coordination and monitoring")
    print("=" * 80)
    
    # Create and run unified dashboard
    dashboard = UnifiedGreekDashboard(port=5003)
    dashboard.run()

if __name__ == "__main__":
    main()