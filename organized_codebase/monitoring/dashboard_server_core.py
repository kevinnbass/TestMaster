#!/usr/bin/env python3
"""
Dashboard Server Core - Atomic Component
Core dashboard server infrastructure
Agent Z - STEELCLAD Frontend Atomization
"""

import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS


class ServerStatus(Enum):
    """Server status states"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServerConfig:
    """Server configuration"""
    port: int = 5001
    host: str = "0.0.0.0"
    debug: bool = False
    threaded: bool = True
    cors_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class DashboardServerCore:
    """
    Dashboard server infrastructure component
    Provides core server functionality for dashboards
    """
    
    def __init__(self, config: ServerConfig = None):
        self.config = config or ServerConfig()
        self.app = Flask(__name__)
        
        # Enable CORS if configured
        if self.config.cors_enabled:
            CORS(self.app)
        
        # Server state
        self.status = ServerStatus.STOPPED
        self.start_time = None
        self.request_count = 0
        
        # Route registry
        self.registered_routes: Dict[str, Callable] = {}
        
        # Server metrics
        self.server_metrics = {
            'requests_handled': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'uptime_seconds': 0,
            'active_connections': 0
        }
        
        # Background threads
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Setup default routes
        self._setup_default_routes()
    
    def _setup_default_routes(self):
        """Setup default server routes"""
        
        @self.app.route('/')
        def index():
            """Default dashboard page"""
            return render_template_string(self._get_default_template())
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'server': self.status.value,
                'uptime': self._get_uptime(),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/metrics')
        def metrics():
            """Server metrics endpoint"""
            return jsonify(self.get_server_metrics())
        
        @self.app.route('/status')
        def status():
            """Server status endpoint"""
            return jsonify(self.get_server_status())
    
    def register_route(self, path: str, handler: Callable, methods: List[str] = None):
        """
        Register a route with the server
        Main interface for adding dashboard routes
        """
        if methods is None:
            methods = ['GET']
        
        # Store in registry
        self.registered_routes[path] = handler
        
        # Add to Flask app
        self.app.add_url_rule(
            path,
            endpoint=f"custom_{path.replace('/', '_')}",
            view_func=handler,
            methods=methods
        )
    
    def start_server(self) -> bool:
        """Start the dashboard server"""
        if self.status != ServerStatus.STOPPED:
            return False
        
        try:
            self.status = ServerStatus.STARTING
            self.start_time = datetime.now()
            
            # Start monitoring thread
            self._start_monitoring()
            
            # Start Flask server in thread
            server_thread = threading.Thread(
                target=self._run_flask_server,
                daemon=True
            )
            server_thread.start()
            
            # Wait briefly for server to start
            time.sleep(1)
            
            self.status = ServerStatus.RUNNING
            return True
            
        except Exception:
            self.status = ServerStatus.ERROR
            return False
    
    def stop_server(self) -> bool:
        """Stop the dashboard server"""
        if self.status != ServerStatus.RUNNING:
            return False
        
        self.status = ServerStatus.STOPPING
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        
        # Note: Flask server in thread will stop when main program exits
        self.status = ServerStatus.STOPPED
        
        return True
    
    def _run_flask_server(self):
        """Run Flask server"""
        try:
            self.app.run(
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug,
                threaded=self.config.threaded,
                use_reloader=False
            )
        except Exception:
            self.status = ServerStatus.ERROR
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Update uptime
                if self.start_time:
                    uptime = (datetime.now() - self.start_time).total_seconds()
                    self.server_metrics['uptime_seconds'] = uptime
                
                # Sleep interval
                time.sleep(5)
                
            except Exception:
                pass
    
    def _get_uptime(self) -> str:
        """Get server uptime"""
        if not self.start_time:
            return "0:00:00"
        
        uptime = datetime.now() - self.start_time
        return str(uptime).split('.')[0]
    
    def _get_default_template(self) -> str:
        """Get default dashboard template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard Server</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                }
                h1 {
                    text-align: center;
                }
                .info {
                    margin: 20px 0;
                    padding: 15px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Dashboard Server Running</h1>
                <div class="info">
                    <p>Server Status: Active</p>
                    <p>Port: """ + str(self.config.port) + """</p>
                    <p>Available Endpoints:</p>
                    <ul>
                        <li>/health - Health check</li>
                        <li>/metrics - Server metrics</li>
                        <li>/status - Server status</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status"""
        return {
            'status': self.status.value,
            'config': self.config.to_dict(),
            'uptime': self._get_uptime(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'registered_routes': len(self.registered_routes),
            'request_count': self.request_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_server_metrics(self) -> Dict[str, Any]:
        """Get server performance metrics"""
        return {
            **self.server_metrics,
            'status': self.status.value,
            'port': self.config.port,
            'latency_target_met': self.server_metrics['avg_response_time'] < 50
        }
    
    def handle_request(self):
        """Track request handling"""
        self.request_count += 1
        self.server_metrics['requests_handled'] += 1
    
    def handle_error(self):
        """Track error handling"""
        self.server_metrics['errors'] += 1
    
    def update_response_time(self, response_time_ms: float):
        """Update average response time"""
        current_avg = self.server_metrics['avg_response_time']
        requests = max(self.server_metrics['requests_handled'], 1)
        
        self.server_metrics['avg_response_time'] = (
            (current_avg * (requests - 1) + response_time_ms) / requests
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return {
            'server_status': self.status.value,
            'port': self.config.port,
            'registered_routes': len(self.registered_routes),
            'requests_handled': self.server_metrics['requests_handled'],
            'errors': self.server_metrics['errors'],
            'uptime_seconds': self.server_metrics['uptime_seconds'],
            'latency_target_met': self.server_metrics['avg_response_time'] < 50
        }