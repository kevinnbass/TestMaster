"""
AGENT X STEELCLAD: Atomic Master Dashboard
Dynamically composed dashboard using atomic components
"""

from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from web.dashboard_modules.specialized.atoms.atomic_registry import (
    AtomicComponentRegistry,
    AtomicDashboardBuilder,
    load_atomic_assets,
    get_atomization_report
)

class AtomicMasterDashboard:
    """
    Master dashboard that dynamically composes UI from atomic components
    """
    
    def __init__(self, port: int = 5020):
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'atomic-dashboard-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize atomic registry
        self.registry = AtomicComponentRegistry()
        self.builder = AtomicDashboardBuilder()
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
    
    def _setup_routes(self):
        """Setup Flask routes with atomic component loading"""
        
        @self.app.route('/')
        def index():
            """Dynamically composed dashboard"""
            # Build atomic component imports
            self.builder.add_css("dashboard_styles", "unified_gamma_styles")
            self.builder.add_js("dashboard_scripts", "unified_gamma_scripts")
            
            html_imports = self.builder.build_html_imports()
            
            # Dynamic HTML template
            template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Atomic Master Dashboard - Dynamically Composed</title>
                
                <!-- ATOMIC COMPONENTS DYNAMICALLY LOADED -->
                {html_imports}
                
                <!-- External Libraries -->
                <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            </head>
            <body>
                <div id="connectionStatus" class="disconnected">Connecting...</div>
                
                <div class="container">
                    <header>
                        <h1>⚛️ Atomic Master Dashboard</h1>
                        <div class="subtitle">Dynamically Composed from {self.registry.get_atomic_stats()['total_components']} Atomic Components</div>
                    </header>
                    
                    <div class="dashboard-grid">
                        <!-- Atomic Component Status -->
                        <div class="card">
                            <h2>⚛️ Atomic Components</h2>
                            <div class="metric">
                                <span class="metric-label">CSS Atoms</span>
                                <span class="metric-value">{len(self.registry.registry['css'])}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">JS Atoms</span>
                                <span class="metric-value">{len(self.registry.registry['js'])}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Python Atoms</span>
                                <span class="metric-value">{len(self.registry.registry['python'])}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">HTML Cores</span>
                                <span class="metric-value">{len(self.registry.registry['html_cores'])}</span>
                            </div>
                        </div>
                        
                        <!-- Dynamic Dashboard Selector -->
                        <div class="card">
                            <h2>🎨 Dashboard Themes</h2>
                            <div style="display: flex; flex-direction: column; gap: 10px;">
                                <button onclick="loadTheme('dashboard')" class="btn btn-primary">Classic Dashboard</button>
                                <button onclick="loadTheme('unified_gamma')" class="btn btn-primary">Gamma Intelligence</button>
                                <button onclick="loadTheme('charts')" class="btn btn-primary">Advanced Charts</button>
                                <button onclick="loadTheme('unified_template')" class="btn btn-primary">3D Visualization</button>
                            </div>
                        </div>
                        
                        <!-- Live Metrics -->
                        <div class="card">
                            <h2>📊 Live Metrics</h2>
                            <div id="liveMetrics">
                                <div class="metric">
                                    <span class="metric-label">Components Loaded</span>
                                    <span class="metric-value" id="componentsLoaded">0</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Active Connections</span>
                                    <span class="metric-value" id="activeConnections">0</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Data Updates</span>
                                    <span class="metric-value" id="dataUpdates">0</span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Component Inspector -->
                        <div class="card">
                            <h2>🔍 Component Inspector</h2>
                            <div id="componentList" style="max-height: 200px; overflow-y: auto;">
                                <div class="insight">
                                    <div class="insight-type">Loaded Components</div>
                                    <div id="loadedComponents" style="font-size: 0.9em; opacity: 0.8;">
                                        Initializing...
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <script>
                    // Atomic Dashboard Controller
                    let socket = io();
                    let componentsLoaded = 0;
                    let dataUpdates = 0;
                    
                    socket.on('connect', () => {{
                        document.getElementById('connectionStatus').textContent = 'Connected';
                        document.getElementById('connectionStatus').className = 'connected';
                        document.getElementById('activeConnections').textContent = '1';
                    }});
                    
                    socket.on('disconnect', () => {{
                        document.getElementById('connectionStatus').textContent = 'Disconnected';
                        document.getElementById('connectionStatus').className = 'disconnected';
                        document.getElementById('activeConnections').textContent = '0';
                    }});
                    
                    socket.on('component_loaded', (data) => {{
                        componentsLoaded++;
                        document.getElementById('componentsLoaded').textContent = componentsLoaded;
                        
                        // Update component list
                        const list = document.getElementById('loadedComponents');
                        const newItem = document.createElement('div');
                        newItem.textContent = `✅ ${{data.name}} (${{data.type}})`;
                        list.appendChild(newItem);
                    }});
                    
                    socket.on('data_update', (data) => {{
                        dataUpdates++;
                        document.getElementById('dataUpdates').textContent = dataUpdates;
                    }});
                    
                    function loadTheme(theme) {{
                        socket.emit('load_theme', {{theme: theme}});
                        alert(`Loading ${{theme}} atomic components...`);
                    }}
                    
                    // Initialize
                    window.addEventListener('load', () => {{
                        // Notify server of loaded components
                        const scripts = document.querySelectorAll('script[src]');
                        const styles = document.querySelectorAll('link[rel="stylesheet"]');
                        
                        scripts.forEach(script => {{
                            socket.emit('component_loaded', {{
                                name: script.src.split('/').pop(),
                                type: 'javascript'
                            }});
                        }});
                        
                        styles.forEach(style => {{
                            socket.emit('component_loaded', {{
                                name: style.href.split('/').pop(),
                                type: 'stylesheet'
                            }});
                        }});
                    }});
                </script>
            </body>
            </html>
            """
            
            return template
        
        @self.app.route('/api/atomic-stats')
        def atomic_stats():
            """Get atomic component statistics"""
            return jsonify(self.registry.get_atomic_stats())
        
        @self.app.route('/api/atomic-manifest')
        def atomic_manifest():
            """Get complete atomic component manifest"""
            return self.registry.generate_import_manifest()
        
        @self.app.route('/api/atomization-report')
        def atomization_report():
            """Get atomization report"""
            return {"report": get_atomization_report()}
        
        @self.app.route('/api/load-dashboard/<dashboard_type>')
        def load_dashboard(dashboard_type):
            """Dynamically load a specific dashboard configuration"""
            assets = self.registry.get_dashboard_assets(dashboard_type)
            return jsonify({
                "dashboard": dashboard_type,
                "assets": assets,
                "loaded": True
            })
    
    def _setup_socketio(self):
        """Setup WebSocket handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected")
            emit('connection_established', {
                'atomic_stats': self.registry.get_atomic_stats()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected")
        
        @self.socketio.on('load_theme')
        def handle_load_theme(data):
            theme = data.get('theme')
            assets = self.registry.get_dashboard_assets(theme)
            
            # Emit loaded components
            for stylesheet in assets.get('stylesheets', []):
                emit('component_loaded', {
                    'name': stylesheet.split('/')[-1],
                    'type': 'stylesheet',
                    'url': stylesheet
                })
            
            for script in assets.get('scripts', []):
                emit('component_loaded', {
                    'name': script.split('/')[-1],
                    'type': 'javascript',
                    'url': script
                })
            
            emit('theme_loaded', {
                'theme': theme,
                'components': len(assets.get('stylesheets', [])) + len(assets.get('scripts', []))
            })
        
        @self.socketio.on('component_loaded')
        def handle_component_loaded(data):
            # Broadcast to all clients
            emit('component_loaded', data, broadcast=True)
    
    def run(self):
        """Run the atomic master dashboard"""
        print(f"""
{'=' * 60}
⚛️  ATOMIC MASTER DASHBOARD
{'=' * 60}
Port: {self.port}
Components: {self.registry.get_atomic_stats()['total_components']} atomic components
Status: STEELCLAD ATOMIZATION COMPLETE ✅
{'=' * 60}
        """)
        
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=True)


if __name__ == '__main__':
    dashboard = AtomicMasterDashboard()
    dashboard.run()