#!/usr/bin/env python3
"""
API Documentation Server
Serves interactive Swagger UI for generated OpenAPI specifications
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, send_from_directory

app = Flask(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / 'generated_api_docs'

# Swagger UI HTML template
SWAGGER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content="TestMaster API Documentation" />
    <title>TestMaster API Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        body { margin: 0; padding: 0; }
        .swagger-ui .topbar { display: none; }
        .info { margin-bottom: 20px; }
        .api-selector { 
            background: #f8f9fa; 
            padding: 15px; 
            border-bottom: 1px solid #dee2e6; 
            margin-bottom: 20px;
        }
        .api-selector h2 { margin: 0 0 10px 0; color: #333; }
        .api-selector select { 
            padding: 8px 12px; 
            border: 1px solid #ccc; 
            border-radius: 4px; 
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="api-selector">
        <h2>TestMaster API Documentation</h2>
        <label for="api-select">Select API:</label>
        <select id="api-select" onchange="loadAPI()">
            <option value="consolidated_api">🌐 All APIs (Consolidated)</option>
            <option value="agent_coordination_dashboard_openapi">🎛️ Agent Coordination Dashboard</option>
            <option value="shared_flask_framework_openapi">🔧 Shared Flask Framework</option>
            <option value="api_tracking_service_openapi">📊 API Tracking Service</option>
        </select>
    </div>
    <div id="swagger-ui"></div>
    
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        let ui;
        
        function loadAPI() {
            const select = document.getElementById('api-select');
            const apiName = select.value;
            
            fetch(`/api/spec/${apiName}`)
                .then(response => response.json())
                .then(spec => {
                    if (ui) {
                        ui.getSystem().specActions.updateSpec(JSON.stringify(spec));
                    } else {
                        initSwaggerUI(spec);
                    }
                })
                .catch(error => {
                    console.error('Error loading API spec:', error);
                    alert('Error loading API specification');
                });
        }
        
        function initSwaggerUI(spec) {
            ui = SwaggerUIBundle({
                spec: spec,
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tryItOutEnabled: true,
                filter: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                onComplete: function() {
                    console.log("Swagger UI loaded successfully");
                }
            });
        }
        
        // Load default API on page load
        window.addEventListener('load', loadAPI);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main documentation interface"""
    return render_template_string(SWAGGER_TEMPLATE)

@app.route('/api/spec/<spec_name>')
def get_api_spec(spec_name):
    """Get OpenAPI specification for a given API"""
    try:
        spec_file = DOCS_DIR / f'{spec_name}.json'
        
        if not spec_file.exists():
            return jsonify({'error': f'API specification {spec_name} not found'}), 404
            
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)
            
        return jsonify(spec)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list')
def list_apis():
    """List all available API specifications"""
    try:
        if not DOCS_DIR.exists():
            return jsonify({'error': 'Documentation directory not found'}), 404
            
        apis = []
        for spec_file in DOCS_DIR.glob('*.json'):
            spec_name = spec_file.stem
            
            # Try to read spec info
            try:
                with open(spec_file, 'r', encoding='utf-8') as f:
                    spec = json.load(f)
                    
                apis.append({
                    'name': spec_name,
                    'title': spec.get('info', {}).get('title', spec_name),
                    'version': spec.get('info', {}).get('version', '1.0.0'),
                    'description': spec.get('info', {}).get('description', ''),
                    'paths_count': len(spec.get('paths', {}))
                })
            except:
                apis.append({
                    'name': spec_name,
                    'title': spec_name,
                    'version': 'unknown',
                    'description': 'Failed to parse specification',
                    'paths_count': 0
                })
                
        return jsonify({'apis': apis})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'API Documentation Server',
        'docs_available': DOCS_DIR.exists(),
        'specs_count': len(list(DOCS_DIR.glob('*.json'))) if DOCS_DIR.exists() else 0
    })

if __name__ == '__main__':
    print("Starting TestMaster API Documentation Server...")
    print(f"Documentation directory: {DOCS_DIR}")
    
    if DOCS_DIR.exists():
        spec_count = len(list(DOCS_DIR.glob('*.json')))
        print(f"Found {spec_count} API specifications")
    else:
        print("WARNING: Documentation directory not found")
        
    print("Server starting at: http://localhost:5020")
    print("Interactive documentation available at: http://localhost:5020/")
    
    app.run(host='0.0.0.0', port=5020, debug=True)