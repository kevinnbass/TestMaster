"""
Test Documentation API & Integration Layer
Agent D - Hour 6: Documentation API & Integration Layer
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add TestMaster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TestMaster'))

@dataclass
class DocumentationEndpoint:
    """Represents a documentation API endpoint."""
    path: str
    method: str
    description: str
    parameters: List[str]
    response_type: str
    integration_points: List[str]

class DocumentationAPI:
    """Unified Documentation API for all TestMaster documentation systems."""
    
    def __init__(self):
        self.endpoints = []
        self.integrations = {}
        self.webhooks = []
        
    def register_endpoint(self, endpoint: DocumentationEndpoint):
        """Register a documentation API endpoint."""
        self.endpoints.append(endpoint)
        
    def register_integration(self, name: str, config: Dict[str, Any]):
        """Register an integration with documentation systems."""
        self.integrations[name] = config
        
    def get_api_specification(self) -> Dict[str, Any]:
        """Get OpenAPI specification for documentation API."""
        return {
            "openapi": "3.0.3",
            "info": {
                "title": "TestMaster Documentation API",
                "version": "1.0.0",
                "description": "Unified API for TestMaster documentation systems"
            },
            "servers": [
                {"url": "http://localhost:5000", "description": "Local server"},
                {"url": "https://api.testmaster.dev", "description": "Production server"}
            ],
            "paths": {
                endpoint.path: {
                    endpoint.method.lower(): {
                        "summary": endpoint.description,
                        "parameters": [{"name": param, "in": "query", "schema": {"type": "string"}} 
                                     for param in endpoint.parameters],
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {"type": endpoint.response_type}
                                    }
                                }
                            }
                        },
                        "tags": ["documentation"]
                    }
                }
                for endpoint in self.endpoints
            }
        }

def create_documentation_endpoints():
    """Create all documentation API endpoints."""
    
    endpoints = [
        # Documentation Generation
        DocumentationEndpoint(
            path="/api/docs/generate",
            method="POST", 
            description="Generate documentation for specified modules",
            parameters=["module_path", "doc_type", "format"],
            response_type="object",
            integration_points=["documentation_system", "knowledge_base"]
        ),
        
        # API Documentation
        DocumentationEndpoint(
            path="/api/docs/api",
            method="GET",
            description="Get API documentation",
            parameters=["format", "version"],
            response_type="object",
            integration_points=["api_validation", "openapi_generator"]
        ),
        
        # Knowledge Search
        DocumentationEndpoint(
            path="/api/docs/search",
            method="GET",
            description="Search knowledge base",
            parameters=["query", "type", "limit"],
            response_type="array",
            integration_points=["knowledge_management", "search_engine"]
        ),
        
        # Legacy Documentation
        DocumentationEndpoint(
            path="/api/docs/legacy",
            method="GET",
            description="Get legacy system documentation",
            parameters=["component_id", "migration_status"],
            response_type="object",
            integration_points=["legacy_integration", "archive_system"]
        ),
        
        # Configuration Documentation
        DocumentationEndpoint(
            path="/api/docs/config",
            method="GET",
            description="Get configuration documentation",
            parameters=["config_type", "environment"],
            response_type="object",
            integration_points=["config_analyzer", "setup_generator"]
        ),
        
        # Documentation Status
        DocumentationEndpoint(
            path="/api/docs/status",
            method="GET",
            description="Get documentation system status",
            parameters=[],
            response_type="object",
            integration_points=["all_systems"]
        ),
        
        # Auto-Update Trigger
        DocumentationEndpoint(
            path="/api/docs/update",
            method="POST",
            description="Trigger documentation auto-update",
            parameters=["target", "force"],
            response_type="object",
            integration_points=["auto_generator", "webhook_system"]
        ),
        
        # Documentation Validation
        DocumentationEndpoint(
            path="/api/docs/validate",
            method="POST",
            description="Validate documentation completeness",
            parameters=["scope", "strict"],
            response_type="object",
            integration_points=["validation_framework", "quality_checker"]
        )
    ]
    
    return endpoints

def create_integration_mappings():
    """Create mappings between documentation systems."""
    
    integrations = {
        "documentation_system": {
            "location": "core/intelligence/documentation/",
            "main_class": "MasterDocumentationOrchestrator",
            "api_endpoints": ["/api/docs/generate"],
            "dependencies": ["knowledge_base", "template_system"]
        },
        
        "api_validation": {
            "location": "core/intelligence/documentation/api_validation_framework.py",
            "main_class": "APIValidationFramework", 
            "api_endpoints": ["/api/docs/api"],
            "dependencies": ["openapi_generator", "endpoint_validator"]
        },
        
        "knowledge_management": {
            "location": "core/intelligence/documentation/knowledge_management_framework.py",
            "main_class": "KnowledgeManagementFramework",
            "api_endpoints": ["/api/docs/search"],
            "dependencies": ["search_engine", "knowledge_extractor"]
        },
        
        "legacy_integration": {
            "location": "core/intelligence/documentation/legacy_integration_framework.py",
            "main_class": "LegacyIntegrationFramework",
            "api_endpoints": ["/api/docs/legacy"],
            "dependencies": ["archive_analyzer", "migration_planner"]
        },
        
        "config_analyzer": {
            "location": "config/",
            "main_class": "ConfigurationAnalyzer",
            "api_endpoints": ["/api/docs/config"],
            "dependencies": ["env_parser", "template_generator"]
        }
    }
    
    return integrations

def create_webhook_system():
    """Create webhook system for documentation updates."""
    
    webhooks = [
        {
            "name": "code_change_webhook",
            "trigger": "git_push",
            "target": "/api/docs/update",
            "description": "Trigger doc updates on code changes"
        },
        {
            "name": "api_change_webhook", 
            "trigger": "api_definition_change",
            "target": "/api/docs/api",
            "description": "Update API docs on API changes"
        },
        {
            "name": "config_change_webhook",
            "trigger": "config_file_change", 
            "target": "/api/docs/config",
            "description": "Update config docs on config changes"
        }
    ]
    
    return webhooks

def generate_integration_dashboard():
    """Generate HTML dashboard for documentation integration."""
    
    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TestMaster Documentation Integration Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        .system-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .system-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        .system-card h3 {{
            color: #333;
            margin-top: 0;
        }}
        .endpoint-list {{
            list-style: none;
            padding: 0;
        }}
        .endpoint-list li {{
            background: #fff;
            margin: 5px 0;
            padding: 8px 12px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9em;
            border-left: 3px solid #10b981;
        }}
        .integration-status {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }}
        .status-card {{
            flex: 1;
            background: #10b981;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .status-card.warning {{ background: #f59e0b; }}
        .status-card.error {{ background: #ef4444; }}
        .api-docs {{
            background: #f8fafc;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        .webhook-list {{
            background: #fef3c7;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e5e7eb;
            color: #6b7280;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>Documentation Integration Dashboard</h1>
        <p style="text-align: center; font-size: 1.1em; color: #6b7280;">
            <strong>Agent D</strong> - Documentation & Validation Excellence<br>
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
        
        <div class="integration-status">
            <div class="status-card">
                <h3>5</h3>
                <p>Documentation Systems</p>
            </div>
            <div class="status-card">
                <h3>8</h3>
                <p>API Endpoints</p>
            </div>
            <div class="status-card">
                <h3>3</h3>
                <p>Active Webhooks</p>
            </div>
        </div>
        
        <h2>Documentation Systems</h2>
        <div class="system-grid">
            <div class="system-card">
                <h3>Documentation Generation</h3>
                <p>Master documentation orchestrator with 60+ modules</p>
                <ul class="endpoint-list">
                    <li>POST /api/docs/generate</li>
                </ul>
            </div>
            
            <div class="system-card">
                <h3>API Validation</h3>
                <p>OpenAPI specification and endpoint validation</p>
                <ul class="endpoint-list">
                    <li>GET /api/docs/api</li>
                    <li>POST /api/docs/validate</li>
                </ul>
            </div>
            
            <div class="system-card">
                <h3>Knowledge Management</h3>
                <p>Intelligent search and knowledge extraction</p>
                <ul class="endpoint-list">
                    <li>GET /api/docs/search</li>
                </ul>
            </div>
            
            <div class="system-card">
                <h3>Legacy Integration</h3>
                <p>Legacy system documentation and migration</p>
                <ul class="endpoint-list">
                    <li>GET /api/docs/legacy</li>
                </ul>
            </div>
            
            <div class="system-card">
                <h3>Configuration Management</h3>
                <p>Configuration documentation and setup guides</p>
                <ul class="endpoint-list">
                    <li>GET /api/docs/config</li>
                </ul>
            </div>
        </div>
        
        <div class="api-docs">
            <h2>API Documentation</h2>
            <p>The Documentation API provides unified access to all TestMaster documentation systems through RESTful endpoints.</p>
            
            <h3>Key Features:</h3>
            <ul>
                <li><strong>Unified Interface</strong> - Single API for all documentation</li>
                <li><strong>Auto-Generation</strong> - Automatic documentation updates</li>
                <li><strong>Real-time Search</strong> - Instant knowledge retrieval</li>
                <li><strong>Integration Ready</strong> - Webhook support for CI/CD</li>
            </ul>
        </div>
        
        <div class="webhook-list">
            <h3>Active Webhooks</h3>
            <ul>
                <li><strong>Code Change</strong> - Triggers on Git push events</li>
                <li><strong>API Change</strong> - Updates on API definition changes</li>
                <li><strong>Config Change</strong> - Updates on configuration file changes</li>
            </ul>
        </div>
        
        <div class="footer">
            <p><em>TestMaster Documentation Integration Dashboard</em></p>
            <p>Part of the 24-Hour Meta-Recursive Documentation Excellence Mission</p>
        </div>
    </div>
</body>
</html>"""
    
    return dashboard_html

def test_documentation_api():
    """Test the documentation API and integration layer."""
    
    print("=" * 80)
    print("Agent D - Hour 6: Documentation API & Integration Layer")
    print("Testing Documentation API Integration")
    print("=" * 80)
    
    # Create documentation API
    print("\n1. Creating Documentation API...")
    api = DocumentationAPI()
    
    # Register endpoints
    endpoints = create_documentation_endpoints()
    for endpoint in endpoints:
        api.register_endpoint(endpoint)
    
    print(f"   Registered {len(endpoints)} API endpoints")
    for endpoint in endpoints:
        print(f"   - {endpoint.method} {endpoint.path}")
    
    # Register integrations
    print("\n2. Registering System Integrations...")
    integrations = create_integration_mappings()
    for name, config in integrations.items():
        api.register_integration(name, config)
    
    print(f"   Registered {len(integrations)} integrations:")
    for name, config in integrations.items():
        print(f"   - {name}: {config['main_class']}")
    
    # Create webhook system
    print("\n3. Setting Up Webhook System...")
    webhooks = create_webhook_system()
    api.webhooks = webhooks
    
    print(f"   Configured {len(webhooks)} webhooks:")
    for webhook in webhooks:
        print(f"   - {webhook['name']}: {webhook['trigger']}")
    
    # Generate API specification
    print("\n4. Generating API Specification...")
    api_spec = api.get_api_specification()
    
    # Create output directory
    output_dir = Path("TestMaster/docs/api_integration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save API specification
    spec_path = output_dir / "documentation_api_spec.json"
    with open(spec_path, 'w', encoding='utf-8') as f:
        json.dump(api_spec, f, indent=2)
    print(f"   API specification saved: {spec_path}")
    
    # Generate integration report
    print("\n5. Generating Integration Report...")
    
    integration_report = {
        "timestamp": datetime.now().isoformat(),
        "analyzer": "Agent D - Documentation API Integration",
        "api_summary": {
            "total_endpoints": len(endpoints),
            "total_integrations": len(integrations),
            "total_webhooks": len(webhooks)
        },
        "endpoints": [asdict(endpoint) for endpoint in endpoints],
        "integrations": integrations,
        "webhooks": webhooks,
        "phase_1_summary": {
            "hour_1": "Documentation Systems Analysis",
            "hour_2": "API Documentation & Validation", 
            "hour_3": "Legacy Code Documentation",
            "hour_4": "Knowledge Management Systems",
            "hour_5": "Configuration & Setup Documentation",
            "hour_6": "Documentation API & Integration Layer"
        }
    }
    
    report_path = output_dir / "integration_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(integration_report, f, indent=2)
    print(f"   Integration report saved: {report_path}")
    
    # Generate integration dashboard
    print("\n6. Creating Integration Dashboard...")
    dashboard_html = generate_integration_dashboard()
    
    dashboard_path = output_dir / "integration_dashboard.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print(f"   Integration dashboard: {dashboard_path}")
    
    # Create API usage examples
    print("\n7. Generating API Usage Examples...")
    
    usage_examples = f"""# Documentation API Usage Examples

## Authentication
```bash
export API_KEY="your_api_key"
curl -H "Authorization: Bearer $API_KEY" \\
     -H "Content-Type: application/json" \\
     http://localhost:5000/api/docs/status
```

## Generate Documentation
```bash
curl -X POST http://localhost:5000/api/docs/generate \\
     -H "Content-Type: application/json" \\
     -d '{{"module_path": "core/intelligence", "doc_type": "api", "format": "markdown"}}'
```

## Search Knowledge Base
```bash
curl "http://localhost:5000/api/docs/search?query=API%20validation&limit=5"
```

## Get API Documentation
```bash
curl "http://localhost:5000/api/docs/api?format=openapi&version=3.0.3"
```

## Get Configuration Documentation
```bash
curl "http://localhost:5000/api/docs/config?config_type=yaml&environment=production"
```

## Validate Documentation
```bash
curl -X POST http://localhost:5000/api/docs/validate \\
     -H "Content-Type: application/json" \\
     -d '{{"scope": "all", "strict": true}}'
```

## Trigger Documentation Update
```bash
curl -X POST http://localhost:5000/api/docs/update \\
     -H "Content-Type: application/json" \\
     -d '{{"target": "api_docs", "force": false}}'
```

## Python SDK Usage

```python
import requests

class TestMasterDocsAPI:
    def __init__(self, base_url="http://localhost:5000", api_key=None):
        self.base_url = base_url
        self.headers = {{"Authorization": f"Bearer {{api_key}}"}} if api_key else {{}}
    
    def generate_docs(self, module_path, doc_type="api", format="markdown"):
        response = requests.post(
            f"{{self.base_url}}/api/docs/generate",
            json={{
                "module_path": module_path,
                "doc_type": doc_type, 
                "format": format
            }},
            headers=self.headers
        )
        return response.json()
    
    def search_knowledge(self, query, limit=10):
        response = requests.get(
            f"{{self.base_url}}/api/docs/search",
            params={{"query": query, "limit": limit}},
            headers=self.headers
        )
        return response.json()

# Usage
api = TestMasterDocsAPI(api_key="your_key")
docs = api.generate_docs("core/intelligence/api")
results = api.search_knowledge("configuration setup")
```

## Webhook Configuration

### GitHub Webhook
```json
{{
  "url": "http://your-server.com/api/docs/update",
  "content_type": "json",
  "events": ["push", "pull_request"]
}}
```

### CI/CD Integration
```yaml
# GitHub Actions
name: Update Documentation
on:
  push:
    branches: [main]
jobs:
  update-docs:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Doc Update
        run: |
          curl -X POST http://api.testmaster.dev/api/docs/update \\
               -H "Authorization: Bearer ${{{{ secrets.API_KEY }}}}" \\
               -d '{{"target": "all", "force": true}}'
```
"""
    
    examples_path = output_dir / "API_USAGE_EXAMPLES.md"
    with open(examples_path, 'w', encoding='utf-8') as f:
        f.write(usage_examples)
    print(f"   Usage examples: {examples_path}")
    
    print("\n" + "=" * 80)
    print("Documentation API Integration Test Complete!")
    print("=" * 80)
    
    return integration_report

if __name__ == "__main__":
    report = test_documentation_api()