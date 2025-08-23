#!/usr/bin/env python3
"""
Simple OpenAPI Schema Generator
Generates basic OpenAPI schemas from Flask route discovery
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

class SimpleAPISchemaGenerator:
    """Simple generator for OpenAPI schemas from Flask files"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def discover_routes_in_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Discover Flask routes in a Python file using regex"""
        routes = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Pattern for @app.route decorators
            app_route_pattern = r'@app\.route\([\'"]([^\'"]+)[\'"](?:,\s*methods\s*=\s*\[([^\]]+)\])?\)'
            
            # Pattern for @bp.route decorators (blueprints)
            bp_route_pattern = r'@\w+\.route\([\'"]([^\'"]+)[\'"](?:,\s*methods\s*=\s*\[([^\]]+)\])?\)'
            
            # Find all route decorators
            for pattern in [app_route_pattern, bp_route_pattern]:
                matches = re.finditer(pattern, content, re.MULTILINE)
                
                for match in matches:
                    route_path = match.group(1)
                    methods_str = match.group(2) if match.group(2) else 'GET'
                    
                    # Clean up methods string
                    methods = []
                    if methods_str:
                        methods_clean = methods_str.replace("'", "").replace('"', "").replace(' ', '')
                        methods = [m.strip() for m in methods_clean.split(',')]
                    else:
                        methods = ['GET']
                    
                    # Try to find the function name following the decorator
                    route_start = match.end()
                    following_content = content[route_start:route_start + 500]
                    func_match = re.search(r'def\s+(\w+)\s*\(', following_content)
                    function_name = func_match.group(1) if func_match else 'unknown'
                    
                    routes.append({
                        'path': route_path,
                        'methods': methods,
                        'function_name': function_name,
                        'file': file_path.name
                    })
                    
            self.logger.info(f"Found {len(routes)} routes in {file_path.name}")
            return routes
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return []
            
    def generate_openapi_spec(self, routes: List[Dict[str, Any]], app_name: str) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification from routes"""
        
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": f"{app_name.replace('_', ' ').title()} API",
                "version": "1.0.0",
                "description": f"Auto-generated API documentation for {app_name}"
            },
            "servers": [
                {"url": "http://localhost:5000", "description": "Main server"},
                {"url": "http://localhost:5002", "description": "Secondary server"},
                {"url": "http://localhost:5003", "description": "Tertiary server"},
                {"url": "http://localhost:5005", "description": "Performance server"},
                {"url": "http://localhost:5010", "description": "Monitoring server"},
                {"url": "http://localhost:5015", "description": "Dashboard server"},
                {"url": "http://localhost:9090", "description": "Prometheus metrics"}
            ],
            "paths": {},
            "components": {
                "schemas": {
                    "Error": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "message": {"type": "string"}
                        }
                    },
                    "Success": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean"},
                            "data": {"type": "object"}
                        }
                    }
                }
            }
        }
        
        # Add paths
        for route in routes:
            path = route['path']
            methods = route['methods']
            function_name = route['function_name']
            
            if path not in spec['paths']:
                spec['paths'][path] = {}
                
            for method in methods:
                method_lower = method.lower()
                if method_lower in ['get', 'post', 'put', 'delete', 'patch']:
                    spec['paths'][path][method_lower] = self._create_operation(
                        function_name, method, path
                    )
                    
        return spec
        
    def _create_operation(self, function_name: str, method: str, path: str) -> Dict[str, Any]:
        """Create OpenAPI operation object"""
        
        operation = {
            "summary": function_name.replace('_', ' ').title(),
            "description": f"Auto-generated documentation for {function_name}",
            "parameters": [],
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Success"}
                        }
                    }
                },
                "400": {
                    "description": "Bad request",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"}
                        }
                    }
                },
                "500": {
                    "description": "Internal server error",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"}
                        }
                    }
                }
            }
        }
        
        # Add path parameters
        path_params = re.findall(r'<([^>]+)>', path)
        for param in path_params:
            param_name = param.split(':')[-1] if ':' in param else param
            param_type = param.split(':')[0] if ':' in param else 'string'
            
            # Map Flask types to OpenAPI types
            type_mapping = {
                'int': 'integer',
                'float': 'number',
                'string': 'string',
                'path': 'string',
                'uuid': 'string'
            }
            
            operation['parameters'].append({
                "name": param_name,
                "in": "path",
                "required": True,
                "schema": {
                    "type": type_mapping.get(param_type, 'string')
                }
            })
            
        # Add request body for POST/PUT/PATCH
        if method.upper() in ['POST', 'PUT', 'PATCH']:
            operation['requestBody'] = {
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                }
            }
            
        return operation
        
    def generate_schemas_for_known_apps(self) -> Dict[str, Dict[str, Any]]:
        """Generate schemas for our known Flask applications"""
        
        known_apps = [
            ('agent_coordination_dashboard.py', 'Agent Coordination Dashboard'),
            ('core/api/shared/shared_flask_framework.py', 'Shared Flask Framework'),
            ('core/intelligence/api_tracking_service.py', 'API Tracking Service'),
            ('core/monitoring/api_usage_tracker.py', 'API Usage Tracker'),
            ('web/unified_gamma_dashboard.py', 'Unified Dashboard'),
            ('performance_monitoring_infrastructure.py', 'Performance Monitor')
        ]
        
        schemas = {}
        
        for app_path, app_title in known_apps:
            full_path = self.project_root / app_path
            
            if full_path.exists():
                self.logger.info(f"Processing {app_title}...")
                routes = self.discover_routes_in_file(full_path)
                
                if routes:
                    app_key = Path(app_path).stem
                    schema = self.generate_openapi_spec(routes, app_title)
                    schemas[app_key] = schema
                    self.logger.info(f"Generated schema for {app_title} with {len(routes)} routes")
                else:
                    self.logger.warning(f"No routes found in {app_title}")
            else:
                self.logger.warning(f"File not found: {app_path}")
                
        return schemas
        
    def save_schemas(self, schemas: Dict[str, Dict[str, Any]]) -> List[Path]:
        """Save generated schemas to files"""
        
        output_dir = self.project_root / 'generated_api_docs'
        output_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Save individual schemas
        for app_name, schema in schemas.items():
            output_file = output_dir / f'{app_name}_openapi.json'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
                
            saved_files.append(output_file)
            self.logger.info(f"Saved: {output_file}")
            
        # Create consolidated schema
        if schemas:
            consolidated = self._create_consolidated_schema(schemas)
            consolidated_file = output_dir / 'consolidated_api.json'
            
            with open(consolidated_file, 'w', encoding='utf-8') as f:
                json.dump(consolidated, f, indent=2, ensure_ascii=False)
                
            saved_files.append(consolidated_file)
            self.logger.info(f"Saved consolidated schema: {consolidated_file}")
            
        return saved_files
        
    def _create_consolidated_schema(self, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create consolidated schema from individual schemas"""
        
        consolidated = {
            "openapi": "3.0.0",
            "info": {
                "title": "TestMaster API Suite",
                "version": "1.0.0",
                "description": "Consolidated API documentation for all TestMaster services"
            },
            "servers": [
                {"url": "http://localhost:5000", "description": "Main application server"},
                {"url": "http://localhost:5002", "description": "Secondary server"},
                {"url": "http://localhost:5003", "description": "Tertiary server"},
                {"url": "http://localhost:5005", "description": "Performance server"},
                {"url": "http://localhost:5010", "description": "Monitoring server"},
                {"url": "http://localhost:5015", "description": "Dashboard server"},
                {"url": "http://localhost:9090", "description": "Prometheus metrics"}
            ],
            "paths": {},
            "components": {
                "schemas": {
                    "Error": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "message": {"type": "string"}
                        }
                    },
                    "Success": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean"},
                            "data": {"type": "object"}
                        }
                    }
                }
            },
            "tags": []
        }
        
        # Merge all schemas
        for app_name, schema in schemas.items():
            tag_name = app_name.replace('_', '-')
            consolidated["tags"].append({
                "name": tag_name,
                "description": f"Endpoints from {schema['info']['title']}"
            })
            
            # Merge paths
            for path, path_spec in schema.get('paths', {}).items():
                # Handle path conflicts by prefixing
                final_path = path
                if final_path in consolidated['paths']:
                    final_path = f'/{tag_name}{path}'
                    
                consolidated['paths'][final_path] = path_spec
                
                # Add tags to operations
                for method, operation in path_spec.items():
                    if 'tags' not in operation:
                        operation['tags'] = []
                    operation['tags'].append(tag_name)
                    
        return consolidated

def main():
    """Main execution"""
    project_root = Path(__file__).parent.parent.parent
    generator = SimpleAPISchemaGenerator(str(project_root))
    
    print("Generating OpenAPI schemas for discovered Flask applications...")
    schemas = generator.generate_schemas_for_known_apps()
    
    if schemas:
        print(f"Generated schemas for {len(schemas)} applications")
        saved_files = generator.save_schemas(schemas)
        
        print("\nGenerated files:")
        for file_path in saved_files:
            rel_path = file_path.relative_to(project_root)
            print(f"  - {rel_path}")
            
        print("\nOpenAPI schema generation complete!")
        
    else:
        print("No schemas generated")
        
if __name__ == '__main__':
    main()