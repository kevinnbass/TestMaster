#!/usr/bin/env python3
"""
OpenAPI Schema Generation Runner
Generates comprehensive OpenAPI schemas from existing Flask applications
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.api_documentation.flask_blueprint_analyzer import FlaskCodeAnalyzer
from core.api_documentation.openapi_generator import OpenAPIGenerator

class APISchemaGenerator:
    """Generates OpenAPI schemas from discovered Flask applications"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzer = FlaskCodeAnalyzer()
        self.openapi_generator = OpenAPIGenerator()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for schema generation"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def discover_flask_apps(self) -> List[Path]:
        """Discover Flask applications in the project"""
        flask_apps = []
        
        # Known Flask applications from Feature Discovery
        known_apps = [
            'core/monitoring/api_usage_tracker.py',
            'web/unified_gamma_dashboard.py',
            'performance_monitoring_infrastructure.py'
        ]
        
        for app_path in known_apps:
            full_path = self.project_root / app_path
            if full_path.exists():
                flask_apps.append(full_path)
                self.logger.info(f"Found Flask app: {app_path}")
                
        # Additional discovery pattern
        for py_file in self.project_root.rglob("*.py"):
            if py_file.name.startswith('.'):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(pattern in content for pattern in [
                    'from flask import Flask',
                    'import flask',
                    'Flask(__name__)',
                    '@app.route',
                    'Blueprint('
                ]):
                    if py_file not in flask_apps:
                        flask_apps.append(py_file)
                        self.logger.info(f"Discovered Flask app: {py_file.relative_to(self.project_root)}")
                        
            except (UnicodeDecodeError, PermissionError) as e:
                self.logger.debug(f"Could not read {py_file}: {e}")
                continue
                
        return flask_apps
        
    def generate_schema_for_app(self, app_path: Path) -> Optional[Dict[str, Any]]:
        """Generate OpenAPI schema for a single Flask application"""
        try:
            self.logger.info(f"Analyzing Flask app: {app_path.name}")
            
            # Analyze the Flask application
            routes = self.analyzer.analyze_file(str(app_path))
            
            if not routes:
                self.logger.warning(f"No routes found in {app_path.name}")
                return None
                
            # Generate OpenAPI specification
            app_name = app_path.stem
            openapi_spec = {
                "openapi": "3.0.0",
                "info": {
                    "title": f"{app_name.replace('_', ' ').title()} API",
                    "version": "1.0.0",
                    "description": f"Auto-generated API documentation for {app_name}"
                },
                "servers": [
                    {"url": "http://localhost:5000", "description": "Development server"},
                    {"url": "http://localhost:5002", "description": "Secondary server"},
                    {"url": "http://localhost:5003", "description": "Tertiary server"},
                    {"url": "http://localhost:5005", "description": "Performance server"},
                    {"url": "http://localhost:5010", "description": "Monitoring server"},
                    {"url": "http://localhost:5015", "description": "Dashboard server"},
                    {"url": "http://localhost:9090", "description": "Prometheus metrics"}
                ],
                "paths": {},
                "components": {
                    "schemas": {},
                    "parameters": {},
                    "responses": {}
                }
            }
            
            # Convert routes to OpenAPI paths
            for route_info in routes:
                path = route_info.get('path', '/')
                methods = route_info.get('methods', ['GET'])
                
                if path not in openapi_spec['paths']:
                    openapi_spec['paths'][path] = {}
                    
                for method in methods:
                    if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                        openapi_spec['paths'][path][method.lower()] = self._create_operation_spec(
                            route_info, method
                        )
                        
            self.logger.info(f"Generated schema with {len(openapi_spec['paths'])} paths")
            return openapi_spec
            
        except Exception as e:
            self.logger.error(f"Error generating schema for {app_path}: {e}")
            return None
            
    def _create_operation_spec(self, route_info: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Create OpenAPI operation specification for a route"""
        operation = {
            "summary": route_info.get('function_name', 'Unknown').replace('_', ' ').title(),
            "description": f"Auto-generated documentation for {route_info.get('function_name', 'unknown')} endpoint",
            "parameters": [],
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {"type": "object"}
                        }
                    }
                },
                "400": {
                    "description": "Bad request"
                },
                "500": {
                    "description": "Internal server error"
                }
            }
        }
        
        # Add path parameters
        path = route_info.get('path', '/')
        if '<' in path and '>' in path:
            import re
            params = re.findall(r'<([^>]+)>', path)
            for param in params:
                param_name = param.split(':')[-1] if ':' in param else param
                operation['parameters'].append({
                    "name": param_name,
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"}
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
        
    def generate_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Generate OpenAPI schemas for all discovered Flask applications"""
        flask_apps = self.discover_flask_apps()
        schemas = {}
        
        for app_path in flask_apps:
            app_name = app_path.stem
            schema = self.generate_schema_for_app(app_path)
            if schema:
                schemas[app_name] = schema
                
        return schemas
        
    def save_schemas(self, schemas: Dict[str, Dict[str, Any]], output_dir: Optional[str] = None) -> List[Path]:
        """Save generated schemas to JSON files"""
        if output_dir is None:
            output_dir = self.project_root / 'generated_api_docs'
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True)
        saved_files = []
        
        for app_name, schema in schemas.items():
            output_file = output_dir / f'{app_name}_openapi.json'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
                
            saved_files.append(output_file)
            self.logger.info(f"Saved OpenAPI schema: {output_file}")
            
        # Create consolidated schema
        consolidated_schema = self._create_consolidated_schema(schemas)
        consolidated_file = output_dir / 'consolidated_api.json'
        
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_schema, f, indent=2, ensure_ascii=False)
            
        saved_files.append(consolidated_file)
        self.logger.info(f"Saved consolidated schema: {consolidated_file}")
        
        return saved_files
        
    def _create_consolidated_schema(self, schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create consolidated OpenAPI schema from multiple applications"""
        consolidated = {
            "openapi": "3.0.0",
            "info": {
                "title": "TestMaster Consolidated API",
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
                "schemas": {},
                "parameters": {},
                "responses": {}
            },
            "tags": []
        }
        
        # Merge paths from all schemas
        for app_name, schema in schemas.items():
            tag = {"name": app_name.replace('_', '-'), "description": f"Endpoints from {app_name}"}
            consolidated["tags"].append(tag)
            
            for path, path_spec in schema.get('paths', {}).items():
                # Prefix path with service name if there are conflicts
                consolidated_path = path
                if consolidated_path in consolidated['paths']:
                    consolidated_path = f'/{app_name}{path}'
                    
                consolidated['paths'][consolidated_path] = path_spec
                
                # Add tags to operations
                for method, operation in path_spec.items():
                    if 'tags' not in operation:
                        operation['tags'] = []
                    operation['tags'].append(app_name.replace('_', '-'))
                    
            # Merge components
            for component_type in ['schemas', 'parameters', 'responses']:
                if component_type in schema.get('components', {}):
                    consolidated['components'][component_type].update(
                        schema['components'][component_type]
                    )
                    
        return consolidated

def main():
    """Main execution function"""
    project_root = Path(__file__).parent.parent.parent
    generator = APISchemaGenerator(str(project_root))
    
    print("🔍 Discovering Flask applications...")
    schemas = generator.generate_all_schemas()
    
    if schemas:
        print(f"📊 Generated schemas for {len(schemas)} applications")
        saved_files = generator.save_schemas(schemas)
        
        print("\n📁 Generated OpenAPI documentation files:")
        for file_path in saved_files:
            print(f"  • {file_path}")
            
        print(f"\n✅ OpenAPI schema generation complete!")
        print(f"📂 Output directory: {project_root / 'generated_api_docs'}")
        
    else:
        print("❌ No Flask applications found or no schemas generated")

if __name__ == '__main__':
    main()