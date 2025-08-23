#!/usr/bin/env python3
"""
Flask Blueprint Analysis System - Agent Delta Implementation
===========================================================

Automated analysis system for extracting API endpoint information from Flask applications.
Analyzes existing codebase to generate comprehensive OpenAPI documentation.

Agent Delta - Phase 1, Hour 3
Created: 2025-08-23 17:30:00
"""

import os
import ast
import sys
import json
import logging
import inspect
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class EndpointInfo:
    """Information about a discovered API endpoint"""
    path: str
    methods: List[str]
    function_name: str
    module_path: str
    line_number: int
    docstring: Optional[str] = None
    parameters: List[Dict[str, Any]] = None
    decorators: List[str] = None
    type_hints: Dict[str, Any] = None
    return_type: Optional[str] = None


@dataclass
class BlueprintInfo:
    """Information about a discovered Flask blueprint"""
    name: str
    url_prefix: str
    module_path: str
    endpoints: List[EndpointInfo]
    import_statements: List[str] = None


class FlaskCodeAnalyzer:
    """
    Analyzes Python source code to extract Flask application structure
    """
    
    def __init__(self):
        self.discovered_routes = []
        self.discovered_blueprints = []
        self.app_instances = []
        self.route_decorators = ['@app.route', '@bp.route', '@blueprint.route']
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for Flask patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            analysis = {
                'file_path': str(file_path),
                'routes': self._extract_routes_from_ast(tree, content),
                'blueprints': self._extract_blueprints_from_ast(tree, content),
                'app_instances': self._extract_app_instances_from_ast(tree, content),
                'imports': self._extract_imports_from_ast(tree)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return {'error': str(e), 'file_path': str(file_path)}
    
    def _extract_routes_from_ast(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract route definitions from AST"""
        routes = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for route decorators
                route_info = self._analyze_route_decorators(node, lines)
                if route_info:
                    # Extract function details
                    func_info = {
                        'function_name': node.name,
                        'line_number': node.lineno,
                        'docstring': ast.get_docstring(node),
                        'parameters': self._extract_function_parameters(node),
                        'decorators': [self._get_decorator_string(d, lines) for d in node.decorator_list],
                        'type_hints': self._extract_type_hints(node),
                        **route_info
                    }
                    routes.append(func_info)
        
        return routes
    
    def _analyze_route_decorators(self, func_node: ast.FunctionDef, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze route decorators on a function"""
        route_info = None
        
        for decorator in func_node.decorator_list:
            decorator_str = self._get_decorator_string(decorator, lines)
            
            # Check for Flask route patterns
            if any(pattern in decorator_str for pattern in ['@app.route', '@bp.route', '@blueprint.route']):
                route_data = self._parse_route_decorator(decorator, decorator_str)
                if route_data:
                    route_info = route_data
                    break
        
        return route_info
    
    def _parse_route_decorator(self, decorator_node: ast.AST, decorator_str: str) -> Optional[Dict[str, Any]]:
        """Parse route decorator to extract path and methods"""
        try:
            # Extract route path
            path = None
            methods = ['GET']  # Default HTTP method
            
            if isinstance(decorator_node, ast.Call):
                # Get the first argument (route path)
                if decorator_node.args:
                    if isinstance(decorator_node.args[0], ast.Constant):
                        path = decorator_node.args[0].value
                    elif isinstance(decorator_node.args[0], ast.Str):  # Python < 3.8
                        path = decorator_node.args[0].s
                
                # Check for methods keyword argument
                for keyword in decorator_node.keywords:
                    if keyword.arg == 'methods':
                        if isinstance(keyword.value, ast.List):
                            methods = []
                            for method_node in keyword.value.elts:
                                if isinstance(method_node, ast.Constant):
                                    methods.append(method_node.value)
                                elif isinstance(method_node, ast.Str):  # Python < 3.8
                                    methods.append(method_node.s)
            
            if path:
                return {
                    'path': path,
                    'methods': methods,
                    'decorator_type': 'route'
                }
                
        except Exception as e:
            logger.debug(f"Failed to parse route decorator: {e}")
        
        return None
    
    def _extract_blueprints_from_ast(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract Blueprint definitions from AST"""
        blueprints = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Look for Blueprint assignments
                if (len(node.targets) == 1 and 
                    isinstance(node.targets[0], ast.Name) and
                    isinstance(node.value, ast.Call)):
                    
                    # Check if it's a Blueprint constructor call
                    call_str = self._get_ast_string(node.value, lines)
                    if 'Blueprint' in call_str:
                        blueprint_info = self._parse_blueprint_constructor(node, lines)
                        if blueprint_info:
                            blueprints.append(blueprint_info)
        
        return blueprints
    
    def _parse_blueprint_constructor(self, assign_node: ast.Assign, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse Blueprint constructor call"""
        try:
            blueprint_var = assign_node.targets[0].name
            constructor = assign_node.value
            
            name = None
            url_prefix = None
            
            # Extract blueprint name (first argument)
            if constructor.args:
                if isinstance(constructor.args[0], ast.Constant):
                    name = constructor.args[0].value
                elif isinstance(constructor.args[0], ast.Str):
                    name = constructor.args[0].s
            
            # Extract url_prefix from kwargs
            for keyword in constructor.keywords:
                if keyword.arg == 'url_prefix':
                    if isinstance(keyword.value, ast.Constant):
                        url_prefix = keyword.value.value
                    elif isinstance(keyword.value, ast.Str):
                        url_prefix = keyword.value.s
            
            if name:
                return {
                    'variable_name': blueprint_var,
                    'name': name,
                    'url_prefix': url_prefix or '',
                    'line_number': assign_node.lineno
                }
                
        except Exception as e:
            logger.debug(f"Failed to parse blueprint constructor: {e}")
        
        return None
    
    def _extract_app_instances_from_ast(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract Flask app instances from AST"""
        apps = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if (len(node.targets) == 1 and 
                    isinstance(node.targets[0], ast.Name) and
                    isinstance(node.value, ast.Call)):
                    
                    call_str = self._get_ast_string(node.value, lines)
                    if 'Flask' in call_str and 'Blueprint' not in call_str:
                        app_info = {
                            'variable_name': node.targets[0].name,
                            'line_number': node.lineno,
                            'constructor_call': call_str
                        }
                        apps.append(app_info)
        
        return apps
    
    def _extract_imports_from_ast(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = [alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in node.names]
                imports.append(f"from {module} import {', '.join(names)}")
        
        return imports
    
    def _extract_function_parameters(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameter information"""
        parameters = []
        
        for arg in func_node.args.args:
            param_info = {
                'name': arg.arg,
                'type_hint': None,
                'default': None
            }
            
            # Extract type annotation
            if arg.annotation:
                param_info['type_hint'] = self._get_type_annotation_string(arg.annotation)
            
            parameters.append(param_info)
        
        # Handle defaults
        defaults = func_node.args.defaults
        if defaults:
            # Defaults apply to the last N parameters
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                param_idx = len(parameters) - num_defaults + i
                if param_idx >= 0 and param_idx < len(parameters):
                    parameters[param_idx]['default'] = self._get_default_value_string(default)
        
        return parameters
    
    def _extract_type_hints(self, func_node: ast.FunctionDef) -> Dict[str, str]:
        """Extract type hints from function"""
        type_hints = {}
        
        # Return type annotation
        if func_node.returns:
            type_hints['return'] = self._get_type_annotation_string(func_node.returns)
        
        # Parameter type annotations (already handled in _extract_function_parameters)
        for arg in func_node.args.args:
            if arg.annotation:
                type_hints[arg.arg] = self._get_type_annotation_string(arg.annotation)
        
        return type_hints
    
    def _get_decorator_string(self, decorator: ast.AST, lines: List[str]) -> str:
        """Get string representation of decorator"""
        try:
            return lines[decorator.lineno - 1].strip()
        except (IndexError, AttributeError):
            return str(decorator)
    
    def _get_ast_string(self, node: ast.AST, lines: List[str]) -> str:
        """Get string representation of AST node"""
        try:
            if hasattr(node, 'lineno'):
                return lines[node.lineno - 1].strip()
        except (IndexError, AttributeError):
            pass
        return str(type(node).__name__)
    
    def _get_type_annotation_string(self, annotation: ast.AST) -> str:
        """Get string representation of type annotation"""
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            elif isinstance(annotation, ast.Subscript):
                # Handle generic types like List[str], Dict[str, int]
                value = self._get_type_annotation_string(annotation.value)
                slice_val = self._get_type_annotation_string(annotation.slice)
                return f"{value}[{slice_val}]"
            elif isinstance(annotation, ast.Tuple):
                # Handle tuple types
                elements = [self._get_type_annotation_string(elt) for elt in annotation.elts]
                return f"({', '.join(elements)})"
            else:
                return str(type(annotation).__name__)
        except Exception:
            return "Any"
    
    def _get_default_value_string(self, default: ast.AST) -> str:
        """Get string representation of default value"""
        try:
            if isinstance(default, ast.Constant):
                return repr(default.value)
            elif isinstance(default, ast.Str):
                return repr(default.s)
            elif isinstance(default, ast.Num):
                return str(default.n)
            elif isinstance(default, ast.NameConstant):
                return str(default.value)
            elif isinstance(default, ast.Name):
                return default.id
            else:
                return "..."
        except Exception:
            return "..."


class FlaskBlueprintAnalyzer:
    """
    Main analyzer that coordinates Flask application analysis across the codebase
    """
    
    def __init__(self, root_path: Path = None):
        self.root_path = root_path or Path.cwd()
        self.code_analyzer = FlaskCodeAnalyzer()
        self.discovered_endpoints = []
        self.discovered_blueprints = []
        
    def analyze_codebase(self, patterns: List[str] = None) -> Dict[str, Any]:
        """Analyze entire codebase for Flask patterns"""
        if patterns is None:
            patterns = ["**/*.py"]
        
        analysis_results = {
            'total_files': 0,
            'analyzed_files': 0,
            'flask_files': 0,
            'endpoints': [],
            'blueprints': [],
            'apps': [],
            'errors': []
        }
        
        # Find all Python files
        python_files = []
        for pattern in patterns:
            python_files.extend(self.root_path.glob(pattern))
        
        analysis_results['total_files'] = len(python_files)
        
        # Analyze each file
        for file_path in python_files:
            try:
                if self._should_skip_file(file_path):
                    continue
                    
                file_analysis = self.code_analyzer.analyze_file(file_path)
                analysis_results['analyzed_files'] += 1
                
                if 'error' in file_analysis:
                    analysis_results['errors'].append(file_analysis)
                    continue
                
                # Check if file contains Flask patterns
                has_flask_content = (
                    file_analysis['routes'] or 
                    file_analysis['blueprints'] or 
                    file_analysis['app_instances']
                )
                
                if has_flask_content:
                    analysis_results['flask_files'] += 1
                    
                    # Collect endpoints
                    for route in file_analysis['routes']:
                        endpoint = EndpointInfo(
                            path=route['path'],
                            methods=route['methods'],
                            function_name=route['function_name'],
                            module_path=str(file_path),
                            line_number=route['line_number'],
                            docstring=route['docstring'],
                            parameters=route['parameters'],
                            decorators=route['decorators'],
                            type_hints=route['type_hints']
                        )
                        analysis_results['endpoints'].append(endpoint)
                    
                    # Collect blueprints
                    for blueprint in file_analysis['blueprints']:
                        blueprint_info = BlueprintInfo(
                            name=blueprint['name'],
                            url_prefix=blueprint['url_prefix'],
                            module_path=str(file_path),
                            endpoints=[],  # Will be populated later
                            import_statements=file_analysis['imports']
                        )
                        analysis_results['blueprints'].append(blueprint_info)
                    
                    # Collect app instances
                    analysis_results['apps'].extend(file_analysis['app_instances'])
                
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                analysis_results['errors'].append({
                    'file_path': str(file_path),
                    'error': str(e)
                })
        
        # Associate endpoints with blueprints
        self._associate_endpoints_with_blueprints(analysis_results)
        
        logger.info(f"Analyzed {analysis_results['analyzed_files']} files, "
                   f"found {len(analysis_results['endpoints'])} endpoints in "
                   f"{analysis_results['flask_files']} Flask files")
        
        return analysis_results
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped during analysis"""
        skip_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'node_modules',
            'venv',
            '.venv',
            'env',
            '.env'
        ]
        
        # Skip files in certain directories
        path_parts = file_path.parts
        if any(pattern in path_parts for pattern in skip_patterns):
            return True
        
        # Skip test files (optional - might want to include them)
        if file_path.name.startswith('test_') or file_path.name.endswith('_test.py'):
            return True
        
        return False
    
    def _associate_endpoints_with_blueprints(self, analysis_results: Dict[str, Any]):
        """Associate discovered endpoints with their blueprints"""
        # Group blueprints by module for easier lookup
        blueprints_by_module = {}
        for blueprint in analysis_results['blueprints']:
            module_path = blueprint.module_path
            if module_path not in blueprints_by_module:
                blueprints_by_module[module_path] = []
            blueprints_by_module[module_path].append(blueprint)
        
        # Associate endpoints with blueprints based on module
        for endpoint in analysis_results['endpoints']:
            module_path = endpoint.module_path
            if module_path in blueprints_by_module:
                # Add endpoint to all blueprints in the same module
                # In practice, would need more sophisticated logic
                for blueprint in blueprints_by_module[module_path]:
                    blueprint.endpoints.append(endpoint)
    
    def generate_openapi_from_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate OpenAPI specification from analysis results"""
        openapi_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "Auto-Generated API Documentation",
                "description": f"Generated from {analysis_results['flask_files']} Flask files",
                "version": "1.0.0"
            },
            "paths": {},
            "components": {
                "schemas": {}
            }
        }
        
        # Process each endpoint
        for endpoint in analysis_results['endpoints']:
            # Determine the OpenAPI path (convert Flask path to OpenAPI format)
            openapi_path = self._convert_flask_path_to_openapi(endpoint.path)
            
            if openapi_path not in openapi_spec["paths"]:
                openapi_spec["paths"][openapi_path] = {}
            
            # Process each HTTP method
            for method in endpoint.methods:
                method_lower = method.lower()
                openapi_spec["paths"][openapi_path][method_lower] = {
                    "summary": self._generate_summary_from_endpoint(endpoint),
                    "description": endpoint.docstring or f"Auto-generated from {endpoint.function_name}",
                    "operationId": f"{endpoint.function_name}_{method_lower}",
                    "parameters": self._generate_parameters_from_endpoint(endpoint),
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        },
                        "500": {
                            "description": "Error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                # Add request body for POST/PUT/PATCH
                if method.upper() in ['POST', 'PUT', 'PATCH']:
                    openapi_spec["paths"][openapi_path][method_lower]["requestBody"] = {
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        }
                    }
        
        return openapi_spec
    
    def _convert_flask_path_to_openapi(self, flask_path: str) -> str:
        """Convert Flask route path to OpenAPI path format"""
        # Convert <param> to {param}
        openapi_path = re.sub(r'<([^>:]+)>', r'{\1}', flask_path)
        # Convert <param:type> to {param}
        openapi_path = re.sub(r'<([^>:]+):[^>]+>', r'{\1}', openapi_path)
        return openapi_path
    
    def _generate_summary_from_endpoint(self, endpoint: EndpointInfo) -> str:
        """Generate a summary for the endpoint"""
        if endpoint.docstring:
            # Use first line of docstring as summary
            first_line = endpoint.docstring.split('\n')[0].strip()
            if first_line:
                return first_line
        
        # Generate from function name
        # Convert snake_case to Title Case
        words = endpoint.function_name.replace('_', ' ').split()
        return ' '.join(word.capitalize() for word in words)
    
    def _generate_parameters_from_endpoint(self, endpoint: EndpointInfo) -> List[Dict[str, Any]]:
        """Generate parameter definitions for the endpoint"""
        parameters = []
        
        # Extract path parameters from the route
        path_params = re.findall(r'<([^>:]+)(?::([^>]+))?>', endpoint.path)
        for param_match in path_params:
            param_name = param_match[0]
            param_type = param_match[1] if param_match[1] else 'string'
            
            parameters.append({
                "name": param_name,
                "in": "path",
                "required": True,
                "schema": {
                    "type": self._convert_flask_type_to_openapi(param_type)
                }
            })
        
        # Add function parameters as query parameters (excluding 'self', path params)
        if endpoint.parameters:
            path_param_names = {match[0] for match in path_params}
            for param in endpoint.parameters:
                if param['name'] not in ['self'] and param['name'] not in path_param_names:
                    parameters.append({
                        "name": param['name'],
                        "in": "query",
                        "required": param['default'] is None,
                        "schema": {
                            "type": self._convert_python_type_to_openapi(param['type_hint'])
                        }
                    })
        
        return parameters
    
    def _convert_flask_type_to_openapi(self, flask_type: str) -> str:
        """Convert Flask parameter type to OpenAPI type"""
        type_mapping = {
            'int': 'integer',
            'float': 'number',
            'string': 'string',
            'path': 'string',
            'uuid': 'string'
        }
        return type_mapping.get(flask_type, 'string')
    
    def _convert_python_type_to_openapi(self, python_type: Optional[str]) -> str:
        """Convert Python type hint to OpenAPI type"""
        if not python_type:
            return 'string'
        
        type_mapping = {
            'int': 'integer',
            'float': 'number',
            'str': 'string',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object',
            'List': 'array',
            'Dict': 'object',
            'Optional': 'string'  # Simplified
        }
        
        # Handle generic types like List[str], Optional[int]
        base_type = python_type.split('[')[0]
        return type_mapping.get(base_type, 'string')


def main():
    """Main function to demonstrate Flask blueprint analysis"""
    print("🔍 AGENT DELTA - Flask Blueprint Analysis System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = FlaskBlueprintAnalyzer(Path.cwd())
    
    print("🔎 Analyzing codebase for Flask patterns...")
    
    # Analyze codebase
    results = analyzer.analyze_codebase()
    
    # Display results
    print(f"📊 ANALYSIS RESULTS:")
    print(f"   • Total files scanned: {results['total_files']}")
    print(f"   • Files analyzed: {results['analyzed_files']}")
    print(f"   • Flask files found: {results['flask_files']}")
    print(f"   • API endpoints discovered: {len(results['endpoints'])}")
    print(f"   • Blueprints found: {len(results['blueprints'])}")
    print(f"   • Flask app instances: {len(results['apps'])}")
    
    if results['errors']:
        print(f"   • Analysis errors: {len(results['errors'])}")
    
    print("\n🔗 DISCOVERED ENDPOINTS:")
    for endpoint in results['endpoints'][:10]:  # Show first 10
        methods_str = ', '.join(endpoint.methods)
        print(f"   {methods_str:20} {endpoint.path:30} ({endpoint.function_name})")
    
    if len(results['endpoints']) > 10:
        print(f"   ... and {len(results['endpoints']) - 10} more endpoints")
    
    print("\n📋 DISCOVERED BLUEPRINTS:")
    for blueprint in results['blueprints']:
        prefix = blueprint.url_prefix or ''
        print(f"   {blueprint.name:20} {prefix:20} ({len(blueprint.endpoints)} endpoints)")
    
    # Generate OpenAPI spec
    print("\n📚 Generating OpenAPI specification...")
    openapi_spec = analyzer.generate_openapi_from_analysis(results)
    
    # Save OpenAPI spec
    output_path = Path("auto_generated_openapi.json")
    with open(output_path, 'w') as f:
        json.dump(openapi_spec, f, indent=2, default=str)
    
    print(f"✅ OpenAPI specification saved to: {output_path}")
    print(f"   • Total endpoints documented: {len(openapi_spec['paths'])}")
    
    print("\n" + "=" * 60)
    print("🎯 FLASK BLUEPRINT ANALYSIS COMPLETE!")
    print("   ✅ Automated AST-based code analysis")
    print("   ✅ Flask route pattern recognition")
    print("   ✅ Blueprint structure extraction") 
    print("   ✅ Type hint analysis and conversion")
    print("   ✅ OpenAPI specification generation")
    print("=" * 60)


if __name__ == "__main__":
    main()