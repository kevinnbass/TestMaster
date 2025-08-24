
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
API and Interface Analysis Module
==================================

Implements comprehensive API and interface analysis:
- REST API contract analysis with OpenAPI validation
- GraphQL schema analysis with N+1 detection
- Function signature complexity assessment
- Interface segregation and cohesion analysis
- API evolution tracking and versioning
- WebSocket pattern analysis
- SDK generation readiness
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class APIAnalyzer(BaseAnalyzer):
    """Analyzer for API and interface patterns."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive API and interface analysis."""
        print("[INFO] Analyzing APIs and Interfaces...")
        
        results = {
            "rest_api_analysis": self._analyze_rest_apis(),
            "graphql_analysis": self._analyze_graphql_schemas(),
            "function_signatures": self._analyze_function_signatures(),
            "interface_segregation": self._analyze_interface_segregation(),
            "api_evolution": self._track_api_evolution(),
            "websocket_patterns": self._analyze_websocket_patterns(),
            "sdk_readiness": self._assess_sdk_generation_readiness(),
            "api_metrics": self._calculate_api_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} API aspects")
        return results
    
    def _analyze_rest_apis(self) -> Dict[str, Any]:
        """Analyze REST API implementations and contracts."""
        rest_apis = {
            "endpoints": [],
            "http_methods": defaultdict(int),
            "status_codes": defaultdict(int),
            "versioning_strategy": None,
            "authentication_patterns": [],
            "rate_limiting": [],
            "openapi_compliance": {}
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect Flask/FastAPI/Django REST endpoints
                endpoints = self._detect_rest_endpoints(tree, content)
                for endpoint in endpoints:
                    endpoint["file"] = file_key
                    rest_apis["endpoints"].append(endpoint)
                    rest_apis["http_methods"][endpoint["method"]] += 1
                
                # Detect API versioning
                if not rest_apis["versioning_strategy"]:
                    strategy = self._detect_versioning_strategy(content)
                    if strategy:
                        rest_apis["versioning_strategy"] = strategy
                
                # Detect authentication patterns
                auth_patterns = self._detect_authentication_patterns(tree, content)
                rest_apis["authentication_patterns"].extend(auth_patterns)
                
                # Detect rate limiting
                rate_limits = self._detect_rate_limiting(tree, content)
                rest_apis["rate_limiting"].extend(rate_limits)
                
            except:
                continue
        
        # Analyze endpoint consistency
        rest_apis["consistency_analysis"] = self._analyze_endpoint_consistency(rest_apis["endpoints"])
        
        # Check OpenAPI/Swagger compliance
        rest_apis["openapi_compliance"] = self._check_openapi_compliance()
        
        return rest_apis
    
    def _detect_rest_endpoints(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect REST API endpoints in the code."""
        endpoints = []
        
        for node in ast.walk(tree):
            # Flask route detection
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr == 'route':
                                # Flask route
                                route_info = self._parse_flask_route(decorator, node)
                                if route_info:
                                    endpoints.append(route_info)
                            elif decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                                # FastAPI route
                                route_info = self._parse_fastapi_route(decorator, node)
                                if route_info:
                                    endpoints.append(route_info)
                    elif isinstance(decorator, ast.Name):
                        # Django REST framework decorators
                        if decorator.id in ['api_view', 'action']:
                            route_info = self._parse_django_route(node, content)
                            if route_info:
                                endpoints.append(route_info)
        
        return endpoints
    
    def _parse_flask_route(self, decorator: ast.Call, func: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Parse Flask route decorator."""
        route_info = {
            "framework": "flask",
            "function": func.name,
            "line": func.lineno,
            "method": "GET",  # Default
            "path": None,
            "parameters": []
        }
        
        # Extract route path
        if decorator.args:
            if isinstance(decorator.args[0], ast.Constant):
                route_info["path"] = decorator.args[0].value
        
        # Extract HTTP methods
        for keyword in decorator.keywords:
            if keyword.arg == 'methods':
                if isinstance(keyword.value, ast.List):
                    methods = []
                    for method in keyword.value.elts:
                        if isinstance(method, ast.Constant):
                            methods.append(method.value)
                    if methods:
                        route_info["method"] = methods[0]  # Primary method
                        route_info["all_methods"] = methods
        
        # Extract function parameters
        route_info["parameters"] = self._extract_function_parameters(func)
        
        return route_info if route_info["path"] else None
    
    def _parse_fastapi_route(self, decorator: ast.Call, func: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Parse FastAPI route decorator."""
        route_info = {
            "framework": "fastapi",
            "function": func.name,
            "line": func.lineno,
            "method": decorator.func.attr.upper(),
            "path": None,
            "parameters": [],
            "response_model": None
        }
        
        # Extract route path
        if decorator.args:
            if isinstance(decorator.args[0], ast.Constant):
                route_info["path"] = decorator.args[0].value
        
        # Extract response model
        for keyword in decorator.keywords:
            if keyword.arg == 'response_model':
                if isinstance(keyword.value, ast.Name):
                    route_info["response_model"] = keyword.value.id
        
        # Extract function parameters with type hints
        route_info["parameters"] = self._extract_function_parameters_with_types(func)
        
        return route_info if route_info["path"] else None
    
    def _parse_django_route(self, func: ast.FunctionDef, content: str) -> Optional[Dict[str, Any]]:
        """Parse Django REST framework route."""
        return {
            "framework": "django",
            "function": func.name,
            "line": func.lineno,
            "method": "MULTIPLE",  # Django often handles multiple methods
            "parameters": self._extract_function_parameters(func)
        }
    
    def _analyze_graphql_schemas(self) -> Dict[str, Any]:
        """Analyze GraphQL schema and resolver patterns."""
        graphql_analysis = {
            "schemas": [],
            "resolvers": [],
            "query_complexity": [],
            "n_plus_one_risks": [],
            "subscription_patterns": []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect GraphQL schemas
                if 'graphene' in content or 'graphql' in content.lower():
                    # Detect schema definitions
                    schemas = self._detect_graphql_schemas(tree, content)
                    graphql_analysis["schemas"].extend(schemas)
                    
                    # Detect resolvers
                    resolvers = self._detect_graphql_resolvers(tree)
                    for resolver in resolvers:
                        resolver["file"] = file_key
                        graphql_analysis["resolvers"].append(resolver)
                        
                        # Check for N+1 problems
                        if self._has_n_plus_one_risk(resolver["node"]):
                            graphql_analysis["n_plus_one_risks"].append({
                                "resolver": resolver["name"],
                                "file": file_key,
                                "line": resolver["line"],
                                "risk_type": "potential_n_plus_one"
                            })
                
            except:
                continue
        
        # Calculate query complexity
        graphql_analysis["query_complexity"] = self._calculate_graphql_complexity(graphql_analysis)
        
        return graphql_analysis
    
    def _detect_graphql_schemas(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect GraphQL schema definitions."""
        schemas = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for Graphene ObjectType
                for base in node.bases:
                    if isinstance(base, ast.Attribute):
                        if base.attr in ['ObjectType', 'Query', 'Mutation', 'Subscription']:
                            schemas.append({
                                "name": node.name,
                                "type": base.attr,
                                "line": node.lineno,
                                "fields": self._extract_graphql_fields(node)
                            })
        
        return schemas
    
    def _detect_graphql_resolvers(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect GraphQL resolver methods."""
        resolvers = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('resolve_'):
                    resolvers.append({
                        "name": node.name,
                        "line": node.lineno,
                        "node": node,
                        "complexity": self._calculate_resolver_complexity(node)
                    })
        
        return resolvers
    
    def _has_n_plus_one_risk(self, resolver_node: ast.FunctionDef) -> bool:
        """Check if resolver has N+1 query risk."""
        # Look for patterns that suggest N+1 problems
        for node in ast.walk(resolver_node):
            if isinstance(node, ast.For):
                # Check for database queries inside loops
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Attribute):
                            # Common ORM query methods
                            if inner_node.func.attr in ['get', 'filter', 'all', 'select_related', 'prefetch_related']:
                                return True
        return False
    
    def _analyze_function_signatures(self) -> Dict[str, Any]:
        """Analyze function signature complexity."""
        signature_analysis = {
            "functions": [],
            "complexity_distribution": defaultdict(int),
            "parameter_statistics": {},
            "type_hint_coverage": 0
        }
        
        all_param_counts = []
        typed_functions = 0
        total_functions = 0
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        
                        # Analyze signature
                        sig_info = self._analyze_single_signature(node)
                        sig_info["file"] = file_key
                        signature_analysis["functions"].append(sig_info)
                        
                        # Track statistics
                        all_param_counts.append(sig_info["parameter_count"])
                        signature_analysis["complexity_distribution"][sig_info["complexity_level"]] += 1
                        
                        if sig_info["has_type_hints"]:
                            typed_functions += 1
                
            except:
                continue
        
        # Calculate statistics
        if all_param_counts:
            signature_analysis["parameter_statistics"] = {
                "average": sum(all_param_counts) / len(all_param_counts),
                "max": max(all_param_counts),
                "min": min(all_param_counts),
                "total_functions": total_functions
            }
        
        if total_functions > 0:
            signature_analysis["type_hint_coverage"] = typed_functions / total_functions
        
        return signature_analysis
    
    def _analyze_single_signature(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a single function signature."""
        args = func_node.args
        
        # Count different parameter types
        param_count = len(args.args) - (1 if args.args and args.args[0].arg == 'self' else 0)
        default_count = len(args.defaults)
        has_varargs = args.vararg is not None
        has_kwargs = args.kwarg is not None
        
        # Check for type hints
        has_type_hints = any(arg.annotation is not None for arg in args.args)
        has_return_type = func_node.returns is not None
        
        # Calculate complexity score
        complexity_score = param_count + (default_count * 0.5) + (2 if has_varargs else 0) + (2 if has_kwargs else 0)
        
        complexity_level = "simple"
        if complexity_score > 10:
            complexity_level = "very_complex"
        elif complexity_score > 6:
            complexity_level = "complex"
        elif complexity_score > 3:
            complexity_level = "moderate"
        
        return {
            "function": func_node.name,
            "line": func_node.lineno,
            "parameter_count": param_count,
            "default_parameters": default_count,
            "has_varargs": has_varargs,
            "has_kwargs": has_kwargs,
            "has_type_hints": has_type_hints,
            "has_return_type": has_return_type,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level
        }
    
    def _analyze_interface_segregation(self) -> Dict[str, Any]:
        """Analyze interface segregation and cohesion."""
        interface_analysis = {
            "interfaces": [],
            "fat_interfaces": [],
            "interface_violations": [],
            "cohesion_scores": {}
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if it's an interface (ABC or Protocol)
                        if self._is_interface(node):
                            interface_info = self._analyze_interface(node)
                            interface_info["file"] = file_key
                            interface_analysis["interfaces"].append(interface_info)
                            
                            # Check for fat interface anti-pattern
                            if interface_info["method_count"] > 7:
                                interface_analysis["fat_interfaces"].append({
                                    "interface": node.name,
                                    "file": file_key,
                                    "method_count": interface_info["method_count"],
                                    "recommendation": "Consider splitting into smaller, more focused interfaces"
                                })
                            
                            # Calculate cohesion score
                            cohesion = self._calculate_interface_cohesion(node)
                            interface_analysis["cohesion_scores"][node.name] = cohesion
                
            except:
                continue
        
        return interface_analysis
    
    def _is_interface(self, class_node: ast.ClassDef) -> bool:
        """Check if a class is an interface (ABC or Protocol)."""
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if base.id in ['ABC', 'Protocol']:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr in ['ABC', 'Protocol']:
                    return True
        
        # Check for abstract methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                        return True
        
        return False
    
    def _analyze_interface(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze an interface definition."""
        methods = []
        properties = []
        abstract_methods = []
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
                
                # Check if abstract
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id == 'abstractmethod':
                            abstract_methods.append(node.name)
                        elif decorator.id == 'property':
                            properties.append(node.name)
        
        return {
            "interface": class_node.name,
            "line": class_node.lineno,
            "method_count": len(methods),
            "methods": methods,
            "abstract_methods": abstract_methods,
            "properties": properties,
            "is_pure_interface": len(abstract_methods) == len(methods)
        }
    
    def _calculate_interface_cohesion(self, class_node: ast.ClassDef) -> float:
        """Calculate interface cohesion score."""
        # Simple cohesion based on method naming and parameters
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods.append({
                    "name": node.name,
                    "params": [arg.arg for arg in node.args.args if arg.arg != 'self']
                })
        
        if len(methods) <= 1:
            return 1.0
        
        # Check for common prefixes (suggests cohesion)
        prefixes = defaultdict(int)
        for method in methods:
            parts = method["name"].split('_')
            if len(parts) > 1:
                prefixes[parts[0]] += 1
        
        # Check for shared parameters (suggests cohesion)
        all_params = []
        for method in methods:
            all_params.extend(method["params"])
        
        param_overlap = len(set(all_params)) / max(len(all_params), 1)
        
        # Calculate cohesion score
        max_prefix_count = max(prefixes.values()) if prefixes else 1
        prefix_cohesion = max_prefix_count / len(methods)
        
        return (prefix_cohesion + param_overlap) / 2
    
    def _track_api_evolution(self) -> Dict[str, Any]:
        """Track API evolution and versioning patterns."""
        evolution_tracking = {
            "versioning_patterns": [],
            "deprecated_apis": [],
            "breaking_changes": [],
            "api_lifecycle": {}
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect deprecation patterns
                deprecated = self._detect_deprecated_apis(tree, content)
                for dep in deprecated:
                    dep["file"] = file_key
                    evolution_tracking["deprecated_apis"].append(dep)
                
                # Detect versioning patterns
                versioning = self._detect_versioning_patterns(content)
                if versioning:
                    evolution_tracking["versioning_patterns"].append({
                        "file": file_key,
                        "pattern": versioning
                    })
                
            except:
                continue
        
        return evolution_tracking
    
    def _detect_deprecated_apis(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect deprecated API markers."""
        deprecated = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                # Check for deprecation decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and 'deprecat' in decorator.id.lower():
                        deprecated.append({
                            "name": node.name,
                            "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                            "line": node.lineno,
                            "marker": "decorator"
                        })
                
                # Check for deprecation in docstring
                docstring = ast.get_docstring(node)
                if docstring and 'deprecated' in docstring.lower():
                    deprecated.append({
                        "name": node.name,
                        "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                        "line": node.lineno,
                        "marker": "docstring"
                    })
        
        return deprecated
    
    def _detect_versioning_patterns(self, content: str) -> Optional[str]:
        """Detect API versioning patterns."""
        # URL versioning
        if re.search(r'/v\d+/', content) or re.search(r'/api/v\d+', content):
            return "url_versioning"
        
        # Header versioning
        if 'api-version' in content.lower() or 'x-api-version' in content.lower():
            return "header_versioning"
        
        # Accept header versioning
        if 'application/vnd' in content:
            return "accept_header_versioning"
        
        return None
    
    def _detect_versioning_strategy(self, content: str) -> Optional[str]:
        """Detect overall versioning strategy."""
        strategies = []
        
        if '/v1/' in content or '/v2/' in content:
            strategies.append("url_path")
        if 'api_version' in content:
            strategies.append("parameter")
        if 'Accept:' in content and 'version' in content:
            strategies.append("header")
        
        return strategies[0] if strategies else None
    
    def _detect_authentication_patterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect authentication patterns."""
        patterns = []
        
        # JWT detection
        if 'jwt' in content.lower() or 'jsonwebtoken' in content:
            patterns.append({"type": "JWT", "confidence": "high"})
        
        # OAuth detection
        if 'oauth' in content.lower():
            patterns.append({"type": "OAuth", "confidence": "high"})
        
        # API Key detection
        if 'api_key' in content.lower() or 'apikey' in content.lower():
            patterns.append({"type": "API_Key", "confidence": "high"})
        
        # Basic Auth detection
        if 'basic' in content.lower() and 'auth' in content.lower():
            patterns.append({"type": "Basic_Auth", "confidence": "medium"})
        
        return patterns
    
    def _detect_rate_limiting(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect rate limiting implementations."""
        rate_limits = []
        
        # Common rate limiting decorators/functions
        if 'ratelimit' in content.lower() or 'rate_limit' in content.lower():
            rate_limits.append({"type": "decorator", "detected": True})
        
        if 'throttle' in content.lower():
            rate_limits.append({"type": "throttling", "detected": True})
        
        return rate_limits
    
    def _analyze_endpoint_consistency(self, endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency across endpoints."""
        if not endpoints:
            return {"consistent": True, "issues": []}
        
        issues = []
        
        # Check path naming consistency
        path_patterns = defaultdict(int)
        for endpoint in endpoints:
            if endpoint.get("path"):
                # Check for camelCase vs snake_case
                if re.search(r'[a-z][A-Z]', endpoint["path"]):
                    path_patterns["camelCase"] += 1
                elif '_' in endpoint["path"]:
                    path_patterns["snake_case"] += 1
                else:
                    path_patterns["other"] += 1
        
        if len(path_patterns) > 1:
            issues.append({
                "type": "naming_inconsistency",
                "description": f"Mixed naming conventions: {dict(path_patterns)}"
            })
        
        # Check HTTP method usage
        method_usage = defaultdict(list)
        for endpoint in endpoints:
            if endpoint.get("path") and endpoint.get("method"):
                method_usage[endpoint["path"]].append(endpoint["method"])
        
        # Check for RESTful patterns
        for path, methods in method_usage.items():
            if 'GET' in methods and 'POST' in methods and not any(m in methods for m in ['PUT', 'PATCH', 'DELETE']):
                issues.append({
                    "type": "incomplete_rest",
                    "path": path,
                    "description": "Partial RESTful implementation"
                })
        
        return {
            "consistent": len(issues) == 0,
            "issues": issues
        }
    
    def _check_openapi_compliance(self) -> Dict[str, Any]:
        """Check for OpenAPI/Swagger specification compliance."""
        compliance = {
            "has_specification": False,
            "specification_files": [],
            "compliance_score": 0
        }
        
        # Check for OpenAPI spec files
        spec_files = [
            "openapi.json", "openapi.yaml", "openapi.yml",
            "swagger.json", "swagger.yaml", "swagger.yml",
            "api-spec.json", "api-spec.yaml"
        ]
        
        for spec_file in spec_files:
            spec_path = self.base_path / spec_file
            if spec_path.exists():
                compliance["has_specification"] = True
                compliance["specification_files"].append(spec_file)
        
        # Check for OpenAPI documentation in code
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                if 'openapi' in content.lower() or 'swagger' in content.lower():
                    compliance["compliance_score"] += 10
            except:
                continue
        
        return compliance
    
    def _analyze_websocket_patterns(self) -> Dict[str, Any]:
        """Analyze WebSocket implementation patterns."""
        websocket_analysis = {
            "implementations": [],
            "patterns": [],
            "issues": []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                
                # Detect WebSocket usage
                if 'websocket' in content.lower() or 'socketio' in content.lower():
                    websocket_analysis["implementations"].append({
                        "file": str(py_file.relative_to(self.base_path)),
                        "library": "socketio" if 'socketio' in content.lower() else "websocket"
                    })
                    
                    # Check for common patterns
                    if 'on_connect' in content or 'on_connection' in content:
                        websocket_analysis["patterns"].append("connection_handling")
                    
                    if 'heartbeat' in content.lower() or 'ping' in content.lower():
                        websocket_analysis["patterns"].append("heartbeat_implementation")
                    
                    if 'broadcast' in content.lower():
                        websocket_analysis["patterns"].append("broadcasting")
                    
                    # Check for potential issues
                    if 'reconnect' not in content.lower():
                        websocket_analysis["issues"].append({
                            "type": "missing_reconnection",
                            "file": str(py_file.relative_to(self.base_path))
                        })
            except:
                continue
        
        return websocket_analysis
    
    def _assess_sdk_generation_readiness(self) -> Dict[str, Any]:
        """Assess readiness for SDK generation."""
        readiness = {
            "score": 0,
            "requirements_met": [],
            "missing_requirements": [],
            "recommendations": []
        }
        
        # Check for consistent naming
        rest_apis = self._analyze_rest_apis()
        if rest_apis["consistency_analysis"]["consistent"]:
            readiness["requirements_met"].append("Consistent endpoint naming")
            readiness["score"] += 20
        else:
            readiness["missing_requirements"].append("Inconsistent endpoint naming")
        
        # Check for OpenAPI spec
        openapi = self._check_openapi_compliance()
        if openapi["has_specification"]:
            readiness["requirements_met"].append("OpenAPI specification present")
            readiness["score"] += 30
        else:
            readiness["missing_requirements"].append("No OpenAPI specification")
            readiness["recommendations"].append("Generate OpenAPI specification for automatic SDK generation")
        
        # Check for versioning
        if rest_apis.get("versioning_strategy"):
            readiness["requirements_met"].append("API versioning implemented")
            readiness["score"] += 15
        else:
            readiness["missing_requirements"].append("No API versioning strategy")
        
        # Check for authentication
        if rest_apis.get("authentication_patterns"):
            readiness["requirements_met"].append("Authentication implemented")
            readiness["score"] += 15
        else:
            readiness["missing_requirements"].append("No authentication patterns detected")
        
        # Check for error standardization
        readiness["score"] += 10  # Partial credit
        readiness["recommendations"].append("Standardize error response format across all endpoints")
        
        # Check for pagination patterns
        readiness["score"] += 10  # Partial credit
        readiness["recommendations"].append("Implement consistent pagination patterns")
        
        return readiness
    
    def _extract_function_parameters(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract function parameter names."""
        return [arg.arg for arg in func_node.args.args if arg.arg != 'self']
    
    def _extract_function_parameters_with_types(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters with type hints."""
        params = []
        for arg in func_node.args.args:
            if arg.arg != 'self':
                param_info = {"name": arg.arg}
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        param_info["type"] = arg.annotation.id
                    elif isinstance(arg.annotation, ast.Constant):
                        param_info["type"] = str(arg.annotation.value)
                params.append(param_info)
        return params
    
    def _extract_graphql_fields(self, class_node: ast.ClassDef) -> List[str]:
        """Extract GraphQL field definitions."""
        fields = []
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    fields.append(node.target.id)
        return fields
    
    def _calculate_resolver_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate complexity of a GraphQL resolver."""
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.While)):
                complexity += 2
            elif isinstance(node, ast.If):
                complexity += 1
            elif isinstance(node, ast.Call):
                complexity += 1
        return complexity
    
    def _calculate_graphql_complexity(self, graphql_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate overall GraphQL query complexity."""
        complexity_analysis = []
        
        for schema in graphql_data.get("schemas", []):
            complexity = len(schema.get("fields", []))
            for resolver in graphql_data.get("resolvers", []):
                if resolver["name"].replace("resolve_", "") in schema.get("fields", []):
                    complexity += resolver.get("complexity", 1)
            
            complexity_analysis.append({
                "schema": schema["name"],
                "complexity_score": complexity,
                "risk_level": "high" if complexity > 20 else "medium" if complexity > 10 else "low"
            })
        
        return complexity_analysis
    
    def _calculate_api_metrics(self) -> Dict[str, Any]:
        """Calculate overall API metrics and recommendations."""
        rest_apis = self._analyze_rest_apis()
        graphql = self._analyze_graphql_schemas()
        signatures = self._analyze_function_signatures()
        interfaces = self._analyze_interface_segregation()
        
        metrics = {
            "total_endpoints": len(rest_apis.get("endpoints", [])),
            "total_graphql_resolvers": len(graphql.get("resolvers", [])),
            "average_signature_complexity": signatures.get("parameter_statistics", {}).get("average", 0),
            "interface_count": len(interfaces.get("interfaces", [])),
            "fat_interface_count": len(interfaces.get("fat_interfaces", [])),
            "type_hint_coverage": signatures.get("type_hint_coverage", 0),
            "recommendations": []
        }
        
        # Generate recommendations
        if metrics["average_signature_complexity"] > 5:
            metrics["recommendations"].append("Consider simplifying function signatures with parameter objects")
        
        if metrics["fat_interface_count"] > 0:
            metrics["recommendations"].append("Refactor fat interfaces using Interface Segregation Principle")
        
        if metrics["type_hint_coverage"] < 0.8:
            metrics["recommendations"].append("Increase type hint coverage for better API documentation")
        
        if graphql.get("n_plus_one_risks"):
            metrics["recommendations"].append("Address N+1 query issues in GraphQL resolvers using DataLoader")
        
        return metrics