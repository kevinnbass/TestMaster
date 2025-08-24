
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

#!/usr/bin/env python3
"""
Agent C - API Dependency Analysis Tool (Hours 16-18)
Comprehensive API endpoint mapping and dependency analysis
"""

import os
import ast
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import re


@dataclass
class APIEndpoint:
    """API endpoint data structure"""
    path: str
    method: str
    handler: str
    file_path: str
    line_number: int
    parameters: List[str]
    dependencies: List[str]
    middleware: List[str]
    auth_required: bool


@dataclass
class APIDependency:
    """API dependency relationship"""
    source_endpoint: str
    target_endpoint: str
    dependency_type: str
    strength: float


class APIAnalyzer(ast.NodeVisitor):
    """AST visitor for API endpoint analysis"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.endpoints = []
        self.dependencies = []
        self.current_class = None
        self.decorators = []
        
    def visit_FunctionDef(self, node):
        """Visit function definitions to identify API endpoints"""
        # Check for API decorators
        api_decorators = self._extract_api_decorators(node.decorator_list)
        
        if api_decorators:
            endpoint = self._create_endpoint(node, api_decorators)
            if endpoint:
                self.endpoints.append(endpoint)
                
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Visit class definitions for API classes"""
        old_class = self.current_class
        self.current_class = node.name
        
        self.generic_visit(node)
        
        self.current_class = old_class
        
    def _extract_api_decorators(self, decorators):
        """Extract API-related decorators"""
        api_decorators = []
        
        for decorator in decorators:
            decorator_name = self._get_decorator_name(decorator)
            
            if self._is_api_decorator(decorator_name):
                api_decorators.append({
                    'name': decorator_name,
                    'args': self._extract_decorator_args(decorator)
                })
                
        return api_decorators
        
    def _get_decorator_name(self, decorator):
        """Get decorator name from AST node"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_attr_chain(decorator)}"
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return f"{self._get_attr_chain(decorator.func)}"
        return ""
        
    def _get_attr_chain(self, node):
        """Get attribute chain like app.route"""
        if isinstance(node, ast.Attribute):
            base = self._get_attr_chain(node.value) if hasattr(node.value, 'attr') else getattr(node.value, 'id', '')
            return f"{base}.{node.attr}" if base else node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return ""
        
    def _is_api_decorator(self, decorator_name):
        """Check if decorator indicates API endpoint"""
        api_patterns = [
            'route', 'get', 'post', 'put', 'delete', 'patch',
            'app.route', 'api.route', 'blueprint.route',
            'endpoint', 'view', 'api_view',
            'RequestMapping', 'GetMapping', 'PostMapping',
            'Path', 'Query', 'Body'
        ]
        
        return any(pattern in decorator_name.lower() for pattern in api_patterns)
        
    def _extract_decorator_args(self, decorator):
        """Extract arguments from decorator"""
        args = []
        
        if isinstance(decorator, ast.Call):
            for arg in decorator.args:
                if isinstance(arg, ast.Constant):
                    args.append(str(arg.value))
                elif isinstance(arg, ast.Str):
                    args.append(arg.s)
                    
        return args
        
    def _create_endpoint(self, node, api_decorators):
        """Create API endpoint from function node"""
        try:
            # Extract endpoint path and method
            path, method = self._extract_path_method(api_decorators)
            
            if not path:
                return None
                
            # Extract parameters
            parameters = self._extract_parameters(node)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(node)
            
            # Extract middleware
            middleware = self._extract_middleware(api_decorators)
            
            # Check auth requirements
            auth_required = self._check_auth_required(api_decorators, node)
            
            handler_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
            
            return APIEndpoint(
                path=path,
                method=method,
                handler=handler_name,
                file_path=self.file_path,
                line_number=node.lineno,
                parameters=parameters,
                dependencies=dependencies,
                middleware=middleware,
                auth_required=auth_required
            )
            
        except Exception as e:
            logging.warning(f"Error creating endpoint for {node.name}: {e}")
            return None
            
    def _extract_path_method(self, api_decorators):
        """Extract path and HTTP method from decorators"""
        path = ""
        method = "GET"  # Default
        
        for decorator in api_decorators:
            name = decorator['name'].lower()
            args = decorator['args']
            
            if 'route' in name and args:
                path = args[0]
                if len(args) > 1:
                    method = args[1].upper()
            elif name in ['get', 'post', 'put', 'delete', 'patch']:
                method = name.upper()
                if args:
                    path = args[0]
            elif 'mapping' in name:
                method = name.replace('mapping', '').upper()
                if args:
                    path = args[0]
                    
        return path, method
        
    def _extract_parameters(self, node):
        """Extract function parameters"""
        parameters = []
        
        for arg in node.args.args:
            if arg.arg != 'self':
                parameters.append(arg.arg)
                
        return parameters
        
    def _extract_dependencies(self, node):
        """Extract function dependencies"""
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(self._get_attr_chain(child.func))
                    
        return list(set(dependencies))
        
    def _extract_middleware(self, api_decorators):
        """Extract middleware from decorators"""
        middleware = []
        
        for decorator in api_decorators:
            name = decorator['name'].lower()
            
            if any(mw in name for mw in ['auth', 'cors', 'cache', 'rate_limit', 'validate']):
                middleware.append(decorator['name'])
                
        return middleware
        
    def _check_auth_required(self, api_decorators, node):
        """Check if authentication is required"""
        # Check decorators
        for decorator in api_decorators:
            name = decorator['name'].lower()
            if any(auth in name for auth in ['auth', 'login', 'token', 'jwt']):
                return True
                
        # Check function body for auth calls
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if any(auth in child.func.id.lower() for auth in ['auth', 'login', 'verify']):
                        return True
                        
        return False


class APIDependencyMapper:
    """Main API dependency mapping tool"""
    
    def __init__(self, root_dir: str, output_file: str):
        self.root_dir = Path(root_dir)
        self.output_file = output_file
        self.endpoints = []
        self.dependencies = []
        self.statistics = {
            'total_endpoints': 0,
            'total_dependencies': 0,
            'files_analyzed': 0,
            'frameworks_detected': set(),
            'security_issues': [],
            'performance_issues': []
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def analyze_codebase(self):
        """Analyze entire codebase for API dependencies"""
        print("Agent C - API Dependency Analysis (Hours 16-18)")
        print(f"Analyzing: {self.root_dir}")
        print(f"Output: {self.output_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        self.logger.info(f"Starting API dependency analysis for {self.root_dir}")
        
        # Find Python files
        python_files = list(self.root_dir.rglob("*.py"))
        self.logger.info(f"Analyzing API dependencies in {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
                self.statistics['files_analyzed'] += 1
                
                if self.statistics['files_analyzed'] % 100 == 0:
                    print(f"   Processed {self.statistics['files_analyzed']} files...")
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing {file_path}: {e}")
                
        # Analyze endpoint relationships
        self._analyze_endpoint_relationships()
        
        # Detect frameworks
        self._detect_frameworks()
        
        # Security analysis
        self._analyze_security()
        
        # Performance analysis
        self._analyze_performance()
        
        duration = time.time() - start_time
        
        # Update statistics
        self.statistics['total_endpoints'] = len(self.endpoints)
        self.statistics['total_dependencies'] = len(self.dependencies)
        self.statistics['frameworks_detected'] = list(self.statistics['frameworks_detected'])
        
        self._print_results(duration)
        self._save_results()
        
        self.logger.info(f"API dependency analysis completed in {duration:.2f} seconds")
        self.logger.info(f"API dependency analysis report saved to {self.output_file}")
        
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Analyze with APIAnalyzer
            analyzer = APIAnalyzer(str(file_path))
            analyzer.visit(tree)
            
            # Add endpoints
            self.endpoints.extend(analyzer.endpoints)
            
        except SyntaxError:
            self.logger.warning(f"Syntax error in {file_path}, skipping")
        except Exception as e:
            self.logger.warning(f"Error parsing {file_path}: {e}")
            
    def _analyze_endpoint_relationships(self):
        """Analyze relationships between endpoints"""
        for i, endpoint1 in enumerate(self.endpoints):
            for j, endpoint2 in enumerate(self.endpoints):
                if i != j:
                    dependency = self._check_endpoint_dependency(endpoint1, endpoint2)
                    if dependency:
                        self.dependencies.append(dependency)
                        
    def _check_endpoint_dependency(self, endpoint1, endpoint2):
        """Check if endpoint1 depends on endpoint2"""
        # Check if endpoint1 calls endpoint2
        for dep in endpoint1.dependencies:
            if endpoint2.handler in dep or endpoint2.path in dep:
                return APIDependency(
                    source_endpoint=f"{endpoint1.method} {endpoint1.path}",
                    target_endpoint=f"{endpoint2.method} {endpoint2.path}",
                    dependency_type="api_call",
                    strength=0.8
                )
                
        # Check path relationships
        if endpoint1.path in endpoint2.path or endpoint2.path in endpoint1.path:
            if endpoint1.path != endpoint2.path:
                return APIDependency(
                    source_endpoint=f"{endpoint1.method} {endpoint1.path}",
                    target_endpoint=f"{endpoint2.method} {endpoint2.path}",
                    dependency_type="path_hierarchy",
                    strength=0.6
                )
                
        return None
        
    def _detect_frameworks(self):
        """Detect web frameworks in use"""
        framework_patterns = {
            'flask': ['@app.route', 'Flask', 'request', 'render_template'],
            'django': ['django', 'HttpResponse', 'render', 'reverse'],
            'fastapi': ['FastAPI', '@app.get', '@app.post', 'Depends'],
            'tornado': ['tornado', 'RequestHandler', 'Application'],
            'pyramid': ['pyramid', 'view_config', 'Response'],
            'bottle': ['bottle', '@route', 'Bottle'],
            'cherrypy': ['cherrypy', 'expose', 'Application'],
            'falcon': ['falcon', 'Resource', 'App']
        }
        
        all_content = ""
        for endpoint in self.endpoints:
            all_content += f"{endpoint.handler} {' '.join(endpoint.dependencies)} {' '.join(endpoint.middleware)}"
            
        for framework, patterns in framework_patterns.items():
            if any(pattern in all_content for pattern in patterns):
                self.statistics['frameworks_detected'].add(framework)
                
    def _analyze_security(self):
        """Analyze security aspects of API endpoints"""
        security_issues = []
        
        for endpoint in self.endpoints:
            # Check for endpoints without auth
            if not endpoint.auth_required and endpoint.method in ['POST', 'PUT', 'DELETE']:
                security_issues.append({
                    'type': 'missing_auth',
                    'endpoint': f"{endpoint.method} {endpoint.path}",
                    'file': endpoint.file_path,
                    'line': endpoint.line_number
                })
                
            # Check for potential SQL injection
            if any('sql' in dep.lower() or 'query' in dep.lower() for dep in endpoint.dependencies):
                if not any('sanitize' in dep.lower() or 'escape' in dep.lower() for dep in endpoint.dependencies):
                    security_issues.append({
                        'type': 'potential_sql_injection',
                        'endpoint': f"{endpoint.method} {endpoint.path}",
                        'file': endpoint.file_path,
                        'line': endpoint.line_number
                    })
                    
        self.statistics['security_issues'] = security_issues
        
    def _analyze_performance(self):
        """Analyze performance aspects"""
        performance_issues = []
        
        for endpoint in self.endpoints:
            # Check for missing caching
            if not any('cache' in mw.lower() for mw in endpoint.middleware):
                if endpoint.method == 'GET' and not any('dynamic' in dep.lower() for dep in endpoint.dependencies):
                    performance_issues.append({
                        'type': 'missing_cache',
                        'endpoint': f"{endpoint.method} {endpoint.path}",
                        'file': endpoint.file_path,
                        'recommendation': 'Consider adding caching middleware'
                    })
                    
            # Check for missing rate limiting
            if not any('rate' in mw.lower() or 'limit' in mw.lower() for mw in endpoint.middleware):
                performance_issues.append({
                    'type': 'missing_rate_limit',
                    'endpoint': f"{endpoint.method} {endpoint.path}",
                    'file': endpoint.file_path,
                    'recommendation': 'Consider adding rate limiting'
                })
                
        self.statistics['performance_issues'] = performance_issues
        
    def _print_results(self, duration):
        """Print analysis results"""
        print(f"\nAPI Dependency Analysis Results:")
        print(f"   Total Endpoints: {self.statistics['total_endpoints']}")
        print(f"   API Dependencies: {self.statistics['total_dependencies']}")
        print(f"   Files Analyzed: {self.statistics['files_analyzed']}")
        print(f"   Frameworks: {', '.join(self.statistics['frameworks_detected']) if self.statistics['frameworks_detected'] else 'None detected'}")
        print(f"   Security Issues: {len(self.statistics['security_issues'])}")
        print(f"   Performance Issues: {len(self.statistics['performance_issues'])}")
        print(f"   Scan Duration: {duration:.2f} seconds")
        
        if self.statistics['security_issues']:
            print(f"\nSecurity Recommendations:")
            for issue in self.statistics['security_issues'][:3]:
                print(f"   - {issue['type']}: {issue['endpoint']}")
                
        if self.statistics['performance_issues']:
            print(f"\nPerformance Recommendations:")
            for issue in self.statistics['performance_issues'][:3]:
                print(f"   - {issue['type']}: {issue['endpoint']}")
                
        print(f"\nAPI dependency analysis complete! Report saved to {self.output_file}")
        
    def _save_results(self):
        """Save analysis results to JSON file"""
        results = {
            'metadata': {
                'analysis_type': 'api_dependency_mapping',
                'timestamp': datetime.now().isoformat(),
                'root_directory': str(self.root_dir),
                'agent': 'Agent C',
                'phase': 'Hours 16-18: API Dependency Mapping'
            },
            'statistics': self.statistics,
            'endpoints': [asdict(endpoint) for endpoint in self.endpoints],
            'dependencies': [asdict(dep) for dep in self.dependencies],
            'recommendations': {
                'security': self.statistics['security_issues'],
                'performance': self.statistics['performance_issues'],
                'consolidation': self._generate_consolidation_recommendations()
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    def _generate_consolidation_recommendations(self):
        """Generate API consolidation recommendations"""
        recommendations = []
        
        # Group endpoints by path similarity
        path_groups = {}
        for endpoint in self.endpoints:
            base_path = '/'.join(endpoint.path.split('/')[:3])  # First 3 path segments
            if base_path not in path_groups:
                path_groups[base_path] = []
            path_groups[base_path].append(endpoint)
            
        # Find groups with multiple endpoints
        for base_path, endpoints in path_groups.items():
            if len(endpoints) > 5:
                recommendations.append({
                    'type': 'endpoint_consolidation',
                    'base_path': base_path,
                    'endpoint_count': len(endpoints),
                    'recommendation': f'Consider consolidating {len(endpoints)} endpoints under {base_path}'
                })
                
        return recommendations


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agent C API Dependency Mapper')
    parser.add_argument('--root', required=True, help='Root directory to analyze')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    mapper = APIDependencyMapper(args.root, args.output)
    mapper.analyze_codebase()


if __name__ == "__main__":
    main()