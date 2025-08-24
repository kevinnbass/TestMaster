#!/usr/bin/env python3
"""
API Documentation Generator
===========================

Generates comprehensive API documentation for all components.
This is critical for ensuring no functionality is lost during consolidation.

Part of Phase 1: Comprehensive Analysis & Mapping
"""

import ast
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import inspect


@dataclass
class APIMethod:
    """Documentation for a single API method."""
    name: str
    signature: str
    docstring: Optional[str]
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    decorators: List[str]
    is_public: bool
    is_async: bool
    line_number: int
    complexity_score: int
    examples: List[str] = field(default_factory=list)


@dataclass
class APIClass:
    """Documentation for a class API."""
    name: str
    docstring: Optional[str]
    base_classes: List[str]
    public_methods: List[APIMethod]
    private_methods: List[APIMethod]
    properties: List[Dict[str, Any]]
    class_variables: List[Dict[str, Any]]
    line_number: int
    is_dataclass: bool = False
    is_abc: bool = False


@dataclass
class APIModule:
    """Complete API documentation for a module."""
    module_path: str
    module_name: str
    docstring: Optional[str]
    classes: List[APIClass]
    functions: List[APIMethod]
    constants: List[Dict[str, Any]]
    imports: List[str]
    
    # API characteristics
    public_api_count: int
    has_main_function: bool
    is_script: bool
    api_complexity: str
    
    # Usage patterns
    typical_usage: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)


class APIDocumenter:
    """
    Generates comprehensive API documentation for all components.
    Essential for preserving functionality during consolidation.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.api_docs: Dict[str, APIModule] = {}
        
        # Load previous analysis results
        component_analysis_file = base_path / "phase1_component_analysis.json"
        dependency_analysis_file = base_path / "phase1_dependency_analysis.json"
        
        self.component_analysis = {}
        self.dependency_analysis = {}
        
        if component_analysis_file.exists():
            with open(component_analysis_file, 'r') as f:
                self.component_analysis = json.load(f)
        
        if dependency_analysis_file.exists():
            with open(dependency_analysis_file, 'r') as f:
                self.dependency_analysis = json.load(f)
        
        print(f"[INFO] API Documenter initialized for: {base_path}")
    
    def generate_api_documentation(self, components: List[str] = None) -> Dict[str, APIModule]:
        """Generate comprehensive API documentation for all components."""
        print(f"[INFO] Starting API documentation generation...")
        start_time = time.time()
        
        if components is None:
            # Use components from previous analysis
            if 'components' in self.component_analysis:
                components = list(self.component_analysis['components'].keys())
            else:
                print("[ERROR] No component analysis found. Run analyze_components.py first.")
                return {}
        
        for component_path in components:
            try:
                full_path = self.base_path / component_path
                if full_path.exists() and component_path.endswith('.py'):
                    api_doc = self._document_module_api(full_path, component_path)
                    if api_doc:
                        self.api_docs[component_path] = api_doc
            except Exception as e:
                print(f"[ERROR] Failed to document API for {component_path}: {e}")
        
        duration = time.time() - start_time
        print(f"[INFO] API documentation complete: {len(self.api_docs)} modules documented in {duration:.2f}s")
        
        return self.api_docs
    
    def _document_module_api(self, file_path: Path, component_path: str) -> Optional[APIModule]:
        """Generate API documentation for a single module."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            api_module = APIModule(
                module_path=component_path,
                module_name=file_path.stem,
                docstring=ast.get_docstring(tree),
                classes=[],
                functions=[],
                constants=[],
                imports=[],
                public_api_count=0,
                has_main_function=False,
                is_script=False,
                api_complexity="unknown"
            )
            
            # Analyze module contents
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    api_class = self._document_class_api(node, content)
                    api_module.classes.append(api_class)
                    api_module.public_api_count += len(api_class.public_methods)
                
                elif isinstance(node, ast.FunctionDef):
                    if node.name == "main":
                        api_module.has_main_function = True
                    
                    api_method = self._document_method_api(node, content)
                    api_module.functions.append(api_method)
                    
                    if api_method.is_public:
                        api_module.public_api_count += 1
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._document_import(node)
                    if import_info:
                        api_module.imports.append(import_info)
                
                elif isinstance(node, ast.Assign):
                    # Document module-level constants
                    constant_info = self._document_constant(node, content)
                    if constant_info:
                        api_module.constants.append(constant_info)
            
            # Determine module characteristics
            api_module.is_script = api_module.has_main_function or any(
                line.strip().startswith('if __name__') for line in content.splitlines()
            )
            
            # Estimate API complexity
            api_module.api_complexity = self._estimate_api_complexity(api_module)
            
            # Add usage patterns and integration points
            self._add_usage_patterns(api_module, component_path)
            
            return api_module
            
        except Exception as e:
            print(f"[ERROR] Error documenting API for {file_path}: {e}")
            return None
    
    def _document_class_api(self, node: ast.ClassDef, content: str) -> APIClass:
        """Document the API of a class."""
        public_methods = []
        private_methods = []
        properties = []
        class_variables = []
        
        # Analyze class methods and properties
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = self._document_method_api(item, content)
                if method_doc.is_public:
                    public_methods.append(method_doc)
                else:
                    private_methods.append(method_doc)
            
            elif isinstance(item, ast.AnnAssign) and item.target:
                # Type-annotated class variables
                if isinstance(item.target, ast.Name):
                    class_variables.append({
                        'name': item.target.id,
                        'type': ast.unparse(item.annotation) if item.annotation else None,
                        'line': item.lineno
                    })
        
        # Check for properties (simplified detection)
        for method in public_methods:
            if 'property' in method.decorators:
                properties.append({
                    'name': method.name,
                    'docstring': method.docstring,
                    'line': method.line_number
                })
        
        # Determine class characteristics
        is_dataclass = any('dataclass' in decorator for decorator in 
                          [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list])
        
        is_abc = any(base.id == 'ABC' if isinstance(base, ast.Name) else False 
                    for base in node.bases)
        
        return APIClass(
            name=node.name,
            docstring=ast.get_docstring(node),
            base_classes=[base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
            public_methods=public_methods,
            private_methods=private_methods,
            properties=properties,
            class_variables=class_variables,
            line_number=node.lineno,
            is_dataclass=is_dataclass,
            is_abc=is_abc
        )
    
    def _document_method_api(self, node: ast.FunctionDef, content: str) -> APIMethod:
        """Document the API of a method or function."""
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else None,
                'default': None  # Would need more analysis for defaults
            }
            parameters.append(param_info)
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            else:
                decorators.append(ast.unparse(decorator))
        
        # Determine if public (doesn't start with _)
        is_public = not node.name.startswith('_')
        
        # Calculate complexity (simplified)
        complexity = len(list(ast.walk(node)))
        
        return APIMethod(
            name=node.name,
            signature=f"{node.name}({', '.join(arg.arg for arg in node.args.args)})",
            docstring=ast.get_docstring(node),
            parameters=parameters,
            return_type=ast.unparse(node.returns) if node.returns else None,
            decorators=decorators,
            is_public=is_public,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            line_number=node.lineno,
            complexity_score=complexity
        )
    
    def _document_import(self, node) -> Optional[str]:
        """Document an import statement."""
        try:
            if isinstance(node, ast.Import):
                return f"import {', '.join(alias.name for alias in node.names)}"
            else:  # ImportFrom
                module = node.module or ""
                names = ', '.join(alias.name for alias in node.names)
                return f"from {module} import {names}"
        except Exception:
            return None
    
    def _document_constant(self, node: ast.Assign, content: str) -> Optional[Dict[str, Any]]:
        """Document a module-level constant."""
        try:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.isupper():  # Likely a constant
                    return {
                        'name': name,
                        'value': ast.unparse(node.value),
                        'line': node.lineno
                    }
        except Exception:
            pass
        return None
    
    def _estimate_api_complexity(self, api_module: APIModule) -> str:
        """Estimate the complexity of the module's API."""
        total_methods = api_module.public_api_count
        total_classes = len(api_module.classes)
        
        if total_methods > 50 or total_classes > 10:
            return "high"
        elif total_methods > 20 or total_classes > 5:
            return "medium"
        else:
            return "low"
    
    def _add_usage_patterns(self, api_module: APIModule, component_path: str):
        """Add typical usage patterns and integration points."""
        # Look for usage patterns in component analysis
        if component_path in self.component_analysis.get('components', {}):
            component_info = self.component_analysis['components'][component_path]
            
            # Add typical usage based on component type
            if 'analytics' in component_path.lower():
                api_module.typical_usage.extend([
                    "Import analytics engine",
                    "Initialize with configuration",
                    "Collect metrics",
                    "Generate reports"
                ])
            elif 'test' in component_path.lower():
                api_module.typical_usage.extend([
                    "Import testing utilities", 
                    "Run analysis",
                    "Generate test reports",
                    "Check coverage"
                ])
            elif 'integration' in component_path.lower():
                api_module.typical_usage.extend([
                    "Import integration component",
                    "Configure cross-system connections",
                    "Execute integrations",
                    "Monitor status"
                ])
        
        # Add integration points from dependency analysis
        if component_path in self.dependency_analysis.get('detailed_dependencies', {}):
            dep_info = self.dependency_analysis['detailed_dependencies'][component_path]
            
            # Add API endpoints as integration points
            if dep_info.get('api_endpoints'):
                api_module.integration_points.extend(dep_info['api_endpoints'])
            
            # Add event handlers as integration points
            if dep_info.get('event_handlers'):
                api_module.integration_points.extend(dep_info['event_handlers'])
    
    def generate_consolidation_api_guide(self) -> Dict[str, Any]:
        """Generate API consolidation guide for safe migration."""
        guide = {
            'critical_apis': [],
            'consolidation_groups': {},
            'compatibility_requirements': {},
            'migration_steps': []
        }
        
        # Identify critical APIs (high complexity, many dependents)
        for module_path, api_doc in self.api_docs.items():
            if api_doc.api_complexity == "high" or api_doc.public_api_count > 30:
                guide['critical_apis'].append({
                    'module': module_path,
                    'public_api_count': api_doc.public_api_count,
                    'complexity': api_doc.api_complexity,
                    'classes': len(api_doc.classes),
                    'functions': len(api_doc.functions)
                })
        
        # Group APIs by consolidation target
        analytics_apis = [path for path in self.api_docs.keys() if 'analytics' in path.lower()]
        testing_apis = [path for path in self.api_docs.keys() if 'test' in path.lower() or 'coverage' in path.lower()]
        integration_apis = [path for path in self.api_docs.keys() if 'integration' in path.lower()]
        
        guide['consolidation_groups'] = {
            'analytics': {
                'modules': analytics_apis,
                'target': 'core/intelligence/analytics/',
                'total_public_apis': sum(self.api_docs[path].public_api_count for path in analytics_apis)
            },
            'testing': {
                'modules': testing_apis,
                'target': 'core/intelligence/testing/',
                'total_public_apis': sum(self.api_docs[path].public_api_count for path in testing_apis)
            },
            'integration': {
                'modules': integration_apis,
                'target': 'core/intelligence/integration/',
                'total_public_apis': sum(self.api_docs[path].public_api_count for path in integration_apis)
            }
        }
        
        # Generate compatibility requirements
        for group_name, group_info in guide['consolidation_groups'].items():
            requirements = []
            for module_path in group_info['modules']:
                if module_path in self.api_docs:
                    api_doc = self.api_docs[module_path]
                    requirements.append({
                        'module': module_path,
                        'must_preserve': [cls.name for cls in api_doc.classes],
                        'public_functions': [func.name for func in api_doc.functions if func.is_public],
                        'integration_points': api_doc.integration_points
                    })
            guide['compatibility_requirements'][group_name] = requirements
        
        return guide
    
    def generate_markdown_documentation(self, output_file: str = "phase1_api_documentation.md"):
        """Generate human-readable API documentation in Markdown format."""
        output_path = self.base_path / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# TestMaster API Documentation\n\n")
            f.write("## Overview\n\n")
            f.write(f"This document provides comprehensive API documentation for {len(self.api_docs)} components.\n\n")
            
            # Summary statistics
            total_classes = sum(len(api.classes) for api in self.api_docs.values())
            total_functions = sum(len(api.functions) for api in self.api_docs.values())
            total_public_apis = sum(api.public_api_count for api in self.api_docs.values())
            
            f.write(f"- **Total Modules**: {len(self.api_docs)}\n")
            f.write(f"- **Total Classes**: {total_classes}\n")
            f.write(f"- **Total Functions**: {total_functions}\n")
            f.write(f"- **Total Public APIs**: {total_public_apis}\n\n")
            
            # Group by consolidation target
            f.write("## Components by Consolidation Target\n\n")
            
            analytics_modules = [path for path in self.api_docs.keys() if 'analytics' in path.lower()]
            testing_modules = [path for path in self.api_docs.keys() if 'test' in path.lower() or 'coverage' in path.lower()]
            integration_modules = [path for path in self.api_docs.keys() if 'integration' in path.lower()]
            
            if analytics_modules:
                f.write("### Analytics Components (`core/intelligence/analytics/`)\n\n")
                for module_path in analytics_modules:
                    api_doc = self.api_docs[module_path]
                    f.write(f"#### {api_doc.module_name}\n")
                    f.write(f"- **Path**: `{module_path}`\n")
                    f.write(f"- **Public APIs**: {api_doc.public_api_count}\n")
                    f.write(f"- **Classes**: {len(api_doc.classes)}\n")
                    f.write(f"- **Functions**: {len(api_doc.functions)}\n")
                    f.write(f"- **Complexity**: {api_doc.api_complexity}\n")
                    if api_doc.docstring:
                        f.write(f"- **Description**: {api_doc.docstring.split('.')[0]}\n")
                    f.write("\n")
            
            if testing_modules:
                f.write("### Testing Components (`core/intelligence/testing/`)\n\n")
                for module_path in testing_modules:
                    api_doc = self.api_docs[module_path]
                    f.write(f"#### {api_doc.module_name}\n")
                    f.write(f"- **Path**: `{module_path}`\n")
                    f.write(f"- **Public APIs**: {api_doc.public_api_count}\n")
                    f.write(f"- **Classes**: {len(api_doc.classes)}\n")
                    f.write(f"- **Functions**: {len(api_doc.functions)}\n")
                    f.write(f"- **Complexity**: {api_doc.api_complexity}\n")
                    if api_doc.docstring:
                        f.write(f"- **Description**: {api_doc.docstring.split('.')[0]}\n")
                    f.write("\n")
            
            if integration_modules:
                f.write("### Integration Components (`core/intelligence/integration/`)\n\n")
                for module_path in integration_modules:
                    api_doc = self.api_docs[module_path]
                    f.write(f"#### {api_doc.module_name}\n")
                    f.write(f"- **Path**: `{module_path}`\n")
                    f.write(f"- **Public APIs**: {api_doc.public_api_count}\n")
                    f.write(f"- **Classes**: {len(api_doc.classes)}\n")
                    f.write(f"- **Functions**: {len(api_doc.functions)}\n")
                    f.write(f"- **Complexity**: {api_doc.api_complexity}\n")
                    if api_doc.docstring:
                        f.write(f"- **Description**: {api_doc.docstring.split('.')[0]}\n")
                    f.write("\n")
        
        print(f"[INFO] Markdown API documentation generated: {output_path}")
    
    def generate_report(self, output_file: str = "phase1_api_documentation.json"):
        """Generate comprehensive API documentation report."""
        
        # Generate consolidation guide
        consolidation_guide = self.generate_consolidation_api_guide()
        
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_modules': len(self.api_docs),
                'documentation_generated_for': list(self.api_docs.keys())
            },
            'api_statistics': {
                'total_classes': sum(len(api.classes) for api in self.api_docs.values()),
                'total_functions': sum(len(api.functions) for api in self.api_docs.values()),
                'total_public_apis': sum(api.public_api_count for api in self.api_docs.values()),
                'modules_by_complexity': {
                    'high': len([api for api in self.api_docs.values() if api.api_complexity == 'high']),
                    'medium': len([api for api in self.api_docs.values() if api.api_complexity == 'medium']),
                    'low': len([api for api in self.api_docs.values() if api.api_complexity == 'low'])
                }
            },
            'consolidation_guide': consolidation_guide,
            'detailed_api_docs': {path: asdict(api_doc) for path, api_doc in self.api_docs.items()}
        }
        
        # Write report
        output_path = self.base_path / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[INFO] API documentation report generated: {output_path}")
        return report


def main():
    """Main execution function."""
    print("=" * 80)
    print("API DOCUMENTATION - PHASE 1: COMPREHENSIVE API MAPPING")
    print("=" * 80)
    
    # Initialize documenter
    base_path = Path(".")
    documenter = APIDocumenter(base_path)
    
    # Generate API documentation
    print(f"[INFO] Generating comprehensive API documentation...")
    api_docs = documenter.generate_api_documentation()
    
    # Generate markdown documentation
    documenter.generate_markdown_documentation("phase1_api_documentation.md")
    
    # Generate comprehensive report
    report = documenter.generate_report("phase1_api_documentation.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("API DOCUMENTATION SUMMARY")
    print("=" * 80)
    
    metadata = report['analysis_metadata']
    stats = report['api_statistics']
    guide = report['consolidation_guide']
    
    print(f"Modules documented: {metadata['total_modules']}")
    print(f"Total classes: {stats['total_classes']}")
    print(f"Total functions: {stats['total_functions']}")
    print(f"Total public APIs: {stats['total_public_apis']}")
    
    complexity_dist = stats['modules_by_complexity']
    print(f"Complexity distribution:")
    print(f"  High: {complexity_dist['high']} modules")
    print(f"  Medium: {complexity_dist['medium']} modules")
    print(f"  Low: {complexity_dist['low']} modules")
    
    print(f"\nCONSOLIDATION GROUPS:")
    for group_name, group_info in guide['consolidation_groups'].items():
        print(f"  {group_name.upper()}: {len(group_info['modules'])} modules -> {group_info['target']}")
        print(f"    Total public APIs: {group_info['total_public_apis']}")
    
    print(f"\n[INFO] Detailed API docs saved to: phase1_api_documentation.json")
    print(f"[INFO] Markdown documentation saved to: phase1_api_documentation.md")
    print("[INFO] Phase 1 API documentation complete!")
    
    return api_docs


if __name__ == '__main__':
    main()