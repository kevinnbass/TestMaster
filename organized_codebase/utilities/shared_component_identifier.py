#!/usr/bin/env python3
"""
Agent C - Shared Component Identification Tool (Hours 29-31)
Identify and categorize shared components for extraction and unification
"""

import os
import ast
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re


@dataclass
class SharedComponent:
    """Shared component data structure"""
    component_id: str
    component_type: str  # ui_component, business_module, data_layer, algorithm
    name: str
    usage_count: int
    file_locations: List[str]
    common_patterns: List[str]
    extraction_potential: str  # high, medium, low
    unification_strategy: str
    estimated_savings: Dict[str, int]


@dataclass
class ComponentUsage:
    """Component usage tracking"""
    component_name: str
    file_path: str
    usage_context: str
    line_number: int
    usage_type: str  # import, inheritance, instantiation, call


@dataclass
class ExtractionRecommendation:
    """Component extraction recommendation"""
    component_id: str
    target_module: str
    extraction_type: str  # utility_function, base_class, mixin, service
    dependencies: List[str]
    migration_plan: List[str]
    risk_assessment: str  # low, medium, high


class ComponentAnalyzer(ast.NodeVisitor):
    """AST analyzer for component identification"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.imports = []
        self.classes = []
        self.functions = []
        self.component_usages = []
        self.current_class = None
        
    def visit_Import(self, node):
        """Track import statements"""
        for alias in node.names:
            self.imports.append({
                'name': alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'type': 'import'
            })
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track from imports"""
        module = node.module or ""
        for alias in node.names:
            self.imports.append({
                'name': f"{module}.{alias.name}" if module else alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'type': 'from_import',
                'module': module
            })
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Track class definitions and inheritance"""
        old_class = self.current_class
        self.current_class = node.name
        
        # Track inheritance
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(self._get_attr_name(base))
                
        self.classes.append({
            'name': node.name,
            'line': node.lineno,
            'bases': base_classes,
            'methods': [],
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
        })
        
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node):
        """Track function definitions"""
        function_info = {
            'name': node.name,
            'line': node.lineno,
            'class': self.current_class,
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'args': [arg.arg for arg in node.args.args],
            'calls': []
        }
        
        # Track function calls within this function
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if call_name:
                    function_info['calls'].append(call_name)
                    
        self.functions.append(function_info)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        """Track function/method calls"""
        call_name = self._get_call_name(node)
        if call_name:
            usage = ComponentUsage(
                component_name=call_name,
                file_path=self.file_path,
                usage_context=self.current_class or "global",
                line_number=node.lineno,
                usage_type="call"
            )
            self.component_usages.append(usage)
        self.generic_visit(node)
        
    def _get_attr_name(self, node):
        """Get attribute name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_attr_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""
        
    def _get_decorator_name(self, decorator):
        """Get decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_attr_name(decorator)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return self._get_attr_name(decorator.func)
        return ""
        
    def _get_call_name(self, node):
        """Get function call name"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attr_name(node.func)
        return ""


class SharedComponentIdentifier:
    """Main shared component identification tool"""
    
    def __init__(self, root_dir: str, output_file: str):
        self.root_dir = Path(root_dir)
        self.output_file = output_file
        self.shared_components = []
        self.component_usages = []
        self.extraction_recommendations = []
        
        # Component tracking
        self.ui_components = defaultdict(list)
        self.business_modules = defaultdict(list)
        self.data_layers = defaultdict(list)
        self.algorithms = defaultdict(list)
        self.utility_functions = defaultdict(list)
        
        self.statistics = {
            'total_files': 0,
            'ui_components_found': 0,
            'business_modules_found': 0,
            'data_layers_found': 0,
            'algorithms_found': 0,
            'utility_functions_found': 0,
            'extraction_opportunities': 0,
            'potential_savings': {
                'lines': 0,
                'files': 0,
                'modules': 0
            }
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def identify_components(self):
        """Identify shared components across the codebase"""
        print("Agent C - Shared Component Identification (Hours 29-31)")
        print(f"Analyzing: {self.root_dir}")
        print(f"Output: {self.output_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        self.logger.info(f"Starting shared component identification for {self.root_dir}")
        
        # Analyze all Python files
        self._analyze_codebase()
        
        # Identify UI components
        self._identify_ui_components()
        
        # Identify business modules
        self._identify_business_modules()
        
        # Identify data layers
        self._identify_data_layers()
        
        # Identify algorithms
        self._identify_algorithms()
        
        # Identify utility functions
        self._identify_utility_functions()
        
        # Generate extraction recommendations
        self._generate_extraction_recommendations()
        
        duration = time.time() - start_time
        
        self._print_results(duration)
        self._save_results()
        
        self.logger.info(f"Shared component identification completed in {duration:.2f} seconds")
        self.logger.info(f"Component analysis report saved to {self.output_file}")
        
    def _analyze_codebase(self):
        """Analyze the entire codebase for component patterns"""
        python_files = list(self.root_dir.rglob("*.py"))
        self.statistics['total_files'] = len(python_files)
        
        self.logger.info(f"Analyzing {len(python_files)} Python files for component patterns")
        
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
                
                if len(self.component_usages) % 5000 == 0:
                    print(f"   Analyzed {len(self.component_usages)} component usages...")
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing {file_path}: {e}")
                
    def _analyze_file(self, file_path: Path):
        """Analyze a single file for component patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse with AST analyzer
            tree = ast.parse(content)
            analyzer = ComponentAnalyzer(str(file_path))
            analyzer.visit(tree)
            
            # Store component usages
            self.component_usages.extend(analyzer.component_usages)
            
            # Categorize components based on file patterns
            self._categorize_file_components(file_path, analyzer)
            
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            self.logger.warning(f"Error parsing {file_path}: {e}")
            
    def _categorize_file_components(self, file_path: Path, analyzer: ComponentAnalyzer):
        """Categorize components based on file analysis"""
        file_str = str(file_path).lower()
        
        # UI Components
        ui_indicators = ['ui', 'component', 'widget', 'view', 'template', 'render', 'display']
        if any(indicator in file_str for indicator in ui_indicators):
            for cls in analyzer.classes:
                self.ui_components[cls['name']].append(str(file_path))
                
        # Business Modules
        business_indicators = ['business', 'logic', 'service', 'manager', 'handler', 'processor']
        if any(indicator in file_str for indicator in business_indicators):
            for cls in analyzer.classes:
                self.business_modules[cls['name']].append(str(file_path))
                
        # Data Layers
        data_indicators = ['data', 'model', 'entity', 'repository', 'dao', 'orm', 'database']
        if any(indicator in file_str for indicator in data_indicators):
            for cls in analyzer.classes:
                self.data_layers[cls['name']].append(str(file_path))
                
        # Algorithms
        algo_indicators = ['algorithm', 'algo', 'math', 'compute', 'calculate', 'optimize']
        if any(indicator in file_str for indicator in algo_indicators):
            for func in analyzer.functions:
                if not func['class']:  # Standalone functions
                    self.algorithms[func['name']].append(str(file_path))
                    
        # Utility Functions
        util_indicators = ['util', 'helper', 'common', 'shared', 'tools']
        if any(indicator in file_str for indicator in util_indicators):
            for func in analyzer.functions:
                if not func['class']:  # Standalone functions
                    self.utility_functions[func['name']].append(str(file_path))
                    
    def _identify_ui_components(self):
        """Identify shared UI components"""
        print("   Identifying UI components...")
        
        for component_name, locations in self.ui_components.items():
            if len(locations) > 1:  # Shared across multiple files
                component = self._create_shared_component(
                    component_name, "ui_component", locations
                )
                if component:
                    self.shared_components.append(component)
                    
        self.statistics['ui_components_found'] = len([c for c in self.shared_components if c.component_type == "ui_component"])
        
    def _identify_business_modules(self):
        """Identify shared business logic modules"""
        print("   Identifying business modules...")
        
        for module_name, locations in self.business_modules.items():
            if len(locations) > 1:
                component = self._create_shared_component(
                    module_name, "business_module", locations
                )
                if component:
                    self.shared_components.append(component)
                    
        self.statistics['business_modules_found'] = len([c for c in self.shared_components if c.component_type == "business_module"])
        
    def _identify_data_layers(self):
        """Identify shared data access layers"""
        print("   Identifying data layers...")
        
        for layer_name, locations in self.data_layers.items():
            if len(locations) > 1:
                component = self._create_shared_component(
                    layer_name, "data_layer", locations
                )
                if component:
                    self.shared_components.append(component)
                    
        self.statistics['data_layers_found'] = len([c for c in self.shared_components if c.component_type == "data_layer"])
        
    def _identify_algorithms(self):
        """Identify shared algorithms"""
        print("   Identifying algorithms...")
        
        for algo_name, locations in self.algorithms.items():
            if len(locations) > 1:
                component = self._create_shared_component(
                    algo_name, "algorithm", locations
                )
                if component:
                    self.shared_components.append(component)
                    
        self.statistics['algorithms_found'] = len([c for c in self.shared_components if c.component_type == "algorithm"])
        
    def _identify_utility_functions(self):
        """Identify shared utility functions"""
        print("   Identifying utility functions...")
        
        for func_name, locations in self.utility_functions.items():
            if len(locations) > 1:
                component = self._create_shared_component(
                    func_name, "utility_function", locations
                )
                if component:
                    self.shared_components.append(component)
                    
        self.statistics['utility_functions_found'] = len([c for c in self.shared_components if c.component_type == "utility_function"])
        
    def _create_shared_component(self, name: str, component_type: str, locations: List[str]) -> Optional[SharedComponent]:
        """Create a shared component object"""
        try:
            usage_count = len(locations)
            unique_locations = list(set(locations))
            
            # Analyze patterns
            common_patterns = self._identify_patterns(name, locations)
            
            # Determine extraction potential
            extraction_potential = self._assess_extraction_potential(name, component_type, usage_count)
            
            # Determine unification strategy
            unification_strategy = self._determine_unification_strategy(component_type, usage_count)
            
            # Estimate savings
            estimated_savings = self._estimate_savings(usage_count, len(unique_locations))
            
            component_id = f"{component_type}_{name}_{len(unique_locations)}"
            
            return SharedComponent(
                component_id=component_id,
                component_type=component_type,
                name=name,
                usage_count=usage_count,
                file_locations=unique_locations,
                common_patterns=common_patterns,
                extraction_potential=extraction_potential,
                unification_strategy=unification_strategy,
                estimated_savings=estimated_savings
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating shared component for {name}: {e}")
            return None
            
    def _identify_patterns(self, name: str, locations: List[str]) -> List[str]:
        """Identify common patterns in component usage"""
        patterns = []
        
        # Naming patterns
        if 'base' in name.lower():
            patterns.append("base_class_pattern")
        if 'abstract' in name.lower():
            patterns.append("abstract_pattern")
        if 'mixin' in name.lower():
            patterns.append("mixin_pattern")
        if name.endswith('Manager'):
            patterns.append("manager_pattern")
        if name.endswith('Handler'):
            patterns.append("handler_pattern")
        if name.endswith('Service'):
            patterns.append("service_pattern")
            
        # Location patterns
        location_dirs = [Path(loc).parent.name for loc in locations]
        dir_counter = Counter(location_dirs)
        if len(dir_counter) == 1:
            patterns.append("single_directory_pattern")
        elif len(dir_counter) > len(locations) * 0.8:
            patterns.append("distributed_pattern")
        else:
            patterns.append("clustered_pattern")
            
        return patterns
        
    def _assess_extraction_potential(self, name: str, component_type: str, usage_count: int) -> str:
        """Assess extraction potential for component"""
        if usage_count >= 10:
            return "high"
        elif usage_count >= 5:
            return "medium" 
        elif usage_count >= 3:
            if component_type in ["algorithm", "utility_function"]:
                return "medium"
            else:
                return "low"
        else:
            return "low"
            
    def _determine_unification_strategy(self, component_type: str, usage_count: int) -> str:
        """Determine unification strategy"""
        strategies = {
            "ui_component": "extract_base_component",
            "business_module": "extract_service_layer",
            "data_layer": "extract_repository_pattern",
            "algorithm": "extract_utility_module",
            "utility_function": "extract_common_utilities"
        }
        
        base_strategy = strategies.get(component_type, "extract_shared_module")
        
        if usage_count >= 10:
            return f"{base_strategy}_with_factory"
        else:
            return base_strategy
            
    def _estimate_savings(self, usage_count: int, file_count: int) -> Dict[str, int]:
        """Estimate potential savings from component extraction"""
        # Conservative estimates
        lines_per_instance = 20  # Average lines saved per usage
        lines_saved = (usage_count - 1) * lines_per_instance
        
        return {
            'lines': lines_saved,
            'files': file_count - 1,  # One canonical implementation
            'maintenance_reduction_percent': min(usage_count * 5, 50)
        }
        
    def _generate_extraction_recommendations(self):
        """Generate recommendations for component extraction"""
        print("   Generating extraction recommendations...")
        
        # Sort by extraction potential and usage count
        high_priority = [c for c in self.shared_components if c.extraction_potential == "high"]
        medium_priority = [c for c in self.shared_components if c.extraction_potential == "medium"]
        
        for component in sorted(high_priority + medium_priority, key=lambda x: x.usage_count, reverse=True):
            recommendation = self._create_extraction_recommendation(component)
            if recommendation:
                self.extraction_recommendations.append(recommendation)
                
        self.statistics['extraction_opportunities'] = len(self.extraction_recommendations)
        
        # Calculate total potential savings
        total_lines = sum(c.estimated_savings['lines'] for c in self.shared_components)
        total_files = sum(c.estimated_savings['files'] for c in self.shared_components)
        unique_modules = len(set(Path(loc).parent for c in self.shared_components for loc in c.file_locations))
        
        self.statistics['potential_savings'] = {
            'lines': total_lines,
            'files': total_files,
            'modules': unique_modules
        }
        
    def _create_extraction_recommendation(self, component: SharedComponent) -> Optional[ExtractionRecommendation]:
        """Create extraction recommendation for component"""
        try:
            # Determine target module
            target_module = self._determine_target_module(component)
            
            # Determine extraction type
            extraction_type = self._determine_extraction_type(component)
            
            # Identify dependencies
            dependencies = self._identify_dependencies(component)
            
            # Create migration plan
            migration_plan = self._create_migration_plan(component, extraction_type)
            
            # Assess risk
            risk_assessment = self._assess_risk(component)
            
            return ExtractionRecommendation(
                component_id=component.component_id,
                target_module=target_module,
                extraction_type=extraction_type,
                dependencies=dependencies,
                migration_plan=migration_plan,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating extraction recommendation for {component.name}: {e}")
            return None
            
    def _determine_target_module(self, component: SharedComponent) -> str:
        """Determine target module for extraction"""
        # Find common parent directory
        common_parent = Path(os.path.commonpath(component.file_locations))
        
        module_names = {
            "ui_component": "common_ui",
            "business_module": "shared_business",
            "data_layer": "common_data",
            "algorithm": "algorithms",
            "utility_function": "utils"
        }
        
        module_name = module_names.get(component.component_type, "shared")
        return str(common_parent / f"{module_name}.py")
        
    def _determine_extraction_type(self, component: SharedComponent) -> str:
        """Determine extraction type"""
        type_mappings = {
            "ui_component": "base_class",
            "business_module": "service",
            "data_layer": "mixin",
            "algorithm": "utility_function",
            "utility_function": "utility_function"
        }
        
        return type_mappings.get(component.component_type, "utility_function")
        
    def _identify_dependencies(self, component: SharedComponent) -> List[str]:
        """Identify component dependencies"""
        # Simplified dependency identification
        dependencies = []
        
        for location in component.file_locations:
            try:
                with open(location, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for imports
                imports = re.findall(r'^(?:from\s+\S+\s+)?import\s+(.+)', content, re.MULTILINE)
                dependencies.extend(imports)
                
            except Exception as e:
                self.logger.warning(f"Error reading dependencies from {location}: {e}")
                
        # Return unique dependencies
        return list(set(dep.split(',')[0].strip() for dep in dependencies if dep))
        
    def _create_migration_plan(self, component: SharedComponent, extraction_type: str) -> List[str]:
        """Create migration plan for component extraction"""
        plans = {
            "base_class": [
                f"1. Create base class {component.name} in target module",
                "2. Identify common attributes and methods",
                "3. Implement base class with proper inheritance",
                "4. Update existing classes to inherit from base",
                "5. Remove duplicate code from child classes",
                "6. Add unit tests for base class",
                "7. Verify all functionality preserved"
            ],
            "service": [
                f"1. Create service class {component.name}Service in target module",
                "2. Extract common business logic methods",
                "3. Implement dependency injection pattern",
                "4. Update callers to use service instance",
                "5. Remove duplicate business logic",
                "6. Add integration tests",
                "7. Verify business rules maintained"
            ],
            "mixin": [
                f"1. Create mixin class {component.name}Mixin in target module",
                "2. Extract common functionality into mixin methods",
                "3. Add mixin to existing class inheritance",
                "4. Remove duplicate methods from classes",
                "5. Ensure method resolution order is correct",
                "6. Add mixin-specific tests",
                "7. Verify all mixed-in functionality works"
            ],
            "utility_function": [
                f"1. Create utility function {component.name} in target module",
                "2. Extract common logic with proper parameters",
                "3. Add comprehensive docstring with examples",
                "4. Replace duplicate code with function calls",
                "5. Handle edge cases and error conditions",
                "6. Add unit tests with full coverage",
                "7. Update all callers to use utility function"
            ]
        }
        
        return plans.get(extraction_type, plans["utility_function"])
        
    def _assess_risk(self, component: SharedComponent) -> str:
        """Assess risk of component extraction"""
        if component.usage_count >= 10:
            return "medium"  # High usage = higher risk if something breaks
        elif len(component.file_locations) >= 5:
            return "medium"  # Distributed across many files
        else:
            return "low"
            
    def _print_results(self, duration):
        """Print component identification results"""
        print(f"\nShared Component Identification Results:")
        print(f"   Files Analyzed: {self.statistics['total_files']:,}")
        print(f"   UI Components: {self.statistics['ui_components_found']}")
        print(f"   Business Modules: {self.statistics['business_modules_found']}")
        print(f"   Data Layers: {self.statistics['data_layers_found']}")
        print(f"   Algorithms: {self.statistics['algorithms_found']}")
        print(f"   Utility Functions: {self.statistics['utility_functions_found']}")
        print(f"   Analysis Duration: {duration:.2f} seconds")
        
        print(f"\nPotential Savings:")
        print(f"   Lines of Code: {self.statistics['potential_savings']['lines']:,}")
        print(f"   Files Affected: {self.statistics['potential_savings']['files']}")
        print(f"   Modules: {self.statistics['potential_savings']['modules']}")
        
        print(f"\nExtraction Opportunities: {self.statistics['extraction_opportunities']}")
        
        if self.extraction_recommendations:
            print(f"\nTop Extraction Recommendations:")
            for rec in sorted(self.extraction_recommendations, key=lambda x: x.component_id)[:5]:
                component = next(c for c in self.shared_components if c.component_id == rec.component_id)
                print(f"   - {rec.extraction_type}: {component.name} ({component.usage_count} usages)")
                
        print(f"\nShared component analysis complete! Report saved to {self.output_file}")
        
    def _save_results(self):
        """Save component identification results to JSON file"""
        results = {
            'metadata': {
                'analysis_type': 'shared_component_identification',
                'timestamp': datetime.now().isoformat(),
                'root_directory': str(self.root_dir),
                'agent': 'Agent C',
                'phase': 'Hours 29-31: Shared Component Identification'
            },
            'statistics': self.statistics,
            'shared_components': [asdict(component) for component in self.shared_components],
            'extraction_recommendations': [asdict(rec) for rec in self.extraction_recommendations],
            'component_categories': {
                'ui_components': len([c for c in self.shared_components if c.component_type == "ui_component"]),
                'business_modules': len([c for c in self.shared_components if c.component_type == "business_module"]),
                'data_layers': len([c for c in self.shared_components if c.component_type == "data_layer"]),
                'algorithms': len([c for c in self.shared_components if c.component_type == "algorithm"]),
                'utility_functions': len([c for c in self.shared_components if c.component_type == "utility_function"])
            },
            'next_steps': [
                'Prioritize high-usage components for extraction',
                'Create shared modules for each component category',
                'Implement extraction recommendations with migration plans',
                'Establish component governance and documentation standards'
            ]
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agent C Shared Component Identifier')
    parser.add_argument('--root', required=True, help='Root directory to analyze')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    identifier = SharedComponentIdentifier(args.root, args.output)
    identifier.identify_components()


if __name__ == "__main__":
    main()