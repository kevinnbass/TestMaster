"""
Modularization Validation Script

Validates that modularized components preserve all functionality 
from the original modules by comparing APIs, methods, and capabilities.
"""

import ast
import inspect
from typing import Dict, List, Set, Any
from pathlib import Path
import importlib.util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModularizationValidator:
    """Validates modularized components against originals"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_ml_modularization(self) -> Dict[str, Any]:
        """Validate ML code analysis modularization"""
        logger.info("Validating ML code analysis modularization...")
        
        # Paths
        original_path = "testmaster/analysis/comprehensive_analysis/archive/ml_code_analysis_original.py"
        modular_core_path = "testmaster/analysis/comprehensive_analysis/ml_analysis/ml_core_analyzer.py"
        modular_tensor_path = "testmaster/analysis/comprehensive_analysis/ml_analysis/ml_tensor_analyzer.py"
        
        # Extract capabilities from original
        original_capabilities = self._extract_module_capabilities(original_path)
        
        # Extract capabilities from modular components
        core_capabilities = self._extract_module_capabilities(modular_core_path)
        tensor_capabilities = self._extract_module_capabilities(modular_tensor_path)
        
        # Combine modular capabilities
        combined_capabilities = self._merge_capabilities([core_capabilities, tensor_capabilities])
        
        # Compare capabilities
        validation = self._compare_capabilities(original_capabilities, combined_capabilities)
        
        # Check for missing functionality
        missing_methods = original_capabilities['methods'] - combined_capabilities['methods']
        missing_classes = original_capabilities['classes'] - combined_capabilities['classes']
        missing_constants = original_capabilities['constants'] - combined_capabilities['constants']
        
        validation.update({
            'missing_methods': list(missing_methods),
            'missing_classes': list(missing_classes), 
            'missing_constants': list(missing_constants),
            'coverage_percentage': self._calculate_coverage(original_capabilities, combined_capabilities)
        })
        
        logger.info(f"ML modularization coverage: {validation['coverage_percentage']:.1f}%")
        
        return validation
    
    def _extract_module_capabilities(self, file_path: str) -> Dict[str, Set[str]]:
        """Extract capabilities (methods, classes, constants) from a module"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            capabilities = {
                'methods': set(),
                'classes': set(),
                'constants': set(),
                'imports': set(),
                'functions': set()
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    capabilities['functions'].add(node.name)
                    if not node.name.startswith('_'):  # Public methods
                        capabilities['methods'].add(node.name)
                
                elif isinstance(node, ast.ClassDef):
                    capabilities['classes'].add(node.name)
                    
                    # Extract class methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = f"{node.name}.{item.name}"
                            capabilities['methods'].add(method_name)
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id.isupper():  # Constants
                                capabilities['constants'].add(target.id)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            capabilities['imports'].add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        capabilities['imports'].add(node.module)
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Error extracting capabilities from {file_path}: {e}")
            return {'methods': set(), 'classes': set(), 'constants': set(), 'imports': set(), 'functions': set()}
    
    def _merge_capabilities(self, capability_lists: List[Dict[str, Set[str]]]) -> Dict[str, Set[str]]:
        """Merge capabilities from multiple modules"""
        merged = {
            'methods': set(),
            'classes': set(),
            'constants': set(),
            'imports': set(),
            'functions': set()
        }
        
        for capabilities in capability_lists:
            for key in merged:
                merged[key].update(capabilities.get(key, set()))
        
        return merged
    
    def _compare_capabilities(self, original: Dict[str, Set[str]], modular: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Compare original vs modular capabilities"""
        comparison = {}
        
        for capability_type in ['methods', 'classes', 'constants', 'functions']:
            original_set = original.get(capability_type, set())
            modular_set = modular.get(capability_type, set())
            
            comparison[capability_type] = {
                'original_count': len(original_set),
                'modular_count': len(modular_set),
                'preserved': len(original_set & modular_set),
                'missing': list(original_set - modular_set),
                'added': list(modular_set - original_set),
                'coverage': len(original_set & modular_set) / len(original_set) if original_set else 1.0
            }
        
        return comparison
    
    def _calculate_coverage(self, original: Dict[str, Set[str]], modular: Dict[str, Set[str]]) -> float:
        """Calculate overall coverage percentage"""
        total_original = sum(len(original.get(key, set())) for key in ['methods', 'classes', 'constants'])
        total_preserved = sum(len(original.get(key, set()) & modular.get(key, set())) 
                            for key in ['methods', 'classes', 'constants'])
        
        return (total_preserved / total_original * 100) if total_original > 0 else 100.0
    
    def validate_all_modularizations(self) -> Dict[str, Any]:
        """Validate all planned modularizations"""
        results = {}
        
        # Validate ML modularization (if files exist)
        ml_original = Path("testmaster/analysis/comprehensive_analysis/archive/ml_code_analysis_original.py")
        if ml_original.exists():
            results['ml_analysis'] = self.validate_ml_modularization()
        
        # Add validation for other modules as they're modularized
        # results['business_rules'] = self.validate_business_rules_modularization()
        # results['semantic_analysis'] = self.validate_semantic_modularization()
        # etc.
        
        return results
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable validation report"""
        report = ["# Modularization Validation Report\n"]
        
        for module_name, validation in results.items():
            report.append(f"## {module_name.replace('_', ' ').title()}")
            
            if 'coverage_percentage' in validation:
                coverage = validation['coverage_percentage']
                status = "✅ PASS" if coverage >= 95 else "⚠️  PARTIAL" if coverage >= 80 else "❌ FAIL"
                report.append(f"**Coverage:** {coverage:.1f}% {status}\n")
            
            # Method coverage
            if 'methods' in validation:
                methods = validation['methods']
                report.append(f"**Methods:** {methods['preserved']}/{methods['original_count']} preserved ({methods['coverage']*100:.1f}%)")
                
                if methods['missing']:
                    report.append(f"**Missing Methods:** {', '.join(methods['missing'])}")
                
                if methods['added']:
                    report.append(f"**Added Methods:** {', '.join(methods['added'])}")
                
                report.append("")
            
            # Class coverage
            if 'classes' in validation:
                classes = validation['classes']
                report.append(f"**Classes:** {classes['preserved']}/{classes['original_count']} preserved ({classes['coverage']*100:.1f}%)")
                
                if classes['missing']:
                    report.append(f"**Missing Classes:** {', '.join(classes['missing'])}")
                
                report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    validator = ModularizationValidator()
    
    # Validate all modularizations
    results = validator.validate_all_modularizations()
    
    # Generate and print report
    report = validator.generate_validation_report(results)
    print(report)
    
    # Save report to file
    with open("modularization_validation_report.md", 'w') as f:
        f.write(report)
    
    logger.info("Validation complete. Report saved to modularization_validation_report.md")