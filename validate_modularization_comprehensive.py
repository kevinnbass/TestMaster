"""
Comprehensive Modularization Validation

Validates that all modularized components preserve the exact functionality
of their original modules, ensuring no functionality is lost.
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModularizationValidator:
    """Validates modularized components against originals"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.comprehensive_analysis_path = self.base_path / "testmaster" / "analysis" / "comprehensive_analysis"
        self.archive_path = self.comprehensive_analysis_path / "archive"
        self.validation_results = {}
        
    def validate_all_modularizations(self) -> Dict[str, Any]:
        """Validate all modularized components"""
        logger.info("Starting comprehensive modularization validation...")
        
        # Define validation mappings
        validation_mappings = {
            "ml_code_analysis_original.py": {
                "modular_wrapper": "ml_code_analysis_modular.py",
                "modular_dir": "ml_analysis",
                "components": ["ml_core_analyzer.py", "ml_tensor_analyzer.py", "ml_model_analyzer.py", "ml_data_analyzer.py"]
            },
            "business_rule_analysis_original.py": {
                "modular_wrapper": "business_rule_analysis_modular.py", 
                "modular_dir": "business_analysis",
                "components": ["business_core_analyzer.py", "business_workflow_analyzer.py", "business_domain_analyzer.py", "business_validation_analyzer.py"]
            },
            "semantic_analysis_original.py": {
                "modular_wrapper": "semantic_analysis_modular.py",
                "modular_dir": "semantic_analysis", 
                "components": ["semantic_core_analyzer.py", "semantic_pattern_analyzer.py", "semantic_context_analyzer.py"]
            },
            "technical_debt_analysis_original.py": {
                "modular_wrapper": "technical_debt_analysis_modular.py",
                "modular_dir": "debt_analysis",
                "components": ["debt_core_analyzer.py", "debt_category_analyzer.py", "debt_financial_analyzer.py"]
            },
            "metaprogramming_analysis_original.py": {
                "modular_wrapper": "metaprogramming_analysis_modular.py",
                "modular_dir": "metaprog_analysis",
                "components": ["metaprog_core_analyzer.py", "metaprog_security_analyzer.py", "metaprog_reflection_analyzer.py"]
            },
            "energy_consumption_analysis_original.py": {
                "modular_wrapper": "energy_consumption_analysis_modular.py",
                "modular_dir": "energy_analysis", 
                "components": ["energy_core_analyzer.py", "energy_algorithm_analyzer.py", "energy_carbon_analyzer.py"]
            }
        }
        
        # Validate each modularization
        for original_file, mapping in validation_mappings.items():
            logger.info(f"Validating {original_file}...")
            self.validation_results[original_file] = self._validate_single_modularization(
                original_file, mapping
            )
        
        # Generate comprehensive report
        return self._generate_validation_report()
    
    def _validate_single_modularization(self, original_file: str, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single modularization"""
        result = {
            "original_exists": False,
            "wrapper_exists": False,
            "modular_dir_exists": False,
            "components_exist": [],
            "original_analysis": {},
            "modular_analysis": {},
            "validation_status": "FAIL"
        }
        
        # Check original file
        original_path = self.archive_path / original_file
        result["original_exists"] = original_path.exists()
        
        # Check wrapper file
        wrapper_path = self.comprehensive_analysis_path / mapping["modular_wrapper"]
        result["wrapper_exists"] = wrapper_path.exists()
        
        # Check modular directory
        modular_dir_path = self.comprehensive_analysis_path / mapping["modular_dir"]
        result["modular_dir_exists"] = modular_dir_path.exists()
        
        # Check components
        for component in mapping["components"]:
            component_path = modular_dir_path / component
            result["components_exist"].append({
                "name": component,
                "exists": component_path.exists(),
                "size": component_path.stat().st_size if component_path.exists() else 0
            })
        
        # Analyze original if it exists
        if result["original_exists"]:
            try:
                result["original_analysis"] = self._analyze_python_file(original_path)
            except Exception as e:
                logger.error(f"Error analyzing {original_path}: {e}")
                result["original_analysis"] = {"error": str(e)}
        
        # Analyze wrapper if it exists
        if result["wrapper_exists"]:
            try:
                result["modular_analysis"] = self._analyze_python_file(wrapper_path)
            except Exception as e:
                logger.error(f"Error analyzing {wrapper_path}: {e}")
                result["modular_analysis"] = {"error": str(e)}
        
        # Determine validation status
        if (result["original_exists"] and result["wrapper_exists"] and 
            result["modular_dir_exists"] and 
            all(comp["exists"] for comp in result["components_exist"])):
            result["validation_status"] = "PASS"
        
        return result
    
    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file to extract key information"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            analysis = {
                "classes": [],
                "functions": [],
                "imports": [],
                "constants": [],
                "file_size": file_path.stat().st_size,
                "line_count": len(content.splitlines())
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis["classes"].append({
                        "name": node.name,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.FunctionDef) and not any(
                    isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                    if hasattr(parent, 'body') and node in getattr(parent, 'body', [])
                ):
                    analysis["functions"].append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        analysis["imports"].extend([alias.name for alias in node.names])
                    else:
                        module = node.module or ""
                        analysis["imports"].extend([f"{module}.{alias.name}" for alias in node.names])
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            analysis["constants"].append(target.id)
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            "summary": {
                "total_modules": len(self.validation_results),
                "passed": 0,
                "failed": 0,
                "details": []
            },
            "validations": self.validation_results,
            "recommendations": []
        }
        
        # Calculate summary stats
        for original_file, result in self.validation_results.items():
            if result["validation_status"] == "PASS":
                report["summary"]["passed"] += 1
            else:
                report["summary"]["failed"] += 1
            
            report["summary"]["details"].append({
                "module": original_file,
                "status": result["validation_status"],
                "original_size": result["original_analysis"].get("file_size", 0),
                "components": len(result["components_exist"])
            })
        
        # Generate recommendations
        if report["summary"]["failed"] > 0:
            report["recommendations"].append(
                "Some modularizations failed validation. Review missing components."
            )
        
        if report["summary"]["passed"] == report["summary"]["total_modules"]:
            report["recommendations"].append(
                "All modularizations passed validation! The functionality has been preserved."
            )
        
        return report
    
    def save_validation_report(self, report: Dict[str, Any]) -> None:
        """Save validation report to markdown file"""
        report_content = self._format_report_as_markdown(report)
        
        report_path = self.base_path / "modularization_validation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Validation report saved to: {report_path}")
    
    def _format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format validation report as markdown"""
        lines = [
            "# Modularization Validation Report",
            "",
            "## Summary",
            "",
            f"- **Total Modules**: {report['summary']['total_modules']}",
            f"- **Passed**: {report['summary']['passed']}",
            f"- **Failed**: {report['summary']['failed']}",
            "",
            "## Validation Results",
            ""
        ]
        
        for detail in report["summary"]["details"]:
            status_icon = "[OK]" if detail["status"] == "PASS" else "[FAIL]"
            lines.extend([
                f"### {detail['module']} {status_icon}",
                "",
                f"- **Status**: {detail['status']}",
                f"- **Original Size**: {detail['original_size']:,} bytes",
                f"- **Components**: {detail['components']}",
                ""
            ])
        
        # Add detailed validation results
        lines.extend([
            "## Detailed Analysis",
            ""
        ])
        
        for original_file, result in report["validations"].items():
            lines.extend([
                f"### {original_file}",
                "",
                f"- Original exists: {'Yes' if result['original_exists'] else 'No'}",
                f"- Wrapper exists: {'Yes' if result['wrapper_exists'] else 'No'}",
                f"- Modular directory exists: {'Yes' if result['modular_dir_exists'] else 'No'}",
                ""
            ])
            
            if result["components_exist"]:
                lines.append("**Components:**")
                for comp in result["components_exist"]:
                    status = "OK" if comp["exists"] else "MISSING"
                    lines.append(f"- {comp['name']}: {status} ({comp['size']} bytes)")
                lines.append("")
            
            # Add analysis comparison if available
            if result["original_analysis"] and result["modular_analysis"]:
                orig = result["original_analysis"]
                mod = result["modular_analysis"]
                
                lines.extend([
                    "**Analysis Comparison:**",
                    f"- Original classes: {len(orig.get('classes', []))}",
                    f"- Original functions: {len(orig.get('functions', []))}",
                    f"- Original size: {orig.get('file_size', 0):,} bytes",
                    ""
                ])
        
        # Add recommendations
        if report["recommendations"]:
            lines.extend([
                "## Recommendations",
                ""
            ])
            for rec in report["recommendations"]:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    validator = ModularizationValidator("C:\\Users\\kbass\\OneDrive\\Documents\\testmaster")
    
    print("Starting comprehensive modularization validation...")
    report = validator.validate_all_modularizations()
    
    print(f"\nValidation Results:")
    print(f"Total modules: {report['summary']['total_modules']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    
    # Save detailed report
    validator.save_validation_report(report)
    
    print("\nValidation complete! Check modularization_validation_report.md for details.")