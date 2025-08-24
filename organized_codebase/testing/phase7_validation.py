#!/usr/bin/env python3
"""
Phase C7 Validation Script
==========================

Manual validation to ensure 100% feature preservation during Phase C7 consolidation.
Checks that all features from archived files are present in the unified systems.

Author: TestMaster Enhancement System
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Any
import json


class FeatureExtractor:
    """Extract features from code files"""
    
    @staticmethod
    def extract_python_features(file_path: str) -> Dict[str, Set[str]]:
        """Extract classes, functions, and imports from Python files"""
        features = {
            "classes": set(),
            "functions": set(),
            "imports": set(),
            "enums": set(),
            "dataclasses": set()
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    features["classes"].add(node.name)
                    # Check if it's an enum or dataclass
                    for decorator in getattr(node, 'decorator_list', []):
                        if isinstance(decorator, ast.Name):
                            if decorator.id == 'dataclass':
                                features["dataclasses"].add(node.name)
                        
                elif isinstance(node, ast.FunctionDef):
                    features["functions"].add(node.name)
                    
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        features["imports"].add(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            features["imports"].add(f"{node.module}.{alias.name}")
            
            # Check for Enums by looking for Enum inheritance
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == 'Enum':
                            features["enums"].add(node.name)
                        elif isinstance(base, ast.Attribute) and base.attr == 'Enum':
                            features["enums"].add(node.name)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return features
    
    @staticmethod
    def extract_jsx_features(file_path: str) -> Dict[str, Set[str]]:
        """Extract components, functions, and imports from JSX files"""
        features = {
            "components": set(),
            "functions": set(),
            "imports": set(),
            "hooks": set(),
            "constants": set()
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract React components (function and class components)
            component_patterns = [
                r'const\s+([A-Z][a-zA-Z0-9_]*)\s*=\s*\(',
                r'function\s+([A-Z][a-zA-Z0-9_]*)\s*\(',
                r'class\s+([A-Z][a-zA-Z0-9_]*)\s+extends',
                r'export\s+default\s+([A-Z][a-zA-Z0-9_]*)'
            ]
            
            for pattern in component_patterns:
                matches = re.findall(pattern, content)
                features["components"].update(matches)
            
            # Extract functions
            function_patterns = [
                r'const\s+([a-z][a-zA-Z0-9_]*)\s*=\s*\(',
                r'function\s+([a-z][a-zA-Z0-9_]*)\s*\(',
                r'const\s+([a-z][a-zA-Z0-9_]*)\s*=\s*useCallback'
            ]
            
            for pattern in function_patterns:
                matches = re.findall(pattern, content)
                features["functions"].update(matches)
            
            # Extract React hooks
            hook_patterns = [
                r'use([A-Z][a-zA-Z0-9_]*)',
                r'const\s+\[.*?\]\s*=\s*useState',
                r'const\s+\[.*?\]\s*=\s*useEffect'
            ]
            
            for pattern in hook_patterns:
                matches = re.findall(pattern, content)
                features["hooks"].update(matches)
            
            # Extract imports
            import_matches = re.findall(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]', content)
            features["imports"].update(import_matches)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return features


class Phase7Validator:
    """Validate Phase C7 consolidation"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.archive_path = self.base_path / "archive" / "phase7_archive" / "replaced_code"
        self.unified_dashboard_path = self.base_path / "ui" / "unified_dashboard.py"
        self.nocode_enhancement_path = self.base_path / "ui" / "nocode_enhancement.py"
        self.orchestration_dashboard_path = self.base_path / "dashboard" / "src" / "components" / "OrchestrationDashboard.jsx"
        
        self.extractor = FeatureExtractor()
        
    def validate_phase_c7(self) -> Dict[str, Any]:
        """Validate Phase C7 consolidation"""
        print("Phase C7 Validation: UI/Dashboard Consolidation")
        print("=" * 60)
        
        validation_results = {
            "phase": "C7",
            "status": "VALIDATING",
            "features_preserved": True,
            "missing_features": [],
            "consolidated_systems": [],
            "validation_summary": {}
        }
        
        # 1. Extract features from archived files
        archived_features = self._extract_archived_features()
        
        # 2. Extract features from consolidated systems
        consolidated_features = self._extract_consolidated_features()
        
        # 3. Validate feature preservation
        validation_results.update(self._validate_feature_preservation(
            archived_features, consolidated_features
        ))
        
        # 4. Generate validation report
        self._generate_validation_report(validation_results)
        
        return validation_results
    
    def _extract_archived_features(self) -> Dict[str, Dict[str, Set[str]]]:
        """Extract features from archived files"""
        print("Extracting features from archived files...")
        
        archived_features = {}
        
        if not self.archive_path.exists():
            print(f"⚠️  Archive path not found: {self.archive_path}")
            return archived_features
        
        # Extract from archived Python files
        python_files = list(self.archive_path.glob("*.py"))
        for file_path in python_files:
            print(f"  Analyzing archived Python file: {file_path.name}")
            features = self.extractor.extract_python_features(str(file_path))
            archived_features[file_path.name] = features
        
        # Extract from archived JSX files
        jsx_files = list(self.archive_path.glob("*.jsx"))
        for file_path in jsx_files:
            print(f"  Analyzing archived JSX file: {file_path.name}")
            features = self.extractor.extract_jsx_features(str(file_path))
            archived_features[file_path.name] = features
        
        print(f"Extracted features from {len(archived_features)} archived files")
        return archived_features
    
    def _extract_consolidated_features(self) -> Dict[str, Dict[str, Set[str]]]:
        """Extract features from consolidated systems"""
        print("Extracting features from consolidated systems...")
        
        consolidated_features = {}
        
        # Unified dashboard system
        if self.unified_dashboard_path.exists():
            print(f"  Analyzing unified dashboard: {self.unified_dashboard_path.name}")
            features = self.extractor.extract_python_features(str(self.unified_dashboard_path))
            consolidated_features["unified_dashboard.py"] = features
        
        # No-code enhancement
        if self.nocode_enhancement_path.exists():
            print(f"  Analyzing no-code enhancement: {self.nocode_enhancement_path.name}")
            features = self.extractor.extract_python_features(str(self.nocode_enhancement_path))
            consolidated_features["nocode_enhancement.py"] = features
        
        # Current orchestration dashboard (should be enhanced)
        if self.orchestration_dashboard_path.exists():
            print(f"  Analyzing orchestration dashboard: {self.orchestration_dashboard_path.name}")
            features = self.extractor.extract_jsx_features(str(self.orchestration_dashboard_path))
            consolidated_features["OrchestrationDashboard.jsx"] = features
        
        print(f"Extracted features from {len(consolidated_features)} consolidated systems")
        return consolidated_features
    
    def _validate_feature_preservation(self, archived_features: Dict, consolidated_features: Dict) -> Dict:
        """Validate that all archived features are preserved in consolidated systems"""
        print("Validating feature preservation...")
        
        validation_summary = {}
        missing_features = []
        features_preserved = True
        
        # Check each archived file
        for archived_file, archived_file_features in archived_features.items():
            print(f"  Validating features from {archived_file}")
            
            file_validation = {
                "total_features": 0,
                "preserved_features": 0,
                "missing": []
            }
            
            # Count all features in archived file
            total_features = sum(len(features) for features in archived_file_features.values())
            file_validation["total_features"] = total_features
            
            # Check if features exist in any consolidated system
            for feature_type, features in archived_file_features.items():
                for feature in features:
                    found = False
                    
                    # Search in all consolidated systems
                    for consolidated_file, consolidated_file_features in consolidated_features.items():
                        if feature_type in consolidated_file_features:
                            if feature in consolidated_file_features[feature_type]:
                                found = True
                                break
                    
                    if found:
                        file_validation["preserved_features"] += 1
                    else:
                        file_validation["missing"].append(f"{feature_type}: {feature}")
                        missing_features.append(f"{archived_file} -> {feature_type}: {feature}")
                        features_preserved = False
            
            validation_summary[archived_file] = file_validation
            
            # Report validation for this file
            preservation_rate = (file_validation["preserved_features"] / max(1, total_features)) * 100
            status = "PRESERVED" if preservation_rate == 100 else "INCOMPLETE"
            print(f"    {status} - {preservation_rate:.1f}% features preserved ({file_validation['preserved_features']}/{total_features})")
            
            if file_validation["missing"]:
                print(f"    Missing features: {len(file_validation['missing'])}")
                for missing in file_validation["missing"][:3]:  # Show first 3
                    print(f"      - {missing}")
                if len(file_validation["missing"]) > 3:
                    print(f"      ... and {len(file_validation['missing']) - 3} more")
        
        return {
            "features_preserved": features_preserved,
            "missing_features": missing_features,
            "validation_summary": validation_summary
        }
    
    def _generate_validation_report(self, validation_results: Dict):
        """Generate comprehensive validation report"""
        print("\nPHASE C7 VALIDATION REPORT")
        print("=" * 60)
        
        overall_status = "PASSED" if validation_results["features_preserved"] else "FAILED"
        print(f"Overall Status: {overall_status}")
        print(f"Features Preserved: {validation_results['features_preserved']}")
        print(f"Missing Features: {len(validation_results['missing_features'])}")
        
        # Detailed summary
        total_files = len(validation_results["validation_summary"])
        passed_files = sum(1 for v in validation_results["validation_summary"].values() 
                          if len(v["missing"]) == 0)
        
        print(f"\nFile Validation Summary:")
        print(f"  Total archived files: {total_files}")
        print(f"  Fully preserved: {passed_files}")
        print(f"  Partially preserved: {total_files - passed_files}")
        
        # Show detailed results for each file
        for file_name, file_results in validation_results["validation_summary"].items():
            total = file_results["total_features"]
            preserved = file_results["preserved_features"]
            rate = (preserved / max(1, total)) * 100
            
            status = "PASS" if rate == 100 else "WARN" if rate > 80 else "FAIL"
            print(f"\n  {status} {file_name}: {preserved}/{total} features ({rate:.1f}%)")
            
            if file_results["missing"]:
                print(f"    Missing ({len(file_results['missing'])}):")
                for missing in file_results["missing"][:5]:  # Show first 5
                    print(f"      - {missing}")
                if len(file_results["missing"]) > 5:
                    print(f"      ... and {len(file_results['missing']) - 5} more")
        
        # Save validation report
        report_file = "phase7_validation_report.json"
        with open(report_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable_results = json.loads(json.dumps(validation_results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nDetailed validation report saved: {report_file}")
        
        if validation_results["features_preserved"]:
            print("\nVALIDATION PASSED: All features from archived files are preserved!")
            print("   Phase C7 consolidation completed successfully with 100% feature preservation.")
        else:
            print(f"\nVALIDATION CONCERNS: {len(validation_results['missing_features'])} features appear missing.")
            print("   However, these may be:")
            print("   • Renamed or refactored equivalents")
            print("   • Moved to different locations")
            print("   • Consolidated into broader functionality")
            print("   • Internal implementation details not exposed")
            print("\n   Manual review recommended for:")
            for missing in validation_results["missing_features"][:10]:
                print(f"     - {missing}")
            if len(validation_results["missing_features"]) > 10:
                print(f"     ... and {len(validation_results['missing_features']) - 10} more")


def main():
    """Run Phase C7 validation"""
    validator = Phase7Validator()
    results = validator.validate_phase_c7()
    
    # Return appropriate exit code
    return 0 if results["features_preserved"] else 1


if __name__ == "__main__":
    exit(main())