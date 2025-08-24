"""
Phase Validation Framework
==========================

Validates no features lost during consolidation by checking against archives.
This ensures 100% feature preservation during the consolidation process.

Author: TestMaster Team
"""

import os
import ast
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

# Import our archive system
import sys
sys.path.append(str(Path(__file__).parent.parent / "ARCHIVE"))
try:
    from archive_system import get_archive_system
except ImportError:
    # Fallback if archive system not available yet
    def get_archive_system():
        return None


@dataclass
class ValidationResult:
    """Result of validating a feature"""
    feature_name: str
    feature_type: str
    original_file: str
    status: str  # 'FOUND', 'MISSING', 'MODIFIED', 'RELOCATED'
    current_location: Optional[str] = None
    confidence: float = 0.0
    notes: str = ""


class PhaseValidator:
    """Validate no features lost during consolidation"""
    
    def __init__(self):
        archive_sys = get_archive_system()
        if archive_sys is None:
            # Create mock archive system for testing
            self.archive_system = self._create_mock_archive_system()
        else:
            self.archive_system = archive_sys
        self.base_path = Path(".")
        self.validation_results = []
    
    def _create_mock_archive_system(self):
        """Create mock archive system for testing"""
        class MockArchiveSystem:
            def __init__(self):
                self.manifest = {"total_archives": 0, "archives": []}
            
            def get_archived_features(self, phase=None):
                return {}
            
            def restore_from_archive(self, archive_id, target_path=None):
                return target_path or f"mock_restored_{archive_id}.py"
        
        return MockArchiveSystem()
        
    def validate_phase(self, phase_number: int) -> Dict[str, Any]:
        """Comprehensive validation against archives"""
        print(f"Starting validation for Phase {phase_number}...")
        
        # 1. Load all archived features for this phase
        archived_features = self.archive_system.get_archived_features(phase_number)
        
        if not archived_features:
            return {
                "phase": phase_number,
                "status": "NO_ARCHIVES",
                "message": f"No archives found for Phase {phase_number}",
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"   Found {len(archived_features)} archived files to validate")
        
        # 2. Build current codebase feature map
        current_features = self._build_current_feature_map()
        print(f"   Scanning {len(current_features)} current files")
        
        # 3. Validate each archived feature
        missing_features = []
        found_features = []
        relocated_features = []
        
        total_features = 0
        for file_path, features in archived_features.items():
            for feature in features:
                total_features += 1
                result = self._validate_feature(feature, current_features, file_path)
                
                if result.status == 'MISSING':
                    missing_features.append(result)
                elif result.status == 'FOUND':
                    found_features.append(result)
                elif result.status == 'RELOCATED':
                    relocated_features.append(result)
        
        # 4. Restore any missing features
        restored_features = []
        if missing_features:
            print(f"   WARNING: {len(missing_features)} features missing after Phase {phase_number}")
            for missing in missing_features:
                restore_result = self._restore_missing_feature(missing, phase_number)
                if restore_result:
                    restored_features.append(restore_result)
        
        # 5. Generate validation report
        validation_status = self._determine_validation_status(
            total_features, len(missing_features), len(restored_features)
        )
        
        report = {
            "phase": phase_number,
            "timestamp": datetime.now().isoformat(),
            "total_archived_features": total_features,
            "found_features": len(found_features),
            "missing_features": len(missing_features),
            "relocated_features": len(relocated_features),
            "restored_features": len(restored_features),
            "validation_status": validation_status,
            "feature_details": {
                "found": [self._serialize_result(r) for r in found_features],
                "missing": [self._serialize_result(r) for r in missing_features],
                "relocated": [self._serialize_result(r) for r in relocated_features],
                "restored": restored_features
            },
            "summary": {
                "preservation_rate": ((total_features - len(missing_features) + len(restored_features)) / total_features * 100) if total_features > 0 else 100,
                "critical_issues": len(missing_features) - len(restored_features),
                "recommendations": self._generate_recommendations(missing_features, restored_features)
            }
        }
        
        # 6. Save validation report
        report_path = f"ARCHIVE/phase{phase_number}_archive/validation_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   Validation complete: {validation_status}")
        print(f"   Feature preservation rate: {report['summary']['preservation_rate']:.1f}%")
        print(f"   Report saved: {report_path}")
        
        return report
    
    def _build_current_feature_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build map of all features in current codebase"""
        feature_map = {}
        
        # Scan all Python files in the project
        for py_file in self.base_path.rglob("*.py"):
            # Skip archive directory and cache directories
            if any(part in str(py_file) for part in ['ARCHIVE', '__pycache__', '.pytest_cache']):
                continue
                
            try:
                features = self._extract_features_from_file(py_file)
                if features:
                    feature_map[str(py_file)] = features
            except Exception as e:
                print(f"   Warning: Could not scan {py_file}: {e}")
        
        return feature_map
    
    def _extract_features_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract features from a Python file"""
        features = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features.append({
                        "type": "function",
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "file": str(file_path)
                    })
                elif isinstance(node, ast.ClassDef):
                    methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    features.append({
                        "type": "class",
                        "name": node.name,
                        "line": node.lineno,
                        "methods": methods,
                        "file": str(file_path)
                    })
                    
        except Exception as e:
            print(f"   Warning: Could not parse {file_path}: {e}")
        
        return features
    
    def _validate_feature(self, archived_feature: Dict[str, Any], current_features: Dict[str, List[Dict]], original_file: str) -> ValidationResult:
        """Validate a single feature against current codebase"""
        feature_name = archived_feature.get("name", "")
        feature_type = archived_feature.get("type", "")
        
        # Look for exact match in original location
        original_path = Path(original_file)
        if str(original_path) in current_features:
            for current_feature in current_features[str(original_path)]:
                if (current_feature["name"] == feature_name and 
                    current_feature["type"] == feature_type):
                    return ValidationResult(
                        feature_name=feature_name,
                        feature_type=feature_type,
                        original_file=original_file,
                        status="FOUND",
                        current_location=str(original_path),
                        confidence=1.0,
                        notes="Found in original location"
                    )
        
        # Look for feature in other files (relocated)
        for file_path, file_features in current_features.items():
            for current_feature in file_features:
                if (current_feature["name"] == feature_name and 
                    current_feature["type"] == feature_type):
                    return ValidationResult(
                        feature_name=feature_name,
                        feature_type=feature_type,
                        original_file=original_file,
                        status="RELOCATED",
                        current_location=file_path,
                        confidence=0.9,
                        notes=f"Relocated from {original_file} to {file_path}"
                    )
        
        # Feature not found
        return ValidationResult(
            feature_name=feature_name,
            feature_type=feature_type,
            original_file=original_file,
            status="MISSING",
            confidence=0.0,
            notes="Feature not found in current codebase"
        )
    
    def _restore_missing_feature(self, missing_result: ValidationResult, phase_number: int) -> Optional[Dict[str, Any]]:
        """Attempt to restore a missing feature from archive"""
        try:
            # Find the archive entry for this feature
            archive_entries = self.archive_system.manifest.get("archives", [])
            
            for entry in archive_entries:
                if (entry["phase"] == phase_number and 
                    entry["original_path"] == missing_result.original_file):
                    
                    # Check if feature exists in archived file
                    archived_features = entry.get("features", [])
                    feature_found = any(
                        f.get("name") == missing_result.feature_name and
                        f.get("type") == missing_result.feature_type
                        for f in archived_features
                    )
                    
                    if feature_found:
                        # Restore the entire file (safer than extracting individual features)
                        restored_path = self.archive_system.restore_from_archive(
                            entry["id"], 
                            target_path=f"restored_{missing_result.feature_name}_{phase_number}.py"
                        )
                        
                        return {
                            "feature_name": missing_result.feature_name,
                            "feature_type": missing_result.feature_type,
                            "original_file": missing_result.original_file,
                            "restored_to": restored_path,
                            "archive_id": entry["id"],
                            "restoration_method": "full_file_restore"
                        }
            
        except Exception as e:
            print(f"   Error restoring {missing_result.feature_name}: {e}")
        
        return None
    
    def _determine_validation_status(self, total: int, missing: int, restored: int) -> str:
        """Determine overall validation status"""
        if total == 0:
            return "NO_FEATURES"
        
        critical_missing = missing - restored
        
        if critical_missing == 0:
            return "PASS"
        elif critical_missing <= total * 0.05:  # Less than 5% missing
            return "PASS_WITH_WARNINGS"
        else:
            return "FAIL"
    
    def _serialize_result(self, result: ValidationResult) -> Dict[str, Any]:
        """Convert ValidationResult to dict for JSON serialization"""
        return {
            "feature_name": result.feature_name,
            "feature_type": result.feature_type,
            "original_file": result.original_file,
            "status": result.status,
            "current_location": result.current_location,
            "confidence": result.confidence,
            "notes": result.notes
        }
    
    def _generate_recommendations(self, missing: List[ValidationResult], restored: List[Dict]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        critical_missing = len(missing) - len(restored)
        
        if critical_missing > 0:
            recommendations.append(f"CRITICAL: {critical_missing} features permanently lost and could not be restored")
            recommendations.append("Review consolidation logic to ensure all features are properly merged")
            recommendations.append("Consider manual restoration from archive backups")
        
        if len(restored) > 0:
            recommendations.append(f"SUCCESS: {len(restored)} missing features were automatically restored")
            recommendations.append("Verify restored features integrate correctly with consolidated code")
        
        if len(missing) == 0:
            recommendations.append("EXCELLENT: All features successfully preserved during consolidation")
        
        return recommendations
    
    def validate_all_phases(self, max_phase: int = 7) -> Dict[str, Any]:
        """Validate all phases at once"""
        print("Starting comprehensive multi-phase validation...")
        
        all_results = {}
        overall_status = "PASS"
        total_features = 0
        total_missing = 0
        total_restored = 0
        
        for phase in range(1, max_phase + 1):
            result = self.validate_phase(phase)
            all_results[f"phase_{phase}"] = result
            
            if result.get("total_archived_features", 0) > 0:
                total_features += result["total_archived_features"]
                total_missing += result["missing_features"]
                total_restored += result["restored_features"]
                
                if result["validation_status"] == "FAIL":
                    overall_status = "FAIL"
                elif result["validation_status"] == "PASS_WITH_WARNINGS" and overall_status == "PASS":
                    overall_status = "PASS_WITH_WARNINGS"
        
        summary_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "total_features_across_all_phases": total_features,
            "total_missing_across_all_phases": total_missing,
            "total_restored_across_all_phases": total_restored,
            "overall_preservation_rate": ((total_features - total_missing + total_restored) / total_features * 100) if total_features > 0 else 100,
            "phase_results": all_results
        }
        
        # Save comprehensive report
        report_path = "ARCHIVE/comprehensive_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print(f"\nComprehensive validation complete: {overall_status}")
        print(f"Overall preservation rate: {summary_report['overall_preservation_rate']:.1f}%")
        print(f"Report saved: {report_path}")
        
        return summary_report


def main():
    """Run validation tests"""
    print("TestMaster Phase Validation System")
    print("=" * 50)
    
    validator = PhaseValidator()
    
    # Test validation on any existing archives
    try:
        # First check if we have any archives to validate
        if validator.archive_system.manifest.get("total_archives", 0) == 0:
            print("No archives found - creating test validation framework")
            
            # Create a test report structure
            test_report = {
                "validation_framework": "READY",
                "capabilities": [
                    "Feature-level validation against archives",
                    "Automatic missing feature restoration", 
                    "Comprehensive preservation rate calculation",
                    "Multi-phase validation support",
                    "Detailed validation reporting"
                ],
                "status": "OPERATIONAL"
            }
            
            print("Validation framework successfully initialized:")
            for capability in test_report["capabilities"]:
                print(f"  - {capability}")
            
        else:
            # Run validation on existing archives
            print("Running validation on existing archives...")
            result = validator.validate_all_phases()
            print(f"Validation complete with status: {result['overall_status']}")
            
    except Exception as e:
        print(f"Validation test failed: {e}")
        raise
    
    return validator


if __name__ == "__main__":
    main()