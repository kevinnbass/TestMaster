#!/usr/bin/env python3
"""
Comprehensive Feature Audit System
==================================

Deep analysis to verify which features are truly lost vs. renamed/consolidated.
Only recommends restoration for genuinely missing functionality.

Author: TestMaster Analysis System
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FeatureSignature:
    """Detailed signature of a code feature"""
    name: str
    feature_type: str  # class, function, method, constant, enum
    signature: str
    docstring: Optional[str] = None
    complexity_score: int = 0
    dependencies: Set[str] = field(default_factory=set)
    functionality_hash: str = ""
    usage_patterns: List[str] = field(default_factory=list)


@dataclass
class FeatureMapping:
    """Maps archived features to consolidated equivalents"""
    archived_feature: FeatureSignature
    consolidated_equivalent: Optional[FeatureSignature] = None
    mapping_confidence: float = 0.0
    mapping_type: str = "unknown"  # exact, renamed, consolidated, enhanced, missing
    justification: str = ""


class ComprehensiveFeatureAuditor:
    """Systematic analysis to verify truly lost features"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.archive_base = self.base_path / "archive"
        
        # Consolidated systems to analyze
        self.consolidated_systems = {
            "observability": [
                "observability/unified_monitor.py",
                "core/observability/agent_ops.py"
            ],
            "state_config": [
                "state/unified_state_manager.py", 
                "config/yaml_config_enhancer.py"
            ],
            "orchestration": [
                "orchestration/unified_orchestrator.py",
                "orchestration/swarm_router_enhancement.py"
            ],
            "ui_dashboard": [
                "ui/unified_dashboard.py",
                "ui/nocode_enhancement.py",
                "dashboard/src/components/OrchestrationDashboard.jsx"
            ]
        }
        
        # Feature extraction results
        self.archived_features: Dict[str, List[FeatureSignature]] = {}
        self.consolidated_features: Dict[str, List[FeatureSignature]] = {}
        self.feature_mappings: List[FeatureMapping] = []
        
    def perform_comprehensive_audit(self) -> Dict[str, Any]:
        """Perform complete feature audit"""
        print("COMPREHENSIVE FEATURE AUDIT")
        print("=" * 80)
        print("Systematic verification of truly lost vs. consolidated features")
        print()
        
        audit_results = {
            "audit_timestamp": "2025-08-19T21:00:00",
            "analysis_type": "COMPREHENSIVE_VERIFICATION",
            "phases_analyzed": {},
            "truly_missing_features": [],
            "renamed_features": [],
            "consolidated_features": [],
            "enhanced_features": [],
            "restoration_recommendations": []
        }
        
        # Analyze each consolidation phase
        phases = [
            ("C4", "phase4_archive", "observability"),
            ("C5", "phase5_archive", "state_config"), 
            ("C6", "phase6_archive", "orchestration"),
            ("C7", "phase7_archive", "ui_dashboard")
        ]
        
        for phase_id, archive_dir, system_key in phases:
            print(f"Analyzing Phase {phase_id}: {archive_dir}")
            print("-" * 60)
            
            phase_result = self._analyze_phase(phase_id, archive_dir, system_key)
            audit_results["phases_analyzed"][phase_id] = phase_result
            
            # Accumulate results
            audit_results["truly_missing_features"].extend(phase_result["missing_features"])
            audit_results["renamed_features"].extend(phase_result["renamed_features"])
            audit_results["consolidated_features"].extend(phase_result["consolidated_features"])
            audit_results["enhanced_features"].extend(phase_result["enhanced_features"])
            
            print()
        
        # Generate final assessment
        audit_results.update(self._generate_final_assessment(audit_results))
        
        # Save detailed results
        self._save_audit_results(audit_results)
        
        return audit_results
    
    def _analyze_phase(self, phase_id: str, archive_dir: str, system_key: str) -> Dict[str, Any]:
        """Analyze a specific consolidation phase"""
        phase_result = {
            "phase_id": phase_id,
            "archive_files_analyzed": 0,
            "consolidated_files_analyzed": 0,
            "total_archived_features": 0,
            "missing_features": [],
            "renamed_features": [],
            "consolidated_features": [],
            "enhanced_features": [],
            "mapping_confidence_avg": 0.0
        }
        
        # Extract archived features
        archive_path = self.archive_base / archive_dir / "replaced_code"
        archived_features = self._extract_archived_features(archive_path)
        
        # Extract consolidated features
        consolidated_features = self._extract_consolidated_features(system_key)
        
        if not archived_features:
            print(f"  No archived features found in {archive_path}")
            return phase_result
        
        print(f"  Found {len(archived_features)} archived features")
        print(f"  Found {len(consolidated_features)} consolidated features")
        
        # Perform intelligent feature mapping
        mappings = self._perform_intelligent_mapping(archived_features, consolidated_features)
        
        # Categorize mappings
        for mapping in mappings:
            if mapping.mapping_type == "missing":
                phase_result["missing_features"].append({
                    "name": mapping.archived_feature.name,
                    "type": mapping.archived_feature.feature_type,
                    "signature": mapping.archived_feature.signature,
                    "complexity": mapping.archived_feature.complexity_score,
                    "justification": mapping.justification
                })
            elif mapping.mapping_type == "renamed":
                phase_result["renamed_features"].append({
                    "original": mapping.archived_feature.name,
                    "new": mapping.consolidated_equivalent.name if mapping.consolidated_equivalent else "unknown",
                    "confidence": mapping.mapping_confidence
                })
            elif mapping.mapping_type == "consolidated":
                phase_result["consolidated_features"].append({
                    "original": mapping.archived_feature.name,
                    "consolidated_into": mapping.consolidated_equivalent.name if mapping.consolidated_equivalent else "unknown",
                    "confidence": mapping.mapping_confidence
                })
            elif mapping.mapping_type == "enhanced":
                phase_result["enhanced_features"].append({
                    "original": mapping.archived_feature.name,
                    "enhanced_as": mapping.consolidated_equivalent.name if mapping.consolidated_equivalent else "unknown",
                    "confidence": mapping.mapping_confidence
                })
        
        # Calculate statistics
        phase_result["total_archived_features"] = len(archived_features)
        phase_result["mapping_confidence_avg"] = sum(m.mapping_confidence for m in mappings) / max(1, len(mappings))
        
        # Report findings
        missing_count = len(phase_result["missing_features"])
        total_count = len(archived_features)
        preservation_rate = ((total_count - missing_count) / total_count) * 100 if total_count > 0 else 100
        
        print(f"  Feature Analysis Results:")
        print(f"    Total archived features: {total_count}")
        print(f"    Truly missing features: {missing_count}")
        print(f"    Preservation rate: {preservation_rate:.1f}%")
        print(f"    Average mapping confidence: {phase_result['mapping_confidence_avg']:.2f}")
        
        if missing_count > 0:
            print(f"  Potentially missing features:")
            for missing in phase_result["missing_features"][:3]:
                print(f"    - {missing['type']}: {missing['name']} (complexity: {missing['complexity']})")
            if missing_count > 3:
                print(f"    ... and {missing_count - 3} more")
        
        return phase_result
    
    def _extract_archived_features(self, archive_path: Path) -> List[FeatureSignature]:
        """Extract features from archived files"""
        features = []
        
        if not archive_path.exists():
            return features
        
        # Analyze Python files
        for py_file in archive_path.glob("*.py"):
            features.extend(self._extract_python_features(py_file))
        
        # Analyze JavaScript/JSX files
        for js_file in list(archive_path.glob("*.js")) + list(archive_path.glob("*.jsx")):
            features.extend(self._extract_js_features(js_file))
        
        return features
    
    def _extract_consolidated_features(self, system_key: str) -> List[FeatureSignature]:
        """Extract features from consolidated systems"""
        features = []
        
        for file_path in self.consolidated_systems.get(system_key, []):
            full_path = self.base_path / file_path
            if full_path.exists():
                if full_path.suffix == ".py":
                    features.extend(self._extract_python_features(full_path))
                elif full_path.suffix in [".js", ".jsx"]:
                    features.extend(self._extract_js_features(full_path))
        
        return features
    
    def _extract_python_features(self, file_path: Path) -> List[FeatureSignature]:
        """Extract detailed Python features"""
        features = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                feature = None
                
                if isinstance(node, ast.ClassDef):
                    feature = FeatureSignature(
                        name=node.name,
                        feature_type="class",
                        signature=self._generate_class_signature(node),
                        docstring=ast.get_docstring(node),
                        complexity_score=self._calculate_complexity(node)
                    )
                
                elif isinstance(node, ast.FunctionDef):
                    feature = FeatureSignature(
                        name=node.name,
                        feature_type="function",
                        signature=self._generate_function_signature(node),
                        docstring=ast.get_docstring(node),
                        complexity_score=self._calculate_complexity(node)
                    )
                
                elif isinstance(node, ast.Assign):
                    # Extract constants and important assignments
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            feature = FeatureSignature(
                                name=target.id,
                                feature_type="constant",
                                signature=f"{target.id} = ...",
                                complexity_score=1
                            )
                
                if feature:
                    feature.functionality_hash = self._generate_functionality_hash(feature)
                    features.append(feature)
        
        except Exception as e:
            print(f"  Error analyzing {file_path}: {e}")
        
        return features
    
    def _extract_js_features(self, file_path: Path) -> List[FeatureSignature]:
        """Extract JavaScript/JSX features"""
        features = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract React components
            component_patterns = [
                (r'const\s+([A-Z][a-zA-Z0-9_]*)\s*=\s*\(', "component"),
                (r'function\s+([A-Z][a-zA-Z0-9_]*)\s*\(', "component"),
                (r'class\s+([A-Z][a-zA-Z0-9_]*)\s+extends', "component")
            ]
            
            for pattern, feature_type in component_patterns:
                for match in re.finditer(pattern, content):
                    feature = FeatureSignature(
                        name=match.group(1),
                        feature_type=feature_type,
                        signature=f"{feature_type}: {match.group(1)}",
                        complexity_score=self._estimate_js_complexity(content, match.group(1))
                    )
                    feature.functionality_hash = self._generate_functionality_hash(feature)
                    features.append(feature)
            
            # Extract functions
            function_patterns = [
                (r'const\s+([a-z][a-zA-Z0-9_]*)\s*=\s*\(', "function"),
                (r'function\s+([a-z][a-zA-Z0-9_]*)\s*\(', "function")
            ]
            
            for pattern, feature_type in function_patterns:
                for match in re.finditer(pattern, content):
                    feature = FeatureSignature(
                        name=match.group(1),
                        feature_type=feature_type,
                        signature=f"{feature_type}: {match.group(1)}",
                        complexity_score=self._estimate_js_complexity(content, match.group(1))
                    )
                    feature.functionality_hash = self._generate_functionality_hash(feature)
                    features.append(feature)
        
        except Exception as e:
            print(f"  Error analyzing {file_path}: {e}")
        
        return features
    
    def _perform_intelligent_mapping(self, archived_features: List[FeatureSignature], 
                                   consolidated_features: List[FeatureSignature]) -> List[FeatureMapping]:
        """Intelligent mapping between archived and consolidated features"""
        mappings = []
        
        for archived_feature in archived_features:
            best_match = None
            best_confidence = 0.0
            mapping_type = "missing"
            justification = ""
            
            for consolidated_feature in consolidated_features:
                confidence, match_type, reason = self._calculate_feature_similarity(
                    archived_feature, consolidated_feature
                )
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = consolidated_feature
                    mapping_type = match_type
                    justification = reason
            
            # Determine final mapping type based on confidence
            if best_confidence >= 0.9:
                final_type = "exact" if mapping_type == "exact" else "enhanced"
            elif best_confidence >= 0.7:
                final_type = "consolidated" if "consolidat" in justification.lower() else "renamed"
            elif best_confidence >= 0.4:
                final_type = "consolidated"
            else:
                final_type = "missing"
                justification = f"No suitable equivalent found (best match: {best_confidence:.2f})"
            
            mapping = FeatureMapping(
                archived_feature=archived_feature,
                consolidated_equivalent=best_match,
                mapping_confidence=best_confidence,
                mapping_type=final_type,
                justification=justification
            )
            
            mappings.append(mapping)
        
        return mappings
    
    def _calculate_feature_similarity(self, archived: FeatureSignature, 
                                    consolidated: FeatureSignature) -> Tuple[float, str, str]:
        """Calculate similarity between two features"""
        confidence = 0.0
        match_type = "unknown"
        reason = ""
        
        # Exact name match
        if archived.name == consolidated.name:
            confidence = 0.95
            match_type = "exact"
            reason = "Exact name match"
        
        # Similar name patterns
        elif self._names_are_similar(archived.name, consolidated.name):
            confidence = 0.8
            match_type = "renamed"
            reason = f"Similar name: {archived.name} -> {consolidated.name}"
        
        # Type compatibility
        elif archived.feature_type == consolidated.feature_type:
            confidence = 0.6
            match_type = "consolidated"
            reason = f"Same type, different name"
        
        # Functionality similarity (based on docstrings, signatures)
        elif self._functionality_similar(archived, consolidated):
            confidence = 0.7
            match_type = "consolidated"
            reason = "Similar functionality detected"
        
        # Enhanced version (more complex consolidated version)
        elif (consolidated.complexity_score > archived.complexity_score * 1.5 and
              archived.feature_type == consolidated.feature_type):
            confidence = 0.5
            match_type = "enhanced"
            reason = "Potentially enhanced version"
        
        return confidence, match_type, reason
    
    def _names_are_similar(self, name1: str, name2: str) -> bool:
        """Check if two names are similar (accounting for common renaming patterns)"""
        # Convert to lowercase for comparison
        n1, n2 = name1.lower(), name2.lower()
        
        # Common renaming patterns - check substring matches instead of regex backreferences
        pattern_pairs = [
            ('interactive', 'unified'),
            ('enhanced', 'unified'),
            ('realtime', 'widget'),
            ('panel', 'widget'),
            ('dashboard', 'unified'),
            ('monitor', 'unified')
        ]
        
        for old_word, new_word in pattern_pairs:
            if old_word in n1 and new_word in n2:
                return True
            if old_word in n2 and new_word in n1:
                return True
        
        # Levenshtein distance for similar names
        return self._levenshtein_ratio(n1, n2) > 0.7
    
    def _functionality_similar(self, feature1: FeatureSignature, feature2: FeatureSignature) -> bool:
        """Check if two features have similar functionality"""
        if not (feature1.docstring and feature2.docstring):
            return False
        
        # Simple keyword overlap check
        words1 = set(re.findall(r'\w+', feature1.docstring.lower()))
        words2 = set(re.findall(r'\w+', feature2.docstring.lower()))
        
        common_words = words1 & words2
        total_words = words1 | words2
        
        if len(total_words) == 0:
            return False
        
        overlap_ratio = len(common_words) / len(total_words)
        return overlap_ratio > 0.3
    
    def _generate_class_signature(self, node: ast.ClassDef) -> str:
        """Generate signature for a class"""
        bases = [ast.unparse(base) for base in node.bases] if node.bases else []
        base_str = f"({', '.join(bases)})" if bases else ""
        return f"class {node.name}{base_str}"
    
    def _generate_function_signature(self, node: ast.FunctionDef) -> str:
        """Generate signature for a function"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"def {node.name}({', '.join(args)})"
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate complexity score for an AST node"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.FunctionDef):
                complexity += 1
        
        return complexity
    
    def _estimate_js_complexity(self, content: str, feature_name: str) -> int:
        """Estimate complexity for JavaScript features"""
        # Simple heuristic based on keywords
        complexity_keywords = ['if', 'for', 'while', 'try', 'catch', 'function', 'class']
        
        # Find the feature's section in the content
        lines = content.split('\n')
        feature_lines = []
        in_feature = False
        brace_count = 0
        
        for line in lines:
            if feature_name in line and ('=' in line or 'function' in line):
                in_feature = True
            
            if in_feature:
                feature_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count <= 0 and in_feature and ('{' in ''.join(feature_lines)):
                    break
        
        feature_content = '\n'.join(feature_lines)
        complexity = 1
        
        for keyword in complexity_keywords:
            complexity += feature_content.count(keyword)
        
        return complexity
    
    def _generate_functionality_hash(self, feature: FeatureSignature) -> str:
        """Generate a hash representing the feature's functionality"""
        content = f"{feature.name}_{feature.feature_type}_{feature.signature}"
        if feature.docstring:
            content += f"_{feature.docstring[:100]}"
        
        return str(hash(content))
    
    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein ratio between two strings"""
        if len(s1) == 0:
            return 0.0 if len(s2) == 0 else 1.0
        if len(s2) == 0:
            return 1.0
        
        # Simple approximation
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
        
        # Count common characters
        common = sum(1 for c in shorter if c in longer)
        return common / len(longer)
    
    def _generate_final_assessment(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final assessment and restoration recommendations"""
        total_missing = len(audit_results["truly_missing_features"])
        total_analyzed = sum(
            phase["total_archived_features"] 
            for phase in audit_results["phases_analyzed"].values()
        )
        
        # Filter truly critical missing features
        critical_missing = []
        for feature in audit_results["truly_missing_features"]:
            if (feature["complexity"] > 5 or 
                feature["type"] in ["class", "component"] or
                any(keyword in feature["name"].lower() for keyword in 
                    ["critical", "essential", "core", "manager", "controller"])):
                critical_missing.append(feature)
        
        restoration_recommendations = []
        
        if len(critical_missing) > 0:
            restoration_recommendations.append({
                "priority": "HIGH",
                "action": "Restore critical missing features",
                "features": critical_missing[:5],  # Top 5 most critical
                "justification": "Features with high complexity or core functionality"
            })
        
        if total_missing > len(critical_missing):
            remaining_missing = total_missing - len(critical_missing)
            restoration_recommendations.append({
                "priority": "MEDIUM", 
                "action": "Review remaining missing features",
                "count": remaining_missing,
                "justification": "Lower complexity features that may be intentionally removed"
            })
        
        # Overall assessment
        if total_missing == 0:
            overall_status = "PERFECT_PRESERVATION"
        elif len(critical_missing) == 0:
            overall_status = "EXCELLENT_PRESERVATION"
        elif len(critical_missing) < 5:
            overall_status = "GOOD_PRESERVATION_MINOR_GAPS"
        else:
            overall_status = "NEEDS_TARGETED_RESTORATION"
        
        return {
            "overall_assessment": {
                "status": overall_status,
                "total_features_analyzed": total_analyzed,
                "total_missing_features": total_missing,
                "critical_missing_features": len(critical_missing),
                "preservation_rate": ((total_analyzed - total_missing) / max(1, total_analyzed)) * 100
            },
            "restoration_recommendations": restoration_recommendations,
            "summary": {
                "action_required": len(restoration_recommendations) > 0,
                "priority_actions": len([r for r in restoration_recommendations if r["priority"] == "HIGH"]),
                "confidence_level": "HIGH" if total_missing < total_analyzed * 0.05 else "MEDIUM"
            }
        }
    
    def _save_audit_results(self, audit_results: Dict[str, Any]):
        """Save comprehensive audit results"""
        # Save main results
        with open("comprehensive_feature_audit_results.json", 'w') as f:
            json.dump(audit_results, f, indent=2, default=str)
        
        # Generate human-readable report
        self._generate_human_readable_report(audit_results)
    
    def _generate_human_readable_report(self, audit_results: Dict[str, Any]):
        """Generate human-readable audit report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE FEATURE AUDIT REPORT")
        print("=" * 80)
        
        assessment = audit_results["overall_assessment"]
        print(f"Overall Status: {assessment['status']}")
        print(f"Preservation Rate: {assessment['preservation_rate']:.1f}%")
        print(f"Total Features Analyzed: {assessment['total_features_analyzed']}")
        print(f"Missing Features: {assessment['total_missing_features']}")
        print(f"Critical Missing: {assessment['critical_missing_features']}")
        print()
        
        if audit_results["restoration_recommendations"]:
            print("RESTORATION RECOMMENDATIONS:")
            print("-" * 40)
            for i, rec in enumerate(audit_results["restoration_recommendations"], 1):
                print(f"{i}. {rec['priority']} PRIORITY: {rec['action']}")
                print(f"   Justification: {rec['justification']}")
                
                if "features" in rec:
                    print(f"   Features to restore:")
                    for feature in rec["features"]:
                        print(f"     - {feature['type']}: {feature['name']} (complexity: {feature['complexity']})")
                elif "count" in rec:
                    print(f"   Additional features: {rec['count']}")
                print()
        else:
            print("NO RESTORATION NEEDED - All features properly preserved!")
        
        print(f"Summary: {'ACTION REQUIRED' if audit_results['summary']['action_required'] else 'NO ACTION NEEDED'}")
        print(f"Confidence Level: {audit_results['summary']['confidence_level']}")
        
        print("\nDetailed audit results saved to: comprehensive_feature_audit_results.json")


def main():
    """Run comprehensive feature audit"""
    auditor = ComprehensiveFeatureAuditor()
    results = auditor.perform_comprehensive_audit()
    
    # Return exit code based on whether restoration is needed
    return 1 if results["summary"]["action_required"] else 0


if __name__ == "__main__":
    exit(main())