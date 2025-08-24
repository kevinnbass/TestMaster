"""
Feature Discovery System
========================

Discovers all duplicate features for consolidation based on the roadmap.
This will find the 15 duplicate features across different files.

Author: TestMaster Team
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FeatureLocation:
    """Represents where a feature is found"""
    file_path: str
    line_number: int
    feature_type: str  # 'function', 'class', 'method', 'api_endpoint'
    feature_name: str
    signature: str
    context: str  # Additional context about the feature


class FeatureDiscovery:
    """Discover all duplicate features for consolidation"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.duplicates = {}
        self.feature_map = defaultdict(list)
        
    def discover_duplicates(self) -> Dict[str, List[str]]:
        """Main entry point - discover all duplicate feature groups"""
        print("Starting comprehensive feature discovery...")
        
        # Predefined duplicate groups from roadmap analysis
        duplicates = {
            "performance_monitoring": [
                "core/observability/agent_ops.py",
                "monitoring/enhanced_monitor.py", 
                "dashboard/api/analytics/performance_analyzer.py"
            ],
            "agent_orchestration": [
                "core/orchestration/agent_graph.py",
                "deployment/swarm_orchestrator.py",
                "dashboard/api/swarm_orchestration.py"
            ],
            "dashboard_widgets": [
                "ui_ux/interactive_dashboard.py",
                "dashboard/src/components/",
                "monitoring/enhanced_monitor.py"
            ],
            "state_management": [
                "agents/team/testing_team.py",
                "deployment/enterprise_deployment.py",
                "core/orchestration/agent_graph.py"
            ],
            "api_endpoints": [
                "api/orchestration_api.py",
                "api/phase2_api.py", 
                "api/phase3_api.py",
                "dashboard/api/"
            ]
        }
        
        # Verify each file/directory exists and analyze features
        verified_duplicates = {}
        for group_name, file_list in duplicates.items():
            verified_files = []
            group_features = []
            
            for file_path in file_list:
                full_path = self.base_path / file_path
                if full_path.exists():
                    verified_files.append(file_path)
                    if full_path.is_file():
                        features = self._analyze_file(full_path, group_name)
                        group_features.extend(features)
                    elif full_path.is_dir():
                        features = self._analyze_directory(full_path, group_name)
                        group_features.extend(features)
                else:
                    print(f"   WARNING: {file_path} not found - skipping")
            
            if verified_files:
                verified_duplicates[group_name] = {
                    "files": verified_files,
                    "features": group_features,
                    "duplicate_count": len(verified_files)
                }
                print(f"   SUCCESS: {group_name}: {len(verified_files)} locations, {len(group_features)} features")
        
        self.duplicates = verified_duplicates
        return verified_duplicates
    
    def _analyze_file(self, file_path: Path, group_name: str) -> List[FeatureLocation]:
        """Analyze a single Python file for features"""
        features = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    signature = f"def {node.name}({', '.join(arg.arg for arg in node.args.args)})"
                    features.append(FeatureLocation(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        feature_type="function",
                        feature_name=node.name,
                        signature=signature,
                        context=f"Group: {group_name}"
                    ))
                
                elif isinstance(node, ast.ClassDef):
                    methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    signature = f"class {node.name}"
                    features.append(FeatureLocation(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        feature_type="class",
                        feature_name=node.name,
                        signature=signature,
                        context=f"Group: {group_name}, Methods: {methods}"
                    ))
            
            # Look for API endpoints (Flask routes, FastAPI endpoints)
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line.startswith('@app.route') or line.startswith('@router.') or 'endpoint' in line.lower():
                    features.append(FeatureLocation(
                        file_path=str(file_path),
                        line_number=line_num,
                        feature_type="api_endpoint",
                        feature_name=line[:50] + "..." if len(line) > 50 else line,
                        signature=line,
                        context=f"Group: {group_name}, API endpoint"
                    ))
                        
        except Exception as e:
            print(f"   WARNING: Could not analyze {file_path}: {e}")
        
        return features
    
    def _analyze_directory(self, dir_path: Path, group_name: str) -> List[FeatureLocation]:
        """Analyze all Python files in a directory"""
        features = []
        
        for file_path in dir_path.rglob("*.py"):
            file_features = self._analyze_file(file_path, group_name)
            features.extend(file_features)
        
        return features
    
    def find_functional_duplicates(self) -> Dict[str, List[FeatureLocation]]:
        """Find features with similar functionality across files"""
        functional_duplicates = defaultdict(list)
        
        # Build feature map
        all_features = []
        for group_data in self.duplicates.values():
            all_features.extend(group_data["features"])
        
        # Group by similar names (simple heuristic)
        for feature in all_features:
            # Look for similar function/class names
            base_name = self._normalize_name(feature.feature_name)
            functional_duplicates[base_name].append(feature)
        
        # Filter to only actual duplicates (more than one occurrence)
        actual_duplicates = {
            name: locations for name, locations in functional_duplicates.items()
            if len(locations) > 1
        }
        
        return actual_duplicates
    
    def _normalize_name(self, name: str) -> str:
        """Normalize function/class names to find similar ones"""
        # Remove common prefixes/suffixes
        name = name.lower()
        for prefix in ['test_', 'enhanced_', 'unified_', 'improved_']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        for suffix in ['_v2', '_enhanced', '_improved', '_new']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        return name
    
    def generate_consolidation_plan(self) -> Dict[str, Any]:
        """Generate detailed consolidation plan"""
        plan = {
            "timestamp": "2025-08-20",
            "total_duplicate_groups": len(self.duplicates),
            "consolidation_targets": {},
            "archive_requirements": [],
            "validation_checks": []
        }
        
        # Create consolidation targets for each group
        for group_name, group_data in self.duplicates.items():
            target_file = self._get_consolidation_target(group_name)
            
            plan["consolidation_targets"][group_name] = {
                "target_file": target_file,
                "source_files": group_data["files"],
                "features_to_merge": len(group_data["features"]),
                "priority": self._get_priority(group_name)
            }
            
            # Archive requirements
            for file_path in group_data["files"]:
                plan["archive_requirements"].append({
                    "file": file_path,
                    "group": group_name,
                    "reason": f"Consolidation of {group_name}"
                })
            
            # Validation checks
            plan["validation_checks"].append({
                "group": group_name,
                "feature_count": len(group_data["features"]),
                "files_count": len(group_data["files"])
            })
        
        return plan
    
    def _get_consolidation_target(self, group_name: str) -> str:
        """Determine target file for consolidation"""
        targets = {
            "performance_monitoring": "observability/unified_monitor.py",
            "agent_orchestration": "orchestration/unified_orchestrator.py",
            "dashboard_widgets": "ui/unified_dashboard.py",
            "state_management": "state/unified_state.py",
            "api_endpoints": "api/unified_api.py"
        }
        return targets.get(group_name, f"consolidated/{group_name}.py")
    
    def _get_priority(self, group_name: str) -> int:
        """Get consolidation priority (1=highest, 5=lowest)"""
        priorities = {
            "performance_monitoring": 1,  # Critical for system health
            "state_management": 2,        # Core system state
            "agent_orchestration": 3,     # Core orchestration
            "api_endpoints": 4,           # External interfaces
            "dashboard_widgets": 5        # UI components
        }
        return priorities.get(group_name, 3)
    
    def save_discovery_report(self, output_path: str = "consolidation/discovery_report.json"):
        """Save comprehensive discovery report"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert FeatureLocation objects to dicts for JSON serialization
        serializable_duplicates = {}
        for group_name, group_data in self.duplicates.items():
            serializable_features = []
            for feature in group_data["features"]:
                serializable_features.append({
                    "file_path": feature.file_path,
                    "line_number": feature.line_number,
                    "feature_type": feature.feature_type,
                    "feature_name": feature.feature_name,
                    "signature": feature.signature,
                    "context": feature.context
                })
            
            serializable_duplicates[group_name] = {
                "files": group_data["files"],
                "features": serializable_features,
                "duplicate_count": group_data["duplicate_count"]
            }
        
        report = {
            "discovery_timestamp": "2025-08-20",
            "total_groups": len(self.duplicates),
            "duplicate_groups": serializable_duplicates,
            "consolidation_plan": self.generate_consolidation_plan(),
            "functional_duplicates": self._serialize_functional_duplicates(),
            "summary": {
                "total_files_with_duplicates": sum(
                    len(group["files"]) for group in self.duplicates.values()
                ),
                "total_features_found": sum(
                    len(group["features"]) for group in self.duplicates.values()
                ),
                "highest_priority_group": min(
                    self.duplicates.keys(),
                    key=lambda x: self._get_priority(x)
                ) if self.duplicates else None
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Discovery report saved to: {output_path}")
        return output_path
    
    def _serialize_functional_duplicates(self) -> Dict[str, List[Dict]]:
        """Serialize functional duplicates for JSON output"""
        functional_dups = self.find_functional_duplicates()
        serialized = {}
        
        for name, locations in functional_dups.items():
            serialized[name] = []
            for loc in locations:
                serialized[name].append({
                    "file_path": loc.file_path,
                    "line_number": loc.line_number,
                    "feature_type": loc.feature_type,
                    "feature_name": loc.feature_name,
                    "signature": loc.signature,
                    "context": loc.context
                })
        
        return serialized


def main():
    """Run feature discovery and generate report"""
    print("TestMaster Feature Discovery System")
    print("=" * 50)
    
    discovery = FeatureDiscovery()
    
    # Discover all duplicates
    duplicates = discovery.discover_duplicates()
    
    # Generate and save report
    report_path = discovery.save_discovery_report()
    
    # Print summary
    print("\nDISCOVERY SUMMARY")
    print("=" * 30)
    print(f"Duplicate groups found: {len(duplicates)}")
    
    for group_name, group_data in duplicates.items():
        print(f"  {group_name}:")
        print(f"    Files: {group_data['duplicate_count']}")
        print(f"    Features: {len(group_data['features'])}")
        print(f"    Priority: {discovery._get_priority(group_name)}")
    
    print(f"\nFull report: {report_path}")
    
    return discovery


if __name__ == "__main__":
    main()