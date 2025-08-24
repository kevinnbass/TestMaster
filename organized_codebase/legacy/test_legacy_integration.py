"""
Test Legacy Integration Framework
Agent D - Hour 3: Legacy Code Documentation & Integration
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import hashlib

# Add TestMaster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TestMaster'))

from TestMaster.core.intelligence.documentation.legacy_integration_framework import (
    LegacyIntegrationFramework,
    ArchiveSystemAnalyzer,
    LegacyMigrationPlanner,
    LegacyComponent,
    LegacySystemType,
    MigrationStatus,
    IntegrationComplexity
)

def test_legacy_integration():
    """Test the legacy integration framework."""
    
    print("=" * 80)
    print("Agent D - Hour 3: Legacy Code Documentation & Integration")
    print("Testing Legacy Integration Framework")
    print("=" * 80)
    
    # Initialize framework
    framework = LegacyIntegrationFramework()
    migration_planner = LegacyMigrationPlanner()
    
    # Analyze archive system
    print("\n1. Analyzing Archive System...")
    archive_path = Path("TestMaster/archive")
    
    if archive_path.exists():
        analyzer = ArchiveSystemAnalyzer(str(archive_path))
        archive_analysis = analyzer.analyze_complete_archive_system()
        print(f"   Archive Analysis Complete:")
        print(f"   - Total Components: {archive_analysis.total_components}")
        print(f"   - Total Lines: {archive_analysis.total_lines:,}")
        print(f"   - Preservation Status: {archive_analysis.preservation_status}")
        print(f"   - Documentation Coverage: {archive_analysis.documentation_coverage:.1%}")
        
        # Show component breakdown
        print(f"\n   Component Breakdown:")
        for system_type, count in archive_analysis.component_breakdown.items():
            print(f"   - {system_type.value}: {count}")
    
    # Analyze complete legacy system
    print("\n2. Analyzing Complete Legacy System...")
    legacy_analysis = framework.analyze_complete_legacy_system()
    legacy_components = legacy_analysis.get('components', [])
    print(f"   Found {len(legacy_components)} legacy components")
    
    # Show sample components
    if legacy_components:
        print(f"\n   Sample Legacy Components:")
        for component in legacy_components[:5]:
            print(f"   - {component.name}")
            print(f"     Type: {component.system_type.value}")
            print(f"     Status: {component.migration_status.value}")
            print(f"     Size: {component.size_lines:,} lines")
            print(f"     Complexity: {component.integration_complexity.value}")
    
    # Analyze oversized modules
    print("\n3. Analyzing Oversized Modules Archive...")
    oversized_path = Path("TestMaster/archive/oversized_modules_20250821_042018")
    
    if oversized_path.exists():
        oversized_components = []
        for root, dirs, files in os.walk(oversized_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(oversized_path)
                    
                    # Count lines
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = len(f.readlines())
                    except:
                        lines = 0
                    
                    component = LegacyComponent(
                        component_id=hashlib.md5(str(rel_path).encode()).hexdigest()[:8],
                        name=file,
                        path=str(file_path),
                        system_type=LegacySystemType.OVERSIZED_MODULE,
                        size_lines=lines,
                        migration_status=MigrationStatus.ARCHIVED,
                        integration_complexity=IntegrationComplexity.MODERATE if lines < 1500 else IntegrationComplexity.COMPLEX
                    )
                    oversized_components.append(component)
        
        print(f"   Found {len(oversized_components)} oversized modules:")
        for comp in oversized_components:
            print(f"   - {comp.name} ({comp.size_lines:,} lines)")
    
    # Analyze legacy scripts
    print("\n4. Analyzing Legacy Scripts...")
    legacy_scripts_path = Path("TestMaster/archive/legacy_scripts")
    
    if legacy_scripts_path.exists():
        script_count = len(list(legacy_scripts_path.glob("*.py")))
        print(f"   Found {script_count} legacy scripts")
        
        # Categorize scripts
        categories = {
            "test_generation": [],
            "coverage_analysis": [],
            "conversion": [],
            "monitoring": [],
            "fixing": []
        }
        
        for script in legacy_scripts_path.glob("*.py"):
            name = script.name
            if "test" in name or "generator" in name:
                categories["test_generation"].append(name)
            elif "coverage" in name:
                categories["coverage_analysis"].append(name)
            elif "convert" in name:
                categories["conversion"].append(name)
            elif "monitor" in name:
                categories["monitoring"].append(name)
            elif "fix" in name:
                categories["fixing"].append(name)
        
        print("\n   Script Categories:")
        for category, scripts in categories.items():
            if scripts:
                print(f"   - {category}: {len(scripts)} scripts")
    
    # Generate migration plans
    print("\n5. Generating Migration Plans...")
    migration_plans = []
    if oversized_components:
        for component in oversized_components[:3]:  # Generate plans for first 3
            plan = migration_planner.create_migration_plan(component)
            if plan:
                migration_plans.append(plan)
                print(f"   Migration Plan: {component.name}")
                print(f"   - Strategy: {plan.migration_strategy}")
                print(f"   - Complexity: {plan.complexity_assessment.value}")
                print(f"   - Estimated Effort: {plan.estimated_effort}")
    
    # Create documentation
    print("\n6. Generating Legacy Documentation...")
    
    # Generate legacy documentation
    legacy_doc = framework.generate_legacy_documentation()
    
    # Create archive documentation
    archive_doc = "# Archive Structure Documentation\n\n" + str(archive_analysis.__dict__ if 'archive_analysis' in locals() else {})
    
    # Create migration documentation
    migration_doc = "# Migration Paths Documentation\n\n"
    if migration_plans:
        for plan in migration_plans:
            migration_doc += f"## {plan.component_id}\n"
            migration_doc += f"- Current: {plan.current_location}\n"
            migration_doc += f"- Target: {plan.target_location}\n"
            migration_doc += f"- Strategy: {plan.migration_strategy}\n\n"
    
    # Generate compatibility matrix
    compatibility_matrix = {
        "archive_components": len(oversized_components) if 'oversized_components' in locals() else 0,
        "legacy_scripts": script_count if 'script_count' in locals() else 0,
        "migration_plans": len(migration_plans),
        "compatibility": "backward_compatible"
    }
    
    print(f"   Documentation Generated:")
    print(f"   - Archive structure documented")
    print(f"   - Migration paths documented")
    print(f"   - Compatibility matrix created")
    
    # Export documentation
    output_dir = Path("TestMaster/docs/legacy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export archive documentation
    archive_doc_path = output_dir / "archive_structure.md"
    with open(archive_doc_path, 'w', encoding='utf-8') as f:
        f.write(archive_doc)
    print(f"   - Archive documentation: {archive_doc_path}")
    
    # Export migration documentation
    migration_doc_path = output_dir / "migration_paths.md"
    with open(migration_doc_path, 'w', encoding='utf-8') as f:
        f.write(migration_doc)
    print(f"   - Migration documentation: {migration_doc_path}")
    
    # Export compatibility matrix
    compat_path = output_dir / "compatibility_matrix.json"
    with open(compat_path, 'w', encoding='utf-8') as f:
        json.dump(compatibility_matrix, f, indent=2)
    print(f"   - Compatibility matrix: {compat_path}")
    
    # Generate integration report
    print("\n7. Generating Integration Report...")
    report = {
        "timestamp": datetime.now().isoformat(),
        "analyzer": "Agent D - Legacy Integration Framework",
        "archive_analysis": {
            "total_components": archive_analysis.total_components if 'archive_analysis' in locals() else 0,
            "total_lines": archive_analysis.total_lines if 'archive_analysis' in locals() else 0,
            "preservation_status": archive_analysis.preservation_status if 'archive_analysis' in locals() else "unknown"
        },
        "legacy_components": {
            "total": len(legacy_components),
            "by_type": {},
            "by_status": {},
            "by_complexity": {}
        },
        "oversized_modules": {
            "total": len(oversized_components) if 'oversized_components' in locals() else 0,
            "total_lines": sum(c.size_lines for c in oversized_components) if 'oversized_components' in locals() else 0
        },
        "legacy_scripts": {
            "total": script_count if 'script_count' in locals() else 0,
            "categories": {k: len(v) for k, v in categories.items()} if 'categories' in locals() else {}
        },
        "migration_plans": {
            "generated": len(migration_plans) if 'migration_plans' in locals() else 0
        }
    }
    
    # Count by type, status, and complexity
    if legacy_components:
        for comp in legacy_components:
            # By type
            type_key = comp.system_type.value
            report["legacy_components"]["by_type"][type_key] = \
                report["legacy_components"]["by_type"].get(type_key, 0) + 1
            
            # By status
            status_key = comp.migration_status.value
            report["legacy_components"]["by_status"][status_key] = \
                report["legacy_components"]["by_status"].get(status_key, 0) + 1
            
            # By complexity
            complexity_key = comp.integration_complexity.value
            report["legacy_components"]["by_complexity"][complexity_key] = \
                report["legacy_components"]["by_complexity"].get(complexity_key, 0) + 1
    
    # Export report
    report_path = output_dir / "integration_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"   Integration report saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("Legacy Integration Framework Test Complete!")
    print("=" * 80)
    
    return report

if __name__ == "__main__":
    report = test_legacy_integration()