#!/usr/bin/env python3
"""
Reorganization Planner Coordinator
==================================

Coordinates the reorganization planning using specialized modules.
Creates actionable reorganization plans based on integrated intelligence.
"""

# Import specialized modules
from reorganization_core import ReorganizationPlanner as CoreReorganizationPlanner
from reorganization_data import DetailedReorganizationPlan


def create_reorganization_plan(llm_intelligence_map: dict, integrated_intelligence: list) -> DetailedReorganizationPlan:
    """
    Create a detailed reorganization plan from integrated intelligence.

    Args:
        llm_intelligence_map: Raw LLM intelligence map
        integrated_intelligence: Processed integrated intelligence

    Returns:
        Detailed reorganization plan with executable tasks
    """
    # Initialize planner with current directory
    from pathlib import Path
    planner = CoreReorganizationPlanner(Path.cwd())
    return planner.create_reorganization_plan(llm_intelligence_map, integrated_intelligence)


def main():
    """
    Main function demonstrating reorganization planning capabilities.

    This function showcases the reorganization planning functionality
    and serves as an example of how to use the reorganization planner.
    """
    print("📋 REORGANIZATION PLANNER")
    print("========================")
    print("Creates actionable reorganization plans based on integrated intelligence")
    print()

    # Example usage (would normally have real intelligence data)
    sample_intelligence_map = {
        'scan_id': 'example_scan_001',
        'timestamp': '2024-01-01T12:00:00',
        'total_files': 150,
        'high_confidence_modules': 45,
        'medium_confidence_modules': 85,
        'low_confidence_modules': 20
    }

    sample_integrated_intelligence = [
        # This would normally contain actual IntegratedIntelligence objects
        # For demonstration, we'll just show the structure
    ]

    print("📊 Sample Intelligence Data:")
    print(f"   Scan ID: {sample_intelligence_map['scan_id']}")
    print(f"   Total Files: {sample_intelligence_map['total_files']}")
    print(f"   High Confidence Modules: {sample_intelligence_map['high_confidence_modules']}")
    print(f"   Medium Confidence Modules: {sample_intelligence_map['medium_confidence_modules']}")
    print(f"   Low Confidence Modules: {sample_intelligence_map['low_confidence_modules']}")
    print()

    print("🔧 Reorganization Planning Capabilities:")
    print("   • Confidence-based decision making")
    print("   • Risk assessment and mitigation")
    print("   • Phased implementation strategies")
    print("   • Automated task generation")
    print("   • Batch execution planning")
    print("   • Rollback planning")
    print("   • Success metrics tracking")
    print()

    print("📝 Key Features:")
    print("   • Creates executable reorganization plans")
    print("   • Groups tasks into risk-appropriate batches")
    print("   • Provides detailed execution guidelines")
    print("   • Includes comprehensive risk mitigation")
    print("   • Supports dry-run execution")
    print("   • Generates success metrics")
    print()

    print("✅ Reorganization planner ready for use!")
    print()
    print("To use with real intelligence data:")
    print("   plan = create_reorganization_plan(intelligence_map, integrated_intelligence)")
    print("   planner.save_plan(plan, Path('reorganization_plan.json'))")


if __name__ == "__main__":
    main()

